"""Document converter using markitdown.

Converts PDF, DOCX, PPTX, XLSX, HTML and other formats to markdown.
.md and .txt files pass through directly.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions that pass through without conversion
PASSTHROUGH_EXTS = {".md", ".txt", ".markdown"}

# Extensions that markitdown can handle
CONVERTIBLE_EXTS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm", ".csv",
    ".json", ".xml", ".zip",
}

# Minimum font size difference (points) between body text and a heading
_HEADING_SIZE_THRESHOLD = 2.0
# Minimum text length for a heading candidate (avoid noise)
_HEADING_MIN_TEXT_LEN = 2
# Maximum heading levels to emit (H1–H6)
_MAX_HEADING_LEVELS = 6

# ---------------------------------------------------------------------------
# Heading-tree validation thresholds
#
# Calibrated against ONE real Oracle manual (tests/fixtures/test-doc.pdf,
# a 350-page Oracle RAG whitepaper) and four synthetic fixture PDFs.
# Real FCCS PDFs were NOT available in any configured corpus at calibration
# time — thresholds are marked PROVISIONAL where real-FCCS data is missing.
# See docs/heading-validation.md for the full calibration report.
#
# Measurements on available documents:
#
#   Document              Collapse  Saturation  MinGap  OrderAnomaly  Valid?
#   ───────────────────   ────────  ──────────  ──────  ────────────  ───────
#   test-doc.pdf (real)     1.00*     0.00       8.0     False         YES
#   deep-legit.pdf          1.00*     0.00       1.5     False         YES
#   good-hierarchy.pdf      1.00*     0.00       4.0     False         YES
#   oracle-manual-sim       1.00*     0.20       0.5     True          NO
#   oracle-template-sim     1.00**    0.00       N/A     False         NO
#
# * Collapse ratio = fraction of H2 sections with NO H3+ children.
#   All three "good" docs have 0 H3+ headings, so their collapse ratio is
#   1.0 — but they are still valid because they have ≤2 heading levels
#   (a flat but correct hierarchy). The signal only fires when there are
#   ≥3 heading levels AND >50% of H2 sections lack H3 children.
#
# ** oracle-template-sim has only 1 heading level (all H1) — collapse = 1.0
#    by the "no H2 headings" rule.
#
# A heading tree is considered INVALID if ANY of these signals fire.
# When invalid, the converter falls back to markitdown for that document.
# ---------------------------------------------------------------------------

# Signal 1: Collapse ratio. Two sub-checks:
#   (a) No H2 headings at all → collapse = 1.0 (everything at one level).
#   (b) ≥3 heading levels exist but >50% of H2 sections have no H3+
#       children → the tree is flat where it should be deep.
# Sub-check (a) catches single-level docs (oracle-template-sim).
# Sub-check (b) catches the Ch.21 failure mode where many sizes map to
# levels but H2 sections don't nest into H3 subsections.
_COLLAPSE_RATIO_THRESHOLD = 0.50

# Signal 2: If >15% of all headings are at the max level (H6), the level
# mapping is saturating — too many distinct sizes crammed into 6 levels.
# PROVISIONAL: calibrated on oracle-manual-sim (0.20) vs good docs (0.00).
# Real FCCS measurement needed.
_DEPTH_SATURATION_THRESHOLD = 0.15

# Signal 3: If any two adjacent heading levels have font sizes within 1.0pt,
# the levels are too close to distinguish reliably (size clustering).
# Calibrated: oracle-manual-sim has 0.5pt gaps; deep-legit has 1.5pt;
# test-doc has 8.0pt; good-hierarchy has 4.0pt.
_MIN_SIZE_GAP_THRESHOLD = 1.0

# Signal 4: If the deepest heading level appears in the document before any
# shallower heading (e.g. H6 before H1), the level ordering is wrong.
# This is a boolean check — any such anomaly fails.
# Calibrated: oracle-manual-sim has H6 as first heading; all good docs
# start with H1 or H2.


class Converter:
    """Document converter using markitdown."""

    def __init__(self, cache_dir: str | None = None):
        """
        Args:
            cache_dir: Directory to cache converted output. If None, no caching.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._markitdown = None
        self._pymupdf_available = None

    @property
    def markitdown(self):
        """Lazy-load markitdown on first use."""
        if self._markitdown is None:
            try:
                from markitdown import MarkItDown
                self._markitdown = MarkItDown()
            except ImportError:
                raise ImportError(
                    "markitdown is not installed. "
                    "Install with: pip install markitdown or markitdown[all]"
                )
        return self._markitdown

    def _file_hash(self, file_path: str) -> str:
        """SHA256 hash of file content for caching."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _cache_path(self, file_path: str, file_hash: str) -> Path | None:
        """Get cache file path for a given source file."""
        if not self.cache_dir:
            return None
        safe_name = Path(file_path).stem.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_name}_{file_hash[:16]}.md"

    def _convert_pdf_with_headings(self, file_path: str) -> str:
        """Convert PDF using pymupdf with font-based heading detection.

        Two-pass approach:
        1. Build a font-size histogram across all pages to identify body text
           vs heading sizes.
        2. Re-iterate pages, emitting markdown with # markers for heading-sized
           text and plain text for body content.

        Falls back to markitdown if pymupdf is unavailable, extracts no text,
        no heading sizes are detected (uniform-font document), or the emitted
        heading tree fails validation (see _validate_heading_tree).
        """
        if self._pymupdf_available is None:
            try:
                import pymupdf
                self._pymupdf_available = True
            except ImportError:
                self._pymupdf_available = False
                logger.warning(
                    "pymupdf not installed; falling back to markitdown for PDF. "
                    "Install with: pip install pymupdf"
                )

        if not self._pymupdf_available:
            result = self.markitdown.convert(file_path)
            return result.text_content or ""

        import pymupdf

        doc = pymupdf.open(file_path)

        # --- Phase 1: Font size histogram ---
        size_histogram: dict[float, int] = {}
        for page in doc:
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type", 0) != 0:  # skip image blocks
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if len(text) < _HEADING_MIN_TEXT_LEN:
                            continue
                        size = round(span["size"], 1)
                        size_histogram[size] = (
                            size_histogram.get(size, 0) + len(text)
                        )

        if not size_histogram:
            doc.close()
            logger.info(f"pymupdf extracted no text from {file_path}, falling back to markitdown")
            result = self.markitdown.convert(file_path)
            return result.text_content or ""

        # Body text = most common size by total character count
        body_size = max(size_histogram, key=size_histogram.get)

        # Heading sizes = sizes significantly larger than body text
        heading_sizes = sorted(
            [s for s in size_histogram if s > body_size + _HEADING_SIZE_THRESHOLD],
            reverse=True,
        )

        # If no heading sizes detected, fall back to markitdown (uniform font)
        if not heading_sizes:
            doc.close()
            logger.info(
                f"No heading sizes detected in {file_path} "
                f"(body={body_size}pt, all text similar size), using markitdown"
            )
            result = self.markitdown.convert(file_path)
            return result.text_content or ""

        # Map heading sizes to levels (largest → H1, next → H2, ...)
        size_to_level = {
            s: min(i + 1, _MAX_HEADING_LEVELS)
            for i, s in enumerate(heading_sizes[:_MAX_HEADING_LEVELS])
        }

        logger.info(
            f"PDF heading detection: body={body_size}pt, "
            f"{len(size_to_level)} heading levels: {size_to_level}"
        )

        # --- Phase 2: Emit markdown with heading markers ---
        output_lines: list[str] = []
        emitted_headings: list[tuple[int, str, float]] = []  # (level, text, size)
        for page in doc:
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s["text"].strip()]
                    if not spans:
                        continue

                    # Use the max font size in the line (handles mixed spans)
                    max_size = max(round(s["size"], 1) for s in spans)
                    line_text = "".join(s["text"] for s in spans).strip()

                    if not line_text:
                        continue

                    if max_size in size_to_level:
                        level = size_to_level[max_size]
                        # Avoid duplicate consecutive headings
                        prefix = "#" * level
                        marker = f"{prefix} {line_text}"
                        if output_lines and output_lines[-1] == marker:
                            continue
                        output_lines.append(marker)
                        output_lines.append("")  # blank line after heading
                        emitted_headings.append((level, line_text, max_size))
                    else:
                        output_lines.append(line_text)
            output_lines.append("")  # page break separator

        doc.close()

        # --- Phase 3: Validate the emitted heading tree ---
        if emitted_headings:
            is_valid, reasons = _validate_heading_tree(emitted_headings, size_to_level)
            if not is_valid:
                logger.warning(
                    f"Heading tree validation failed for {file_path}: "
                    f"{'; '.join(reasons)}. Falling back to markitdown."
                )
                result = self.markitdown.convert(file_path)
                return result.text_content or ""

        return "\n".join(output_lines)

    def convert(self, file_path: str) -> str:
        """
        Convert a file to markdown text.

        Args:
            file_path: Path to the source file.

        Returns:
            Markdown text string.

        Raises:
            ValueError: If file extension is not supported.
            Exception: If conversion fails.
        """
        file_path = os.path.expanduser(file_path)
        ext = Path(file_path).suffix.lower()

        # Passthrough for markdown and text
        if ext in PASSTHROUGH_EXTS:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

        # Check if convertible
        if ext not in CONVERTIBLE_EXTS:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Check cache
        fhash = self._file_hash(file_path)
        cache_file = self._cache_path(file_path, fhash)
        if cache_file and cache_file.exists():
            logger.debug(f"Cache hit: {file_path}")
            return cache_file.read_text(encoding="utf-8")

        # Convert: pymupdf for PDFs (heading detection), markitdown for everything else
        logger.debug(f"Converting: {file_path}")
        try:
            if ext == ".pdf":
                markdown_text = self._convert_pdf_with_headings(file_path)
            else:
                result = self.markitdown.convert(file_path)
                markdown_text = result.text_content or ""
        except Exception as e:
            logger.error(f"Conversion failed for {file_path}: {e}")
            raise

        # Cache result
        if cache_file:
            cache_file.write_text(markdown_text, encoding="utf-8")

        return markdown_text

    def can_convert(self, file_path: str, allowed_extensions: list[str]) -> bool:
        """Check if a file should be converted based on extension allowlist."""
        ext = Path(file_path).suffix.lower()
        return ext in allowed_extensions


def _validate_heading_tree(
    headings: list[tuple[int, str, float]],
    size_to_level: dict[float, int],
) -> tuple[bool, list[str]]:
    """Validate the emitted heading tree for structural correctness.

    Checks four signals:
    1. Collapse ratio: (a) zero H2 headings (all one level → 1.0), or
       (b) ≥3 heading levels but >50% of H2 sections have no H3+ children
       (flat tree where depth is expected).
    2. Depth saturation: >15% of headings at the max level (H6).
    3. Size clustering: any two adjacent heading levels with font sizes
       within 1.0pt of each other.
    4. Order anomaly: a heading level jumps by more than 1 from the
       previous max (e.g. H6 before any H1-H5), or the first heading
       is deeper than H2.

    Args:
        headings: List of (level, text, font_size) tuples in document order.
        size_to_level: Mapping of font sizes to heading levels.

    Returns:
        (is_valid, reasons) — True if the tree passes all checks, else False
        with a list of human-readable failure reasons.
    """
    reasons: list[str] = []
    total = len(headings)

    if total == 0:
        return True, []  # no headings is handled by the caller

    # --- Signal 1: Collapse ratio ---
    h2_headings = [h for h in headings if h[0] == 2]
    distinct_levels = {h[0] for h in headings}

    if not h2_headings:
        # No H2 headings at all — everything is at one level
        collapse_ratio = 1.0
    elif len(distinct_levels) >= 3:
        # ≥3 levels: check if H2 sections actually have H3+ children.
        # Walk the heading list and count H2 sections that are followed
        # by at least one H3+ heading before the next H2.
        h2_with_children = 0
        h2_count = 0
        for i, (level, _, _) in enumerate(headings):
            if level == 2:
                h2_count += 1
                # Look ahead: is there an H3+ before the next H2/H1?
                has_child = False
                for j in range(i + 1, len(headings)):
                    next_level = headings[j][0]
                    if next_level <= 2:
                        break  # next H2 or H1 — section ends
                    if next_level >= 3:
                        has_child = True
                        break
                if has_child:
                    h2_with_children += 1
        collapse_ratio = 1.0 - (h2_with_children / h2_count) if h2_count else 1.0
    else:
        # ≤2 levels: flat but potentially correct. Don't flag.
        collapse_ratio = 0.0

    if collapse_ratio > _COLLAPSE_RATIO_THRESHOLD:
        reasons.append(
            f"collapse ratio {collapse_ratio:.2f} > {_COLLAPSE_RATIO_THRESHOLD}"
        )

    # --- Signal 2: Depth saturation ---
    level_counts = Counter(h[0] for h in headings)
    max_level_count = level_counts.get(_MAX_HEADING_LEVELS, 0)
    depth_saturation_ratio = max_level_count / total if total > 0 else 0.0

    if depth_saturation_ratio > _DEPTH_SATURATION_THRESHOLD:
        reasons.append(
            f"depth saturation {depth_saturation_ratio:.2f} > "
            f"{_DEPTH_SATURATION_THRESHOLD} ({max_level_count}/{total} at H{_MAX_HEADING_LEVELS})"
        )

    # --- Signal 3: Size clustering ---
    sizes_sorted = sorted(size_to_level.keys(), reverse=True)
    if len(sizes_sorted) >= 2:
        for i in range(len(sizes_sorted) - 1):
            gap = round(sizes_sorted[i] - sizes_sorted[i + 1], 1)
            if gap <= _MIN_SIZE_GAP_THRESHOLD:
                reasons.append(
                    f"size clustering: levels {size_to_level[sizes_sorted[i]]} "
                    f"({sizes_sorted[i]}pt) and {size_to_level[sizes_sorted[i + 1]]} "
                    f"({sizes_sorted[i + 1]}pt) are within {gap:.1f}pt"
                )
                break  # one violation is enough

    # --- Signal 4: Order anomaly ---
    max_level_seen = 0
    for i, (level, _, _) in enumerate(headings):
        if i == 0 and level > 2:
            reasons.append(
                f"order anomaly: first heading is H{level} (expected H1 or H2)"
            )
            break
        if max_level_seen > 0 and level > max_level_seen + 1:
            reasons.append(
                f"order anomaly: H{level} appears before H{max_level_seen + 1}"
            )
            break
        max_level_seen = max(max_level_seen, level)

    is_valid = len(reasons) == 0
    return is_valid, reasons