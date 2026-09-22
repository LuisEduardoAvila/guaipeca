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
# Re-calibrated 2026-09-22 against a REAL Oracle FCCS PDF:
#   /tmp/guaipeca-calib/fccs-epm-information-development-team.pdf
#   (1,349 pages, 42 MB) — the primary reference document.
# Plus tests/fixtures/test-doc.pdf (real 5-page Oracle manual excerpt).
# Synthetic fixtures (oracle-*-sim.pdf, deep-legit.pdf, good-hierarchy.pdf)
# are used only as contrast and are labelled as synthetic.
#
# See docs/heading-validation.md for the full calibration report with
# per-document measurements, threshold rationale, and margins.
#
# Measurements on REAL documents:
#
#   Document              Pages  Collapse  Saturation  MinGap  Valid?
#   ───────────────────   ─────  ────────  ──────────  ──────  ───────
#   FCCS Info Dev (real)   1349    0.030     0.149      2.0     YES
#   FCCS DIEPM (real)       871    0.000     0.239      2.0     YES
#   FCCS eCalc (real)       354    0.000     0.459      2.0     YES
#   FCCS FR Web (real)      233    0.167     0.140      2.0     YES
#   test-doc.pdf (real)       5    0.167     0.000      8.0     YES
#
# Synthetic fixtures (contrast only):
#
#   oracle-manual-sim        40    0.025     0.200      0.5     NO
#   oracle-template-sim      20    1.000     0.000      N/A     NO
#   deep-legit.pdf            3    0.167     0.000      1.5     YES
#   good-hierarchy.pdf        5    0.200     0.000      4.0     YES
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

# Signal 2: If the vast majority of headings sit at the max level (H6),
# the size mapping is saturating — too many distinct sizes crammed into
# 6 bins, with multiple sizes merged into H6.
# Calibrated on 4 REAL Oracle FCCS PDFs (2026-09-22):
#   - FCCS eCalc Guide (354p): 304/662 = 0.459 (deepest real doc)
#   - FCCS DIEPM Guide (871p): 171/714 = 0.239
#   - FCCS Info Dev (1349p): 169/1138 = 0.149
#   - FCCS FR Web (233p): 44/315 = 0.140
#   - test-doc.pdf (5p): 0/8 = 0.000
# Threshold 0.60 gives the deepest real doc (0.459) a 0.141 margin.
# The synthetic bad fixture oracle-manual-sim (0.200) is below this
# threshold but is still rejected by size-clustering (0.5pt gaps) and
# first-heading sanity (H6 first).  Depth-saturation is a secondary
# signal that catches only extreme cases (60%+ at max depth).
_DEPTH_SATURATION_THRESHOLD = 0.60

# Signal 3: If any two adjacent heading levels have font sizes within 1.0pt,
# the levels are too close to distinguish reliably (size clustering).
# Calibrated: oracle-manual-sim has 0.5pt gaps; deep-legit has 1.5pt;
# test-doc has 8.0pt; good-hierarchy has 4.0pt.
_MIN_SIZE_GAP_THRESHOLD = 1.0

# Signal 4: First-heading sanity check.  The previous "order anomaly"
# signal flagged ANY deeper heading appearing before a shallower one in
# document order.  This was broken: real Oracle manuals have a Table of
# Contents where chapter numbers are at the H5 font size (18pt), appearing
# before the first H4 body heading — a completely legitimate structure.
#
# The redesigned check only flags the FIRST heading in the document if it
# is unexpectedly deep (deeper than H3), which would indicate a mis-mapped
# size histogram.  This catches oracle-manual-sim (first heading = H6)
# while passing the real FCCS PDF (first heading = H1).
#
# Level jumps (H1 → H6 with no intermediate levels) are NOT flagged here
# because they can occur legitimately when a document has a TOC or
# front-matter section at a different font size than body chapters.


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
    2. Depth saturation: >60% of headings at the max level (H6).
    3. Size clustering: any two adjacent heading levels with font sizes
       within 1.0pt of each other.
    4. First-heading sanity: the first heading in the document is deeper
       than H3, indicating a mis-mapped size histogram. (Previous signal
       flagged any deeper-before-shallower ordering, which false-positived
       on real Oracle manual TOC entries.)

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

    # --- Signal 4: First-heading sanity check ---
    # Only check if the very first heading is unexpectedly deep (H4+),
    # which indicates a mis-mapped size histogram. We do NOT check
    # document-order interleaving (e.g. H5 before H4) because real
    # Oracle manuals legitimately have TOC entries at heading font
    # sizes that appear before body-text headings at a different level.
    # Measured: real FCCS PDF has H5 TOC entries on page 3 before the
    # first H4 on page 48 — this is normal, not an anomaly.
    if headings[0][0] > 3:
        reasons.append(
            f"first heading is H{headings[0][0]} (expected H1-H3); "
            f"size histogram may be mis-mapped"
        )

    is_valid = len(reasons) == 0
    return is_valid, reasons
