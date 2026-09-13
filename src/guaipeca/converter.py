"""Document converter using markitdown.

Converts PDF, DOCX, PPTX, XLSX, HTML and other formats to markdown.
.md and .txt files pass through directly.
"""

from __future__ import annotations

import hashlib
import os
import logging
from pathlib import Path
from typing import Optional

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


class Converter:
    """Document converter using markitdown."""

    def __init__(self, cache_dir: Optional[str] = None):
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

    def _cache_path(self, file_path: str, file_hash: str) -> Optional[Path]:
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
        or no heading sizes are detected (uniform-font document).
        """
        if self._pymupdf_available is None:
            try:
                import pymupdf  # noqa: F401
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
                    else:
                        output_lines.append(line_text)
            output_lines.append("")  # page break separator

        doc.close()
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