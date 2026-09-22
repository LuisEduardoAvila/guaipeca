"""Structure-aware chunking for Guaipeca.

Adapted from a structure-aware chunking strategy:
- Split on ## headings (structure-aware)
- Table-aware: tables treated as atomic units (never split mid-table, even with internal blank lines)
- Code-block-aware: fenced code blocks are never split mid-block
- Configurable overlap between consecutive chunks
- Pure Python regex, zero LLM calls
- ToC-aware: heading_path and section_id tracked for each chunk
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single document chunk."""
    id: str
    heading: str
    text: str
    source_path: str
    char_offset: int = 0
    metadata: dict = field(default_factory=dict)
    heading_path: list[str] = field(default_factory=list)
    section_id: str = ""

    @property
    def summary(self) -> str:
        """Summary of the chunk: heading if available, else first 1-2 non-empty lines."""
        # Prefer the heading as summary -- it's the most informative short label
        if self.heading and self.heading.strip():
            return self.heading.strip()[:200]

        # Fall back to first 1-2 non-empty lines of text
        lines = self.text.strip().split("\n")
        summary_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            summary_lines.append(line)
            if len(summary_lines) >= 2:
                break
        return " ".join(summary_lines)[:200]

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "heading": self.heading,
            "text": self.text,
            "source_path": self.source_path,
            "char_offset": self.char_offset,
            "summary": self.summary,
            "char_count": self.char_count,
            "metadata": self.metadata,
            "heading_path": self.heading_path,
            "section_id": self.section_id,
        }


# Table detection regex (pipe-delimited markdown tables)
TABLE_PATTERN = re.compile(
    r"(\|[^\n]+\|\n\|[\s\-:|]+\|\n(?:\|[^\n]+\|\n?)*)",
    re.MULTILINE,
)

# Heading detection
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Fenced code block detection (``` or ~~~)
CODE_FENCE_PATTERN = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)


def _make_chunk_id(source_path: str, heading: str, offset: int) -> str:
    """Generate a stable chunk ID from source path + heading + offset."""
    raw = f"{source_path}:{heading}:{offset}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_section_id(source_path: str, heading_path: list[str]) -> str:
    """Generate a section ID from source path and top-level heading.

    For structured docs: SHA256(source_path + ":" + section_heading)[:16]
    where section_heading is the ## heading (heading_path[-1]).
    heading_path only contains # and ## headings (deep headings excluded),
    so heading_path[-1] is always the enclosing ## section heading.

    For non-structured docs (empty heading_path): "unstructured:{hash}" where
    hash = SHA256(source_path)[:16].
    """
    if not heading_path:
        return f"unstructured:{hashlib.sha256(source_path.encode()).hexdigest()[:16]}"

    # The section heading is the last entry in heading_path (the ## heading)
    top_heading = heading_path[-1] if heading_path else ""
    raw = f"{source_path}:{top_heading}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_text(
    text: str,
    source_path: str,
    max_size: int = 2000,
    table_aware: bool = True,
    split_on_headings: bool = True,
    overlap: int = 200,
) -> list[Chunk]:
    """
    Chunk text into structured segments.

    Args:
        text: The markdown text to chunk.
        source_path: Source file path (for metadata).
        max_size: Maximum chunk size in characters.
        table_aware: If True, treat tables as atomic units.
        split_on_headings: If True, split on ## headings.
        overlap: Number of characters to carry from the end of one chunk into the next.

    Returns:
        List of Chunk objects.
    """
    if not text or not text.strip():
        return []

    # If not splitting on headings, just chunk by paragraph
    if not split_on_headings:
        return _chunk_by_paragraph(text, source_path, max_size, overlap=overlap)

    # Extract tables first (if table_aware) and replace with placeholders
    tables: dict[str, str] = {}
    working_text = text
    if table_aware:
        def _replace_table(m):
            key = f"__TABLE_{len(tables)}__"
            tables[key] = m.group(0)
            return key + "\n"

        working_text = TABLE_PATTERN.sub(_replace_table, working_text)

    # Split on ## headings (level 2+)
    sections = _split_on_headings(working_text)

    chunks = []
    for heading, section_text, offset, heading_path in sections:
        # Restore tables in section
        for key, table_text in tables.items():
            section_text = section_text.replace(key, table_text)

        # Compute section_id for this section
        section_id = _make_section_id(source_path, heading_path)

        # If section is too large, split further by paragraph (with code-block awareness)
        if len(section_text) > max_size:
            sub_chunks = _chunk_by_paragraph(
                section_text, source_path, max_size,
                heading=heading, base_offset=offset,
                overlap=overlap,
                heading_path=heading_path,
                section_id=section_id,
            )
            chunks.extend(sub_chunks)
        else:
            chunk_id = _make_chunk_id(source_path, heading, offset)
            chunks.append(Chunk(
                id=chunk_id,
                heading=heading,
                text=section_text.strip(),
                source_path=source_path,
                char_offset=offset,
                heading_path=heading_path,
                section_id=section_id,
            ))

    return chunks


def _split_on_headings(text: str) -> list[tuple[str, str, int, list[str]]]:
    """
    Split text on ## (level 2) headings.

    Returns list of (heading, text, char_offset, heading_path) tuples.
    Level 1 (#) is treated as document title, included in first chunk.
    Level 3+ (###, ####) stay with their parent ## section.

    heading_path semantics: the path from the document title (#) down to the
    enclosing ## section heading ONLY. Deep headings (H3/H4/H5/...) do NOT
    appear in heading_path. This ensures:
    - section_id is always derived from the ## section heading (heading_path[-1])
    - ToC tree reflects section-level structure (# -> ##), not deep sub-headings
    - chunks within the same ## section share the same heading_path and section_id

    Deep headings are part of the section content but never mutate the section's
    captured heading_path. Splits occur ONLY on level-2 headings.
    """
    sections = []
    current_heading = ""
    current_text = ""
    current_offset = 0

    # Stack for tracking level-1 (#) and level-2 (##) headings only.
    # Deep headings (H3+) are NOT tracked here — they don't affect heading_path.
    section_stack: list[tuple[int, str]] = []
    # The heading_path for the current section (captured ONCE when ## opens).
    # Frozen: deeper headings inside the section never mutate this.
    current_heading_path: list[str] = []

    lines = text.split("\n")
    pos = 0

    for line in lines:
        match = HEADING_PATTERN.match(line)
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            if level == 2:
                # Save previous section with its captured heading_path
                if current_text.strip():
                    sections.append((current_heading, current_text.strip(), current_offset, current_heading_path))

                # Pop stack entries deeper than or equal to current level
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                # Push current ## heading onto stack
                section_stack.append((level, heading_text))
                # Capture heading_path for the new section ONCE — frozen,
                # will NOT be mutated by deeper headings inside the section.
                current_heading_path = [h for _, h in section_stack]
                current_heading = heading_text
                current_text = line + "\n"
                current_offset = pos
            elif level == 1:
                # Level 1 (#) — document title or new document section.
                # If a ## section is in progress, close it first.
                has_section = any(lvl == 2 for lvl, _ in section_stack)
                if has_section and current_text.strip():
                    sections.append((current_heading, current_text.strip(), current_offset, current_heading_path))
                    current_text = ""
                # Pop stack entries >= level 1
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, heading_text))
                # If no ## section is active, update heading_path so the
                # title appears in the pre-section chunk's path.
                if not has_section:
                    current_heading_path = [h for _, h in section_stack]
                else:
                    # A new # after a ## section: start fresh with just the title
                    current_heading_path = [h for _, h in section_stack]
                    current_heading = heading_text
                    current_offset = pos
                current_text += line + "\n"
            else:
                # Level 3+ (###, ####, etc.) — keep with current section.
                # Do NOT mutate current_heading_path or section_stack.
                # Deep headings are part of section content but do not
                # affect the section's heading_path or section_id.
                current_text += line + "\n"
        else:
            current_text += line + "\n"
        pos += len(line) + 1  # +1 for \n

    # Don't forget the last section
    if current_text.strip():
        sections.append((current_heading, current_text.strip(), current_offset, current_heading_path))

    # If no sections were created (no ## headings), return entire text as one chunk
    if not sections:
        sections.append(("", text.strip(), 0, []))

    return sections


def _split_paragraphs_aware(text: str, max_size: int) -> list[tuple[str, int, int]]:
    """
    Split text into paragraphs, respecting fenced code blocks and markdown tables.

    Never splits inside a fenced code block (``` or ~~~).
    Never splits inside a markdown table (pipe-delimited).
    If a code block or table exceeds max_size, it is kept as one chunk
    (even if oversized) — mirroring the atomic-unit policy for code blocks.

    Returns list of (paragraph_text, start_offset, length) tuples.
    """
    lines = text.split("\n")
    paragraphs: list[tuple[str, int, int]] = []
    current_lines: list[str] = []
    current_start = 0
    pos = 0
    in_code_block = False
    code_fence = None
    code_fence_len = 0  # P3-6: track opening fence length
    in_table = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect code fence start/end
        fence_match = CODE_FENCE_PATTERN.match(stripped)
        if fence_match:
            fence_char = fence_match.group(1)[0]  # ` or ~
            fence_length = len(fence_match.group(1))  # number of ` or ~ chars
            if not in_code_block:
                in_code_block = True
                code_fence = fence_char
                code_fence_len = fence_length
            elif fence_char == code_fence:
                # P3-6: closing fence must have >= opening fence length (CommonMark spec)
                if fence_length >= code_fence_len:
                    in_code_block = False
                    code_fence = None
                    code_fence_len = 0

        # Detect table boundaries: a table row starts with | and the next
        # line is a separator (|---|---|). Table ends at a blank line
        # only if the next non-blank line is not a table row; or at a
        # non-table, non-blank line.
        is_table_row = stripped.startswith("|")
        if not in_code_block:
            if not in_table and is_table_row:
                # Look ahead: is the next line a table separator?
                if i + 1 < len(lines) and re.match(r"\|[\s\-:|]+\|", lines[i + 1].strip()):
                    in_table = True
            elif in_table and not stripped:
                # Blank line: look ahead — if next non-blank line is a
                # table row, we're still inside the table.
                still_table = False
                for j in range(i + 1, len(lines)):
                    next_stripped = lines[j].strip()
                    if not next_stripped:
                        continue
                    still_table = next_stripped.startswith("|")
                    break
                if not still_table:
                    in_table = False
            elif in_table and stripped and not is_table_row:
                # Non-table, non-blank line ends the table
                in_table = False

        # Paragraph boundary: blank line outside code blocks and tables
        if not in_code_block and not in_table and not stripped:
            # Flush current paragraph
            if current_lines:
                para_text = "\n".join(current_lines)
                paragraphs.append((para_text, current_start, len(para_text)))
                current_lines = []
            current_start = pos + len(line) + 1
        else:
            if not current_lines:
                current_start = pos
            current_lines.append(line)

        pos += len(line) + 1  # +1 for \n

    # Flush remaining
    if current_lines:
        para_text = "\n".join(current_lines)
        paragraphs.append((para_text, current_start, len(para_text)))

    return paragraphs


def _chunk_by_paragraph(
    text: str,
    source_path: str,
    max_size: int,
    heading: str = "",
    base_offset: int = 0,
    overlap: int = 200,
    heading_path: list[str] | None = None,
    section_id: str = "",
) -> list[Chunk]:
    """
    Split text by paragraph boundaries when sections exceed max_size.

    Args:
        text: The text to split.
        source_path: Source file path.
        max_size: Maximum chunk size in characters.
        heading: Heading text for this section.
        base_offset: Character offset of this section within the document.
        overlap: Number of characters to carry from the end of the previous chunk
                 into the start of the next chunk.
        heading_path: Heading path for this section (inherited by sub-chunks).
        section_id: Section ID for this section (inherited by sub-chunks).
    """
    if heading_path is None:
        heading_path = []
    if not section_id:
        section_id = _make_section_id(source_path, heading_path)

    paragraphs = _split_paragraphs_aware(text, max_size)
    chunks = []
    current_text = ""
    offset = base_offset
    overlap_text = ""
    # P2-4: Track the length of the original (non-overlap) content in the current chunk
    # so overlap is only extracted from the original content, not from previous overlap
    current_original_len = 0

    for para_text, para_start, para_len in paragraphs:
        para = para_text.strip()
        if not para:
            continue

        # Prepend overlap from previous chunk
        candidate = (overlap_text + "\n\n" + para) if overlap_text else para

        if len(current_text) + len(candidate) + 2 > max_size:
            if current_text:
                chunk_id = _make_chunk_id(source_path, heading, offset)
                chunks.append(Chunk(
                    id=chunk_id,
                    heading=heading,
                    text=current_text.strip(),
                    source_path=source_path,
                    char_offset=offset,
                    heading_path=heading_path,
                    section_id=section_id,
                ))
                # P2-4: Extract overlap only from the original (non-overlap) content
                # to prevent compounding of overlap across multiple splits
                original_content = current_text.strip()
                if current_original_len > 0 and current_original_len < len(original_content):
                    original_content = original_content[:current_original_len]
                if overlap > 0 and len(original_content) > overlap:
                    overlap_text = original_content[-overlap:]
                else:
                    overlap_text = original_content
                offset += len(current_text) + 2
            current_text = candidate
            # Track original content length (candidate = overlap_text + para)
            current_original_len = len(para) if overlap_text else len(candidate)
        else:
            current_text = current_text + "\n\n" + candidate if current_text else candidate
            if not overlap_text:
                current_original_len = len(current_text)

    if current_text:
        chunk_id = _make_chunk_id(source_path, heading, offset)
        chunks.append(Chunk(
            id=chunk_id,
            heading=heading,
            text=current_text.strip(),
            source_path=source_path,
            char_offset=offset,
            heading_path=heading_path,
            section_id=section_id,
        ))

    return chunks


def chunk_file(
    file_path: str,
    markdown_text: str,
    max_size: int = 2000,
    table_aware: bool = True,
    split_on_headings: bool = True,
    overlap: int = 200,
) -> list[Chunk]:
    """
    Convenience: chunk a file's markdown text with standard params.

    Args:
        file_path: Source file path.
        markdown_text: Markdown text (already converted if needed).
        max_size: Max chunk size in characters.
        table_aware: Treat tables as atomic.
        split_on_headings: Split on ## headings.
        overlap: Characters to overlap between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    return chunk_text(
        markdown_text,
        source_path=file_path,
        max_size=max_size,
        table_aware=table_aware,
        split_on_headings=split_on_headings,
        overlap=overlap,
    )
