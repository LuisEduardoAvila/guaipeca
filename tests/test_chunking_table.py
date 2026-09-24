"""Tests for table-aware chunking in _split_paragraphs_aware.

Tests cover:
1. Oversized table atomicity (table > max_size stays as one paragraph)
2. Table with internal blank lines stays atomic
3. Normal-sized table stays atomic
4. Code fence regression (P3-6 CommonMark fence-length handling)
5. Table inside code block is not treated as a table
"""

from __future__ import annotations

from itertools import pairwise

from guaipeca.chunking import _chunk_by_paragraph, _split_paragraphs_aware, chunk_text


class TestTableAwareChunking:
    """Test that markdown tables are treated as atomic units."""

    def test_oversized_table_is_atomic(self):
        """A table larger than max_size should stay as one paragraph."""
        header = "| Col A | Col B | Col C |"
        sep = "|-------|-------|-------|"
        rows = [f"| row-{i:03d}-data | val-{i:03d}-xxx | ext-{i:03d}-yyy |" for i in range(100)]
        table = header + "\n" + sep + "\n" + "\n".join(rows)
        assert len(table) > 2000  # ensure it's oversized

        paras = _split_paragraphs_aware(table, max_size=2000)
        assert len(paras) == 1, f"Oversized table split into {len(paras)} paragraphs"
        assert len(paras[0][0]) == len(table)

    def test_table_with_internal_blank_line_is_atomic(self):
        """A table with a blank line between rows should stay as one paragraph.

        This is the live-bug case: without table-awareness, the blank line
        would cause a paragraph split, breaking the table in half.
        """
        header = "| Col A | Col B |"
        sep = "|-------|-------|"
        rows = []
        for i in range(100):
            rows.append(f"| val-{i:03d}-xxxx | val2-{i:03d}-yyyy |")
            if i == 50:
                rows.append("")  # blank line in the middle
        table = header + "\n" + sep + "\n" + "\n".join(rows)
        assert len(table) > 2000

        paras = _split_paragraphs_aware(table, max_size=2000)
        assert len(paras) == 1, f"Table with blank line split into {len(paras)} paragraphs"

    def test_normal_table_is_atomic(self):
        """A normal-sized table should stay as one paragraph."""
        table = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        paras = _split_paragraphs_aware(table, max_size=2000)
        assert len(paras) == 1

    def test_table_followed_by_text(self):
        """Table followed by a blank line and more text should produce 2 paragraphs."""
        table = "| A | B |\n|---|---|\n| 1 | 2 |"
        text = table + "\n\nSome text after the table."
        paras = _split_paragraphs_aware(text, max_size=2000)
        assert len(paras) == 2
        assert "A" in paras[0][0]  # table paragraph
        assert "Some text" in paras[1][0]  # text paragraph

    def test_oversized_table_in_chunk_text_is_atomic(self):
        """chunk_text with an oversized table should keep it in one chunk."""
        header = "| Col A | Col B |"
        sep = "|-------|-------|"
        rows = [f"| val-{i:03d}-xxxxxxx | val2-{i:03d}-yyyyyyy |" for i in range(120)]
        table = header + "\n" + sep + "\n" + "\n".join(rows)
        text = f"## Section\n\n{table}\n"
        assert len(table) > 2000

        chunks = chunk_text(text, source_path="test.md", max_size=2000, table_aware=True)
        # Find the chunk(s) containing table data
        table_chunks = [c for c in chunks if "| Col A" in c.text or "| val-" in c.text]
        assert len(table_chunks) == 1, f"Table split across {len(table_chunks)} chunks"
        # Verify it contains the full table (header + all rows)
        assert "Col A" in table_chunks[0].text
        assert table_chunks[0].text.count("| val-") == 120


class TestCodeFenceRegression:
    """Regression test for P3-6 CommonMark fence-length handling.

    The previous coordinator's diff accidentally dedented the
    `code_fence = None` / `code_fence_len = 0` lines OUT of the
    `if fence_length >= code_fence_len:` block, causing the fence state
    to reset on every closing fence line regardless of length match.
    """

    def test_closing_fence_shorter_than_opening_keeps_block_open(self):
        """A closing fence with fewer backticks than opening should NOT close the block."""
        # Opening: 4 backticks; closing attempt: 3 backticks (should NOT close)
        text = "````\ncode line 1\n```\nstill in code block\n````\n\nAfter block."
        paras = _split_paragraphs_aware(text, max_size=2000)
        # The entire code block (including the 3-backtick line and "still in code block")
        # should be one paragraph, followed by "After block."
        assert len(paras) == 2
        # First paragraph should contain the 3-backtick line as code (not a close)
        assert "```\nstill in code block" in paras[0][0]
        # Second paragraph should be the text after the block closes
        assert "After block." in paras[1][0]

    def test_closing_fence_equal_length_closes_block(self):
        """A closing fence with same length as opening should close the block."""
        text = "```\ncode line 1\n```\n\nAfter block."
        paras = _split_paragraphs_aware(text, max_size=2000)
        assert len(paras) == 2
        assert "code line 1" in paras[0][0]
        assert "After block." in paras[1][0]

    def test_closing_fence_longer_than_opening_closes_block(self):
        """A closing fence with more backticks than opening should close the block."""
        text = "```\ncode line 1\n````\n\nAfter block."
        paras = _split_paragraphs_aware(text, max_size=2000)
        assert len(paras) == 2

    def test_table_inside_code_block_not_treated_as_table(self):
        """A markdown table inside a code block should NOT trigger table-awareness."""
        text = "```\n| A | B |\n|---|---|\n| 1 | 2 |\n```\n\nAfter."
        paras = _split_paragraphs_aware(text, max_size=2000)
        # The code block (including table lines) should be one paragraph
        # The table-awareness should NOT split inside the code block
        assert len(paras) == 2
        assert "| A | B |" in paras[0][0]  # table lines inside code block
        assert "After." in paras[1][0]

class TestChunkOverlapOffset:
    """Regression: overlap must be taken from the correct original offset.

    Bug: ``current_original_len`` was only updated in the first (no-overlap)
    branch, so once a chunk contained a previous chunk's overlap the length
    never advanced. Overlap was then extracted from a stale, wrong position,
    degrading retrieval at chunk boundaries — and produced chunks slightly
    over ``max_size``.
    """

    @staticmethod
    def _paras(n: int, width: int = 90) -> str:
        return "\n\n".join(f"para-{i}-" + "x" * width for i in range(n))

    def test_chunks_respect_max_size_with_overlap(self):
        """No chunk may exceed max_size once overlap is in play."""
        max_size, overlap = 400, 100
        chunks = _chunk_by_paragraph(
            self._paras(40), "f.md", max_size=max_size, overlap=overlap
        )
        assert len(chunks) > 3
        for c in chunks:
            assert len(c.text) <= max_size, f"chunk over max_size: {len(c.text)}"

    def test_overlap_present_between_consecutive_chunks(self):
        """Each consecutive pair should share overlapping content."""
        overlap = 120
        chunks = _chunk_by_paragraph(
            self._paras(40), "f.md", max_size=420, overlap=overlap
        )
        assert len(chunks) > 3
        for prev, nxt in pairwise(chunks):
            # The tail of the previous chunk should reappear near the start
            # of the next chunk (overlap carries content forward).
            tail = prev.text[-overlap:].strip()
            marker = tail[-40:].strip()
            assert marker, "empty overlap marker"
            assert marker in nxt.text, (
                "expected overlap content from previous chunk in next chunk"
            )

    def test_offsets_are_monotonic_and_consistent(self):
        """chunk offsets must strictly increase and track chunk lengths."""
        chunks = _chunk_by_paragraph(
            self._paras(30), "f.md", max_size=400, overlap=80
        )
        offsets = [c.char_offset for c in chunks]
        assert offsets == sorted(offsets)
        assert len(set(offsets)) == len(offsets)
        # offset advances by (len - overlap) plus separators; must never move
        # backwards and must not advance to/past the whole text.
        for prev, nxt in pairwise(chunks):
            assert nxt.char_offset > prev.char_offset
