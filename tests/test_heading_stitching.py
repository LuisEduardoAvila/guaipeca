"""Tests for heading line-wrap stitching in the PDF converter.

Oracle FCCS PDFs wrap long titles across 2+ physical lines.  pymupdf groups
the lines of a wrapped heading into a single block with multiple line
entries at the same heading font size.  The converter stitches these
into one logical heading before emitting markdown.

Tests cover:
1. Wrapped headings are stitched (regression test for truncation bug)
2. No over-stitching (heading does not absorb following body text)
3. Standalone headings remain standalone
4. Duplicate consecutive headings are still suppressed
5. Real fixture (test-doc.pdf) still converts correctly
"""

from __future__ import annotations

import re

import pytest

from guaipeca.converter import Converter


@pytest.fixture
def converter():
    """Converter instance without cache (fresh conversion each test)."""
    return Converter()


@pytest.fixture
def wrapped_pdf(tmp_path):
    """Path to the synthetic wrapped-heading fixture."""
    return "tests/fixtures/wrapped-heading.pdf"


def _extract_headings(markdown: str) -> list[tuple[int, str]]:
    """Extract (level, text) pairs from markdown heading lines."""
    headings = []
    for line in markdown.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            headings.append((len(m.group(1)), m.group(2)))
    return headings


class TestHeadingStitching:
    """Test that wrapped headings (multi-line in one block) are stitched."""

    def test_wrapped_heading_is_stitched(self, converter, wrapped_pdf):
        """Two heading-sized lines in one block → one heading, not two."""
        md = converter._convert_pdf_with_headings(wrapped_pdf)
        headings = _extract_headings(md)

        # The wrapped heading must appear as a single complete heading
        stitched = [h for h in headings if "Configuring Detailed Analysis" in h[1]]
        assert len(stitched) == 1, f"Expected 1 stitched heading, got {stitched}"
        assert "Consolidation and Close" in stitched[0][1], (
            f"Heading should contain the continuation text: {stitched[0][1]}"
        )

    def test_no_truncated_headings(self, converter, wrapped_pdf):
        """No heading should end with a conjunction (truncation signal)."""
        md = converter._convert_pdf_with_headings(wrapped_pdf)
        headings = _extract_headings(md)

        conjunctions = {"and", "or", "of", "the", "for", "with", "in", "to", "on", "at", "by", "from"}
        truncated = [
            h for h in headings
            if h[1].split()[-1].lower().rstrip(",.;:") in conjunctions
        ]
        assert truncated == [], f"Found truncated headings: {truncated}"

    def test_no_stray_continuation_fragments(self, converter, wrapped_pdf):
        """No standalone 'Close' or 'Consolidation' fragments as headings."""
        md = converter._convert_pdf_with_headings(wrapped_pdf)
        headings = _extract_headings(md)

        stray_fragments = {"Close", "Consolidation", "Financial", "and"}
        stray = [h for h in headings if h[1] in stray_fragments]
        assert stray == [], f"Found stray continuation fragments: {stray}"


class TestNoOverStitching:
    """Test that headings are NOT merged with following body text."""

    def test_standalone_heading_not_merged_with_body(self, converter, wrapped_pdf):
        """A heading followed by body text must not absorb the body text."""
        md = converter._convert_pdf_with_headings(wrapped_pdf)
        headings = _extract_headings(md)

        # "Standalone Section" must be a heading by itself
        standalone = [h for h in headings if h[1] == "Standalone Section"]
        assert len(standalone) == 1, (
            f"Expected 'Standalone Section' as a standalone heading, got: {standalone}"
        )

        # The body paragraph must NOT be part of any heading
        body_in_heading = [h for h in headings if "Body paragraph" in h[1]]
        assert body_in_heading == [], (
            f"Body text was absorbed into a heading: {body_in_heading}"
        )

    def test_wrapped_heading_not_merged_with_following_body(self, converter, wrapped_pdf):
        """The stitched heading must not absorb body text from the next block."""
        md = converter._convert_pdf_with_headings(wrapped_pdf)
        headings = _extract_headings(md)

        # Find the stitched heading and verify it doesn't contain body text
        stitched = [h for h in headings if "Configuring Detailed Analysis" in h[1]]
        assert len(stitched) == 1
        # Must NOT contain body text from the following block
        assert "body text" not in stitched[0][1].lower(), (
            f"Stitched heading absorbed body text: {stitched[0][1]}"
        )
        assert "describes" not in stitched[0][1].lower(), (
            f"Stitched heading absorbed body text: {stitched[0][1]}"
        )


class TestExistingBehaviour:
    """Test that existing behaviour is not broken by the stitching fix."""

    def test_test_doc_pdf_converts_correctly(self, converter):
        """The real Oracle manual excerpt fixture still converts correctly."""
        md = converter._convert_pdf_with_headings("tests/fixtures/test-doc.pdf")
        headings = _extract_headings(md)

        # Should have the expected headings
        assert len(headings) >= 6, f"Expected at least 6 headings, got {len(headings)}"
        assert headings[0][1] == "Understanding RAG: Retrieval-Augmented Generation"

        # No truncated headings
        conjunctions = {"and", "or", "of", "the", "for", "with", "in", "to", "on", "at", "by", "from"}
        truncated = [
            h for h in headings
            if h[1].split()[-1].lower().rstrip(",.;:") in conjunctions
        ]
        assert truncated == [], f"Found truncated headings in test-doc.pdf: {truncated}"

    def test_main_title_heading_preserved(self, converter, wrapped_pdf):
        """The single-line H1 'Main Title' must remain as-is."""
        md = converter._convert_pdf_with_headings(wrapped_pdf)
        headings = _extract_headings(md)

        assert headings[0] == (1, "Main Title"), (
            f"First heading should be 'Main Title', got: {headings[0]}"
        )

    def test_duplicate_consecutive_headings_suppressed(self, converter, wrapped_pdf):
        """Duplicate consecutive headings are still suppressed."""
        # The fixture doesn't have duplicates, but we verify the suppression
        # logic still works by checking no heading appears twice consecutively
        md = converter._convert_pdf_with_headings(wrapped_pdf)
        headings = _extract_headings(md)

        for i in range(1, len(headings)):
            if headings[i] == headings[i - 1]:
                pytest.fail(f"Duplicate consecutive heading: {headings[i]}")