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

# Real Oracle FCCS calibration PDF (see docs/heading-validation.md).
# Tests that need the full document skip when it is absent.
_FCCS_PDF = "/tmp/guaipeca-calib/oracle-fccs.pdf"


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

class TestChapterNumberTitleMerge:
    """Bare section labels (chapter numbers) must merge with their titles.

    Regression: Oracle FCCS typesets the chapter number and the chapter title
    as adjacent heading-sized lines in the SAME block but at DIFFERENT font
    sizes (e.g. 30pt number + 24pt title).  The old size-equality-only
    stitcher emitted "## 5" and "### Managing Security" as two headings.
    """

    def test_bare_label_regex_matches_structural_labels(self):
        from guaipeca.converter import _BARE_SECTION_LABEL

        for label in ["1", "5", "10", "30", "5.2", "5.10.3", "A.1", "Chapter 5", "Section 3.1"]:
            assert _BARE_SECTION_LABEL.match(label), f"should match: {label!r}"

    def test_bare_label_regex_rejects_prose_and_lone_letters(self):
        from guaipeca.converter import _BARE_SECTION_LABEL

        # Prose titles and a lone letter (e.g. an appendix "A" heading, which
        # is usually a real standalone heading) must NOT match.
        for prose in [
            "Managing Security",
            "Creating and Running an EPM Center of Excellence",
            "Overview of the Home Page",
            "1 Introduction",  # label + text on one line = already a title
            "A",
        ]:
            assert not _BARE_SECTION_LABEL.match(prose), f"should NOT match: {prose!r}"

    def test_chapter_number_and_title_merge(self, converter):
        """A bare number heading followed by a different-size title merges.

        Note: on the real FCCS PDF, double-digit chapter numbers (10+) are
        emitted by pymupdf as a SEPARATE block from the title, so this only
        asserts the single-digit chapters that share a block.
        """
        import os

        import pytest

        if not os.path.exists(_FCCS_PDF):
            pytest.skip("real FCCS calibration PDF not present")

        md = converter._convert_pdf_with_headings(_FCCS_PDF)
        headings = _extract_headings(md)

        # Body chapter headings (H1-H3) must carry their title, not be bare
        # numbers.  TOC page numbers (H5) are legitimately bare, so only
        # check levels 1-3.
        bare = [h for h in headings if h[1].strip().isdigit()]
        body_bare = [h for h in bare if h[0] <= 3]
        # Single-digit chapters (1-9) share a block with the title and must
        # merge.  Double-digit chapters are a separate-block case (see
        # test_chapter_number_cross_block) and are excluded here.
        single_digit_bare = [h for h in body_bare if len(h[1].strip()) == 1]
        assert single_digit_bare == [], (
            f"unmerged single-digit chapter numbers: {single_digit_bare}"
        )

        # And a known chapter must carry its name.
        ch5 = [h for h in headings if h[1].startswith("5 ") and "Managing Security" in h[1]]
        assert len(ch5) == 1, f"expected merged '5 Managing Security', got {ch5}"

    def test_chapter_number_cross_block(self, converter):
        """Double-digit chapters (separate block from title) must also merge.

        pymupdf splits the 30pt number and 24pt title into different blocks
        for chapters 10+; the stitcher must bridge that boundary in the same
        way it already does for TOC numbers.
        """
        import os

        import pytest

        if not os.path.exists(_FCCS_PDF):
            pytest.skip("real FCCS calibration PDF not present")

        md = converter._convert_pdf_with_headings(_FCCS_PDF)
        headings = _extract_headings(md)

        # Chapter 10's title must be attached to its number.
        ch10 = [h for h in headings if h[1].startswith("10 ") and "Integrating Cloud EPM" in h[1]]
        assert len(ch10) == 1, f"expected merged '10 Integrating Cloud EPM...', got {ch10}"
