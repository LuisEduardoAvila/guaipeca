"""Regression tests for heading_path / section_id corruption in _split_on_headings.

Bug: deep headings (H3/H4/H5) inside a ## section would overwrite the section's
heading_path, causing section_id to be derived from the wrong heading. This led
to chunks from different sections collapsing under one bogus section_id.

Fix: heading_path is captured ONCE when the ## section opens and never mutated
by deeper headings. section_id is always derived from the ## heading.
"""

from __future__ import annotations

from guaipeca.chunking import _split_on_headings, chunk_file

# ===========================================================================
# Repro: the exact case from the bug report
# ===========================================================================

REPRO_MD = """\
# Doc
## Chapter 21 Consolidating Data
### Intro
some text here
#### Seeded Consolidation Rules
##### Seeded Consolidation Rules - Example (March)
rule text
#### Consolidation and Translation Rules
more text
## Chapter 22 Other
body
"""


class TestReproHeadingPath:
    """Exact reproduction from the bug report."""

    def test_repro_section_paths(self):
        """Chapter 21 section's heading_path must be ['Doc', 'Chapter 21 Consolidating Data'].

        BEFORE fix: ['Doc', 'Chapter 21 Consolidating Data', 'Intro', 'Consolidation and Translation Rules']
        AFTER fix:  ['Doc', 'Chapter 21 Consolidating Data']
        """
        sections = _split_on_headings(REPRO_MD)
        paths = {heading: path for heading, _, _, path in sections}

        assert paths["Chapter 21 Consolidating Data"] == [
            "Doc",
            "Chapter 21 Consolidating Data",
        ], f"Chapter 21 path corrupted: {paths['Chapter 21 Consolidating Data']}"

        assert paths["Chapter 22 Other"] == [
            "Doc",
            "Chapter 22 Other",
        ], f"Chapter 22 path corrupted: {paths['Chapter 22 Other']}"

    def test_repro_pre_section_path(self):
        """Pre-section chunk (before any ##) should have path = ['Doc']."""
        sections = _split_on_headings(REPRO_MD)
        pre_section = sections[0]
        assert pre_section[0] == ""  # empty heading
        assert pre_section[3] == ["Doc"], f"Pre-section path wrong: {pre_section[3]}"

    def test_repro_no_deep_headings_in_path(self):
        """No heading_path should contain H3/H4/H5 headings."""
        sections = _split_on_headings(REPRO_MD)
        deep_headings = {"Intro", "Seeded Consolidation Rules",
                         "Seeded Consolidation Rules - Example (March)",
                         "Consolidation and Translation Rules"}
        for heading, _, _, path in sections:
            for h in path:
                assert h not in deep_headings, (
                    f"Deep heading {h!r} found in path {path} for section {heading!r}"
                )


# ===========================================================================
# section_id derivation: must come from ## heading, not deep headings
# ===========================================================================

class TestSectionIdFromSectionHeading:
    """section_id must be derived from the ## section heading."""

    def test_section_id_distinct_per_section(self):
        """Different ## sections must have different section_ids."""
        chunks = chunk_file("test.md", REPRO_MD, max_size=2000)

        ch21 = [c for c in chunks if c.heading == "Chapter 21 Consolidating Data"]
        ch22 = [c for c in chunks if c.heading == "Chapter 22 Other"]

        assert ch21 and ch22, "Both sections should exist"
        assert ch21[0].section_id != ch22[0].section_id, (
            "Different sections should have different section_ids"
        )

    def test_section_id_stable_within_section(self):
        """All chunks within the same ## section share the same section_id."""
        chunks = chunk_file("test.md", REPRO_MD, max_size=100)  # Force multiple sub-chunks

        ch21_chunks = [c for c in chunks if c.heading == "Chapter 21 Consolidating Data"]
        if len(ch21_chunks) > 1:
            sids = {c.section_id for c in ch21_chunks}
            assert len(sids) == 1, (
                f"Chunks in same section have different section_ids: {sids}"
            )

    def test_section_id_uses_section_heading_not_deep(self):
        """section_id must be hash of source_path + ':' + ## heading, not a deep heading."""
        chunks = chunk_file("test.md", REPRO_MD, max_size=2000)

        ch21 = next(c for c in chunks if c.heading == "Chapter 21 Consolidating Data")
        # Manually compute what section_id should be
        import hashlib
        expected = hashlib.sha256(
            b"test.md:Chapter 21 Consolidating Data"
        ).hexdigest()[:16]
        assert ch21.section_id == expected, (
            f"section_id mismatch: got {ch21.section_id}, expected {expected}"
        )

    def test_section_id_changes_when_section_heading_changes(self):
        """If two sections have the same deep headings but different ## headings,
        section_ids must differ."""
        md = """\
# Title
## Section A
### Shared Sub
content
## Section B
### Shared Sub
content
"""
        chunks = chunk_file("doc.md", md, max_size=2000)
        section_a = next(c for c in chunks if c.heading == "Section A")
        section_b = next(c for c in chunks if c.heading == "Section B")
        assert section_a.section_id != section_b.section_id, (
            "Sections with same deep headings but different ## headings should differ"
        )


# ===========================================================================
# Deep heading nesting: H3/H4/H5 siblings must not evict each other
# ===========================================================================

class TestDeepHeadingNesting:
    """Deep headings should not corrupt each other's section paths."""

    def test_sibling_deep_headings(self):
        """Multiple H3 headings under the same ## section should not corrupt
        the section's heading_path."""
        md = """\
# Doc
## Section A
### Sub 1
text
### Sub 2
text
#### Deep 1
text
#### Deep 2
text
### Sub 3
text
## Section B
content
"""
        sections = _split_on_headings(md)
        paths = {h: p for h, _, _, p in sections}

        assert paths["Section A"] == ["Doc", "Section A"], (
            f"Section A path corrupted: {paths['Section A']}"
        )
        assert paths["Section B"] == ["Doc", "Section B"], (
            f"Section B path corrupted: {paths['Section B']}"
        )

    def test_deep_headings_do_not_affect_section_id(self):
        """section_id for a section with many deep headings should equal
        section_id for a section with the same ## heading but no deep headings."""
        md_with_deep = """\
# Doc
## Section X
### Sub 1
text
#### Deep 1
text
## Section Y
text
"""
        md_without_deep = """\
# Doc
## Section X
text
## Section Y
text
"""
        chunks_with = chunk_file("doc.md", md_with_deep, max_size=2000)
        chunks_without = chunk_file("doc.md", md_without_deep, max_size=2000)

        x_with = next(c for c in chunks_with if c.heading == "Section X")
        x_without = next(c for c in chunks_without if c.heading == "Section X")

        assert x_with.section_id == x_without.section_id, (
            f"Deep headings changed section_id: {x_with.section_id} != {x_without.section_id}"
        )

    def test_h5_headings_in_path(self):
        """H5 headings should not appear in heading_path at all."""
        md = """\
# Doc
## Section
### Sub
#### Deep
##### Deepest
content
"""
        sections = _split_on_headings(md)
        for heading, _, _, path in sections:
            assert "Deepest" not in path, f"H5 heading in path: {path}"
            assert "Deep" not in path, f"H4 heading in path: {path}"
            assert "Sub" not in path, f"H3 heading in path: {path}"


# ===========================================================================
# Multiple H1 headings (edge case)
# ===========================================================================

class TestMultipleH1:
    """Multiple H1 headings should each contribute to heading_path."""

    def test_two_h1_then_h2(self):
        """Second H1 should update the path for subsequent sections."""
        md = """\
# Title One
## Section A
content
# Title Two
## Section B
content
"""
        sections = _split_on_headings(md)
        paths = {h: p for h, _, _, p in sections}

        assert paths["Section A"] == ["Title One", "Section A"], (
            f"Section A path wrong: {paths['Section A']}"
        )
        assert paths["Section B"] == ["Title Two", "Section B"], (
            f"Section B path wrong: {paths['Section B']}"
        )
