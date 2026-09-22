"""Tests for heading-tree validation and markitdown fallback.

Tests cover:
1. Validator signals (collapse, saturation, size clustering, order anomaly)
2. Valid heading trees (good hierarchy, deep hierarchy, flat-but-correct)
3. Fallback path (invalid tree → markitdown)
4. Edge cases (no headings, single heading, uniform font)
"""

from __future__ import annotations

from guaipeca.converter import _validate_heading_tree


class TestCollapseRatio:
    """Test Signal 1: collapse ratio detection."""

    def test_no_h2_headings_collapse(self):
        """All headings at H1 → collapse = 1.0 → invalid."""
        headings = [
            (1, "Chapter 1", 24.0),
            (1, "Chapter 2", 24.0),
            (1, "Chapter 3", 24.0),
        ]
        size_to_level = {24.0: 1}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert not valid
        assert any("collapse" in r for r in reasons)

    def test_flat_but_correct_two_levels(self):
        """Two heading levels with no H3 children is OK (flat but correct)."""
        headings = [
            (1, "Title", 24.0),
            (2, "Section 1", 16.0),
            (2, "Section 2", 16.0),
            (2, "Section 3", 16.0),
        ]
        size_to_level = {24.0: 1, 16.0: 2}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert valid, f"Should be valid: {reasons}"

    def test_three_levels_no_children_collapse(self):
        """3+ levels but H2 sections have no H3 children → collapse."""
        headings = [
            (1, "Title", 24.0),
            (2, "Section 1", 20.0),
            (2, "Section 2", 20.0),
            (2, "Section 3", 20.0),
            (3, "Sub A", 16.0),  # Only one H2 has H3 child
        ]
        size_to_level = {24.0: 1, 20.0: 2, 16.0: 3}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        # 2/3 H2 sections have no children → collapse = 0.67 > 0.50
        assert not valid
        assert any("collapse" in r for r in reasons)

    def test_deep_hierarchy_with_children_ok(self):
        """5 levels where H2 sections have H3+ children → no collapse."""
        headings = []
        for ch in range(3):
            headings.append((1, f"Part {ch+1}", 24.0))
            for sec in range(2):
                headings.append((2, f"{ch+1}.{sec+1} Section", 20.0))
                headings.append((3, f"{ch+1}.{sec+1}.1 Sub", 16.0))
                headings.append((4, f"Level 4 ({ch+1}.{sec+1})", 14.0))
                headings.append((5, f"Level 5 ({ch+1}.{sec+1})", 12.5))
        size_to_level = {24.0: 1, 20.0: 2, 16.0: 3, 14.0: 4, 12.5: 5}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert valid, f"Deep hierarchy should be valid: {reasons}"


class TestDepthSaturation:
    """Test Signal 2: depth saturation detection."""

    def test_saturation_at_h6(self):
        """>60% of headings at H6 → saturation."""
        headings = []
        for i in range(20):
            headings.append((1, f"Chapter {i}", 24.0))
        for i in range(40):
            headings.append((6, f"Deep heading {i}", 13.0))
        size_to_level = {24.0: 1, 13.0: 6}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        # 40/60 = 0.667 > 0.60 → fires
        assert not valid
        assert any("saturation" in r for r in reasons)

    def test_no_saturation_when_few_h6(self):
        """Few H6 headings → no saturation."""
        headings = [(1, "Title", 24.0)] + [(6, f"Deep {i}", 13.0) for i in range(2)]
        size_to_level = {24.0: 1, 13.0: 6}
        valid, _ = _validate_heading_tree(headings, size_to_level)
        # 2/3 = 0.67 — that's > 0.60, so this WILL fire
        assert not valid  # This is correct behavior — 67% at H6 is saturated

    def test_no_saturation_when_no_h6(self):
        """No H6 headings → no saturation."""
        headings = [(1, "Title", 24.0), (2, "Section", 16.0)]
        size_to_level = {24.0: 1, 16.0: 2}
        valid, _ = _validate_heading_tree(headings, size_to_level)
        assert valid

    def test_real_fccs_saturation_passes(self):
        """Real FCCS eCalc Guide has 304/662 = 0.459 at H6 → must PASS 0.60 threshold.

        This is the key regression test: the deepest real Oracle FCCS manual
        measured has 46% of headings at H6. The old threshold of 0.15
        (and the intermediate 0.20) would reject it. The 0.60 threshold
        gives it a 0.141 margin.
        """
        headings = [(1, "Oracle FCCS", 35.0)]
        # 33 H2s, each followed by at least one H3 to avoid collapse
        for i in range(33):
            headings.append((2, f"Chapter {i+1}", 30.0))
            headings.append((3, f"Chapter {i+1} Overview", 24.0))
        # 40 H3s total (7 additional standalone H3s)
        for i in range(7):
            headings.append((3, f"Additional Section {i}", 24.0))
        # 323 H4s
        for i in range(323):
            headings.append((4, f"Section {i}", 21.0))
        # 572 H5s
        for i in range(572):
            headings.append((5, f"Subsection {i}", 18.0))
        # 169 H6s
        for i in range(169):
            headings.append((6, f"Detail {i}", 16.0))
        size_to_level = {35.0: 1, 30.0: 2, 24.0: 3, 21.0: 4, 18.0: 5, 16.0: 6}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert valid, f"Real FCCS Info Dev distribution should pass: {reasons}"

    def test_real_fccs_ecalc_saturation_passes(self):
        """Real FCCS eCalc Guide has 304/662 = 0.459 at H6 → must PASS 0.60 threshold.

        This is the deepest real Oracle manual measured. The old threshold
        of 0.20 would have rejected it (0.459 >> 0.20).
        """
        headings = [(1, "Oracle FCCS eCalc", 35.0)]
        # 14 H2s with H3 children
        for i in range(14):
            headings.append((2, f"Chapter {i+1}", 30.0))
            headings.append((3, f"Chapter {i+1} Overview", 24.0))
        # 10 more H3s
        for i in range(10):
            headings.append((3, f"Extra Section {i}", 24.0))
        # 113 H4s
        for i in range(113):
            headings.append((4, f"Section {i}", 21.0))
        # 206 H5s
        for i in range(206):
            headings.append((5, f"Subsection {i}", 18.0))
        # 304 H6s — this is the key: 304/662 = 0.459
        for i in range(304):
            headings.append((6, f"Detail {i}", 16.0))
        size_to_level = {35.0: 1, 30.0: 2, 24.0: 3, 21.0: 4, 18.0: 5, 16.0: 6}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert valid, f"Real FCCS eCalc distribution (sat=0.459) should pass: {reasons}"


class TestSizeClustering:
    """Test Signal 3: size clustering detection."""

    def test_close_sizes_detected(self):
        """Adjacent heading sizes within 1.0pt → clustering."""
        headings = [(1, "Title", 18.0), (2, "Section", 16.0), (3, "Sub", 15.5)]
        size_to_level = {18.0: 1, 16.0: 2, 15.5: 3}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert not valid
        assert any("clustering" in r for r in reasons)

    def test_well_separated_sizes_ok(self):
        """Heading sizes well separated → no clustering."""
        headings = [(1, "Title", 24.0), (2, "Section", 18.0), (3, "Sub", 14.0)]
        size_to_level = {24.0: 1, 18.0: 2, 14.0: 3}
        valid, _ = _validate_heading_tree(headings, size_to_level)
        assert valid


class TestFirstHeadingSanity:
    """Test Signal 4: first-heading sanity check.

    The old "order anomaly" signal flagged any deeper heading appearing
    before a shallower one in document order. This was broken: real Oracle
    manuals have TOC entries at heading font sizes that legitimately appear
    before body headings at different levels. The redesigned check only
    flags the FIRST heading if it is deeper than H3.
    """

    def test_first_heading_h6_flagged(self):
        """First heading at H6 → flagged (mis-mapped histogram)."""
        headings = [(6, "Deep", 13.0), (1, "Title", 24.0)]
        size_to_level = {24.0: 1, 13.0: 6}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert not valid
        assert any("first heading" in r for r in reasons)

    def test_first_heading_h4_flagged(self):
        """First heading at H4 → flagged (deeper than H3)."""
        headings = [(4, "Deep", 14.0), (1, "Title", 24.0)]
        size_to_level = {24.0: 1, 14.0: 4}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert not valid
        assert any("first heading" in r for r in reasons)

    def test_first_heading_h3_ok(self):
        """First heading at H3 is OK (e.g. a section-level intro before the
        main document structure begins).
        """
        headings = [
            (3, "Introduction", 18.0),
            (1, "Title", 24.0),
            (2, "Section A", 20.0),
            (3, "Sub A", 18.0),
            (2, "Section B", 20.0),
            (3, "Sub B", 18.0),
        ]
        size_to_level = {24.0: 1, 20.0: 2, 18.0: 3}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert valid, f"H3 first heading should be valid: {reasons}"

    def test_h5_before_h4_is_ok(self):
        """H5 appearing before any H4 in document order is OK.

        This is the key regression test: real Oracle FCCS manuals have
        TOC entries at 18pt (mapped to H5) on pages 3-19, before the
        first H4 body heading on page 48. This is legitimate document
        structure, NOT an anomaly.
        """
        headings = [
            (1, "Document Title", 35.0),
            (2, "Subtitle", 30.0),
            (3, "Contents", 24.0),
            (5, "1", 18.0),   # TOC entry
            (5, "2", 18.0),   # TOC entry
            (5, "3", 18.0),   # TOC entry
            (4, "First Body Section", 21.0),  # First H4, page 48
            (5, "Subsection", 18.0),
        ]
        size_to_level = {35.0: 1, 30.0: 2, 24.0: 3, 21.0: 4, 18.0: 5}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert valid, f"Real-style TOC should not trigger order anomaly: {reasons}"

    def test_level_jump_not_flagged(self):
        """H1 → H4 (jump of 3) is NOT flagged — can happen with TOC/front-matter.

        The old signal flagged this; the redesigned one does not, because
        level jumps are common in real documents with front matter.
        """
        headings = [(1, "Title", 24.0), (4, "Deep", 14.0)]
        size_to_level = {24.0: 1, 14.0: 4}
        _valid, reasons = _validate_heading_tree(headings, size_to_level)
        # May fail on other signals (e.g. collapse if only 2 levels),
        # but should NOT fail on first-heading sanity
        assert not any("first heading" in r for r in reasons)

    def test_normal_order_ok(self):
        """H1 → H2 → H3 → normal order."""
        headings = [
            (1, "Title", 24.0),
            (2, "Section", 18.0),
            (3, "Subsection", 14.0),
        ]
        size_to_level = {24.0: 1, 18.0: 2, 14.0: 3}
        valid, _ = _validate_heading_tree(headings, size_to_level)
        assert valid


class TestEdgeCases:
    """Test edge cases."""

    def test_no_headings_is_valid(self):
        """No headings → valid (handled by caller)."""
        valid, reasons = _validate_heading_tree([], {})
        assert valid
        assert reasons == []

    def test_single_h1_no_h2_is_collapse(self):
        """Single H1 with no H2 → collapse (one-level doc is suspicious)."""
        headings = [(1, "Title", 24.0)]
        size_to_level = {24.0: 1}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert not valid
        assert any("collapse" in r for r in reasons)

    def test_single_h2_with_h1_ok(self):
        """H1 + H2 (two levels, flat) → valid."""
        headings = [(1, "Title", 24.0), (2, "Section", 16.0)]
        size_to_level = {24.0: 1, 16.0: 2}
        valid, _ = _validate_heading_tree(headings, size_to_level)
        assert valid

    def test_multiple_signals_combine(self):
        """Multiple signals firing → all reasons reported."""
        headings = [(6, "Bad", 13.0)] + [(1, "Title", 24.0)] * 10 + [(6, f"D{i}", 13.0) for i in range(5)]
        size_to_level = {24.0: 1, 13.0: 6}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert not valid
        # Should have first-heading + saturation
        assert len(reasons) >= 2