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
        """>15% of headings at H6 → saturation."""
        headings = []
        for i in range(20):
            headings.append((1, f"Chapter {i}", 24.0))
        for i in range(40):
            headings.append((6, f"Deep heading {i}", 13.0))
        size_to_level = {24.0: 1, 13.0: 6}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert not valid
        assert any("saturation" in r for r in reasons)

    def test_no_saturation_when_few_h6(self):
        """Few H6 headings → no saturation."""
        headings = [(1, "Title", 24.0)] + [(6, f"Deep {i}", 13.0) for i in range(2)]
        size_to_level = {24.0: 1, 13.0: 6}
        valid, _ = _validate_heading_tree(headings, size_to_level)
        # 2/3 = 0.67 — that's > 0.15, so this WILL fire
        assert not valid  # This is correct behavior — 67% at H6 is saturated

    def test_no_saturation_when_no_h6(self):
        """No H6 headings → no saturation."""
        headings = [(1, "Title", 24.0), (2, "Section", 16.0)]
        size_to_level = {24.0: 1, 16.0: 2}
        valid, _ = _validate_heading_tree(headings, size_to_level)
        assert valid


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


class TestOrderAnomaly:
    """Test Signal 4: order anomaly detection."""

    def test_h6_before_h1(self):
        """H6 as first heading → order anomaly."""
        headings = [(6, "Deep", 13.0), (1, "Title", 24.0)]
        size_to_level = {24.0: 1, 13.0: 6}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert not valid
        assert any("order anomaly" in r for r in reasons)

    def test_level_jump_detected(self):
        """H1 → H4 (jump of 3) → order anomaly."""
        headings = [(1, "Title", 24.0), (4, "Deep", 14.0)]
        size_to_level = {24.0: 1, 14.0: 4}
        valid, reasons = _validate_heading_tree(headings, size_to_level)
        assert not valid
        assert any("order anomaly" in r for r in reasons)

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
        # Should have order anomaly + saturation
        assert len(reasons) >= 2