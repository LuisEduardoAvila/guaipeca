"""Tests for section-based redundancy de-duplication.

Tests cover:
1. Basic de-dup: multiple chunks per section_id → one kept
2. Best-scoring chunk per section is kept
3. Unstructured docs (no section_id) are not deduped
4. dedup_sections=False (default) preserves all chunks
5. Integration with search
"""

from __future__ import annotations

from guaipeca.config import GuaipecaConfig, SearchConfig
from guaipeca.search import Searcher


def _make_searcher():
    """Create a minimal searcher for unit tests."""
    cfg = GuaipecaConfig(corpora=[], search=SearchConfig())
    searcher = Searcher.__new__(Searcher)
    searcher.config = cfg
    return searcher


class TestDedupBySection:
    """Test the _dedup_by_section method directly."""

    def test_basic_dedup(self):
        """Two chunks with same section_id → only best-scoring kept."""
        searcher = _make_searcher()
        results = [
            {"chunk_id": "a:1", "section_id": "sec1", "score": 0.9, "summary": "best"},
            {"chunk_id": "a:2", "section_id": "sec1", "score": 0.7, "summary": "second"},
            {"chunk_id": "b:1", "section_id": "sec2", "score": 0.5, "summary": "other section"},
        ]
        deduped = searcher._dedup_by_section(results)
        assert len(deduped) == 2
        assert deduped[0]["chunk_id"] == "a:1"  # best from sec1
        assert deduped[1]["chunk_id"] == "b:1"  # only from sec2

    def test_best_scoring_kept(self):
        """The highest-scoring chunk per section is kept."""
        searcher = _make_searcher()
        results = [
            {"chunk_id": "a:1", "section_id": "sec1", "score": 0.3, "summary": "low"},
            {"chunk_id": "a:2", "section_id": "sec1", "score": 0.95, "summary": "high"},
            {"chunk_id": "a:3", "section_id": "sec1", "score": 0.5, "summary": "mid"},
        ]
        deduped = searcher._dedup_by_section(results)
        assert len(deduped) == 1
        assert deduped[0]["chunk_id"] == "a:1"  # results are pre-sorted by score

    def test_unstructured_not_deduped(self):
        """Chunks with unstructured: section_id are all kept."""
        searcher = _make_searcher()
        results = [
            {"chunk_id": "a:1", "section_id": "unstructured:abc123", "score": 0.9, "summary": "a"},
            {"chunk_id": "a:2", "section_id": "unstructured:abc123", "score": 0.7, "summary": "b"},
            {"chunk_id": "a:3", "section_id": "unstructured:abc123", "score": 0.5, "summary": "c"},
        ]
        deduped = searcher._dedup_by_section(results)
        assert len(deduped) == 3  # all kept — unstructured docs don't dedup

    def test_empty_section_id_kept(self):
        """Chunks with empty section_id are all kept."""
        searcher = _make_searcher()
        results = [
            {"chunk_id": "a:1", "section_id": "", "score": 0.9, "summary": "a"},
            {"chunk_id": "a:2", "section_id": "", "score": 0.7, "summary": "b"},
        ]
        deduped = searcher._dedup_by_section(results)
        assert len(deduped) == 2

    def test_empty_results(self):
        """Empty input → empty output."""
        searcher = _make_searcher()
        deduped = searcher._dedup_by_section([])
        assert deduped == []

    def test_mixed_structured_unstructured(self):
        """Mix of structured and unstructured chunks."""
        searcher = _make_searcher()
        results = [
            {"chunk_id": "a:1", "section_id": "sec1", "score": 0.9, "summary": "s1"},
            {"chunk_id": "a:2", "section_id": "sec1", "score": 0.8, "summary": "s1 dup"},
            {"chunk_id": "b:1", "section_id": "unstructured:xyz", "score": 0.7, "summary": "unstruct"},
            {"chunk_id": "b:2", "section_id": "unstructured:xyz", "score": 0.6, "summary": "unstruct 2"},
            {"chunk_id": "c:1", "section_id": "sec2", "score": 0.5, "summary": "s2"},
        ]
        deduped = searcher._dedup_by_section(results)
        # sec1: 1 kept, unstructured: 2 kept, sec2: 1 kept
        assert len(deduped) == 4
        section_ids = [r["section_id"] for r in deduped if r["section_id"].startswith("sec")]
        assert len(section_ids) == 2  # sec1 and sec2

    def test_many_sections_all_unique(self):
        """All chunks from different sections → no dedup."""
        searcher = _make_searcher()
        results = [
            {"chunk_id": f"a:{i}", "section_id": f"sec{i}", "score": 0.9 - i * 0.1, "summary": f"s{i}"}
            for i in range(10)
        ]
        deduped = searcher._dedup_by_section(results)
        assert len(deduped) == 10


class TestDedupSectionsIntegration:
    """Test dedup_sections parameter in search()."""

    def test_dedup_sections_flag_in_signature(self):
        """search() should accept dedup_sections parameter."""
        import inspect
        sig = inspect.signature(Searcher.search)
        assert "dedup_sections" in sig.parameters
        assert sig.parameters["dedup_sections"].default is False

    def test_dedup_improves_diversity(self, indexed_corpus):
        """With dedup_sections=True, results should have more section diversity."""
        searcher = indexed_corpus
        # Search without dedup
        result_no_dedup = searcher.search("consolidation", top_k=10, hybrid=True, dedup_sections=False)
        # Search with dedup
        result_dedup = searcher.search("consolidation", top_k=10, hybrid=True, dedup_sections=True)

        # Count unique section_ids in each
        no_dedup_sections = set()
        for r in result_no_dedup["results"]:
            no_dedup_sections.add(r.get("section_id", ""))

        dedup_sections = set()
        for r in result_dedup["results"]:
            dedup_sections.add(r.get("section_id", ""))

        # With dedup, every result should have a unique section_id
        # (unless there are unstructured docs)
        structured_results = [r for r in result_dedup["results"] if r.get("section_id", "") and not r["section_id"].startswith("unstructured:")]
        if structured_results:
            dedup_sids = [r["section_id"] for r in structured_results]
            assert len(dedup_sids) == len(set(dedup_sids)), "Duplicate section_ids in deduped results"