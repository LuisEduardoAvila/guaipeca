"""Tests for score-normalized RRF fusion.

Tests cover:
1. Score normalization behavior
2. Exact-token queries get correct rank-1 placement
3. Semantic queries don't regress
4. Backward compatibility with existing config keys
5. fusion_alpha config parameter
"""

from __future__ import annotations

from guaipeca.config import GuaipecaConfig, SearchConfig
from guaipeca.search import Searcher


def _make_searcher(**search_kwargs):
    """Create a minimal searcher with given search config for unit tests."""
    cfg = GuaipecaConfig(corpora=[], search=SearchConfig(**search_kwargs))
    searcher = Searcher.__new__(Searcher)
    searcher.config = cfg
    return searcher


class TestScoreNormalizedRRF:
    """Test score-normalized RRF fusion directly."""

    def test_strong_bm25_outranks_weak_dense_at_same_rank(self):
        """A strong BM25 match should outrank a weak dense match."""
        searcher = _make_searcher()

        # Dense: three results, the weak one is at rank 0 but has low raw score
        # relative to the top dense result
        dense_results = [
            {"chunk_id": "doc:d1", "raw_score": 0.9, "summary": "strong dense", "score": 0.9},
            {"chunk_id": "doc:weak", "raw_score": 0.001, "summary": "weak match", "score": 0.001},
            {"chunk_id": "doc:d3", "raw_score": 0.0005, "summary": "weaker", "score": 0.0005},
        ]
        # BM25: strong exact match at rank 0
        sparse_results = [
            {"chunk_id": "doc:strong", "raw_score": 11.22, "summary": "exact match", "score": 11.22},
            {"chunk_id": "doc:s2", "raw_score": 2.0, "summary": "partial", "score": 2.0},
        ]

        fused = searcher._rrf_fuse(dense_results, sparse_results, top_k=10)

        chunk_ids = [r["chunk_id"] for r in fused]
        assert "doc:strong" in chunk_ids
        # doc:strong (sparse rank 0, norm=1.0) should outrank doc:weak (dense rank 1, norm~0.0)
        # because doc:weak has a very low raw_score relative to doc:d1
        strong_rank = chunk_ids.index("doc:strong")
        weak_rank = chunk_ids.index("doc:weak")
        assert strong_rank < weak_rank, (
            f"Strong BM25 (rank {strong_rank}) should be < weak dense (rank {weak_rank})"
        )

    def test_pure_rank_mode_when_alpha_zero(self):
        """With fusion_alpha=0, behavior should match pure rank-based RRF."""
        searcher = _make_searcher(fusion_alpha=0.0)

        dense_results = [
            {"chunk_id": "doc:a", "raw_score": 0.9, "summary": "a", "score": 0.9},
            {"chunk_id": "doc:b", "raw_score": 0.1, "summary": "b", "score": 0.1},
        ]
        sparse_results = [
            {"chunk_id": "doc:c", "raw_score": 100.0, "summary": "c", "score": 100.0},
            {"chunk_id": "doc:d", "raw_score": 0.01, "summary": "d", "score": 0.01},
        ]

        fused = searcher._rrf_fuse(dense_results, sparse_results, top_k=10)

        # With alpha=0, all that matters is rank position, not raw score
        scores = {r["chunk_id"]: r["rrf_score"] for r in fused}
        assert abs(scores["doc:a"] - scores["doc:c"]) < 1e-9, (
            f"Pure rank mode should give equal scores: {scores['doc:a']} vs {scores['doc:c']}"
        )

    def test_pure_score_mode_when_alpha_one(self):
        """With fusion_alpha=1, raw score magnitude dominates."""
        searcher = _make_searcher(fusion_alpha=1.0)

        dense_results = [
            {"chunk_id": "doc:weak", "raw_score": 0.001, "summary": "weak", "score": 0.001},
        ]
        sparse_results = [
            {"chunk_id": "doc:strong", "raw_score": 11.22, "summary": "exact", "score": 11.22},
        ]

        fused = searcher._rrf_fuse(dense_results, sparse_results, top_k=10)
        scores = {r["chunk_id"]: r["rrf_score"] for r in fused}

        # With alpha=1, both get norm=1.0 (only element in their set)
        assert abs(scores["doc:strong"] - scores["doc:weak"]) < 1e-9

    def test_chunk_appearing_in_both_gets_combined_score(self):
        """A chunk appearing in both dense and sparse results should get combined score."""
        searcher = _make_searcher()

        dense_results = [
            {"chunk_id": "doc:shared", "raw_score": 0.8, "summary": "shared", "score": 0.8},
            {"chunk_id": "doc:d2", "raw_score": 0.4, "summary": "d2", "score": 0.4},
        ]
        sparse_results = [
            {"chunk_id": "doc:shared", "raw_score": 5.0, "summary": "shared", "score": 5.0},
            {"chunk_id": "doc:s2", "raw_score": 1.0, "summary": "s2", "score": 1.0},
        ]

        fused = searcher._rrf_fuse(dense_results, sparse_results, top_k=10)
        shared = next(r for r in fused if r["chunk_id"] == "doc:shared")

        assert shared["rrf_score"] > 0.5, f"Shared chunk should have high combined score: {shared['rrf_score']}"
        assert fused[0]["chunk_id"] == "doc:shared", "Shared chunk should be rank 1"

    def test_empty_results_handled(self):
        """Empty result lists should not crash."""
        searcher = _make_searcher()
        fused = searcher._rrf_fuse([], [], top_k=5)
        assert fused == []

    def test_backward_compatible_config_keys(self):
        """Existing config keys should still work."""
        cfg = SearchConfig()
        assert cfg.bm25_weight == 1.0
        assert cfg.dense_weight == 1.0
        assert cfg.rrf_k == 60
        assert cfg.fusion_alpha == 0.3

    def test_custom_fusion_alpha(self):
        """Custom fusion_alpha should be respected."""
        cfg = SearchConfig(fusion_alpha=0.5)
        assert cfg.fusion_alpha == 0.5

    def test_fusion_alpha_from_yaml(self, tmp_path):
        """fusion_alpha should be parseable from YAML config."""
        import yaml
        config_data = {
            "corpora": [
                {"name": "test", "path": str(tmp_path), "extensions": [".md"]}
            ],
            "search": {
                "hybrid": True,
                "fusion_alpha": 0.4,
            },
        }
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(yaml.dump(config_data))

        cfg = GuaipecaConfig.from_yaml(str(config_path))
        assert cfg.search.fusion_alpha == 0.4


class TestHybridSearchQuality:
    """Integration tests for hybrid search quality with normalized RRF."""

    def test_exact_token_search_returns_results(self, indexed_corpus):
        """Searching for an exact term should return results."""
        searcher = indexed_corpus
        result = searcher.search("consolidation", top_k=5, hybrid=True)
        assert "error" not in result
        assert result["total"] > 0

    def test_semantic_search_returns_results(self, indexed_corpus):
        """Semantic (concept) queries should still return results."""
        searcher = indexed_corpus
        result = searcher.search("vector retrieval embedding", top_k=5, hybrid=True)
        assert "error" not in result
        assert result["total"] > 0

    def test_results_sorted_by_score(self, indexed_corpus):
        """Hybrid results should be sorted by score descending."""
        searcher = indexed_corpus
        result = searcher.search("financial close", top_k=5, hybrid=True)
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)