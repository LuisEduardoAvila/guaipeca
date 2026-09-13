"""Hybrid BM25 + FAISS search integration tests.

Tests cover:
1. BM25 Index Creation
2. BM25 Search
3. Hybrid Search (RRF Fusion)
4. Backward Compatibility (dense-only)
5. Config-Driven Hybrid Mode
6. MCP Tool Hybrid Parameter
7. BM25 Index Lifecycle
8. Concurrent Hybrid Searches
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
import numpy as np

from guaipeca.config import (
    GuaipecaConfig, CorpusConfig, ChunkingConfig, IndexingConfig,
    SearchConfig,
)
from guaipeca.embedding import EmbeddingService
from guaipeca.search import Searcher
from guaipeca.index import CorpusIndex


# ===========================================================================
# 1. BM25 Index Creation
# ===========================================================================

class TestBM25IndexCreation:
    """Test that BM25 index is created during indexing."""

    def test_bm25_index_dir_exists(self, indexed_corpus):
        """After indexing, BM25 index directory should exist."""
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]
        assert corpus_idx.bm25_path.exists(), (
            f"BM25 index directory not found at {corpus_idx.bm25_path}"
        )

    def test_bm25_retriever_loaded(self, indexed_corpus):
        """After indexing, BM25 retriever should be available."""
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]
        with corpus_idx._rwlock.read_lock():
            corpus_idx._load()
            assert corpus_idx._bm25_retriever is not None, "BM25 retriever is None after indexing"

    def test_bm25_stats_available(self, indexed_corpus):
        """Stats should report BM25 availability."""
        searcher = indexed_corpus
        status = searcher.status
        for c in status["corpora"]:
            assert "bm25_available" in c, "Stats missing 'bm25_available'"
        # At least one corpus should have BM25 available
        assert any(c["bm25_available"] for c in status["corpora"]), (
            "No corpus has BM25 available"
        )


# ===========================================================================
# 2. BM25 Search
# ===========================================================================

class TestBM25Search:
    """Test BM25 keyword search."""

    def test_bm25_search_returns_results(self, indexed_corpus):
        """BM25 search for a common term should return results."""
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]
        results = corpus_idx.search_bm25("consolidation", top_k=5)
        assert len(results) > 0, "No BM25 results for 'consolidation'"

    def test_bm25_search_results_have_required_fields(self, indexed_corpus):
        """BM25 results should have same fields as dense results."""
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]
        results = corpus_idx.search_bm25("financial", top_k=3)
        for r in results:
            assert "chunk_id" in r
            assert "summary" in r
            assert "location" in r
            assert "corpus" in r
            assert "topic" in r
            assert "score" in r
            assert "raw_score" in r

    def test_bm25_search_exact_term(self, indexed_corpus):
        """BM25 search for exact terms should find chunks containing those terms."""
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]
        # Search for terms likely in the test document
        results = corpus_idx.search_bm25("consolidation", top_k=10)
        assert len(results) > 0, "No BM25 results for 'consolidation'"

    def test_bm25_search_respects_top_k(self, indexed_corpus):
        """BM25 search should respect top_k limit."""
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]
        results = corpus_idx.search_bm25("financial", top_k=3)
        assert len(results) <= 3, f"Expected <=3 results, got {len(results)}"

    def test_bm25_search_empty_corpus(self, indexed_corpus):
        """BM25 search on empty query should still return results (or empty)."""
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]
        # Very specific query that might not match
        results = corpus_idx.search_bm25("zzzznonexistentterm12345", top_k=5)
        # Should return empty or very few results
        assert len(results) <= 5, "Should return at most top_k results"

    def test_bm25_results_have_scores(self, indexed_corpus):
        """BM25 results should have numeric scores."""
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]
        results = corpus_idx.search_bm25("financial", top_k=5)
        for r in results:
            assert isinstance(r["score"], float), f"Score is not float: {type(r['score'])}"
            assert isinstance(r["raw_score"], float), f"raw_score is not float: {type(r['raw_score'])}"


# ===========================================================================
# 3. Hybrid Search (RRF Fusion)
# ===========================================================================

class TestHybridSearch:
    """Test hybrid BM25 + FAISS search with RRF fusion."""

    def test_hybrid_search_returns_results(self, indexed_corpus):
        """Hybrid search should return results."""
        searcher = indexed_corpus
        result = searcher.search("consolidation", top_k=5, hybrid=True)
        assert "error" not in result, f"Search error: {result.get('error')}"
        assert result["total"] > 0, "No hybrid search results"
        assert result.get("hybrid") is True, "Result should indicate hybrid mode"

    def test_hybrid_search_has_hybrid_flag(self, indexed_corpus):
        """Hybrid search result should have hybrid=True flag."""
        searcher = indexed_corpus
        result = searcher.search("financial", top_k=3, hybrid=True)
        assert result.get("hybrid") is True

    def test_hybrid_search_fuses_both_methods(self, indexed_corpus):
        """Hybrid search should consider both dense and sparse results."""
        searcher = indexed_corpus
        # Search with hybrid
        hybrid_result = searcher.search("consolidation", top_k=10, hybrid=True)
        # Search with dense only
        dense_result = searcher.search("consolidation", top_k=10, hybrid=False)

        # Both should return results
        assert hybrid_result["total"] > 0
        assert dense_result["total"] > 0

        # Hybrid might find different or additional results due to BM25
        hybrid_ids = {r["chunk_id"] for r in hybrid_result["results"]}
        dense_ids = {r["chunk_id"] for r in dense_result["results"]}

        # They should overlap (many results in common) but might differ
        overlap = hybrid_ids & dense_ids
        assert len(overlap) > 0, "No overlap between hybrid and dense results"

    def test_hybrid_search_exact_term_match(self, indexed_corpus):
        """Hybrid search should surface exact-term matches via BM25."""
        searcher = indexed_corpus
        # Use a specific term that BM25 excels at finding
        result = searcher.search("consolidation", top_k=5, hybrid=True)
        assert result["total"] > 0
        for r in result["results"]:
            assert r["chunk_id"]
            assert r["score"] > 0

    def test_hybrid_search_results_sorted_by_score(self, indexed_corpus):
        """Hybrid search results should be sorted by score descending."""
        searcher = indexed_corpus
        result = searcher.search("financial consolidation", top_k=5, hybrid=True)
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"


# ===========================================================================
# 4. Backward Compatibility (Dense-Only)
# ===========================================================================

class TestBackwardCompatibility:
    """Test that dense-only mode is backward compatible."""

    def test_dense_only_no_hybrid_flag(self, indexed_corpus):
        """Search without hybrid param should work as before."""
        searcher = indexed_corpus
        result = searcher.search("consolidation", top_k=5)
        assert "error" not in result
        assert result["total"] > 0
        # Should not have hybrid=True
        assert result.get("hybrid") is False or "hybrid" not in result or result.get("hybrid") is False

    def test_dense_only_explicit_false(self, indexed_corpus):
        """Search with hybrid=False should behave like dense-only."""
        searcher = indexed_corpus
        result = searcher.search("consolidation", top_k=5, hybrid=False)
        assert "error" not in result
        assert result["total"] > 0
        assert result.get("hybrid") is False

    def test_dense_results_same_as_before(self, indexed_corpus):
        """Dense-only results should match original behavior (no BM25)."""
        searcher = indexed_corpus
        result = searcher.search("consolidation", top_k=5, hybrid=False)
        # Should return same number of results as before hybrid feature
        assert result["total"] > 0
        for r in result["results"]:
            assert "chunk_id" in r
            assert "score" in r
            assert isinstance(r["score"], float)


# ===========================================================================
# 5. Config-Driven Hybrid Mode
# ===========================================================================

class TestConfigHybrid:
    """Test config-driven hybrid mode."""

    def test_search_config_defaults(self):
        """SearchConfig should have safe defaults."""
        cfg = SearchConfig()
        assert cfg.hybrid is False
        assert cfg.bm25_weight == 1.0
        assert cfg.dense_weight == 1.0
        assert cfg.rrf_k == 60

    def test_config_from_yaml_search_section(self, tmp_path):
        """Config should parse search section from YAML."""
        import yaml
        config_data = {
            "corpora": [
                {"name": "test", "path": str(tmp_path), "extensions": [".md"]}
            ],
            "search": {
                "hybrid": True,
                "bm25_weight": 0.8,
                "dense_weight": 1.2,
                "rrf_k": 30,
            },
        }
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(yaml.dump(config_data))

        cfg = GuaipecaConfig.from_yaml(str(config_path))
        assert cfg.search.hybrid is True
        assert cfg.search.bm25_weight == 0.8
        assert cfg.search.dense_weight == 1.2
        assert cfg.search.rrf_k == 30

    def test_config_defaults_without_search_section(self, tmp_path):
        """Config without search section should use defaults."""
        import yaml
        config_data = {
            "corpora": [
                {"name": "test", "path": str(tmp_path), "extensions": [".md"]}
            ],
        }
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(yaml.dump(config_data))

        cfg = GuaipecaConfig.from_yaml(str(config_path))
        assert cfg.search.hybrid is False
        assert cfg.search.bm25_weight == 1.0
        assert cfg.search.dense_weight == 1.0
        assert cfg.search.rrf_k == 60

    def test_config_hybrid_overrides_param(self, indexed_corpus):
        """Explicit hybrid param should override config."""
        searcher = indexed_corpus
        # Config has hybrid=False (default), pass hybrid=True
        result = searcher.search("consolidation", top_k=5, hybrid=True)
        assert result.get("hybrid") is True

    def test_config_hybrid_false_param_true(self, indexed_corpus):
        """If config hybrid is False and param is True, should use hybrid."""
        searcher = indexed_corpus
        assert searcher.config.search.hybrid is False
        result = searcher.search("consolidation", top_k=5, hybrid=True)
        assert result.get("hybrid") is True


# ===========================================================================
# 6. MCP Tool Hybrid Parameter
# ===========================================================================

class TestMCPSearchTool:
    """Test the MCP search tool with hybrid parameter."""

    def test_search_tool_schema_has_hybrid(self):
        """Search tool schema should include hybrid parameter."""
        from guaipeca.mcp_server import TOOLS
        search_tool = next(t for t in TOOLS if t["name"] == "search")
        assert "hybrid" in search_tool["inputSchema"]["properties"]
        assert search_tool["inputSchema"]["properties"]["hybrid"]["type"] == "boolean"

    def test_mcp_search_passes_hybrid(self, indexed_corpus):
        """MCP search should pass hybrid param to searcher."""
        from guaipeca.mcp_server import GuaipecaMCPServer
        server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
        server.config = indexed_corpus.config
        server.searcher = indexed_corpus

        # Call search with hybrid=True
        result = server._handle_tool_call({
            "name": "search",
            "arguments": {"query": "consolidation", "hybrid": True},
        })
        # Parse the result
        import json
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert data.get("hybrid") is True
        assert data["total"] > 0

    def test_mcp_search_without_hybrid(self, indexed_corpus):
        """MCP search without hybrid param should work as before."""
        from guaipeca.mcp_server import GuaipecaMCPServer
        server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
        server.config = indexed_corpus.config
        server.searcher = indexed_corpus

        result = server._handle_tool_call({
            "name": "search",
            "arguments": {"query": "consolidation"},
        })
        import json
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert data["total"] > 0


# ===========================================================================
# 7. BM25 Index Lifecycle
# ===========================================================================

class TestBM25IndexLifecycle:
    """Test BM25 index lifecycle (build, save, load, reset)."""

    def test_bm25_index_resets_on_force(self, indexed_corpus):
        """Force re-index should rebuild BM25 index."""
        searcher = indexed_corpus
        stats = searcher.index_corpus(corpus_name="test-docs", force=True)
        assert "error" not in stats

        # BM25 should still be available after force re-index
        corpus_idx = searcher.corpora["test-docs"]
        with corpus_idx._rwlock.read_lock():
            corpus_idx._load()
            assert corpus_idx._bm25_retriever is not None, "BM25 retriever not available after force re-index"

    def test_bm25_search_after_force_reindex(self, indexed_corpus):
        """BM25 search should work after force re-index."""
        searcher = indexed_corpus
        searcher.index_corpus(corpus_name="test-docs", force=True)

        corpus_idx = searcher.corpora["test-docs"]
        results = corpus_idx.search_bm25("consolidation", top_k=5)
        assert len(results) > 0, "No BM25 results after force re-index"

    def test_bm25_index_persists_to_disk(self, indexed_corpus):
        """BM25 index files should persist to disk after indexing."""
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]
        assert corpus_idx.bm25_path.exists(), "BM25 index dir not on disk"


# ===========================================================================
# 8. Concurrent Hybrid Searches
# ===========================================================================

class TestConcurrentHybridSearch:
    """Test concurrent hybrid searches don't deadlock."""

    def test_concurrent_hybrid_searches_complete(self, indexed_corpus):
        """Multiple concurrent hybrid searches should complete."""
        searcher = indexed_corpus
        queries = [
            "consolidation",
            "financial close",
            "currency translation",
            "journal entry",
            "report",
        ]
        results = {}
        errors = []

        def search_task(query):
            try:
                result = searcher.search(query, top_k=5, hybrid=True)
                results[query] = result
            except Exception as e:
                errors.append((query, str(e)))

        threads = []
        for q in queries:
            t = threading.Thread(target=search_task, args=(q,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert len(errors) == 0, f"Errors in concurrent hybrid searches: {errors}"
        assert len(results) == len(queries)
        for q, r in results.items():
            assert r["total"] > 0, f"No results for concurrent hybrid query: {q}"

    def test_concurrent_mixed_hybrid_and_dense(self, indexed_corpus):
        """Mix of hybrid and dense searches should not deadlock."""
        searcher = indexed_corpus
        results = []
        errors = []
        barrier = threading.Barrier(4)

        def hybrid_task():
            barrier.wait()
            r = searcher.search("consolidation", top_k=3, hybrid=True)
            results.append(("hybrid", r))

        def dense_task():
            barrier.wait()
            r = searcher.search("consolidation", top_k=3, hybrid=False)
            results.append(("dense", r))

        threads = [
            threading.Thread(target=hybrid_task),
            threading.Thread(target=dense_task),
            threading.Thread(target=hybrid_task),
            threading.Thread(target=dense_task),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert len(errors) == 0
        assert len(results) == 4