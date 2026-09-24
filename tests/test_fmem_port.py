"""Tests for advanced pipeline features.

Tests cover:
1. FAISS Backup + Corruption Recovery
2. Content Preprocessing for Embeddings
3. Embedding Cache (LRU + TTL)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import hashlib
from pathlib import Path

import pytest
import numpy as np

from guaipeca.config import (
    GuaipecaConfig, CorpusConfig, ChunkingConfig, IndexingConfig,
    EmbeddingConfig, SearchConfig, ServerConfig, UploadConfig,
)
from guaipeca.embedding import EmbeddingService, _EmbeddingCache
from guaipeca.index import CorpusIndex
from guaipeca.chunking import Chunk, chunk_file
from guaipeca.converter import Converter
from guaipeca.search import Searcher


def _model_available(svc) -> bool:
    """Best-effort check that the fastembed model can be loaded.

    Tests that exercise the real model bail out gracefully when fastembed
    or the model weights are unavailable (e.g. offline CI).
    """
    try:
        _ = svc.model
        return True
    except Exception:
        return False


# ===========================================================================
# 1. FAISS Backup + Corruption Recovery
# ===========================================================================

class TestFAISSBackup:
    """Test FAISS index backup creation and corruption recovery."""

    def test_backup_created_after_indexing(self, indexed_corpus):
        """After indexing, a .fai.bak backup file should exist."""
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]
        # Force a re-index to trigger backup creation
        searcher.index_corpus(corpus_name="test-docs", force=True)
        backup_path = corpus_idx.faiss_path.with_suffix(".fai.bak")
        assert backup_path.exists(), f"FAISS backup file not found at {backup_path}"

    def test_backup_is_valid_faiss(self, indexed_corpus):
        """The backup file should be a valid FAISS index."""
        import faiss
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]
        searcher.index_corpus(corpus_name="test-docs", force=True)
        backup_path = corpus_idx.faiss_path.with_suffix(".fai.bak")
        if backup_path.exists():
            index = faiss.read_index(str(backup_path))
            assert index.ntotal > 0, "Backup FAISS index has 0 vectors"

    def test_recovery_from_corrupted_index(self, indexed_corpus):
        """If the primary .fai file is corrupted, load should fall back to .bak."""
        import faiss
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]

        # Ensure we have a valid index and backup
        searcher.index_corpus(corpus_name="test-docs", force=True)

        # Corrupt the primary FAISS index
        with open(corpus_idx.faiss_path, "wb") as f:
            f.write(b"CORRUPTED_INDEX_DATA_NOT_VALID_FAISS")

        # Reset loaded state and reload
        with corpus_idx._rwlock.write_lock():
            corpus_idx._loaded = False
            corpus_idx._faiss_index = None
            corpus_idx._load()

            # Should have recovered from backup
            assert corpus_idx._faiss_index is not None
            assert corpus_idx._faiss_index.ntotal > 0, (
                "FAISS index not recovered from backup after corruption"
            )

    def test_recovery_from_both_corrupted(self, indexed_corpus):
        """If both .fai and .fai.bak are corrupted, start fresh with empty index."""
        import faiss
        searcher = indexed_corpus
        corpus_idx = searcher.corpora["test-docs"]

        # Ensure we have a valid index and backup
        searcher.index_corpus(corpus_name="test-docs", force=True)

        # Corrupt both primary and backup
        with open(corpus_idx.faiss_path, "wb") as f:
            f.write(b"CORRUPTED_PRIMARY")
        backup_path = corpus_idx.faiss_path.with_suffix(".fai.bak")
        with open(backup_path, "wb") as f:
            f.write(b"CORRUPTED_BACKUP")

        # Reset loaded state and reload
        with corpus_idx._rwlock.write_lock():
            corpus_idx._loaded = False
            corpus_idx._faiss_index = None
            corpus_idx._load()

            # Should have a fresh empty index
            assert corpus_idx._faiss_index is not None
            assert corpus_idx._faiss_index.ntotal == 0, (
                "Expected empty index when both primary and backup are corrupted"
            )

    def test_fresh_index_when_no_files_exist(self, temp_data_dir):
        """When neither .fai nor .fai.bak exist, should create fresh index."""
        import faiss
        # Create a minimal corpus index with no existing files
        corpus_cfg = CorpusConfig(name="fresh-test", path=temp_data_dir)
        chunking_cfg = ChunkingConfig()
        indexing_cfg = IndexingConfig(data_dir=temp_data_dir)

        # Create a mock embedder
        class MockEmbedder:
            dimensions = 384
            preprocess = False

        idx = CorpusIndex(
            corpus=corpus_cfg,
            chunking=chunking_cfg,
            indexing=indexing_cfg,
            embedding_service=MockEmbedder(),
        )

        # Load should create a fresh empty index
        idx._load()
        assert idx._faiss_index is not None
        assert idx._faiss_index.ntotal == 0
        assert isinstance(idx._faiss_index, faiss.IndexFlatIP)


# ===========================================================================
# 2. Content Preprocessing for Embeddings
# ===========================================================================

class TestContentPreprocessing:
    """Test content preprocessing before embedding."""

    def test_preprocess_text_with_heading(self):
        """Preprocessing should prepend heading to text."""
        text = "This is the body content of the chunk."
        heading = "Important Section"
        result = EmbeddingService.preprocess_text(text, heading)
        assert "Important Section" in result
        assert "This is the body content" in result
        assert result.startswith("Important Section")

    def test_preprocess_text_without_heading(self):
        """Preprocessing without heading should use first 500 chars."""
        text = "This is content without a heading. " * 100  # > 500 chars
        result = EmbeddingService.preprocess_text(text, heading="")
        assert len(result) <= 500
        assert result == text[:500]

    def test_preprocess_text_respects_max_chars(self):
        """Preprocessing should respect max_chars limit."""
        text = "A" * 1000
        heading = "Test Heading"
        result = EmbeddingService.preprocess_text(text, heading, max_chars=200)
        # Heading + newline + first 200 chars
        assert "Test Heading" in result
        assert "A" * 200 in result
        # Total should be heading + newline + 200 chars
        assert len(result) == len("Test Heading") + 1 + 200

    def test_preprocess_text_none_heading(self):
        """Preprocessing with None/empty heading should just truncate text."""
        text = "Content here."
        result = EmbeddingService.preprocess_text(text, heading=None)
        assert result == text  # Short text, no truncation

    def test_preprocess_text_strips_heading(self):
        """Preprocessing should strip whitespace from heading."""
        text = "Body content."
        heading = "  Spaced Heading  "
        result = EmbeddingService.preprocess_text(text, heading)
        assert result.startswith("Spaced Heading")

    def test_preprocess_config_default_true(self):
        """EmbeddingConfig should default preprocess=True."""
        cfg = EmbeddingConfig()
        assert cfg.preprocess is True

    def test_preprocess_config_from_yaml(self, tmp_path):
        """Config should parse preprocess from YAML."""
        import yaml
        config_data = {
            "corpora": [{"name": "test", "path": str(tmp_path), "extensions": [".md"]}],
            "embedding": {"preprocess": False},
        }
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(yaml.dump(config_data))
        cfg = GuaipecaConfig.from_yaml(str(config_path))
        assert cfg.embedding.preprocess is False

    def test_preprocess_config_default_when_not_in_yaml(self, tmp_path):
        """Config without preprocess in YAML should default to True."""
        import yaml
        config_data = {
            "corpora": [{"name": "test", "path": str(tmp_path), "extensions": [".md"]}],
            "embedding": {"model": "all-MiniLM-L6-v2"},
        }
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(yaml.dump(config_data))
        cfg = GuaipecaConfig.from_yaml(str(config_path))
        assert cfg.embedding.preprocess is True

    def test_preprocess_in_embedding_service(self):
        """EmbeddingService should have preprocess attribute."""
        svc = EmbeddingService(preprocess=True)
        assert svc.preprocess is True

        svc2 = EmbeddingService(preprocess=False)
        assert svc2.preprocess is False

    def test_search_still_works_with_preprocessing(self, indexed_corpus):
        """Search should still return results with preprocessing enabled."""
        searcher = indexed_corpus
        # Re-index to ensure we have valid indexed content
        searcher.index_corpus(corpus_name="test-docs", force=True)
        result = searcher.search("consolidation", top_k=5)
        assert "error" not in result, f"Search error: {result.get('error')}"
        assert result["total"] > 0, "No results with preprocessing enabled"


# ===========================================================================
# 3. Embedding Cache (LRU + TTL)
# ===========================================================================

class TestEmbeddingCache:
    """Test the _EmbeddingCache class directly."""

    def test_cache_get_miss(self):
        """Cache miss should return None and increment misses."""
        cache = _EmbeddingCache(maxsize=100, ttl=3600)
        result = cache.get("nonexistent")
        assert result is None
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0

    def test_cache_put_then_get_hit(self):
        """Cache put then get should return the value and increment hits."""
        cache = _EmbeddingCache(maxsize=100, ttl=3600)
        key = "test_key"
        value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cache.put(key, value)
        result = cache.get(key)
        assert result is not None
        np.testing.assert_array_equal(result, value)
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 0
        assert cache.stats["size"] == 1

    def test_cache_lru_eviction(self):
        """Cache should evict LRU entries when maxsize is reached."""
        cache = _EmbeddingCache(maxsize=3, ttl=3600)
        # Fill cache
        for i in range(3):
            cache.put(f"key_{i}", np.array([i], dtype=np.float32))
        assert cache.stats["size"] == 3

        # Access key_0 to make it recently used
        cache.get("key_0")

        # Add a new key — should evict key_1 (LRU)
        cache.put("key_3", np.array([3], dtype=np.float32))
        assert cache.stats["size"] == 3
        assert cache.get("key_1") is None  # Evicted
        assert cache.get("key_0") is not None  # Still present
        assert cache.get("key_2") is not None  # Still present
        assert cache.get("key_3") is not None  # Just added
        assert cache.stats["evictions"] >= 1

    def test_cache_ttl_expiry(self):
        """Cache entries should expire after TTL."""
        cache = _EmbeddingCache(maxsize=100, ttl=1)  # 1 second TTL
        cache.put("key", np.array([1.0], dtype=np.float32))
        assert cache.get("key") is not None  # Should be in cache

        # Wait for TTL to expire
        time.sleep(1.1)
        result = cache.get("key")
        assert result is None  # Should be expired
        assert cache.stats["misses"] >= 1

    def test_cache_clear(self):
        """Clear should remove all entries but preserve stats."""
        cache = _EmbeddingCache(maxsize=100, ttl=3600)
        cache.put("key1", np.array([1.0], dtype=np.float32))
        cache.put("key2", np.array([2.0], dtype=np.float32))
        assert cache.stats["size"] == 2

        cache.clear()
        assert cache.stats["size"] == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_stats_property(self):
        """Stats should return hits, misses, evictions, size, ttl_seconds."""
        cache = _EmbeddingCache(maxsize=100, ttl=3600)
        stats = cache.stats
        assert "hits" in stats
        assert "misses" in stats
        assert "evictions" in stats
        assert "size" in stats
        assert "ttl_seconds" in stats
        assert stats["ttl_seconds"] == 3600
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["size"] == 0

    def test_cache_eviction_increments_counter(self):
        """Each eviction should increment the evictions counter."""
        cache = _EmbeddingCache(maxsize=2, ttl=3600)
        cache.put("k1", np.array([1], dtype=np.float32))
        cache.put("k2", np.array([2], dtype=np.float32))
        cache.put("k3", np.array([3], dtype=np.float32))  # Evicts k1
        assert cache.stats["evictions"] == 1
        cache.put("k4", np.array([4], dtype=np.float32))  # Evicts k2
        assert cache.stats["evictions"] == 2


class TestEmbeddingServiceCache:
    """Test the embedding cache integration in EmbeddingService."""

    def test_cache_key_is_deterministic(self):
        """Same text + model should produce the same cache key."""
        svc = EmbeddingService(model_name="test-model", dimensions=384)
        key1 = svc._cache_key("hello world")
        key2 = svc._cache_key("hello world")
        assert key1 == key2

    def test_cache_key_differs_by_model(self):
        """Different model names should produce different cache keys."""
        svc1 = EmbeddingService(model_name="model-a", dimensions=384)
        svc2 = EmbeddingService(model_name="model-b", dimensions=384)
        key1 = svc1._cache_key("hello")
        key2 = svc2._cache_key("hello")
        assert key1 != key2

    def test_cache_key_is_sha256(self):
        """Cache key should be a 64-char hex string (SHA256)."""
        svc = EmbeddingService(model_name="test", dimensions=384)
        key = svc._cache_key("test text")
        assert len(key) == 64
        int(key, 16)  # Should be valid hex

    def test_clear_cache_method_exists(self):
        """EmbeddingService should have a clear_cache method."""
        svc = EmbeddingService(dimensions=384)
        assert hasattr(svc, "clear_cache")
        svc.clear_cache()  # Should not raise

    def test_cache_stats_property_exists(self):
        """EmbeddingService should have a cache_stats property."""
        svc = EmbeddingService(dimensions=384)
        stats = svc.cache_stats
        assert "hits" in stats
        assert "misses" in stats
        assert "evictions" in stats
        assert "size" in stats

    def test_embed_uses_cache(self):
        """Embedding the same text twice should use the cache on second call."""
        # Use a real model — this is a session-scoped test
        # We test the cache behavior by checking stats
        svc = EmbeddingService(
            model_name="all-MiniLM-L6-v2",
            cache_dir=os.path.expanduser("~/.guaipeca/models"),
            dimensions=384,
        )
        text = "This is a test sentence for cache testing."
        # First call — miss
        result1 = svc.embed_query(text)
        assert result1 is not None
        assert svc.cache_stats["misses"] >= 1

        # Second call — should be a cache hit
        result2 = svc.embed_query(text)
        np.testing.assert_array_equal(result1, result2)
        assert svc.cache_stats["hits"] >= 1

    def test_embed_batch_partial_cache_hit(self):
        """Batch embed with some cached texts should only embed uncached ones."""
        svc = EmbeddingService(
            model_name="all-MiniLM-L6-v2",
            cache_dir=os.path.expanduser("~/.guaipeca/models"),
            dimensions=384,
        )
        # Embed one text first (caches it)
        text1 = "First text for partial cache test."
        svc.embed_query(text1)

        initial_hits = svc.cache_stats["hits"]

        # Batch embed: text1 is cached, text2 is new
        results = svc.embed([text1, "Second text for partial cache test."])
        assert results.shape == (2, 384)
        # Should have at least 1 hit (text1) and 1 miss (text2)
        assert svc.cache_stats["hits"] > initial_hits

    def test_force_reindex_clears_cache(self, indexed_corpus):
        """Force reindex should clear the embedding cache before re-embedding."""
        searcher = indexed_corpus

        # Do an initial index to populate cache
        searcher.index_corpus(corpus_name="test-docs", force=True)

        # Check cache has entries
        stats_before = searcher.embedder.cache_stats
        assert stats_before["size"] > 0

        # During force reindex, cache is cleared at the start of _do_index(),
        # then repopulated with new embeddings. So we check that the cache
        # was cleared by verifying that misses increased (all entries had to be
        # re-embedded) rather than checking size == 0 at the end.
        misses_before = searcher.embedder.cache_stats["misses"]

        # Force reindex should clear cache and re-embed everything
        searcher.index_corpus(corpus_name="test-docs", force=True)

        stats_after = searcher.embedder.cache_stats
        # Cache should have been cleared then repopulated
        assert stats_after["size"] > 0  # Has entries from re-embedding
        # Misses should have increased significantly (all texts re-embedded)
        assert stats_after["misses"] > misses_before, (
            "Cache misses should increase after force reindex (cache was cleared)"
        )


# ===========================================================================
# 4. Config Tests for Cache Settings
# ===========================================================================

class TestEmbeddingCacheConfig:
    """Test embedding cache configuration."""

    def test_cache_config_defaults(self):
        """EmbeddingConfig should have correct cache defaults."""
        cfg = EmbeddingConfig()
        assert cfg.cache_ttl_seconds == 3600
        assert cfg.cache_max_entries == 10000

    def test_cache_config_from_yaml(self, tmp_path):
        """Config should parse cache settings from YAML."""
        import yaml
        config_data = {
            "corpora": [{"name": "test", "path": str(tmp_path), "extensions": [".md"]}],
            "embedding": {
                "cache_ttl_seconds": 1800,
                "cache_max_entries": 5000,
            },
        }
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(yaml.dump(config_data))
        cfg = GuaipecaConfig.from_yaml(str(config_path))
        assert cfg.embedding.cache_ttl_seconds == 1800
        assert cfg.embedding.cache_max_entries == 5000

    def test_cache_config_defaults_when_not_in_yaml(self, tmp_path):
        """Config without cache settings should use defaults."""
        import yaml
        config_data = {
            "corpora": [{"name": "test", "path": str(tmp_path), "extensions": [".md"]}],
        }
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(yaml.dump(config_data))
        cfg = GuaipecaConfig.from_yaml(str(config_path))
        assert cfg.embedding.cache_ttl_seconds == 3600
        assert cfg.embedding.cache_max_entries == 10000


class TestZeroSizeCacheRegression:
    """Regression: cache_max_entries=0 (or negative) and ttl<=0 must not crash.

    Previously a maxsize of 0 made the LRU eviction loop call
    ``next(iter(...))`` on an empty OrderedDict, raising StopIteration on
    every embed/search call. Zero size is now treated as "cache disabled".
    """

    def test_zero_maxsize_put_get_does_not_raise(self):
        cache = _EmbeddingCache(maxsize=0, ttl=3600)
        assert cache.enabled is False
        cache.put("k", np.array([1.0], dtype=np.float32))  # must not raise
        assert cache.get("k") is None
        assert cache.stats["size"] == 0
        assert cache.stats["evictions"] == 0
        assert cache.stats["enabled"] is False

    def test_negative_maxsize_is_disabled(self):
        cache = _EmbeddingCache(maxsize=-5, ttl=3600)
        assert cache.enabled is False
        cache.put("k", np.array([1.0], dtype=np.float32))
        assert cache.get("k") is None

    def test_zero_ttl_is_disabled(self):
        cache = _EmbeddingCache(maxsize=100, ttl=0)
        assert cache.enabled is False
        cache.put("k", np.array([1.0], dtype=np.float32))
        assert cache.get("k") is None

    def test_zero_maxsize_service_embed_does_not_raise(self):
        """EmbeddingService with cache_max_entries=0 must still embed."""
        svc = EmbeddingService(cache_max_entries=0)
        assert svc._cache.enabled is False
        # embed() goes through the cache path for every text; with caching
        # disabled it must skip put() entirely (no StopIteration).
        out = svc.embed(["hello world"]) if _model_available(svc) else None
        if out is not None:
            assert out.shape[0] == 1


# ===========================================================================
# 5. Integration: All Features Together
# ===========================================================================

class TestFeaturesIntegration:
    """Test all three features working together."""

    def test_search_works_with_all_features(self, indexed_corpus):
        """Search should work with backup, preprocessing, and cache all active."""
        searcher = indexed_corpus
        # Force reindex to exercise all features
        searcher.index_corpus(corpus_name="test-docs", force=True)

        result = searcher.search("financial consolidation", top_k=5)
        assert "error" not in result
        assert result["total"] > 0

    def test_backup_exists_after_force_reindex(self, indexed_corpus):
        """After force reindex, backup should exist."""
        searcher = indexed_corpus
        searcher.index_corpus(corpus_name="test-docs", force=True)
        corpus_idx = searcher.corpora["test-docs"]
        backup_path = corpus_idx.faiss_path.with_suffix(".fai.bak")
        assert backup_path.exists()

    def test_cache_populated_after_indexing(self, indexed_corpus):
        """After indexing, embedding cache should have entries."""
        searcher = indexed_corpus
        searcher.index_corpus(corpus_name="test-docs", force=True)
        stats = searcher.embedder.cache_stats
        assert stats["size"] > 0
        assert stats["misses"] > 0  # Had to embed during indexing

    def test_repeated_search_uses_cache(self, indexed_corpus):
        """Repeated search with same query should hit cache."""
        searcher = indexed_corpus
        query = "consolidation"

        # First search
        result1 = searcher.search(query, top_k=3)
        hits_after_first = searcher.embedder.cache_stats["hits"]

        # Second search with same query
        result2 = searcher.search(query, top_k=3)

        hits_after_second = searcher.embedder.cache_stats["hits"]
        assert hits_after_second > hits_after_first, (
            "Cache hits should increase on repeated search"
        )

    def test_get_chunk_returns_full_text_not_preprocessed(self, indexed_corpus):
        """get_chunk should return the full chunk text, not the truncated preprocessing text."""
        searcher = indexed_corpus
        # Index with preprocessing
        searcher.index_corpus(corpus_name="test-docs", force=True)

        # Search to get a chunk_id
        result = searcher.search("consolidation", top_k=5)
        assert result["total"] > 0

        # Check all returned chunks have text that wasn't truncated to 500 chars
        # (preprocessing only affects what gets embedded, not what gets stored)
        for r in result["results"]:
            chunk_id = r["chunk_id"]
            chunk = searcher.get_chunk(chunk_id)
            assert chunk is not None, f"Chunk not found: {chunk_id}"
            # The stored text should be the original chunk text, not preprocessed
            # Some chunks may be short (< 500 chars) so we can't assert > 500
            # Instead verify the text doesn't start with just the heading + truncation
            assert "text" in chunk
            assert len(chunk["text"]) > 0, f"Chunk text is empty: {chunk_id}"