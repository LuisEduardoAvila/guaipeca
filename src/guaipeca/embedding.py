"""Embedding service using fastembed (local, no torch dependency).

Loads model once, caches in memory. Supports batch embedding.
Uses ONNX Runtime instead of torch for much smaller image footprint.

Production features:
- LRU + TTL embedding cache (max 10K entries, 1-hour TTL)
- Memory-pressure aware eviction (psutil, optional)
- Content preprocessing for better semantic signal
- Cache stats tracking
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


class _EmbeddingCache:
    """LRU cache with TTL expiration for embedding vectors.

    Prevents unbounded memory growth while improving throughput on
    repeated embeddings (e.g. re-indexing unchanged files).

    Features:
    - LRU eviction when maxsize is reached
    - TTL-based expiration (entries expire after ttl seconds)
    - Memory-pressure aware: evicts aggressively when system RAM > 85%
    - Stats tracking: hits, misses, evictions, size
    """

    def __init__(self, maxsize: int = 10000, ttl: int = 3600):
        """
        Args:
            maxsize: Maximum number of entries (default 10000).
            ttl: Time-to-live in seconds (default 3600 = 1 hour).
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _is_expired(self, key: str) -> bool:
        """Check if a key has expired based on TTL."""
        if key not in self._timestamps:
            return True
        return time.time() - self._timestamps[key] > self.ttl

    def get(self, key: str) -> np.ndarray | None:
        """
        Get item from cache.

        Args:
            key: Cache key (text hash).

        Returns:
            Embedding array or None if not found/expired.
        """
        if key not in self._cache:
            self._misses += 1
            return None

        # Check if expired
        if self._is_expired(key):
            self._evict(key)
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return self._cache[key]

    def put(self, key: str, value: np.ndarray) -> None:
        """
        Add item to cache with LRU eviction and memory pressure checks.

        Args:
            key: Cache key (text hash).
            value: Embedding array.
        """
        # Check memory pressure before adding
        if self._is_memory_pressure_high():
            # Evict more aggressively under memory pressure
            while len(self._cache) > int(self.maxsize * 0.7):
                oldest_key = next(iter(self._cache))
                self._evict(oldest_key)

        # Evict expired entries if cache is full
        while len(self._cache) >= self.maxsize:
            oldest_key = next(iter(self._cache))
            self._evict(oldest_key)

        # Add new entry
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def _is_memory_pressure_high(self, threshold: float = 0.85) -> bool:
        """
        Check if system memory pressure is high.

        Uses psutil if available, falls back to /proc/meminfo on Linux,
        returns False if neither is available.

        Args:
            threshold: Memory usage threshold (0.0-1.0, default 0.85).

        Returns:
            True if memory pressure is high.
        """
        try:
            import psutil
            return psutil.virtual_memory().percent >= threshold * 100
        except ImportError:
            pass

        # Fallback: /proc/meminfo (Linux only)
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            available_kb = None
            total_kb = None
            for line in lines:
                if line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                elif line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
            if available_kb is not None and total_kb is not None:
                used_percent = 100 - (available_kb / total_kb) * 100
                return used_percent >= threshold * 100
        except Exception:
            pass

        return False

    def _evict(self, key: str) -> None:
        """Evict a single key from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
        self._evictions += 1

    def clear(self) -> None:
        """Clear all cache entries. Stats are preserved (cumulative)."""
        self._cache.clear()
        self._timestamps.clear()

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "size": len(self._cache),
            "ttl_seconds": self.ttl,
        }


class EmbeddingService:
    """Local fastembed embedding service with LRU+TTL caching."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | None = None,
        dimensions: int = 384,
        preprocess: bool = True,
        cache_ttl_seconds: int = 3600,
        cache_max_entries: int = 10000,
    ):
        """
        Args:
            model_name: fastembed model name (e.g. "all-MiniLM-L6-v2").
            cache_dir: Where to cache the downloaded ONNX model.
            dimensions: Expected embedding dimensions (for validation).
            preprocess: If True, preprocess text before embedding (heading + first 500 chars).
            cache_ttl_seconds: TTL for embedding cache entries (default 3600 = 1 hour).
            cache_max_entries: Max embedding cache entries (default 10000).
        """
        import os
        self.model_name = model_name
        self.cache_dir = os.path.expanduser(cache_dir) if cache_dir else cache_dir
        self.dimensions = dimensions
        self.preprocess = preprocess
        self._model = None
        self._cache = _EmbeddingCache(
            maxsize=cache_max_entries,
            ttl=cache_ttl_seconds,
        )

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        """Normalize model name for fastembed compatibility.

        fastembed expects full HuggingFace names (e.g. 'sentence-transformers/all-MiniLM-L6-v2').
        Guaipeca config uses short names (e.g. 'all-MiniLM-L6-v2').
        If no '/' is present, prepend 'sentence-transformers/' as the default org.
        """
        if "/" not in name:
            return f"sentence-transformers/{name}"
        return name

    @staticmethod
    def preprocess_text(text: str, heading: str = "", max_chars: int = 500) -> str:
        """Preprocess text for embedding: heading + first max_chars of text.

        This gives the embedding model better semantic signal than the full chunk.
        The full text is still stored in metadata for retrieval.

        Args:
            text: The chunk text.
            heading: Optional heading to prepend.
            max_chars: Maximum characters of text to include (default 500).

        Returns:
            Preprocessed string for embedding.
        """
        if heading and heading.strip():
            return f"{heading.strip()}\n{text[:max_chars]}"
        return text[:max_chars]

    @property
    def model(self):
        """Lazy-load the fastembed model."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                fastembed_name = self._normalize_model_name(self.model_name)
                logger.info(f"Loading embedding model: {self.model_name} (fastembed: {fastembed_name})")
                kwargs = {"model_name": fastembed_name}
                if self.cache_dir:
                    kwargs["cache_dir"] = self.cache_dir
                self._model = TextEmbedding(**kwargs)
                # Validate dimensions with a warmup embedding
                warmup = list(self._model.embed(["warmup"]))
                if warmup:
                    actual_dim = warmup[0].shape[0]
                    if actual_dim != self.dimensions:
                        logger.warning(
                            f"Model dimensions ({actual_dim}) != config dimensions ({self.dimensions}). "
                            f"Using actual: {actual_dim}"
                        )
                        self.dimensions = actual_dim
            except ImportError:
                raise ImportError(
                    "fastembed is not installed. "
                    "Install with: pip install fastembed"
                )
        return self._model

    def _cache_key(self, text: str) -> str:
        """Generate cache key: SHA256 of model_name + text."""
        return hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()

    def embed(self, texts: list[str], headings: list[str] | None = None) -> np.ndarray:
        """
        Embed a batch of texts with caching.

        Args:
            texts: List of text strings to embed.
            headings: Optional list of headings for preprocessing.
                      If provided and self.preprocess is True, each text
                      is preprocessed (heading + first 500 chars) before embedding.

        Returns:
            numpy array of shape (len(texts), dimensions), L2-normalized for
            cosine similarity via inner product.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimensions)

        # Preprocess if enabled (only when headings are provided — for chunk indexing)
        effective_texts = texts
        if self.preprocess and headings is not None:
            effective_texts = [
                self.preprocess_text(t, h) for t, h in zip(texts, headings)
            ]

        # Check cache for each text
        results: list[np.ndarray | None] = [None] * len(effective_texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(effective_texts):
            key = self._cache_key(text)
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts
        if uncached_texts:
            embeddings_gen = self.model.embed(uncached_texts)
            new_embeddings = np.array(list(embeddings_gen), dtype=np.float32)

            # L2 normalize for cosine similarity via FAISS IndexFlatIP
            norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            new_embeddings = new_embeddings / norms

            # Cache and place in results
            for j, idx in enumerate(uncached_indices):
                key = self._cache_key(uncached_texts[j])
                self._cache.put(key, new_embeddings[j])
                results[idx] = new_embeddings[j]

        return np.array(results, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query text (with cache lookup).

        Queries are NOT preprocessed (they're already short user input).

        Args:
            text: Query string.

        Returns:
            numpy array of shape (dimensions,), L2-normalized.
        """
        # Check cache first
        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Embed and cache
        result = self.embed([text])[0]
        # embed() already cached it, but double-check
        return result

    def clear_cache(self) -> None:
        """Clear the embedding cache (e.g. on force reindex)."""
        self._cache.clear()

    @property
    def cache_stats(self) -> dict:
        """Return embedding cache statistics."""
        return self._cache.stats

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None