"""Multi-corpus weighted search.

Searches across all configured corpora, applies corpus weights, merges and sorts results.
Supports hybrid BM25 + FAISS dense search with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import GuaipecaConfig
from .embedding import EmbeddingService
from .index import CorpusIndex

logger = logging.getLogger(__name__)

# Maximum query length to prevent DoS via memory exhaustion
MAX_QUERY_LENGTH = 10_000


class Searcher:
    """Multi-corpus weighted search with optional hybrid BM25 fusion."""

    def __init__(
        self,
        config: GuaipecaConfig,
        embedding_service: EmbeddingService,
        converter=None,
    ):
        """
        Args:
            config: Full Guaipeca config.
            embedding_service: EmbeddingService for query embedding.
            converter: Document converter instance.
        """
        self.config = config
        self.embedder = embedding_service

        # Build corpus indexes
        self.corpora: dict[str, CorpusIndex] = {}
        for corpus_cfg in config.corpora:
            self.corpora[corpus_cfg.name] = CorpusIndex(
                corpus=corpus_cfg,
                chunking=config.chunking,
                indexing=config.indexing,
                embedding_service=embedding_service,
                converter=converter,
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        corpora: Optional[list[str]] = None,
        hybrid: Optional[bool] = None,
    ) -> dict:
        """
        Search across corpora.

        Args:
            query: Search query text.
            top_k: Total results to return (across all corpora).
            corpora: Optional list of corpus names to search. None = all.
            hybrid: Enable BM25 hybrid search. None = use config default (search.hybrid).

        Returns:
            Dict with results, total, query.
        """
        # Validate top_k
        if top_k < 1:
            return {"results": [], "total": 0, "query": query, "error": "top_k must be >= 1"}

        if not query.strip():
            return {"results": [], "total": 0, "query": query, "error": "query required"}

        # Validate query length
        if len(query) > MAX_QUERY_LENGTH:
            return {
                "results": [],
                "total": 0,
                "query": query[:100] + "...",
                "error": f"Query too long ({len(query)} chars, max {MAX_QUERY_LENGTH})",
            }

        # Determine which corpora to search
        search_corpora = corpora if corpora else list(self.corpora.keys())
        for name in search_corpora:
            if name not in self.corpora:
                return {"results": [], "total": 0, "query": query, "error": f"corpus not found: {name}"}

        # Determine hybrid mode: explicit param overrides config
        use_hybrid = hybrid if hybrid is not None else self.config.search.hybrid

        # Check if any corpora have indexed content
        total_indexed = 0
        for name in search_corpora:
            corpus_idx = self.corpora[name]
            # Access stats under read lock to check ntotal
            with corpus_idx._rwlock.read_lock():
                corpus_idx._load()
                if corpus_idx._faiss_index and corpus_idx._faiss_index.ntotal > 0:
                    total_indexed += corpus_idx._faiss_index.ntotal

        if total_indexed == 0:
            return {
                "results": [],
                "total": 0,
                "query": query,
                "error": "No indexed content found. Run 'guaipeca index' first.",
            }

        # Embed query (always needed for dense search)
        query_vector = self.embedder.embed_query(query)

        # Search each corpus (get top_k from each, then merge)
        all_results = []
        per_corpus_k = top_k * 2  # over-fetch for better merge

        if use_hybrid:
            # Hybrid mode: run both dense and BM25, fuse with RRF
            for name in search_corpora:
                corpus_idx = self.corpora[name]
                dense_results = corpus_idx.search(query_vector, top_k=per_corpus_k)
                sparse_results = corpus_idx.search_bm25(query, top_k=per_corpus_k)
                fused = self._rrf_fuse(dense_results, sparse_results, top_k=per_corpus_k)
                all_results.extend(fused)
        else:
            # Dense-only mode (original behavior)
            for name in search_corpora:
                corpus_idx = self.corpora[name]
                results = corpus_idx.search(query_vector, top_k=per_corpus_k)
                all_results.extend(results)

        # Sort by weighted score (descending)
        all_results.sort(key=lambda r: r["score"], reverse=True)

        # Trim to top_k
        final = all_results[:top_k]

        return {
            "results": final,
            "total": len(final),
            "query": query,
            "hybrid": use_hybrid,
        }

    def _rrf_fuse(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Fuse dense and sparse results using Reciprocal Rank Fusion (RRF).

        RRF score = weight / (k + rank)
        where k is a constant (default 60) that smooths the ranking.

        Args:
            dense_results: Results from FAISS dense search (already ranked).
            sparse_results: Results from BM25 sparse search (already ranked).
            top_k: Max results to return.

        Returns:
            Fused and ranked list of result dicts.
        """
        k = self.config.search.rrf_k
        dense_weight = self.config.search.dense_weight
        sparse_weight = self.config.search.bm25_weight

        scores_map: dict[str, dict] = {}

        # Dense results (rank 0-based)
        for rank, r in enumerate(dense_results):
            cid = r["chunk_id"]
            rrf_score = dense_weight / (k + rank + 1)
            if cid not in scores_map:
                scores_map[cid] = dict(r)
                scores_map[cid]["rrf_score"] = 0.0
            scores_map[cid]["rrf_score"] += rrf_score

        # Sparse results (rank 0-based)
        for rank, r in enumerate(sparse_results):
            cid = r["chunk_id"]
            rrf_score = sparse_weight / (k + rank + 1)
            if cid not in scores_map:
                scores_map[cid] = dict(r)
                scores_map[cid]["rrf_score"] = 0.0
            scores_map[cid]["rrf_score"] += rrf_score

        # Sort by RRF score descending
        fused = sorted(
            scores_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # Set score to rrf_score for consistent sorting in merge
        for r in fused:
            r["score"] = r["rrf_score"]

        return fused[:top_k]

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """
        Get full chunk content by chunk_id.

        Args:
            chunk_id: Format "corpus_name:stable_id" (e.g. "epm-docs:a1b2c3d4e5f67890").

        Returns:
            Chunk dict or None if not found.
        """
        if ":" not in chunk_id:
            return None

        corpus_name, stable_id = chunk_id.rsplit(":", 1)
        if corpus_name not in self.corpora:
            return None

        return self.corpora[corpus_name].get_chunk(stable_id)

    def index_corpus(self, corpus_name: str = "all", force: bool = False) -> dict:
        """
        Index or re-index one or all corpora.

        Args:
            corpus_name: "all" or specific corpus name.
            force: Force re-index even if unchanged.

        Returns:
            Stats dict or aggregated stats for "all".
        """
        if corpus_name == "all":
            all_stats = []
            for name, corpus_idx in self.corpora.items():
                stats = corpus_idx.index(force=force)
                all_stats.append(stats)
            return {"corpora": all_stats}
        else:
            if corpus_name not in self.corpora:
                return {"error": f"corpus not found: {corpus_name}"}
            return self.corpora[corpus_name].index(force=force)

    @property
    def status(self) -> dict:
        """Status of all corpora."""
        return {
            "corpora": [self.corpora[name].stats for name in self.corpora],
            "embedding_model": self.embedder.model_name,
            "dimensions": self.embedder.dimensions,
            "hybrid_enabled": self.config.search.hybrid,
        }