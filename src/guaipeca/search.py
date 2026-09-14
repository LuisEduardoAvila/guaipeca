"""Multi-corpus weighted search.

Searches across all configured corpora, applies corpus weights, merges and sorts results.
Supports hybrid BM25 + FAISS dense search with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
import os

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
        corpora: list[str] | None = None,
        hybrid: bool | None = None,
        return_mode: str = "chunks",
        max_chars: int | None = None,
    ) -> dict:
        """
        Search across corpora.

        Args:
            query: Search query text.
            top_k: Total results to return (across all corpora).
            corpora: Optional list of corpus names to search. None = all.
            hybrid: Enable BM25 hybrid search. None = use config default (search.hybrid).
            return_mode: Result granularity — "chunks" (default), "documents", or "auto".
            max_chars: Threshold for auto mode. None = use config default (search.max_chars).

        Returns:
            Dict with results, total, query, return_mode.
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

        # Validate return_mode
        if return_mode not in ("chunks", "documents", "auto"):
            return {
                "results": [],
                "total": 0,
                "query": query,
                "error": f"Invalid return_mode: {return_mode!r}. Must be 'chunks', 'documents', or 'auto'.",
            }

        # Determine max_chars threshold
        chars_threshold = max_chars if max_chars is not None else self.config.search.max_chars

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

        # Apply return_mode
        if return_mode == "documents":
            final = self._collapse_to_documents(final)
            return {
                "results": final,
                "total": len(final),
                "query": query,
                "hybrid": use_hybrid,
                "return_mode": "documents",
            }
        elif return_mode == "auto":
            chunks_size = self._estimate_chunks_size(final)
            if chunks_size < chars_threshold:
                return {
                    "results": final,
                    "total": len(final),
                    "query": query,
                    "hybrid": use_hybrid,
                    "return_mode": "chunks",
                }
            else:
                final = self._collapse_to_documents(final)
                return {
                    "results": final,
                    "total": len(final),
                    "query": query,
                    "hybrid": use_hybrid,
                    "return_mode": "documents",
                }
        else:
            # chunks mode (default)
            return {
                "results": final,
                "total": len(final),
                "query": query,
                "hybrid": use_hybrid,
                "return_mode": "chunks",
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

    def _collapse_to_documents(self, chunk_results: list[dict]) -> list[dict]:
        """Collapse chunk results into document-level results.

        Groups chunk results by source_path, reads the full document text
        from disk, and returns one result per unique source document.

        Document score = best (highest) chunk score from that document.

        Args:
            chunk_results: Sorted list of chunk result dicts.

        Returns:
            List of document-level result dicts, sorted by score descending.
        """
        # Group chunks by source_path
        # Extract file path from location (before # heading separator)
        docs: dict[str, dict] = {}
        for chunk in chunk_results:
            location = chunk.get("location", "")
            source_path = location.split("#")[0] if "#" in location else location

            if not source_path:
                continue

            # Skip if file no longer exists
            if not os.path.exists(source_path):
                continue

            if source_path not in docs:
                docs[source_path] = {
                    "source_path": source_path,
                    "corpus": chunk.get("corpus", ""),
                    "topic": chunk.get("topic", ""),
                    "score": chunk["score"],
                    "matched_chunks": 1,
                }
            else:
                # Update best score and count
                docs[source_path]["score"] = max(docs[source_path]["score"], chunk["score"])
                docs[source_path]["matched_chunks"] += 1

        # Read full document text for each
        doc_results = []
        for source_path, info in docs.items():
            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"Could not read document {source_path}: {e}")
                continue

            doc_results.append({
                "source_path": source_path,
                "filename": os.path.basename(source_path),
                "corpus": info["corpus"],
                "topic": info["topic"],
                "score": info["score"],
                "text": text,
                "char_count": len(text),
                "matched_chunks": info["matched_chunks"],
            })

        # Sort by score descending
        doc_results.sort(key=lambda r: r["score"], reverse=True)
        return doc_results

    def _estimate_chunks_size(self, chunk_results: list[dict]) -> int:
        """Estimate total character count of chunk result texts.

        Uses get_chunk to retrieve full text for each chunk result.

        Args:
            chunk_results: List of chunk result dicts.

        Returns:
            Total character count of all chunk texts.
        """
        total = 0
        for chunk in chunk_results:
            chunk_id = chunk.get("chunk_id", "")
            full_chunk = self.get_chunk(chunk_id)
            if full_chunk and "text" in full_chunk:
                total += len(full_chunk["text"])
            else:
                # Fallback: estimate from summary length
                total += len(chunk.get("summary", ""))
        return total

    def get_chunk(self, chunk_id: str) -> dict | None:
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

    def list_documents(self, corpus: str | None = None) -> dict:
        """
        List indexed documents across all corpora or a specific corpus.

        Args:
            corpus: Optional corpus name to filter. None = all corpora.

        Returns:
            Dict with 'documents' list and 'total' count.
        """
        documents = []

        if corpus is not None:
            if corpus not in self.corpora:
                return {"error": f"corpus not found: {corpus}"}
            documents = self.corpora[corpus].list_documents()
        else:
            for corpus_idx in self.corpora.values():
                documents.extend(corpus_idx.list_documents())

        return {
            "documents": documents,
            "total": len(documents),
        }

    def get_document(self, corpus: str, source_path: str) -> dict | None:
        """
        Get full converted text of a document by corpus and source_path.

        Args:
            corpus: Corpus name.
            source_path: Absolute path to the source file.

        Returns:
            Document dict with text and metadata, or None if not found.
        """
        if corpus not in self.corpora:
            return None

        return self.corpora[corpus].get_document_text(source_path)

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
            for corpus_idx in self.corpora.values():
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