"""Multi-corpus weighted search.

Searches across all configured corpora, applies corpus weights, merges and sorts results.
Supports hybrid BM25 + FAISS dense search with Reciprocal Rank Fusion (RRF).
ToC-aware features: section aggregation, section filtering, keyword re-ranking.
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
        # Document converter for reading non-text sources (PDF, DOCX, ...).
        # Falls back to a default Converter so document reading works even
        # when the caller did not supply one.
        if converter is None:
            from .converter import Converter
            converter = Converter()
        self.converter = converter

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
        section_mode: bool = False,
        section_filter: str | None = None,
        dedup_sections: bool = False,
    ) -> dict:
        """
        Search across corpora.

        Args:
            query: Search query text.
            top_k: Total results to return (across all corpora).
            corpora: Optional list of corpus names to search. None = all.
            hybrid: Enable BM25 hybrid search. None = use config default (search.hybrid).
            return_mode: Result granularity -- "chunks" (default), "documents", or "auto".
            max_chars: Threshold for auto mode. None = use config default (search.max_chars).
            section_mode: If True, group results by section and return full section text.
            section_filter: If specified, restrict search to chunks within the given section.
            dedup_sections: If True, collapse results sharing a section_id, keeping only
                the best-scoring chunk per section. Improves diversity by preventing
                near-duplicate chunks from the same section crowding the results.

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

        # If section_filter is specified, use section-restricted search
        if section_filter:
            return self._search_with_section_filter(
                query, top_k=top_k,
                corpora=search_corpora, hybrid=use_hybrid,
                section_filter=section_filter,
            )

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

        # Apply ToC re-ranking if enabled (Phase 5)
        if getattr(self.config.search, 'toc_rerank', False):
            all_results = self._rerank_by_toc(all_results, query)
            all_results.sort(key=lambda r: r["score"], reverse=True)

        # Apply section de-duplication if requested
        if dedup_sections:
            all_results = self._dedup_by_section(all_results)

        # Trim to top_k
        final = all_results[:top_k]

        # If section_mode is True, aggregate results by section (Phase 2)
        if section_mode:
            return self._aggregate_sections(final, query, use_hybrid)

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
        """Fuse dense and sparse results using score-normalized RRF.

        Standard RRF (score = weight / (k + rank)) treats all rank-0
        results equally, regardless of their raw score magnitude. This
        means a strong exact-token BM25 hit (raw_score ~11) gets the same
        fusion boost as a weak dense match (raw_score ~0.001) if both
        are at rank 0.

        Score-normalized fusion combines rank position with normalized
        raw scores:

            fused = weight * (alpha * norm_raw + (1 - alpha) * rrf_rank)

        where norm_raw is the raw_score min-max normalized to [0, 1]
        within each result set, and rrf_rank = 1 / (k + rank + 1).

        This ensures a strong BM25 match outranks a weak dense match
        at the same rank position, without the problems of simply
        bumping bm25_weight (which would regress semantic queries where
        BM25 scores are naturally low).

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

        # Score-normalization alpha: how much weight to give the raw score
        # vs the rank position. 0.3 means 30% score quality, 70% rank position.
        # This preserves RRF's robustness while letting strong matches rise.
        alpha = getattr(self.config.search, "fusion_alpha", 0.3)

        scores_map: dict[str, dict] = {}

        # Helper: min-max normalize raw_scores to [0, 1]
        def _normalize(results: list[dict]) -> list[float]:
            if not results:
                return []
            raws = [r.get("raw_score", 0.0) for r in results]
            lo, hi = min(raws), max(raws)
            rng = hi - lo
            if rng < 1e-9:
                return [1.0] * len(results)  # all same → all max
            return [(r - lo) / rng for r in raws]

        # Dense results (rank 0-based)
        dense_norm = _normalize(dense_results)
        for rank, r in enumerate(dense_results):
            cid = r["chunk_id"]
            rrf_rank = 1.0 / (k + rank + 1)
            norm_score = dense_norm[rank] if rank < len(dense_norm) else 0.0
            fused = dense_weight * (alpha * norm_score + (1 - alpha) * rrf_rank)
            if cid not in scores_map:
                scores_map[cid] = dict(r)
                scores_map[cid]["rrf_score"] = 0.0
            scores_map[cid]["rrf_score"] += fused

        # Sparse results (rank 0-based)
        sparse_norm = _normalize(sparse_results)
        for rank, r in enumerate(sparse_results):
            cid = r["chunk_id"]
            rrf_rank = 1.0 / (k + rank + 1)
            norm_score = sparse_norm[rank] if rank < len(sparse_norm) else 0.0
            fused = sparse_weight * (alpha * norm_score + (1 - alpha) * rrf_rank)
            if cid not in scores_map:
                scores_map[cid] = dict(r)
                scores_map[cid]["rrf_score"] = 0.0
            scores_map[cid]["rrf_score"] += fused

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
            text = self._read_document_text(source_path)
            if text is None:
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

    def _read_document_text(self, source_path: str) -> str | None:
        """Read a source document's full text, converting non-text formats.

        Plain-text formats (.md/.txt/.markdown) are read directly. All other
        supported formats (PDF, DOCX, PPTX, XLSX, HTML, ...) are routed through
        the converter so they are returned as markdown text rather than being
        mis-decoded as UTF-8 (which would drop the whole document).

        Returns:
            Document text, or None if it could not be read/converted.
        """
        ext = os.path.splitext(source_path)[1].lower()
        try:
            if ext in (".md", ".txt", ".markdown"):
                with open(source_path, "r", encoding="utf-8") as f:
                    return f.read()
            return self.converter.convert(source_path)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Could not read document {source_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Could not convert document {source_path}: {e}")
            return None

    def _dedup_by_section(self, results: list[dict]) -> list[dict]:
        """Collapse results sharing a section_id, keeping the best-scoring chunk per section.

        This improves result diversity by preventing multiple chunks from the
        same document section from crowding the top results. The best-scoring
        chunk from each section is kept; others are dropped.

        Results without a section_id (non-structured docs) are kept as-is.

        Args:
            results: Sorted list of chunk result dicts (descending by score).

        Returns:
            Filtered list with at most one chunk per section_id.
        """
        seen_sections: set[str] = set()
        deduped: list[dict] = []

        for r in results:
            sid = r.get("section_id", "")
            if not sid or sid.startswith("unstructured:"):
                # No section_id or unstructured doc — keep as-is (no dedup)
                deduped.append(r)
                continue
            if sid not in seen_sections:
                seen_sections.add(sid)
                deduped.append(r)

        return deduped

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

    # ---- ToC-Aware Feature Methods ----

    def _search_with_section_filter(
        self,
        query: str,
        top_k: int,
        corpora: list[str],
        hybrid: bool,
        section_filter: str,
    ) -> dict:
        """Search restricted to a specific section (Phase 4).

        Args:
            query: Search query text.
            top_k: Number of results to return.
            corpora: List of corpus names to search.
            hybrid: Whether to use hybrid BM25+FAISS.
            section_filter: Section ID or heading_path prefix to filter by.

        Returns:
            Dict with results, total, query, section_filter.
        """
        query_vector = self.embedder.embed_query(query)

        all_results = []
        section_found = False

        for name in corpora:
            corpus_idx = self.corpora[name]
            results = corpus_idx.search_section(query_vector, section_filter, top_k=top_k)
            if results:
                section_found = True
            # Also try BM25 if hybrid
            if hybrid:
                # For BM25, get all results and filter by section
                bm25_results = corpus_idx.search_bm25(query, top_k=top_k * 3)
                # Filter BM25 results to only those in the matching sections
                with corpus_idx._rwlock.read_lock():
                    corpus_idx._load()
                    section_map = dict(corpus_idx._section_map)
                section_chunk_ids = set()
                for sid, cids in section_map.items():
                    # Direct match
                    if sid == section_filter:
                        section_chunk_ids.update(cids)
                        section_found = True
                        continue
                    # Heading_path prefix match: check if any chunk in this section
                    # has a heading_path starting with the filter text
                    for cid in cids:
                        pos = corpus_idx._stable_id_to_pos.get(cid)
                        if pos is not None and pos < len(corpus_idx._chunks):
                            chunk = corpus_idx._chunks[pos]
                            if chunk.heading_path:
                                # Fuzzy match: check if filter is a substring of any heading in the path,
                                # or if the filter matches as a prefix of the joined heading_path.
                                # This handles headings with emojis, annotations, or extra text.
                                joined = " ".join(chunk.heading_path)
                                filter_lower = section_filter.lower()
                                joined_lower = joined.lower()
                                if (joined_lower.startswith(filter_lower)
                                        or filter_lower in joined_lower
                                        or any(filter_lower in h.lower() for h in chunk.heading_path)):
                                    section_chunk_ids.update(cids)
                                    section_found = True
                                    break

                filtered_bm25 = [
                    r for r in bm25_results
                    if r["chunk_id"].split(":", 1)[1] in section_chunk_ids
                ]
                if filtered_bm25:
                    fused = self._rrf_fuse(results, filtered_bm25, top_k=top_k)
                    all_results.extend(fused)
                else:
                    all_results.extend(results)
            else:
                all_results.extend(results)

        if not section_found:
            return {
                "results": [],
                "total": 0,
                "query": query,
                "hybrid": hybrid,
                "section_filter": section_filter,
                "error": f"section not found: {section_filter}",
            }

        all_results.sort(key=lambda r: r["score"], reverse=True)
        final = all_results[:top_k]

        return {
            "results": final,
            "total": len(final),
            "query": query,
            "hybrid": hybrid,
            "section_filter": section_filter,
        }

    def _aggregate_sections(self, results: list[dict], query: str, hybrid: bool) -> dict:
        """Aggregate search results by section, returning full section text (Phase 2).

        For each unique section in results, fetch all chunks in that section,
        concatenate them (sorted by char_offset), and return the full section text.

        Non-structured documents (empty heading_path) have a single "unstructured:{hash}"
        section_id -- all their chunks are returned as one block.
        """
        section_results = []
        seen_sections = set()

        for r in results:
            section_id = r.get("section_id", "")
            if not section_id or section_id in seen_sections:
                continue
            seen_sections.add(section_id)

            # Determine which corpus this result is from
            corpus_name = r.get("corpus", "")
            if corpus_name not in self.corpora:
                continue

            corpus_idx = self.corpora[corpus_name]
            section_chunks = corpus_idx.get_section_chunks(section_id)

            if not section_chunks:
                continue

            # Sort by char_offset (already done in get_section_chunks, but ensure)
            section_chunks.sort(key=lambda c: c.char_offset)

            # Concatenate all chunk texts
            full_text = "\n\n".join(c.text for c in section_chunks)

            # Get heading info from first chunk
            first_chunk = section_chunks[0]
            heading_path = first_chunk.heading_path
            heading = first_chunk.heading

            section_results.append({
                "section_id": section_id,
                "heading": heading,
                "heading_path": heading_path,
                "text": full_text,
                "chunk_count": len(section_chunks),
                "corpus": corpus_name,
                "location": f"{first_chunk.source_path}#{heading}".rstrip("#"),
                "source_path": first_chunk.source_path,
                "filename": r.get("filename", ""),
                "score": r["score"],
                "summary": heading or " ".join(full_text[:200].split()),
            })

        return {
            "results": section_results,
            "total": len(section_results),
            "query": query,
            "hybrid": hybrid,
            "section_mode": True,
        }

    def _rerank_by_toc(self, results: list[dict], query: str) -> list[dict]:
        """Re-rank results by keyword overlap between query and heading_path terms (Phase 5).

        For each result, compute the overlap between query terms and heading_path terms.
        Score adjustment = (overlapping_terms / total_query_terms) * toc_rerank_weight.
        Results with empty heading_path are not re-ranked (pass through unchanged).
        """
        weight = getattr(self.config.search, 'toc_rerank_weight', 0.3)
        query_terms = set(query.lower().split())
        if not query_terms:
            return results

        for r in results:
            heading_path = r.get("heading_path", [])
            if not heading_path:
                continue  # Non-structured doc: no re-ranking

            # Collect all terms from heading_path
            heading_terms = set()
            for h in heading_path:
                heading_terms.update(h.lower().split())

            # Compute overlap
            overlapping = query_terms & heading_terms
            overlap_ratio = len(overlapping) / len(query_terms) if query_terms else 0
            boost = overlap_ratio * weight

            r["score"] = r["score"] + boost

        return results

    def get_toc(self, corpus: str, document: str | None = None) -> dict:
        """Get table of contents for a corpus or specific document (Phase 3).

        Args:
            corpus: Corpus name.
            document: Optional source path of a specific document.

        Returns:
            ToC tree dict or error dict.
        """
        if corpus not in self.corpora:
            return {"error": f"corpus not found: {corpus}"}

        corpus_idx = self.corpora[corpus]
        return corpus_idx.get_toc_tree(source_path=document)

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