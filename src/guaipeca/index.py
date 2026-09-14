"""FAISS + BM25 index management per corpus.

Each corpus gets its own FAISS index + BM25 index + metadata JSON.
Supports incremental indexing (file hash check) and persistence.

Production safety features:
- Atomic JSON writes (temp + rename) to prevent corruption on crash
- Atomic FAISS writes (temp + rename) to prevent corruption on crash
- Metadata written BEFORE FAISS index (crash-safe ordering)
- Thread-safe access via read-write pattern (concurrent readers, exclusive writer)
- File locking (fcntl.flock) to prevent concurrent indexing of the same corpus
- Stable chunk IDs (SHA256-based) with lookup map for retrieval across rebuilds
- Stale vector cleanup: filters out results from deleted files, rebuilds when deletion ratio is high
- BM25 sparse keyword index alongside FAISS dense vector index
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_file
from .config import ChunkingConfig, CorpusConfig, IndexingConfig
from .converter import Converter
from .embedding import EmbeddingService

logger = logging.getLogger(__name__)

# Deletion ratio threshold for triggering a full index rebuild
STALE_REBUILD_THRESHOLD = 0.10


class _ReadWriteLock:
    """Simple read-write lock using threading.Condition.

    Allows multiple concurrent readers or one exclusive writer.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def acquire_read(self):
        with self._cond:
            while self._writer or self._waiting_writers > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self):
        with self._cond:
            self._waiting_writers += 1
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._waiting_writers -= 1
            self._writer = True

    def release_write(self):
        with self._cond:
            self._writer = False
            self._cond.notify_all()

    class _ReadContext:
        def __init__(self, lock):
            self._lock = lock

        def __enter__(self):
            self._lock.acquire_read()
            return self

        def __exit__(self, *args):
            self._lock.release_read()

    class _WriteContext:
        def __init__(self, lock):
            self._lock = lock

        def __enter__(self):
            self._lock.acquire_write()
            return self

        def __exit__(self, *args):
            self._lock.release_write()

    def read_lock(self):
        return self._ReadContext(self)

    def write_lock(self):
        return self._WriteContext(self)


class CorpusIndex:
    """Manages FAISS index + BM25 index + metadata for a single corpus."""

    def __init__(
        self,
        corpus: CorpusConfig,
        chunking: ChunkingConfig,
        indexing: IndexingConfig,
        embedding_service,
        converter: Converter | None = None,
    ):
        """
        Args:
            corpus: Corpus configuration (name, path, weight, etc.)
            chunking: Chunking parameters.
            indexing: Indexing settings (data dir, incremental, etc.)
            embedding_service: EmbeddingService instance.
            converter: Document converter (markitdown). If None, only .md/.txt supported.
        """
        self.corpus = corpus
        self.chunking = chunking
        self.indexing = indexing
        self.embedder = embedding_service
        self.converter = converter or Converter()

        # Paths
        self.index_dir = Path(indexing.data_dir) / "corpora" / corpus.name
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_path = self.index_dir / "faiss_index.fai"
        self.metadata_path = self.index_dir / "metadata.json"
        self.hashes_path = self.index_dir / "file_hashes.json"
        self.lock_path = self.index_dir / "corpus.lock"

        # BM25 paths
        self.bm25_path = self.index_dir / "bm25_index"  # directory for bm25s save

        # State
        self._faiss_index = None
        self._chunks: list[Chunk] = []
        self._file_hashes: dict[str, str] = {}
        self._loaded = False
        # Stable ID → FAISS positional index lookup map
        self._stable_id_to_pos: dict[str, int] = {}

        # BM25 state
        self._bm25_retriever = None
        self._bm25_corpus_texts: list[str] = []

        # Thread safety: read-write lock for concurrent search / exclusive indexing
        self._rwlock = _ReadWriteLock()

    def _file_hash(self, file_path: str) -> str:
        """SHA256 hash of file content."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _atomic_write_json(path: Path, data) -> None:
        """
        Write JSON to a file atomically.

        Writes to a temp file first, then renames to the final path.
        This ensures a crash during write doesn't corrupt the file.
        """
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=path.name + "."
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, str(path))
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_write_faiss(faiss_index, path: Path) -> None:
        """
        Write FAISS index atomically.

        Writes to a temp file first, then renames to the final path.
        Ensures a crash during write doesn't corrupt the index file.
        """
        import faiss

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=path.name + "."
        )
        try:
            os.close(tmp_fd)
            faiss.write_index(faiss_index, tmp_path)
            os.rename(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _rebuild_stable_id_map(self) -> None:
        """Rebuild the stable_id → FAISS positional index lookup map."""
        self._stable_id_to_pos = {
            chunk.id: pos for pos, chunk in enumerate(self._chunks)
        }

    # ---- BM25 methods ----

    def _build_bm25_index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunk texts.

        Full rebuild on every index() call — BM25 is cheap to build
        (pure token frequency counting, no model inference).
        """
        try:
            import bm25s
        except ImportError:
            logger.warning("bm25s not installed; BM25 hybrid search will be unavailable")
            self._bm25_retriever = None
            self._bm25_corpus_texts = []
            return

        if not chunks:
            self._bm25_retriever = None
            self._bm25_corpus_texts = []
            return

        texts = [c.text for c in chunks]
        try:
            corpus_tokens = bm25s.tokenize(texts, show_progress=False)
            self._bm25_retriever = bm25s.BM25()
            self._bm25_retriever.index(corpus_tokens, show_progress=False)
            self._bm25_corpus_texts = texts
            logger.debug(f"Built BM25 index for '{self.corpus.name}' with {len(texts)} chunks")
        except Exception as e:
            logger.error(f"Failed to build BM25 index for '{self.corpus.name}': {e}")
            self._bm25_retriever = None
            self._bm25_corpus_texts = []

    def _save_bm25(self) -> None:
        """Save BM25 index to disk."""
        if self._bm25_retriever is not None:
            try:
                self._bm25_retriever.save(
                    str(self.bm25_path),
                    corpus=self._bm25_corpus_texts,
                )
            except Exception as e:
                logger.error(f"Failed to save BM25 index for '{self.corpus.name}': {e}")

    def _load_bm25(self) -> None:
        """Load BM25 index from disk."""
        try:
            import bm25s
        except ImportError:
            logger.warning("bm25s not installed; BM25 hybrid search will be unavailable")
            self._bm25_retriever = None
            return

        if self.bm25_path.exists():
            try:
                self._bm25_retriever = bm25s.BM25.load(
                    str(self.bm25_path),
                )
                logger.debug(f"Loaded BM25 index for '{self.corpus.name}'")
            except Exception as e:
                logger.error(f"Failed to load BM25 index for '{self.corpus.name}': {e}")
                self._bm25_retriever = None
        else:
            self._bm25_retriever = None

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        """BM25 keyword search on this corpus.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of result dicts with chunk data and score (same format as search()).
        """
        try:
            import bm25s
        except ImportError:
            return []

        # Acquire read lock, load if needed, snapshot state
        with self._rwlock.read_lock():
            self._load()
            retriever = self._bm25_retriever
            chunks_snapshot = list(self._chunks)

        if retriever is None or not chunks_snapshot:
            return []

        try:
            query_tokens = bm25s.tokenize([query], show_progress=False)
            fetch_k = min(top_k * 3, len(chunks_snapshot))
            results, scores = retriever.retrieve(
                query_tokens, k=fetch_k, show_progress=False
            )
        except Exception as e:
            logger.error(f"BM25 search failed for '{self.corpus.name}': {e}")
            return []

        # Map BM25 results to chunk dicts
        out = []
        result_indices = results[0]  # bm25s returns 2D array
        result_scores = scores[0]

        for idx, score in zip(result_indices, result_scores):
            if idx < 0 or idx >= len(chunks_snapshot):
                continue

            chunk = chunks_snapshot[idx]

            # Filter out stale results
            if not os.path.exists(chunk.source_path):
                continue

            # Apply corpus weight
            weighted_score = float(score) * self.corpus.weight

            out.append({
                "chunk_id": f"{self.corpus.name}:{chunk.id}",
                "summary": chunk.summary,
                "location": f"{chunk.source_path}#{chunk.heading}".rstrip("#"),
                "source_path": chunk.source_path,
                "filename": os.path.basename(chunk.source_path),
                "corpus": self.corpus.name,
                "topic": self.corpus.topic,
                "score": weighted_score,
                "raw_score": float(score),
            })

            if len(out) >= top_k:
                break

        return out

    # ---- End BM25 methods ----

    def _load_faiss_with_recovery(self):
        """Load FAISS index with validation and backup recovery.

        Tries the primary .fai file first. If it's missing or corrupted,
        falls back to the .fai.bak backup. If both fail, starts fresh.
        Logs warnings on all fallback paths.
        """
        import faiss

        backup_path = self.faiss_path.with_suffix(".fai.bak")

        # Try primary index
        if self.faiss_path.exists():
            try:
                index = faiss.read_index(str(self.faiss_path))
                return index
            except Exception as e:
                logger.warning(
                    f"FAISS index load failed for '{self.corpus.name}': {e}, "
                    f"trying backup"
                )

        # Try backup
        if backup_path.exists():
            try:
                index = faiss.read_index(str(backup_path))
                logger.warning(
                    f"Recovered FAISS index from backup for '{self.corpus.name}' "
                    f"({index.ntotal} vectors)"
                )
                return index
            except Exception as e:
                logger.warning(
                    f"FAISS backup also corrupted for '{self.corpus.name}': {e}, "
                    f"starting fresh"
                )

        # Fresh index
        if self.faiss_path.exists() or backup_path.exists():
            logger.warning(
                f"No valid FAISS index found for '{self.corpus.name}', starting fresh"
            )
        return faiss.IndexFlatIP(self.embedder.dimensions)

    def _load(self):
        """Load existing index and metadata from disk."""
        if self._loaded:
            return


        # Load file hashes
        if self.hashes_path.exists():
            with open(self.hashes_path, "r") as f:
                self._file_hashes = json.load(f)

        # Load metadata (chunks with text for get_chunk retrieval)
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                meta = json.load(f)
                self._chunks = [
                    Chunk(
                        id=c["id"],
                        heading=c.get("heading", ""),
                        text=c.get("text", ""),
                        source_path=c.get("source_path", ""),
                        char_offset=c.get("char_offset", 0),
                        metadata=c.get("metadata", {}),
                    )
                    for c in meta
                ]

        # Load FAISS index (with backup recovery)
        self._faiss_index = self._load_faiss_with_recovery()

        # Rebuild stable ID lookup map
        self._rebuild_stable_id_map()

        # Load BM25 index
        self._load_bm25()

        self._loaded = True

    def _save(self):
        """Persist index and metadata to disk with atomic writes.

        Write order (crash-safe):
        1. Metadata JSON (atomic) — if this succeeds but FAISS write fails,
           old FAISS index still matches old metadata (safe, indices align)
        2. FAISS index (atomic) — temp + rename
        3. File hashes (atomic)
        4. BM25 index — best-effort save (not critical for crash safety)
        """
        # 1. Write metadata FIRST (atomic)
        self._atomic_write_json(self.metadata_path, [c.to_dict() for c in self._chunks])

        # 1b. Backup existing FAISS index before overwriting
        if self.faiss_path.exists():
            backup_path = self.faiss_path.with_suffix(".fai.bak")
            try:
                import shutil
                shutil.copy2(str(self.faiss_path), str(backup_path))
            except Exception as e:
                logger.warning(
                    f"Failed to backup FAISS index for '{self.corpus.name}': {e}"
                )

        # 2. Write FAISS index (atomic: temp + rename)
        self._atomic_write_faiss(self._faiss_index, self.faiss_path)

        # 3. Write file hashes (atomic)
        self._atomic_write_json(self.hashes_path, self._file_hashes)

        # 4. Write BM25 index (best-effort)
        self._save_bm25()

    def _acquire_lock(self) -> bool:
        """
        Acquire an exclusive file lock for indexing this corpus.

        Uses fcntl.flock on a lock file. Returns True if lock acquired,
        False if another process holds the lock. On non-Unix systems,
        returns True (no file locking available).
        """
        try:
            import fcntl
        except ImportError:
            # Non-Unix system (e.g. Windows) — no fcntl available
            logger.warning("fcntl not available on this platform; file locking disabled")
            self._lock_fd = None
            return True

        try:
            self._lock_fd = open(self.lock_path, "w")
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            # Lock is held by another process
            if hasattr(self, "_lock_fd") and self._lock_fd:
                self._lock_fd.close()
            return False

    def _release_lock(self) -> None:
        """Release the file lock."""
        if hasattr(self, "_lock_fd") and self._lock_fd:
            try:
                import fcntl
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
                self._lock_fd.close()
            except (OSError, ImportError):
                pass
            self._lock_fd = None

    def _scan_files(self) -> list[str]:
        """Scan corpus directory for files to index."""
        corpus_path = Path(self.corpus.path)
        if not corpus_path.exists():
            logger.warning(f"Corpus path does not exist: {corpus_path}")
            return []

        files = []
        for root, dirs, filenames in os.walk(corpus_path):
            # Filter out excluded dirs
            dirs[:] = [d for d in dirs if d not in self.indexing.exclude_dirs]

            for filename in filenames:
                file_path = os.path.join(root, filename)
                ext = Path(filename).suffix.lower()

                if ext not in self.corpus.extensions:
                    continue

                # Check file size
                try:
                    size = os.path.getsize(file_path)
                    if size > self.indexing.max_file_size:
                        logger.warning(f"File too large ({size} bytes): {file_path}")
                        continue
                except OSError:
                    continue

                files.append(file_path)

        return sorted(files)

    def _rebuild_index(self, keep_files: set[str]) -> None:
        """
        Rebuild the FAISS index from scratch, keeping only chunks from files in keep_files.

        This is triggered when the stale vector ratio exceeds the threshold.
        """
        import faiss

        logger.info(f"Rebuilding index for corpus '{self.corpus.name}' (stale vector cleanup)")

        # Collect chunks that belong to still-existing files
        keep_chunks = []
        keep_texts = []
        keep_headings = []
        for chunk in self._chunks:
            if chunk.source_path in keep_files:
                keep_chunks.append(chunk)
                keep_texts.append(chunk.text)
                keep_headings.append(chunk.heading)

        # Re-embed and rebuild
        if keep_texts:
            if getattr(self.embedder, 'preprocess', False):
                embed_texts = [
                    EmbeddingService.preprocess_text(t, h)
                    for t, h in zip(keep_texts, keep_headings)
                ]
            else:
                embed_texts = keep_texts
            embeddings = self.embedder.embed(embed_texts)
            self._faiss_index = faiss.IndexFlatIP(self.embedder.dimensions)
            self._faiss_index.add(embeddings)
        else:
            self._faiss_index = faiss.IndexFlatIP(self.embedder.dimensions)

        self._chunks = keep_chunks
        # Rebuild stable ID lookup map after reordering
        self._rebuild_stable_id_map()

    def index(self, force: bool = False) -> dict:
        """
        Index (or re-index) this corpus.

        Args:
            force: If True, re-index all files even if unchanged.

        Returns:
            Stats dict: files_indexed, chunks_created, files_skipped, errors, duration_ms
        """
        # Acquire file lock to prevent concurrent indexing of the same corpus
        if not self._acquire_lock():
            return {
                "corpus": self.corpus.name,
                "error": f"indexing in progress for corpus {self.corpus.name}",
            }

        try:
            with self._rwlock.write_lock():
                return self._do_index(force)
        finally:
            self._release_lock()

    def _do_index(self, force: bool) -> dict:
        """Internal indexing logic (called under write lock)."""
        start = time.time()
        self._load()

        files = self._scan_files()
        files_indexed = 0
        chunks_created = 0
        files_skipped = 0
        errors = []

        # When force=True, reset the index to avoid duplicate vectors.
        # We keep existing chunks/hashes for incremental skip logic, but
        # rebuild FAISS + chunks from scratch so re-indexing is idempotent.
        if force:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(self.embedder.dimensions)
            self._chunks = []
            self._file_hashes = {}
            self._rebuild_stable_id_map()
            # Reset BM25 state too
            self._bm25_retriever = None
            self._bm25_corpus_texts = []
            # Clear embedding cache on force reindex
            if hasattr(self.embedder, 'clear_cache'):
                self.embedder.clear_cache()

        new_chunks: list[Chunk] = []
        new_embeddings: list[np.ndarray] = []

        # Keep track of which files still exist (for cleanup)
        current_files = set()

        for file_path in files:
            current_files.add(file_path)

            # Compute file hash once (used for both change detection and storage)
            file_hash = self._file_hash(file_path)

            # Check if file changed (incremental)
            if not force and self.indexing.incremental and \
                    self._file_hashes.get(file_path) == file_hash:
                files_skipped += 1
                continue

            # Convert to markdown
            try:
                md_text = self.converter.convert(file_path)
            except Exception as e:
                errors.append({"file": file_path, "error": str(e)})
                logger.error(f"Failed to convert {file_path}: {e}")
                continue

            # Chunk (pass overlap from config)
            try:
                file_chunks = chunk_file(
                    file_path=file_path,
                    markdown_text=md_text,
                    max_size=self.chunking.max_size,
                    table_aware=self.chunking.table_aware,
                    split_on_headings=self.chunking.split_on_headings,
                    overlap=self.chunking.overlap,
                )
            except Exception as e:
                errors.append({"file": file_path, "error": str(e)})
                logger.error(f"Failed to chunk {file_path}: {e}")
                continue

            # Embed (with optional preprocessing for better semantic signal)
            try:
                if getattr(self.embedder, 'preprocess', False):
                    texts = [
                        EmbeddingService.preprocess_text(c.text, c.heading)
                        for c in file_chunks
                    ]
                else:
                    texts = [c.text for c in file_chunks]
                if texts:
                    embeddings = self.embedder.embed(texts)
                    new_chunks.extend(file_chunks)
                    new_embeddings.append(embeddings)
            except Exception as e:
                errors.append({"file": file_path, "error": str(e)})
                logger.error(f"Failed to embed {file_path}: {e}")
                continue

            # Update hash (reuse cached hash from earlier)
            self._file_hashes[file_path] = file_hash
            files_indexed += 1
            chunks_created += len(file_chunks)

        # Add new embeddings to FAISS index
        if new_embeddings:
            all_new = np.vstack(new_embeddings)
            self._faiss_index.add(all_new)
            self._chunks.extend(new_chunks)

        # Clean up deleted files
        deleted = set(self._file_hashes.keys()) - current_files
        if deleted:
            for f in deleted:
                logger.info(f"File removed: {f}")
                self._file_hashes.pop(f, None)

            # Check if rebuild is needed (stale vector ratio exceeds threshold)
            total_vectors = self._faiss_index.ntotal
            if total_vectors > 0:
                # Count how many chunks belong to deleted files
                stale_count = sum(
                    1 for c in self._chunks if c.source_path not in current_files
                )
                stale_ratio = stale_count / total_vectors if total_vectors > 0 else 0

                if stale_ratio > STALE_REBUILD_THRESHOLD:
                    logger.info(
                        f"Stale vector ratio {stale_ratio:.1%} exceeds threshold "
                        f"{STALE_REBUILD_THRESHOLD:.0%}, rebuilding index"
                    )
                    self._rebuild_index(current_files)
                else:
                    logger.debug(
                        f"Stale vector ratio {stale_ratio:.1%} below threshold, "
                        f"will filter in search"
                    )

        # Rebuild stable ID map (positions may have changed)
        self._rebuild_stable_id_map()

        # Build BM25 index from all current chunks (full rebuild — cheap)
        self._build_bm25_index(self._chunks)

        # Save
        self._save()

        duration_ms = int((time.time() - start) * 1000)
        stats = {
            "corpus": self.corpus.name,
            "files_indexed": files_indexed,
            "chunks_created": chunks_created,
            "files_skipped": files_skipped,
            "errors": errors,
            "duration_ms": duration_ms,
        }
        logger.info(f"Indexed {self.corpus.name}: {stats}")
        return stats

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search this corpus index (dense/FAISS).

        Args:
            query_vector: Pre-embedded query vector (normalized).
            top_k: Number of results to return.

        Returns:
            List of result dicts with chunk data and score.
        """
        # Acquire read lock only long enough to get index reference and copy state
        with self._rwlock.read_lock():
            self._load()
            faiss_index = self._faiss_index
            chunks_snapshot = list(self._chunks)  # shallow copy for consistent reads

        if faiss_index is None or faiss_index.ntotal == 0:
            return []

        # Over-fetch to compensate for stale results that will be filtered out
        fetch_k = min(top_k * 3, faiss_index.ntotal)

        # Search FAISS (without holding the lock)
        scores, indices = faiss_index.search(
            query_vector.reshape(1, -1).astype(np.float32),
            fetch_k,
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks_snapshot):
                continue

            chunk = chunks_snapshot[idx]

            # Filter out stale results: skip chunks whose source file no longer exists
            if not os.path.exists(chunk.source_path):
                continue

            # Apply corpus weight
            weighted_score = float(score) * self.corpus.weight

            # Use stable chunk ID for retrieval
            stable_id = chunk.id

            results.append({
                "chunk_id": f"{self.corpus.name}:{stable_id}",
                "summary": chunk.summary,
                "location": f"{chunk.source_path}#{chunk.heading}".rstrip("#"),
                "source_path": chunk.source_path,
                "filename": os.path.basename(chunk.source_path),
                "corpus": self.corpus.name,
                "topic": self.corpus.topic,
                "score": weighted_score,
                "raw_score": float(score),
            })

            # Stop once we have enough valid results
            if len(results) >= top_k:
                break

        return results

    def get_chunk(self, stable_id: str) -> dict | None:
        """Get full chunk content by stable chunk ID.

        Args:
            stable_id: The chunk's stable SHA256-based ID.

        Returns:
            Chunk dict or None if not found.
        """
        with self._rwlock.read_lock():
            self._load()

            pos = self._stable_id_to_pos.get(stable_id)
            if pos is None or pos < 0 or pos >= len(self._chunks):
                return None

            chunk = self._chunks[pos]
            return {
                "chunk_id": f"{self.corpus.name}:{stable_id}",
                "text": chunk.text,
                "source": chunk.source_path,
                "heading": chunk.heading,
                "corpus": self.corpus.name,
                "char_count": chunk.char_count,
            }

    def list_documents(self) -> list[dict]:
        """List all indexed documents in this corpus.

        Returns metadata for each tracked file: source_path, filename,
        chunk count, file size, and last_indexed timestamp.
        """
        with self._rwlock.read_lock():
            self._load()

            # Count chunks per source_path
            chunk_counts: dict[str, int] = {}
            for chunk in self._chunks:
                chunk_counts[chunk.source_path] = (
                    chunk_counts.get(chunk.source_path, 0) + 1
                )

            docs = []
            for source_path in self._file_hashes:
                filename = os.path.basename(source_path)
                try:
                    file_size = os.path.getsize(source_path)
                except OSError:
                    file_size = 0

                # Get last_indexed from metadata file mtime
                try:
                    mtime = os.path.getmtime(str(self.metadata_path))
                    last_indexed = int(mtime)
                except OSError:
                    last_indexed = 0

                docs.append({
                    "source_path": source_path,
                    "filename": filename,
                    "corpus": self.corpus.name,
                    "chunk_count": chunk_counts.get(source_path, 0),
                    "file_size": file_size,
                    "last_indexed": last_indexed,
                })
            return docs

    def get_document_text(self, source_path: str) -> dict | None:
        """Get full converted text of a document by source_path.

        Args:
            source_path: Absolute path to the source file within this corpus.

        Returns:
            Dict with text, source_path, filename, corpus, file_size,
            chunk_count, or None if not found.
        """
        with self._rwlock.read_lock():
            self._load()

            # Check if this file is tracked
            if source_path not in self._file_hashes:
                return None

            # Check if file exists on disk
            if not os.path.exists(source_path):
                return None

            # Convert the file to markdown text
            try:
                text = self.converter.convert(source_path)
            except Exception as e:
                logger.error(f"Failed to convert {source_path}: {e}")
                return None

            # Count chunks for this document
            chunk_count = sum(
                1 for c in self._chunks if c.source_path == source_path
            )

            try:
                file_size = os.path.getsize(source_path)
            except OSError:
                file_size = 0

            return {
                "source_path": source_path,
                "filename": os.path.basename(source_path),
                "corpus": self.corpus.name,
                "text": text,
                "char_count": len(text),
                "file_size": file_size,
                "chunk_count": chunk_count,
            }

    @property
    def stats(self) -> dict:
        """Index statistics."""
        with self._rwlock.read_lock():
            self._load()
            return {
                "corpus": self.corpus.name,
                "total_chunks": len(self._chunks),
                "total_vectors": self._faiss_index.ntotal if self._faiss_index else 0,
                "files_tracked": len(self._file_hashes),
                "bm25_available": self._bm25_retriever is not None,
            }