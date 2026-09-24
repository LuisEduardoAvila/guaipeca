"""Regression tests for index.py concurrency fixes (P1 races).

Covers three related FAISS/lock synchronization bugs:

1. search_section() read self._faiss_index without holding the read lock
   (now snapshots the native index reference under the read lock).
2. search() released the read lock before calling FAISS search, racing with
   concurrent index.add() mutating the same native faiss.IndexFlatIP
   (now holds the read lock across the native search call).
3. _load() allowed two threads to populate shared state simultaneously
   (now serialized via a dedicated _load_lock with double-checked guard).

The tests use a lightweight deterministic mock embedder so no model has to
load — keeping them fast on a Pi.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from guaipeca.chunking import Chunk
from guaipeca.config import ChunkingConfig, CorpusConfig, IndexingConfig
from guaipeca.index import CorpusIndex

DIM = 32


class MockEmbedder:
    """Deterministic embedder — no model load, no external deps."""

    dimensions = DIM
    preprocess = False

    def embed(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(DIM).astype(np.float32)
            n = np.linalg.norm(v)
            out.append(v / n if n > 0 else v)
        return np.vstack(out)


def _make_index(tmp_path, name="conc-test"):
    corpus_cfg = CorpusConfig(name=name, path=str(tmp_path))
    chunking_cfg = ChunkingConfig()
    indexing_cfg = IndexingConfig(data_dir=str(tmp_path))
    return CorpusIndex(
        corpus=corpus_cfg,
        chunking=chunking_cfg,
        indexing=indexing_cfg,
        embedding_service=MockEmbedder(),
    ), indexing_cfg


def _seed(idx: CorpusIndex, n: int = 20) -> None:
    """Populate the in-memory index + chunks (bypassing disk scan)."""
    import faiss

    idx._faiss_index = faiss.IndexFlatIP(DIM)
    chunks = []
    vecs = []
    for i in range(n):
        c = Chunk(
            id=f"chunk-{i}",
            heading=f"H{i}",
            text=f"text {i}",
            source_path=str(idx.corpus.path),
            char_offset=i,
        )
        chunks.append(c)
        v = np.zeros(DIM, dtype=np.float32)
        v[i % DIM] = 1.0
        vecs.append(v)
    idx._faiss_index.add(np.vstack(vecs))
    idx._chunks = chunks
    idx._rebuild_stable_id_map()
    idx._rebuild_toc_metadata()
    idx._loaded = True


# ---------------------------------------------------------------------------
# Finding 1 — search_section uses a lock-protected native index snapshot
# ---------------------------------------------------------------------------

class TestSearchSectionLocking:
    def test_search_section_works_with_seeded_index(self, tmp_path):
        idx, _ = _make_index(tmp_path)
        _seed(idx)
        q = np.zeros(DIM, dtype=np.float32)
        q[0] = 1.0
        out = idx.search_section(q, section_filter="H0", top_k=5)
        # Section lookup returns *something* (either direct id or heading match)
        assert isinstance(out, list)

    def test_search_section_survives_concurrent_search(self, tmp_path):
        """search_section must not read self._faiss_index unlocked while
        another reader/search touches it."""
        idx, _ = _make_index(tmp_path)
        _seed(idx)
        q = np.zeros(DIM, dtype=np.float32)
        q[0] = 1.0

        errors = []

        def run_section():
            try:
                for _ in range(50):
                    idx.search_section(q, section_filter="H0", top_k=5)
            except Exception as e:  # pragma: no cover - failure reporting
                errors.append(e)

        def run_search():
            try:
                for _ in range(50):
                    idx.search(q, top_k=5)
            except Exception as e:  # pragma: no cover - failure reporting
                errors.append(e)

        threads = [threading.Thread(target=run_section) for _ in range(3)]
        threads += [threading.Thread(target=run_search) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent search_section raised: {errors}"


# ---------------------------------------------------------------------------
# Finding 2 — search() holds the read lock across the native FAISS call so it
# mutually excludes index.add() (we prove the lock is actually held).
# ---------------------------------------------------------------------------

class TestSearchHoldsReadLock:
    def test_search_holds_read_lock_during_faiss_call(self, tmp_path):
        idx, _ = _make_index(tmp_path)
        _seed(idx)

        observed = {}

        class SpyIndex:
            """Wraps the native index and records whether a writer could enter."""

            def __init__(self, native, rwlock):
                self._native = native
                self._rwlock = rwlock

            @property
            def ntotal(self):
                return self._native.ntotal

            def search(self, *a, **kw):
                # A writer must not be able to acquire the write lock while we
                # are inside the native search call.
                got_write = self._rwlock._cond.acquire(timeout=0)
                try:
                    observed["writer_blocked"] = not (
                        self._rwlock._writer or self._rwlock._readers == 0
                    )
                finally:
                    if got_write:
                        self._rwlock._cond.release()
                return self._native.search(*a, **kw)

        native = idx._faiss_index
        idx._faiss_index = SpyIndex(native, idx._rwlock)

        q = np.zeros(DIM, dtype=np.float32)
        q[0] = 1.0
        idx.search(q, top_k=5)

        # Inside search() at least one read lock must have been held.
        assert observed.get("writer_blocked") is True, (
            "search() did not hold the read lock across the FAISS search call"
        )


# ---------------------------------------------------------------------------
# Finding 3 — concurrent _load() is serialized
# ---------------------------------------------------------------------------

class TestConcurrentLoad:
    def test_only_one_thread_populates_state(self, tmp_path):
        idx, _ = _make_index(tmp_path)

        calls = []
        real_load = idx._load_unlocked

        def counting_load():
            calls.append(threading.current_thread().name)
            # simulate a slow disk load so other threads pile up on the lock
            time.sleep(0.05)
            real_load()

        idx._load_unlocked = counting_load

        start = threading.Barrier(8)

        def worker():
            start.wait()
            idx._load()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(calls) == 1, (
            f"_load() ran the population path {len(calls)} times; expected 1"
        )
        assert idx._loaded is True
        assert idx._faiss_index is not None

    def test_load_is_idempotent(self, tmp_path):
        idx, _ = _make_index(tmp_path)
        idx._load()
        first = idx._faiss_index
        idx._load()
        assert idx._faiss_index is first
