"""Regression tests for the corpus index lock (issue #18).

The lock had two defects:

1. ``_acquire_lock`` opened the lock file with ``"w"``, truncating it before
   any lock was taken. On Windows such a truncation can be rejected by the OS
   while another process holds byte 0, and the surrounding ``except OSError``
   turned that into "proceeding unlocked" -- two indexers then wrote
   metadata.json / FAISS concurrently, the corruption #15 fixed.
2. Lock-open failures failed OPEN, so indexing ran without a lock.

The lock now opens ``"a+b"`` (creates, never truncates) and fails closed with a
distinguishable reason. The one deliberate fail-open path -- no locking
primitive at all -- is kept and documented by test 5.

No embedding model is loaded: every test either exercises ``_acquire_lock``
directly or is refused by the lock before ``_do_index`` runs.
"""

from __future__ import annotations

import sys
import types

import pytest

from guaipeca.config import ChunkingConfig, CorpusConfig, IndexingConfig
from guaipeca.index import CorpusIndex


class _NoEmbedder:
    """Never called: these tests stop at the lock, before any embedding."""

    dimensions = 8
    preprocess = False

    def embed(self, texts):  # pragma: no cover - must not be reached
        raise AssertionError("lock failure paths must not reach the embedder")

    def clear_cache(self):  # pragma: no cover
        pass


def _make_index(tmp_path, name="lock-test"):
    corpus_cfg = CorpusConfig(name=name, path=str(tmp_path))
    chunking_cfg = ChunkingConfig()
    indexing_cfg = IndexingConfig(data_dir=str(tmp_path))
    idx = CorpusIndex(
        corpus=corpus_cfg,
        chunking=chunking_cfg,
        indexing=indexing_cfg,
        embedding_service=_NoEmbedder(),
    )
    return idx


# ---------------------------------------------------------------------------
# 1. The lock file is opened without truncation ("a+b", never "w")
# ---------------------------------------------------------------------------

def test_lock_file_not_truncated(tmp_path):
    idx = _make_index(tmp_path)
    sentinel = "sentinel-lock-content-must-survive\n"
    idx.lock_path.write_text(sentinel, encoding="utf-8")

    try:
        assert idx._acquire_lock() is True
        assert idx._lock_failure is None
    finally:
        idx._release_lock()

    # A "w" open would have zeroed the file before any lock was taken.
    assert idx.lock_path.read_text(encoding="utf-8") == sentinel


# ---------------------------------------------------------------------------
# 2. An unopenable lock file fails CLOSED with the open-failure reason
# ---------------------------------------------------------------------------

def test_open_failure_fails_closed(tmp_path):
    idx = _make_index(tmp_path)
    # A directory at the lock path is portable and works even as root;
    # chmod 000 is unreliable in containers and when running as root.
    idx.lock_path.mkdir()

    assert idx._acquire_lock() is False
    assert idx._lock_failure is not None
    assert "could not be opened" in idx._lock_failure

    result = idx.index()
    assert "error" in result
    assert "could not acquire lock" in result["error"]
    assert "refusing to index" in result["error"]
    # The failure reason is surfaced, not just the generic in-progress text.
    assert "indexing in progress" not in result["error"]


# ---------------------------------------------------------------------------
# 3. Real contention still reports the backwards-compatible message
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="uses a second POSIX fd")
def test_contention_reports_in_progress(tmp_path):
    import fcntl

    holder = _make_index(tmp_path)
    try:
        assert holder._acquire_lock() is True
        # A second instance targets the same corpus dir, hence the same lock
        # file; POSIX flock excludes it.
        contender = _make_index(tmp_path)

        result = contender.index()

        assert "error" in result
        assert result["error"] == f"indexing in progress for corpus {contender.corpus.name}"
        assert contender._lock_failure == "held by another process"
    finally:
        holder._release_lock()

    # Sanity: the lock file survived the failed attempt and is still lockable.
    with open(holder.lock_path, "rb") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# 4. The Windows branch is reachable and fails closed on contention
# ---------------------------------------------------------------------------

def test_msvcrt_branch_reachable_via_mock(tmp_path, monkeypatch):
    idx = _make_index(tmp_path)
    calls = []

    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.LK_NBLCK = 2
    fake_msvcrt.LK_UNLCK = 0

    def locking(fd, mode, nbytes):
        calls.append((fd, mode, nbytes))
        raise OSError(36, "Resource deadlock avoided")

    fake_msvcrt.locking = locking

    # The function imports fcntl/msvcrt inline. A None entry in sys.modules
    # makes ``import fcntl`` raise ImportError, redirecting to the Windows
    # branch on Linux.
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setitem(sys.modules, "fcntl", None)

    assert idx._acquire_lock() is False
    assert idx._lock_failure == "held by another process"
    assert calls and calls[0][1] == fake_msvcrt.LK_NBLCK

    result = idx.index()
    assert result["error"] == f"indexing in progress for corpus {idx.corpus.name}"


# ---------------------------------------------------------------------------
# 5. No locking primitive at all: deliberate fail-open, still indexes
# ---------------------------------------------------------------------------

def test_no_mechanism_still_indexes(tmp_path, monkeypatch):
    idx = _make_index(tmp_path)

    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setitem(sys.modules, "msvcrt", None)

    assert idx._acquire_lock() is True
    # Fail-open is not a failure: there is no reason to report.
    assert idx._lock_failure is None
    idx._release_lock()
