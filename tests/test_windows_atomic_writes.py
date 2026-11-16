"""Windows-compatibility regressions: atomic writes and file locking.

Two issues reported on 0.3.5 / Windows 11 / Python 3.14:

1. ``os.rename()`` raised ``FileExistsError: [WinError 183]`` when the
   destination already existed. On POSIX ``os.rename`` silently overwrites;
   on Windows it fails. Every metadata/FAISS/upload write targets a path that
   usually already exists on a re-index, so this broke the normal path.

2. ``fcntl`` is unavailable on Windows, so the index lock silently degraded
   to "no locking", allowing concurrent indexing to corrupt the index.

On Linux ``os.rename`` overwrites silently, so a real WinError 183 cannot be
provoked here. The regression guards therefore assert *which call* is used
(``os.replace``, with ``os.rename`` patched to fail), which is what actually
differs between a fixed and an unfixed checkout. The overwrite tests exercise
the behaviour both platforms must satisfy.
"""

from __future__ import annotations

import json
import os

import pytest

from guaipeca.index import CorpusIndex


class TestAtomicWriteUsesReplace:
    def test_write_json_overwrites_existing(self, tmp_path):
        """Re-writing metadata over an existing file must succeed."""
        path = tmp_path / "metadata.json"
        path.write_text('{"generation": 1}')

        CorpusIndex._atomic_write_json(path, {"generation": 2})

        assert json.loads(path.read_text())["generation"] == 2

    def test_write_json_leaves_no_temp_files(self, tmp_path):
        path = tmp_path / "metadata.json"
        path.write_text("{}")
        CorpusIndex._atomic_write_json(path, {"a": 1})

        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []

    def test_write_json_uses_os_replace(self, tmp_path, monkeypatch):
        """Regression guard: os.rename fails on Windows when the target exists."""
        calls = []
        real_rename = os.rename
        real_replace = os.replace

        def fake_rename(src, dst, *a, **kw):
            calls.append("rename")
            return real_rename(src, dst, *a, **kw)

        def fake_replace(src, dst, *a, **kw):
            calls.append("replace")
            return real_replace(src, dst, *a, **kw)

        monkeypatch.setattr(os, "rename", fake_rename)
        monkeypatch.setattr(os, "replace", fake_replace)
        CorpusIndex._atomic_write_json(tmp_path / "m.json", {"k": 1})

        assert calls == ["replace"], f"expected os.replace, saw {calls}"

    def test_write_json_cleans_up_temp_on_error(self, tmp_path):
        path = tmp_path / "m.json"

        class Unserialisable:
            pass

        with pytest.raises(TypeError):
            CorpusIndex._atomic_write_json(path, {"bad": Unserialisable()})

        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []
        assert not path.exists()


class TestUploadUsesReplace:
    def test_upload_uses_os_replace(self, monkeypatch, tmp_path):
        """Re-uploading a file under an existing name must not use os.rename.

        On Linux this succeeds either way; the guard is that os.rename is
        patched to raise, mirroring its Windows behaviour on an existing target.
        """
        from guaipeca import uploads
        from guaipeca.config import CorpusConfig, GuaipecaConfig, UploadConfig

        calls = []
        real_replace = os.replace

        def fake_replace(src, dst, *a, **kw):
            calls.append("replace")
            return real_replace(src, dst, *a, **kw)

        def fail_rename(*a, **kw):
            raise AssertionError("uploads must not call os.rename")

        monkeypatch.setattr(os, "replace", fake_replace)
        monkeypatch.setattr(os, "rename", fail_rename)

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        cfg = GuaipecaConfig(
            corpora=[CorpusConfig(name="test", path=str(corpus_dir))],
            upload=UploadConfig(allow=["test"]),
        )

        # Write twice: the second write targets an existing filename, which is
        # exactly the case that raised WinError 183 under os.rename.
        first = uploads.save_upload(cfg, "test", "note.md", b"# hello\n", do_index=False)
        second = uploads.save_upload(cfg, "test", "note.md", b"# hello again\n", do_index=False)

        assert "error" not in first, first
        assert "error" not in second, second
        assert calls == ["replace", "replace"]
        assert (corpus_dir / "note.md").read_bytes() == b"# hello again\n"


class TestFileLocking:
    def test_lock_acquired_and_released_fcntl(self, tmp_path):
        """POSIX path still works: acquire, then release, then re-acquire."""
        idx, _ = _make_index(tmp_path)
        assert idx._acquire_lock() is True
        idx._release_lock()
        assert idx._acquire_lock() is True
        idx._release_lock()

    def test_second_lock_is_refused(self, tmp_path):
        """A held lock must refuse a second acquirer."""
        import os
        import subprocess
        import sys
        from pathlib import Path

        idx, _ = _make_index(tmp_path)
        assert idx._acquire_lock() is True
        try:
            # Same process re-opening + flock: flock is per-fd on POSIX, so use
            # a subprocess to model the real "another process is indexing" case.
            code = f"""
import sys
sys.path.insert(0, {str(Path.cwd() / "src")!r})
from guaipeca.config import CorpusConfig, ChunkingConfig, IndexingConfig
from guaipeca.index import CorpusIndex
idx = CorpusIndex(
    # Must match _make_index()'s corpus name: the lock path is
    # <data_dir>/corpora/<name>/corpus.lock, so a different name means a
    # different lock file and the child would acquire happily.
    corpus=CorpusConfig(name="win-test", path={str(tmp_path)!r}),
    chunking=ChunkingConfig(),
    indexing=IndexingConfig(data_dir={str(tmp_path)!r}),
    embedding_service=None,
)
print(idx._acquire_lock())
"""
            out = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=120,
                check=False,
                # Absolute src path: a relative one lets the child import a
                # different installed guaipeca, which made this test lie.
                env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
            )
            assert out.returncode == 0, out.stderr
            assert out.stdout.strip().splitlines()[-1] == "False", out.stdout
        finally:
            idx._release_lock()

    def test_msvcrt_fallback_used_when_fcntl_missing(self, tmp_path, monkeypatch):
        """On Windows fcntl is absent; msvcrt.locking must be used instead."""
        import builtins
        import sys

        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "fcntl":
                raise ImportError("no fcntl on Windows")
            return real_import(name, *a, **kw)

        calls = []

        class FakeMsvcrt:
            LK_NBLCK = 2
            LK_UNLCK = 0

            @staticmethod
            def locking(fd, mode, nbytes):
                calls.append(mode)

        monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        idx, _ = _make_index(tmp_path)
        assert idx._acquire_lock() is True
        assert idx._lock_kind == "msvcrt"
        idx._release_lock()

        assert calls == [FakeMsvcrt.LK_NBLCK, FakeMsvcrt.LK_UNLCK]


DIM = 8


class _NullEmbedder:
    dimensions = DIM
    preprocess = False

    def embed(self, texts):
        import numpy as np
        return np.zeros((len(texts), DIM), dtype=np.float32)


def _make_index(tmp_path):
    from guaipeca.config import ChunkingConfig, CorpusConfig, IndexingConfig

    return CorpusIndex(
        corpus=CorpusConfig(name="win-test", path=str(tmp_path)),
        chunking=ChunkingConfig(),
        indexing=IndexingConfig(data_dir=str(tmp_path)),
        embedding_service=_NullEmbedder(),
    ), None
