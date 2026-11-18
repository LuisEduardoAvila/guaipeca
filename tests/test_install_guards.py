"""Fresh-install guards (issue #30).

A fresh clone silently indexed nothing: ``cp config/guaipeca.yaml
~/.guaipeca/guaipeca.yaml`` fails when the directory does not exist, the
loader falls back to the bundled sample, whose corpus paths are placeholders
(``/path/to/your/docs``), and the run exited 0 having indexed 0 files.

These tests pin the three states that must stay distinguishable:

1. the unedited sample is detected by CONTENT and ``index`` refuses with 2;
2. a corpus path that does not exist is refused with 1, naming every bad path;
3. a corpus that exists but matches no file warns, and still exits 0;
4. an incremental re-index with nothing changed stays silent and exits 0.

No embedding model and no network: ``_create_services`` is monkeypatched, and
``cmd_status`` is exercised with its Searcher and EmbeddingService replaced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from guaipeca import cli, search
from guaipeca.cli import _is_unedited_sample, cmd_index, cmd_status

# The bundled sample, located the same way the guard locates it: repo root is
# two levels above src/guaipeca/cli.py, i.e. tests/.. here.
SAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "guaipeca.yaml"


# ---- Helpers ---------------------------------------------------------------


def _status_embedder_stub():
    """EmbeddingService replacement that never loads a model.

    Built at module level so both the fake ``cli._create_services`` and the
    ``cmd_status`` patch point share one class.
    """

    class _StubEmbedder:
        """cmd_status only reads these attributes; no embed call happens."""

        def __init__(self, *args, **kwargs):
            self.cache_dir = None
            self.model_name = "stub"
            self.dimensions = 384
            self.preprocess = False
            self.cache_ttl_seconds = 3600
            self.cache_max_entries = 10000

    return _StubEmbedder


def _status_searcher_stub():
    """Searcher replacement exposing only the ``status`` dict cmd_status reads."""

    class _StubSearcher:
        def __init__(self, *args, **kwargs):
            self.status = {
                "embedding_model": "stub",
                "dimensions": 384,
                "hybrid_enabled": False,
                "max_distance": None,
                "corpora": [],
            }

    return _StubSearcher


class _SearcherStub:
    """Stands in for Searcher: returns one canned result, records calls.

    ``cmd_index`` only calls ``index_corpus(corpus_name=..., force=...)``, so
    this is the whole contract. ``calls`` is how the missing-path tests prove
    the guard fired BEFORE any indexing work was attempted.
    """

    def __init__(self, result: dict):
        self.result = result
        self.calls: list[str] = []

    def index_corpus(self, corpus_name: str, force: bool = False) -> dict:
        self.calls.append(corpus_name)
        return self.result


def _write_config(tmp_path: Path, corpora: list[dict]) -> Path:
    """Write a minimal config file pointing at *corpora*."""
    path = tmp_path / "guaipeca.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "corpora": corpora,
                "indexing": {"data_dir": str(tmp_path / "data")},
            }
        ),
        encoding="utf-8",
    )
    return path


def _fail_services(monkeypatch) -> None:
    """Make any attempt to build services fail the test.

    Indexing needs fastembed and faiss. A test that accidentally gets that far
    would be slow and environment-dependent, and one that does so on a refusal
    path would hide the guard being bypassed.
    """

    def _boom(config):  # pragma: no cover - only on a broken test
        raise AssertionError("_create_services must not be called in this test")

    monkeypatch.setattr(cli, "_create_services", _boom)


def _stub_searcher(monkeypatch, result: dict) -> _SearcherStub:
    stub = _SearcherStub(result)

    class _Unused:
        pass

    def _fake_create_services(config):
        return _Unused(), _Unused(), stub

    monkeypatch.setattr(cli, "_create_services", _fake_create_services)
    return stub


def _stats(corpus: str, indexed: int, skipped: int, chunks: int = 0) -> dict:
    """One corpus's stats, in the shape CorpusIndex._do_index returns."""
    return {
        "corpus": corpus,
        "files_indexed": indexed,
        "chunks_created": chunks,
        "files_skipped": skipped,
        "errors": [],
        "duration_ms": 1,
    }


def _args(**kwargs) -> argparse.Namespace:
    """argparse.Namespace with the defaults cmd_index/cmd_status rely on."""
    base = {"config": None, "corpus": None, "force": False}
    base.update(kwargs)
    return argparse.Namespace(**base)


def _copy_sample(tmp_path: Path) -> Path:
    copy = tmp_path / "guaipeca.yaml"
    copy.write_bytes(SAMPLE_CONFIG.read_bytes())
    return copy


def _simulate_fallback(monkeypatch, tmp_path: Path, copy: Path) -> None:
    """Make ``_find_config`` land on *copy* exactly as the fallback would.

    The guard deliberately ignores an explicit ``--config`` pointing at the
    sample: passing that flag is a deliberate choice, while issue #30 is about
    falling through to it by accident. So the tests have to reproduce the
    accident — no ``--config`` — rather than hand the path in.
    """
    monkeypatch.setattr("guaipeca.cli._find_config", lambda explicit=None: str(copy))


# ---- 1. Sample detection ---------------------------------------------------


def test_bundled_sample_is_located():
    """The comparison target exists where the guard expects it.

    If this fails, every other test here exercises the ``return False``
    fallback branch and proves nothing about the placeholder case.
    """
    assert SAMPLE_CONFIG.is_file(), f"bundled sample missing: {SAMPLE_CONFIG}"


def test_unedited_sample_config_is_detected(tmp_path):
    """A byte-identical copy is the sample; one edited path is not."""
    copy = _copy_sample(tmp_path)
    assert _is_unedited_sample(str(copy)) is True

    text = SAMPLE_CONFIG.read_text(encoding="utf-8")
    assert "/path/to/your/docs" in text, (
        "the bundled sample no longer carries a placeholder path; this test "
        "needs a new value to edit"
    )
    edited = tmp_path / "edited.yaml"
    edited.write_text(text.replace("/path/to/your/docs", str(tmp_path), 1), encoding="utf-8")
    assert _is_unedited_sample(str(edited)) is False


def test_unreadable_config_is_not_the_sample(tmp_path):
    """A path that cannot be compared returns False instead of raising."""
    assert _is_unedited_sample(str(tmp_path / "does-not-exist.yaml")) is False


# ---- 2. Refusal on the sample ---------------------------------------------


def test_sample_config_index_refuses(tmp_path, monkeypatch, capsys):
    """The unedited sample never reaches an index run: exit 2, no services."""
    copy = _copy_sample(tmp_path)
    _simulate_fallback(monkeypatch, tmp_path, copy)
    _fail_services(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cmd_index(_args(config=None))

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "sample config" in err
    # The advice must point at the file the user is meant to create. Telling
    # them to edit the bundled config/ sample would silently undo itself on the
    # next `git pull`.
    assert "~/.guaipeca/guaipeca.yaml" in err


def test_sample_config_read_command_warns_only(tmp_path, monkeypatch, capsys):
    """status on the unedited sample warns but does not exit non-zero.

    Read commands must stay usable: the warning names the file to edit, and
    nothing about ``status`` depends on the corpora holding documents.
    """
    copy = _copy_sample(tmp_path)
    _simulate_fallback(monkeypatch, tmp_path, copy)
    _fail_services(monkeypatch)

    # cmd_status imports both inside the function, so the patch goes on the
    # source modules: neither name is a cli attribute to replace.
    import guaipeca.embedding as embedding_module

    monkeypatch.setattr(embedding_module, "EmbeddingService", _status_embedder_stub())
    monkeypatch.setattr(search, "Searcher", _status_searcher_stub())

    cmd_status(_args(config=None))  # must not raise SystemExit

    assert "sample config" in capsys.readouterr().err


def test_explicit_config_at_the_sample_is_not_refused(tmp_path, monkeypatch, capsys):
    """Passing the sample with --config is deliberate, and must not be blocked.

    The guard is about the *accidental* fallback. A user who names a path has
    made a decision; refusing them would be the tool overriding a stated intent,
    and would also make the sample unusable for the one legitimate purpose it
    has — inspecting what a config looks like.

    It still fails, but for the *honest* reason: the sample's corpus paths do not
    exist, so the missing-path guard catches it with exit 1 rather than the
    sample guard's exit 2. The distinction matters — exit 2 says "this file is a
    placeholder", exit 1 says "these folders are wrong", and only the second is
    true when the user named the file themselves.
    """
    copy = _copy_sample(tmp_path)
    _fail_services(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cmd_index(_args(config=str(copy)))

    assert exc.value.code == 1, "explicit --config must not trigger the sample guard"
    out = capsys.readouterr()
    assert "sample config" not in out.err
    assert "path does not exist" in out.err


# ---- 3. Missing corpus paths ---------------------------------------------


def test_missing_corpus_path_exits_1(tmp_path, monkeypatch, capsys):
    """corpus=all: every missing path is reported, then exit 1, no indexing."""
    real = tmp_path / "real-docs"
    real.mkdir()
    (real / "a.md").write_text("# A\ntext\n", encoding="utf-8")
    gone_a = tmp_path / "gone-a"
    gone_b = tmp_path / "gone-b"

    config_path = _write_config(
        tmp_path,
        [
            {"name": "real", "path": str(real)},
            {"name": "gone-a", "path": str(gone_a)},
            {"name": "gone-b", "path": str(gone_b)},
        ],
    )
    stub = _stub_searcher(monkeypatch, {"corpora": []})

    with pytest.raises(SystemExit) as exc:
        cmd_index(_args(config=str(config_path)))

    assert exc.value.code == 1
    err = capsys.readouterr().err
    # A path that does not exist is reported as a missing directory, not as a
    # path-shaped corpus name: the error already quotes the name.
    assert "Error: corpus 'gone-a': path does not exist:" in err
    assert "Error: corpus 'gone-b': path does not exist:" in err
    # Every bad path is reported: stopping at the first would make the user
    # rediscover the second only on the next run.
    #
    # Count only the CLI's own message. The indexer also warns
    # "Corpus path does not exist" per corpus from _scan_files, so a bare
    # substring count would reach 4 and say nothing about our reporting.
    assert err.count("Error: corpus 'gone-") == 2
    # The healthy corpus is not blamed.
    assert "'real'" not in err
    # The fix names the config that actually won.
    assert f"Fix: edit paths in {config_path}" in err
    # Refused before doing any work.
    assert stub.calls == []


def test_missing_single_corpus_exits_1(tmp_path, monkeypatch, capsys):
    """A named corpus with a bad path is refused too: exit 1, no indexing."""
    real = tmp_path / "real-docs"
    real.mkdir()
    gone = tmp_path / "gone"
    config_path = _write_config(
        tmp_path,
        [
            {"name": "real", "path": str(real)},
            {"name": "gone", "path": str(gone)},
        ],
    )
    stub = _stub_searcher(monkeypatch, _stats("gone", 0, 0))

    with pytest.raises(SystemExit) as exc:
        cmd_index(_args(config=str(config_path), corpus="gone"))

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Error: corpus 'gone': path does not exist:" in err
    assert f"Fix: edit paths in {config_path}" in err
    assert stub.calls == []


# ---- 4. Empty-but-existing corpus, and incremental no-op -----------------


def test_empty_existing_corpus_warns_exit_0(tmp_path, monkeypatch, capsys):
    """Existing folder, 0 matched: a warning, not a failure.

    The index run is stubbed to the 0/0 stats a real run produces for a folder
    whose files do not match the corpus extensions: nothing is broken, so the
    exit stays 0, but the user must not be told the corpus is indexed.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.xyz").write_text("not an indexed extension\n", encoding="utf-8")
    config_path = _write_config(tmp_path, [{"name": "docs", "path": str(docs)}])
    _stub_searcher(monkeypatch, _stats("docs", 0, 0))

    cmd_index(_args(config=str(config_path)))  # no SystemExit

    err = capsys.readouterr().err
    assert "0 files matched" in err
    assert "'docs'" in err


def test_incremental_no_change_exits_0(tmp_path, monkeypatch, capsys):
    """Nothing changed (files_skipped > 0) is success, and silent.

    Stubbed rather than a real second index: the behaviour under test is how
    the CLI reads the stats, and a real run would need fastembed and write an
    index to disk for no extra coverage. ``_stats`` is the exact shape
    CorpusIndex._do_index returns.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n", encoding="utf-8")
    config_path = _write_config(tmp_path, [{"name": "docs", "path": str(docs)}])
    _stub_searcher(monkeypatch, _stats("docs", 0, 1))

    cmd_index(_args(config=str(config_path)))  # no SystemExit

    assert "0 files matched" not in capsys.readouterr().err


def test_warning_is_per_corpus_not_per_run(tmp_path, monkeypatch, capsys):
    """In a corpus=all result only the empty corpus warns.

    A guard reading aggregate stats would blame every corpus for one empty one.
    """
    docs_a = tmp_path / "a"
    docs_b = tmp_path / "b"
    docs_a.mkdir()
    docs_b.mkdir()
    config_path = _write_config(
        tmp_path,
        [{"name": "a", "path": str(docs_a)}, {"name": "b", "path": str(docs_b)}],
    )
    _stub_searcher(monkeypatch, {"corpora": [_stats("a", 2, 0, 5), _stats("b", 0, 0)]})

    cmd_index(_args(config=str(config_path)))

    err = capsys.readouterr().err
    assert "'b'" in err
    assert "'a'" not in err
