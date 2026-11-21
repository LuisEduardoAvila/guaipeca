"""CLI ``--json`` output for the ``search``, ``status`` and ``index`` commands.

Issue #42: the three most-scripted read/index commands only emitted human text,
so a caller piping ``guaipeca search`` into ``jq`` got prose, not JSON. This adds
``--json`` to each and pins the contract a script can rely on:

* ``--json`` writes exactly one JSON document to stdout (``json.loads`` accepts
  it, and nothing follows it) - the same payload the MCP tool would return;
* an error payload exits 1, writes to stderr, and leaves stdout EMPTY so a
  ``jq`` pipe never sees a half-JSON document;
* without ``--json`` the human output is unchanged.

No embedding model and no FAISS index: the searcher and the service factory are
monkeypatched the same way ``tests/test_install_guards.py`` and
``tests/test_status_output_parity.py`` do, so nothing here loads a model or
touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from guaipeca import cli
from guaipeca.cli import build_parser

# ---- Canned payloads -------------------------------------------------------
# The exact shapes the real Searcher returns, so the JSON the CLI prints is the
# same document a script would get from the MCP tool.

SEARCH_OK = {
    "query": "q",
    "total": 1,
    "hybrid": False,
    "return_mode": "chunks",
    "results": [
        {
            "corpus": "c",
            "summary": "s",
            "location": "/x#h",
            "topic": "t",
            "score": 0.5,
            "chunk_id": "c:abc",
        }
    ],
}

STATUS_OK = {
    "corpora": [
        {
            "corpus": "alpha",
            "total_chunks": 10,
            "total_vectors": 10,
            "files_tracked": 2,
            "bm25_available": True,
            "max_distance": None,
        }
    ],
    "embedding_model": "all-MiniLM-L6-v2",
    "dimensions": 384,
    "hybrid_enabled": True,
    "max_distance": None,
}

INDEX_OK = {
    "corpus": "docs",
    "files_indexed": 2,
    "chunks_created": 5,
    "files_skipped": 0,
    "errors": [],
    "duration_ms": 1,
}


# ---- Stubs -----------------------------------------------------------------


class _FakeSearcher:
    """Stands in for Searcher on the ``search`` path.

    Returns *ok* for a normal query, and mimics the real ``top_k < 1`` guard so
    the error case is exercised the way a real ``--top-k 0`` would trigger it.
    """

    def __init__(self, ok: dict):
        self._ok = ok

    def search(self, query, top_k=5, **kwargs):
        if top_k < 1:
            return {
                "results": [],
                "total": 0,
                "query": query,
                "error": "top_k must be >= 1",
            }
        return self._ok


class _FakeIndexSearcher:
    """Stands in for Searcher on the ``index`` path.

    Mirrors ``Searcher.index_corpus``: ``"all"`` aggregates, a known corpus
    returns its stats, and an unknown corpus returns the ``corpus not found``
    error - the case the error test drives.
    """

    def __init__(self, known: set[str], ok: dict):
        self._known = known
        self._ok = ok

    def index_corpus(self, corpus_name="all", force=False):
        if corpus_name == "all":
            return {"corpora": [self._ok]}
        if corpus_name not in self._known:
            return {"error": f"corpus not found: {corpus_name}"}
        return self._ok


def _write_config(tmp_path: Path, corpora: list[dict]) -> Path:
    """Write a minimal config file pointing at *corpora* (real dirs)."""
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


def _assert_json_only(stdout: str, expected: dict) -> None:
    """stdout is exactly one JSON document: parses, and nothing follows it."""
    parsed = json.loads(stdout)
    assert parsed == expected
    # No human prose before or after the document: re-serialising the parsed
    # value must reproduce stdout byte-for-byte (modulo the trailing newline).
    assert stdout.strip() == json.dumps(parsed, indent=2, ensure_ascii=False)


# ---- search ----------------------------------------------------------------


def _run_search(monkeypatch, capsys, ok: dict, extra_argv=None):
    monkeypatch.setattr(cli, "_load_config", lambda *a, **k: object())
    monkeypatch.setattr(
        cli, "_create_services", lambda cfg: (None, None, _FakeSearcher(ok))
    )
    args = build_parser().parse_args(["search", "q"] + (extra_argv or []))
    cli.cmd_search(args)
    return capsys.readouterr()


def test_search_json_is_valid_json(monkeypatch, capsys):
    out = _run_search(monkeypatch, capsys, SEARCH_OK, ["--json"])
    _assert_json_only(out.out, SEARCH_OK)
    # Human markers must not leak into the JSON stream.
    assert "Query:" not in out.out
    assert "Results:" not in out.out


def test_search_json_error_exits_1_empty_stdout(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_search(monkeypatch, capsys, SEARCH_OK, ["--json", "--top-k", "0"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == "", "an error payload must leave stdout empty for the pipe"
    assert "Error:" in captured.err
    assert "top_k must be >= 1" in captured.err


def test_search_without_json_is_human(monkeypatch, capsys):
    out = _run_search(monkeypatch, capsys, SEARCH_OK)
    assert "Query: q" in out.out
    assert "Results: 1" in out.out
    assert "Chunk ID: c:abc" in out.out


# ---- status ----------------------------------------------------------------


def _run_status(monkeypatch, capsys, status: dict, extra_argv=None):
    import guaipeca.embedding as embedding_module
    import guaipeca.search as search_module
    from guaipeca.config import GuaipecaConfig

    class _StubEmbedder:
        def __init__(self, *a, **k):
            pass

    class _StubSearcher:
        def __init__(self, *a, **k):
            self.status = status

    # cmd_status reads config.embedding.* to build the (stubbed) embedder, so
    # the config must be a real GuaipecaConfig, not a bare object.
    monkeypatch.setattr(cli, "_load_config", lambda *a, **k: GuaipecaConfig())
    monkeypatch.setattr(embedding_module, "EmbeddingService", _StubEmbedder)
    monkeypatch.setattr(search_module, "Searcher", _StubSearcher)
    args = build_parser().parse_args(["status"] + (extra_argv or []))
    cli.cmd_status(args)
    return capsys.readouterr()


def test_status_json_is_valid_json(monkeypatch, capsys):
    out = _run_status(monkeypatch, capsys, STATUS_OK, ["--json"])
    _assert_json_only(out.out, STATUS_OK)
    assert "Guaipeca Status" not in out.out


def test_status_json_error_exits_1_empty_stdout(monkeypatch, capsys):
    # A status payload carrying an error must behave like the other commands:
    # exit 1, stderr, empty stdout.
    bad = {"error": "index unreadable"}
    with pytest.raises(SystemExit) as exc:
        _run_status(monkeypatch, capsys, bad, ["--json"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err


def test_status_without_json_is_human(monkeypatch, capsys):
    out = _run_status(monkeypatch, capsys, STATUS_OK)
    assert "Guaipeca Status" in out.out
    assert "Total chunks: 10" in out.out


# ---- index -----------------------------------------------------------------


def _run_index(monkeypatch, capsys, tmp_path, corpus="all", extra_argv=None):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n", encoding="utf-8")
    config_path = _write_config(tmp_path, [{"name": "docs", "path": str(docs)}])

    monkeypatch.setattr(
        cli,
        "_create_services",
        lambda cfg: (None, None, _FakeIndexSearcher({"docs"}, INDEX_OK)),
    )
    # --config is a top-level flag and must precede the subcommand; --json and
    # friends are index-subcommand flags and follow it.
    argv = ["--config", str(config_path), "index", corpus] + (extra_argv or [])
    args = build_parser().parse_args(argv)
    cli.cmd_index(args)
    return capsys.readouterr()


def test_index_json_is_valid_json(monkeypatch, capsys, tmp_path):
    out = _run_index(monkeypatch, capsys, tmp_path, corpus="all", extra_argv=["--json"])
    _assert_json_only(out.out, {"corpora": [INDEX_OK]})
    assert "Files indexed:" not in out.out


def test_index_json_single_corpus_is_valid_json(monkeypatch, capsys, tmp_path):
    out = _run_index(monkeypatch, capsys, tmp_path, corpus="docs", extra_argv=["--json"])
    _assert_json_only(out.out, INDEX_OK)


def test_index_json_unknown_corpus_exits_1_empty_stdout(monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run_index(
            monkeypatch, capsys, tmp_path, corpus="nope", extra_argv=["--json"]
        )
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == "", "an error payload must leave stdout empty for the pipe"
    assert "Error:" in captured.err
    assert "corpus not found: nope" in captured.err


def test_index_without_json_is_human(monkeypatch, capsys, tmp_path):
    out = _run_index(monkeypatch, capsys, tmp_path, corpus="docs")
    assert "Files indexed: 2" in out.out
    assert "Chunks created: 5" in out.out
