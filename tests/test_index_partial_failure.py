"""A per-corpus index failure must not read as success (#52 review).

`Searcher.index_corpus("all")` returns `{"corpora": [...]}` whose entries can
carry an `error` key — lock contention returns
`{"corpus": ..., "error": "indexing in progress for corpus ..."}`. That is a real
failure mode, not a hypothetical: a systemd timer or a second process holding the
lock produces it routinely.

Neither output mode surfaced it as a failure. `_emit` only inspects the top-level
`error` key, and the human path printed "Error: ..." to stderr and carried on,
both exiting 0. So `guaipeca index --json all` reported success on a run where
one corpus was silently skipped — the same silent-success class as #30.

Contract after this fix: an index result whose corpora carry errors exits 1, with
the errors on stderr. JSON mode still prints nothing to stdout, so a jq pipe
never sees a half-document.
"""

from __future__ import annotations

import pytest
import yaml

from guaipeca import cli


class _Searcher:
    def __init__(self, result):
        self._result = result

    def index_corpus(self, corpus_name="all", force=False):
        return self._result


LOCK_FAILURE = {
    "corpora": [
        {"corpus": "epm-docs", "error": "indexing in progress for corpus epm-docs"},
        {"corpus": "lily", "files_indexed": 0, "chunks_created": 0,
         "files_skipped": 0, "errors": [], "duration_ms": 1},
    ]
}


def _run(monkeypatch, capsys, tmp_path, *, as_json: bool):
    cfg = tmp_path / "guaipeca.yaml"
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "a.md").write_text("# a\n\nhello\n", encoding="utf-8")
    cfg.write_text(
        yaml.safe_dump({"corpora": [{"name": "epm-docs", "path": str(docs)}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "_load_config", lambda _p: __import__(
        "guaipeca.config", fromlist=["GuaipecaConfig"]
    ).GuaipecaConfig.from_yaml(str(cfg)))
    monkeypatch.setattr(cli, "_create_services", lambda _c: (None, None, _Searcher(LOCK_FAILURE)))
    monkeypatch.setattr(cli, "_find_config", lambda _p: str(cfg))

    parser = cli.build_parser()
    argv = ["--config", str(cfg), "index"]
    if as_json:
        argv.append("--json")
    args = parser.parse_args(argv)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_index(args)
    return exc.value.code, capsys.readouterr()


@pytest.mark.parametrize("as_json", [True, False])
def test_per_corpus_error_exits_nonzero(monkeypatch, capsys, tmp_path, as_json):
    """A locked corpus must fail the run in both modes, not just print."""
    code, captured = _run(monkeypatch, capsys, tmp_path, as_json=as_json)
    assert code == 1, f"per-corpus failure exited {code}"
    assert "indexing in progress" in captured.err


def test_json_mode_keeps_stdout_empty_on_per_corpus_error(monkeypatch, capsys, tmp_path):
    """Pipe safety: a failure must not leave a partial JSON document on stdout."""
    _code, captured = _run(monkeypatch, capsys, tmp_path, as_json=True)
    assert captured.out == ""
