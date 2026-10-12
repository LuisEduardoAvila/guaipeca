"""Tests for `guaipeca search` output handling across return/section modes.

Regression guard: `return_mode=documents` / `auto`->documents and
`section_mode` produce result dicts with a different shape than chunk results.
`cmd_search` must handle all of them instead of assuming chunk keys.
"""

from __future__ import annotations

from guaipeca import cli
from guaipeca.cli import build_parser


class FakeSearcher:
    def __init__(self, result: dict):
        self._result = result

    def search(self, *args, **kwargs) -> dict:
        return self._result


def _run(monkeypatch, capsys, result: dict, extra_argv=None) -> str:
    monkeypatch.setattr(cli, "_load_config", lambda *a, **k: object())
    monkeypatch.setattr(
        cli, "_create_services", lambda cfg: (None, None, FakeSearcher(result))
    )
    args = build_parser().parse_args(["search", "q"] + (extra_argv or []))
    cli.cmd_search(args)
    return capsys.readouterr().out


def test_chunks_mode_prints_chunk_id(monkeypatch, capsys):
    result = {
        "query": "q", "total": 1, "hybrid": False, "return_mode": "chunks",
        "results": [{
            "corpus": "c", "summary": "s", "location": "/x#h",
            "topic": "t", "score": 0.5, "chunk_id": "c:abc",
        }],
    }
    out = _run(monkeypatch, capsys, result)
    assert "Chunk ID: c:abc" in out


def test_documents_mode_prints_path_and_text(monkeypatch, capsys):
    result = {
        "query": "q", "total": 1, "hybrid": False, "return_mode": "documents",
        "results": [{
            "corpus": "c", "source_path": "/x/doc.pdf", "filename": "doc.pdf",
            "topic": "t", "score": 0.7, "text": "FULL DOC TEXT",
            "char_count": 13, "matched_chunks": 2,
        }],
    }
    out = _run(monkeypatch, capsys, result)
    assert "Return mode: documents" in out
    assert "Path: /x/doc.pdf" in out
    assert "FULL DOC TEXT" in out
    assert "Chunk ID:" not in out


def test_section_mode_prints_section(monkeypatch, capsys):
    result = {
        "query": "q", "total": 1, "hybrid": False, "section_mode": True,
        "results": [{
            "corpus": "c", "section_id": "s1", "heading": "Chapter 1",
            "heading_path": ["Chapter 1"], "text": "SECTION TEXT",
            "chunk_count": 3, "location": "/x#Chapter 1", "score": 0.9,
        }],
    }
    out = _run(monkeypatch, capsys, result)
    assert "Mode: sections" in out
    assert "Chapter 1" in out
    assert "SECTION TEXT" in out
    assert "Chunk ID:" not in out


def test_section_filter_prints_filter(monkeypatch, capsys):
    result = {
        "query": "q", "total": 1, "hybrid": False, "section_filter": "Chapter 3",
        "results": [{
            "corpus": "c", "summary": "s", "location": "/x#h",
            "topic": "t", "score": 0.5, "chunk_id": "c:abc",
        }],
    }
    out = _run(monkeypatch, capsys, result, ["--section-filter", "Chapter 3"])
    assert "Section filter: Chapter 3" in out
    assert "Chunk ID: c:abc" in out
