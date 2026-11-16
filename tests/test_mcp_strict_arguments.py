"""MCP tool-call hardening: unknown arguments and error reporting.

Two defects, both found while debugging a caller whose agent believed it had
scoped a search to one corpus and had not:

1. ``_handle_tool_call`` read arguments with ``arguments.get(...)`` and never
   validated their names. A caller passing ``corpus`` (singular) instead of
   ``corpora`` had the key silently dropped; ``search`` then treated a missing
   ``corpora`` as "search everything". Verified against a live deployment: the
   bogus argument returned hits from two corpora instead of one, with no error.

   Silent scope widening is the worst failure shape here because the answer
   looks valid.

2. ``search()`` reports validation failures as ``{"results": [], "error": ...}``.
   The MCP layer wrapped that in a normal result, so a wrong corpus name was
   indistinguishable from a legitimate zero-hit search over the wire.
"""

from __future__ import annotations

import json

import pytest

from guaipeca.mcp_server import TOOLS, GuaipecaMCPServer


def _server(searcher=None):
    server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
    server.config = None
    server.searcher = searcher
    return server


class _RecordingSearcher:
    """Records the corpora it was asked to search; never loads a model."""

    def __init__(self, result=None):
        self.calls = []
        self._result = result or {"results": [], "total": 0, "query": "q", "return_mode": "chunks"}

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self._result


class TestUnknownArgumentsRejected:
    def test_singular_corpus_is_rejected(self):
        """The reported bug: ``corpus`` instead of ``corpora`` must not be ignored."""
        result = GuaipecaMCPServer._reject_unknown_arguments(
            "search", {"query": "x", "corpus": "sam"}
        )
        assert result is not None
        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "'corpus'" in text
        assert "corpora" in text  # suggests the intended name

    def test_rejected_before_searching(self):
        """A rejected call must not reach the searcher at all."""
        searcher = _RecordingSearcher()
        server = _server(searcher)

        result = server._handle_tool_call(
            {"name": "search", "arguments": {"query": "x", "corpus": "sam"}}
        )

        assert result["isError"] is True
        assert searcher.calls == []

    def test_camel_case_key_is_rejected(self):
        result = GuaipecaMCPServer._reject_unknown_arguments(
            "get_chunk", {"chunkId": "sam:abc"}
        )
        assert result["isError"] is True
        assert "chunk_id" in result["content"][0]["text"]

    def test_empty_schema_rejects_any_argument(self):
        result = GuaipecaMCPServer._reject_unknown_arguments("status", {"foo": 1})
        assert result["isError"] is True

    @pytest.mark.parametrize(
        "tool,args",
        [
            ("search", {"query": "x"}),
            ("search", {"query": "x", "corpora": ["a"], "top_k": 3}),
            ("status", {}),
            ("list_folders", {}),
            ("get_chunk", {"chunk_id": "a:b"}),
            ("get_document", {"corpus": "c", "source_path": "/p"}),
            ("index", {"corpus": "all", "force": True}),
            ("delete", {"corpus": "c", "filename": "f"}),
            ("upload", {"corpus": "c", "filename": "f", "content": "eA=="}),
            ("get_toc", {"corpus": "c"}),
        ],
    )
    def test_declared_arguments_pass(self, tool, args):
        assert GuaipecaMCPServer._reject_unknown_arguments(tool, args) is None

    def test_every_tool_is_reachable_by_its_own_schema(self):
        """Every declared property name must be accepted for its own tool."""
        for tool in TOOLS:
            props = (tool.get("inputSchema") or {}).get("properties", {})
            args = {name: None for name in props}
            assert GuaipecaMCPServer._reject_unknown_arguments(tool["name"], args) is None


class TestSearchErrorsAreMCPErrors:
    def test_unknown_corpus_sets_is_error(self):
        """A wrong corpus name must not look like a valid empty search."""
        searcher = _RecordingSearcher(
            result={"results": [], "total": 0, "query": "q", "error": "corpus not found: nope"}
        )
        server = _server(searcher)

        result = server._handle_tool_call(
            {"name": "search", "arguments": {"query": "q", "corpora": ["nope"]}}
        )

        assert result.get("isError") is True
        assert "corpus not found: nope" in result["content"][0]["text"]

    def test_successful_search_is_not_flagged(self):
        searcher = _RecordingSearcher(
            result={"results": [{"chunk_id": "a:b"}], "total": 1, "query": "q"}
        )
        server = _server(searcher)

        result = server._handle_tool_call(
            {"name": "search", "arguments": {"query": "q", "corpora": ["sam"]}}
        )

        assert not result.get("isError")
        assert json.loads(result["content"][0]["text"])["total"] == 1

    def test_correct_arguments_reach_searcher(self):
        searcher = _RecordingSearcher()
        server = _server(searcher)

        server._handle_tool_call(
            {"name": "search", "arguments": {"query": "q", "corpora": ["sam"], "top_k": 2}}
        )

        assert searcher.calls[0]["corpora"] == ["sam"]
        assert searcher.calls[0]["top_k"] == 2
