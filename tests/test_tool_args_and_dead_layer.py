"""Acceptance tests for #55 (argument validation + error guard) and #58 (dead layer).

Both issues land in one PR because deleting the dead pre-migration JSON-RPC
block removes the last trace of the sanitising intent without fixing the leak.
This file is the gate for both: the #55 tests must pass after the guard commit,
and the #58 tests after the deletion commit.

These tests fail on unfixed code.
"""

from __future__ import annotations

import importlib
import json

import pytest

from guaipeca.mcp_server import GuaipecaMCPServer

# --------------------------------------------------------------------------
# #55.1 — arguments are validated, not passed through
# --------------------------------------------------------------------------

@pytest.fixture
def server(tmp_path, monkeypatch):
    """A server whose searcher is a stub, so no model loads."""
    from guaipeca.config import CorpusConfig, GuaipecaConfig, UploadConfig

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nalpha\n")

    cfg = GuaipecaConfig(
        corpora=[CorpusConfig(name="docs", path=str(corpus), extensions=[".md"])],
        upload=UploadConfig(allow=["docs"], allowed_extensions=[".md"]),
    )
    cfg.indexing.data_dir = str(tmp_path / "idx")
    cfg.embedding.cache_dir = str(tmp_path / "emb")

    srv = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
    srv.config = cfg

    calls = {}

    class _StubSearcher:
        def search(self, query, **kwargs):
            calls["search"] = {"query": query, **kwargs}
            return {"results": [], "total": 0, "query": query}

        def list_documents(self, corpus=None):
            return {"documents": [], "total": 0}

        def index_corpus(self, corpus_name="all", force=False):
            return {"corpus": corpus_name, "indexed": True}

        def status(self, corpus=None):
            return {"corpora": []}

    srv.searcher = _StubSearcher()
    srv._calls = calls
    return srv


def _call(server, name, **arguments):
    return server._handle_tool_call({"name": name, "arguments": arguments})


def test_top_k_as_string_is_coerced_not_crashed(server):
    """`top_k="5"` raised TypeError out of the handler before the guard."""
    result = _call(server, "search", query="alpha", top_k="5")

    assert not isinstance(result, Exception)
    assert server._calls["search"]["top_k"] == 5, "a numeric string should coerce to int"


def test_top_k_non_numeric_is_a_clear_error(server):
    result = _call(server, "search", query="alpha", top_k="abc")

    assert result.get("isError") or "error" in result, (
        f"expected an error result, got {result!r}"
    )
    text = json.dumps(result).lower()
    assert "top_k" in text
    assert "not supported between" not in text, "a Python TypeError leaked to the client"


@pytest.mark.parametrize("value,expected", [(0, 1), (-1, 1), (10**6, 200)])
def test_top_k_is_clamped(server, value, expected):
    _call(server, "search", query="alpha", top_k=value)

    assert server._calls["search"]["top_k"] == expected


def test_bare_string_corpora_is_wrapped(server):
    """`corpora="docs"` was iterated character by character."""
    _call(server, "search", query="alpha", corpora="docs")

    assert server._calls["search"]["corpora"] == ["docs"]


def test_non_string_corpora_entries_are_rejected(server):
    result = _call(server, "search", query="alpha", corpora=["docs", 123])

    assert result.get("isError") or "error" in result
    assert "corpora" in json.dumps(result).lower()


def test_return_mode_is_validated(server):
    result = _call(server, "search", query="alpha", return_mode="foobar")

    assert result.get("isError") or "error" in result
    assert "return_mode" in json.dumps(result).lower()


def test_valid_return_mode_still_works(server):
    _call(server, "search", query="alpha", return_mode="documents")

    assert server._calls["search"]["return_mode"] == "documents"


# --------------------------------------------------------------------------
# #55.2 — an exception from a tool is caught and sanitised
# --------------------------------------------------------------------------

def test_tool_exception_returns_sanitised_error(server):
    """A raised exception must become a result, not escape to the transport."""
    def boom(*a, **k):
        raise RuntimeError("/home/secret/path/to/file.py exploded")

    server.searcher.search = boom

    result = _call(server, "search", query="alpha")

    assert result.get("isError") or "error" in result, (
        f"the exception escaped: {result!r}"
    )
    text = json.dumps(result)
    assert "/home/secret" not in text, f"an absolute path leaked to the client: {text}"
    assert "RuntimeError" not in text or "exploded" not in text


# --------------------------------------------------------------------------
# #58 — the dead pre-migration layer is gone
# --------------------------------------------------------------------------

DEAD_SYMBOLS = [
    "_process_jsonrpc",
    "_sanitize_error_message",
    "handle_request",
]

DEAD_CONSTANTS = [
    "MCP_VERSION",
    "KNOWN_PROTOCOL_VERSIONS",
    "MIN_STREAMABLE_VERSION",
]


@pytest.mark.parametrize("name", DEAD_SYMBOLS)
def test_dead_symbol_is_removed(name):
    mod = importlib.import_module("guaipeca.mcp_server")
    assert not hasattr(mod, name), f"{name} is still defined"
    assert not hasattr(GuaipecaMCPServer, name), f"{name} is still a method"


@pytest.mark.parametrize("name", DEAD_CONSTANTS)
def test_dead_constant_is_removed(name):
    mod = importlib.import_module("guaipeca.mcp_server")
    assert not hasattr(mod, name), f"{name} is still defined"


def test_module_docstring_no_longer_advertises_the_hand_rolled_transport():
    """The header described transports that were removed in the SDK migration."""
    mod = importlib.import_module("guaipeca.mcp_server")
    doc = (mod.__doc__ or "").lower()
    # The hand-rolled layer was newline-delimited JSON-RPC over stdin/stdout.
    # The SDK now owns both transports, so the header must not promise it.
    assert "newline-delimited json-rpc over stdin" not in doc, (
        "the module docstring still describes the removed hand-rolled transport"
    )
    assert "hand-rolled" not in doc


def test_live_dispatch_still_works_after_deletion(server):
    """The SDK's on_call_tool delegates to _handle_tool_call; it must survive."""
    result = _call(server, "search", query="alpha")
    assert "error" not in result or not result.get("isError")
    assert "search" in server._calls
