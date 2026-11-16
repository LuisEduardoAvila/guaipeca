"""MCP tool descriptions must state the contract a weak model needs.

Reported failure: an agent called ``search`` with ``corpus`` (singular) for a
corpus that did not exist. Two things made that likely, and neither was the
model being stupid:

* ``search`` takes ``corpora`` (array) while six other tools take ``corpus``
  (string), and no description said so.
* ``corpora`` had no ``enum`` and no pointer to ``status``, so a caller with
  no knowledge of the deployment had to invent a name.

These tests pin the wording that fixes both, plus the runtime enum.
"""

from __future__ import annotations

import json

from guaipeca.mcp_server import TOOLS, GuaipecaMCPServer


def _tool(name: str) -> dict:
    return next(t for t in TOOLS if t["name"] == name)


def _server(corpora=("alpha", "beta")):
    class FakeSearcher:
        pass

    searcher = FakeSearcher()
    searcher.corpora = {name: None for name in corpora}

    server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
    server.config = None
    server.searcher = searcher
    return server


class TestSearchDescription:
    def test_states_corpora_is_a_list(self):
        text = _tool("search")["description"].lower()
        assert "corpora=" in text or "corpora=[" in text
        assert "list" in text

    def test_warns_omitting_searches_everything(self):
        """The actual footgun: no corpora means all corpora."""
        text = _tool("search")["description"].lower()
        assert "every corpus" in text

    def test_points_at_status_for_names(self):
        assert "status" in _tool("search")["description"].lower()

    def test_corpora_param_is_named_plural(self):
        desc = _tool("search")["inputSchema"]["properties"]["corpora"]["description"]
        assert "list" in desc.lower()
        assert "singular" in desc.lower()

    def test_hybrid_does_not_claim_default_false(self):
        """Config may enable hybrid; a hard default:false in the schema lies."""
        prop = _tool("search")["inputSchema"]["properties"]["hybrid"]
        assert prop.get("default") is not False
        assert "config" in prop["description"].lower()

    def test_section_filter_documents_section_mode_dependency(self):
        desc = _tool("search")["inputSchema"]["properties"]["section_filter"]["description"]
        assert "section_mode" in desc

    def test_status_describes_itself_as_discovery(self):
        assert "corpus names" in _tool("status")["description"].lower()


class TestSingularCorpusTools:
    def test_every_singular_tool_says_not_a_list(self):
        holders = [
            t for t in TOOLS if "corpus" in (t.get("inputSchema") or {}).get("properties", {})
        ]
        assert holders, "expected tools with a corpus parameter"
        for tool in holders:
            desc = tool["inputSchema"]["properties"]["corpus"]["description"].lower()
            assert "not a list" in desc, f"{tool['name']} does not warn it is singular"


class TestCorpusEnum:
    def test_search_advertises_live_corpus_names(self):
        tools = _server(["epm-docs", "sam"])._tools_with_corpus_enum()
        search = next(t for t in tools if t["name"] == "search")
        assert search["inputSchema"]["properties"]["corpora"]["items"]["enum"] == [
            "epm-docs",
            "sam",
        ]

    def test_singular_tools_advertise_names(self):
        tools = _server(["a", "b"])._tools_with_corpus_enum()
        for name in ("get_document", "list_documents", "get_toc"):
            tool = next(t for t in tools if t["name"] == name)
            assert tool["inputSchema"]["properties"]["corpus"]["enum"] == ["a", "b"]

    def test_index_keeps_all_sentinel(self):
        tools = _server(["a"])._tools_with_corpus_enum()
        index = next(t for t in tools if t["name"] == "index")
        assert index["inputSchema"]["properties"]["corpus"]["enum"] == ["a", "all"]

    def test_global_tools_are_unchanged(self):
        """TOOLS is module-level and shared; enum injection must not mutate it."""
        _server(["x"])._tools_with_corpus_enum()
        assert "enum" not in json.dumps(TOOLS)

    def test_falls_back_when_corpora_unavailable(self):
        server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
        server.config = None
        server.searcher = None
        assert server._tools_with_corpus_enum() == TOOLS
