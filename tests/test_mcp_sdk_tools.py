"""Tests for the SDK tool bridge (``guaipeca.mcp_sdk.tools``).

Phase 2 of the MCP SDK migration: the ten existing tools must be exposed
through the official SDK, with schemas identical to today's server, and with
the synchronous tool dispatch kept off the event loop.

No embedding model is loaded here — a stub server instance supplies
``_tools_with_corpus_enum`` and ``_handle_tool_call`` (same idea as
``MockMCPServer`` in ``test_streamable_http.py``).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from mcp import types

from guaipeca import __version__
from guaipeca.mcp_sdk.tools import build_server
from guaipeca.mcp_server import TOOLS

PROJECT_ROOT = Path(__file__).parent.parent
GOLDEN_PATH = PROJECT_ROOT / "tests" / "fixtures" / "tools_list_golden.json"


# ---- Stub server instance ---------------------------------------------------


class StubMCPServer:
    """Stands in for ``GuaipecaMCPServer`` without the model or searcher."""

    def __init__(self, call_result: Any = None, call_delay: float = 0.0):
        self.call_result = call_result
        self.call_delay = call_delay
        self.calls: list[dict] = []

    def _tools_with_corpus_enum(self) -> list[dict]:
        # No corpora known: falls back to TOOLS verbatim, which is exactly the
        # shape the golden file was generated from.
        return [dict(tool) for tool in TOOLS]

    def _handle_tool_call(self, params: dict) -> Any:
        self.calls.append(params)
        if self.call_delay:
            time.sleep(self.call_delay)
        return self.call_result


def _get_handler(server: Any, name: str) -> Any:
    """Pull a request handler off the SDK server without depending on storage.

    ``Server._request_handlers`` maps method -> ``HandlerEntry``, which wraps
    the callable in ``.handler``. Unwrap it, but tolerate a bare callable in
    case the SDK's storage changes.
    """
    for attr in ("request_handlers", "_request_handlers"):
        handlers = getattr(server, attr, None)
        if handlers:
            for key, handler in handlers.items():
                key_name = getattr(key, "value", key)
                if str(key_name).lower() == name.lower():
                    return getattr(handler, "handler", handler)
    raise AssertionError(f"No {name!r} handler registered on the SDK server")


def _list_tools_payload(server: Any) -> dict:
    """Serialize the SDK's tools/list to the wire shape clients actually receive.

    `model_dump_json(by_alias=True)` matters: without it pydantic emits
    snake_case (input_schema) and the comparison against the golden file would
    silently be comparing different key names.
    """
    handler = _get_handler(server, "tools/list")
    result = asyncio.run(handler(None, None))
    return json.loads(result.model_dump_json(exclude_none=True, by_alias=True))


def _call_tool(server: Any, name: str, arguments: dict | None = None) -> Any:
    handler = _get_handler(server, "tools/call")
    params = types.CallToolRequestParams(name=name, arguments=arguments or {})
    return asyncio.run(handler(None, params))


# ---- 1. Golden parity -------------------------------------------------------


def test_sdk_tool_list_matches_golden():
    """The SDK's tools/list must be identical to today's server output."""
    server = build_server(config=None, server_instance=StubMCPServer())
    actual = _list_tools_payload(server)

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    actual_by_name = {t["name"]: t for t in actual["tools"]}
    golden_by_name = {t["name"]: t for t in golden["tools"]}

    assert set(actual_by_name) == set(golden_by_name), (
        f"tool names differ: only in SDK={sorted(set(actual_by_name) - set(golden_by_name))}, "
        f"only in golden={sorted(set(golden_by_name) - set(actual_by_name))}"
    )

    problems = []
    for name, expected in golden_by_name.items():
        got = actual_by_name[name]
        if got.get("description") != expected.get("description"):
            problems.append(
                f"{name}: description differs\n"
                f"expected: {expected.get('description')!r}\n"
                f"     got: {got.get('description')!r}"
            )
        if got.get("inputSchema") != expected.get("inputSchema"):
            problems.append(
                f"{name}: inputSchema differs\nexpected: "
                f"{json.dumps(expected.get('inputSchema'), indent=2, sort_keys=True)}\n     got: "
                f"{json.dumps(got.get('inputSchema'), indent=2, sort_keys=True)}"
            )

    assert not problems, "SDK tool list drifted from golden:\n\n" + "\n\n".join(problems)


def test_sdk_server_metadata():
    server = build_server(config=None, server_instance=StubMCPServer())
    assert server.name == "guaipeca"
    assert server.version == __version__


# ---- 2. Call bridge ---------------------------------------------------------


def test_call_tool_bridges_result_text():
    stub = StubMCPServer(
        call_result={"content": [{"type": "text", "text": '{"results": ["a"]}'}]}
    )
    server = build_server(config=None, server_instance=stub)

    result = _call_tool(server, "search", {"query": "rag"})

    assert isinstance(result, types.CallToolResult)
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == '{"results": ["a"]}'
    assert result.is_error is False
    # The unchanged dispatcher received an MCP-shaped params dict.
    assert stub.calls == [{"name": "search", "arguments": {"query": "rag"}}]


def test_call_tool_unexpected_return_is_tool_error():
    stub = StubMCPServer(call_result="not a dict")
    server = build_server(config=None, server_instance=stub)

    result = _call_tool(server, "search", {"query": "rag"})

    assert isinstance(result, types.CallToolResult)
    assert result.is_error is True
    assert result.content[0].text == '"not a dict"'


# ---- 3. Sync-to-thread ------------------------------------------------------


def test_call_tool_does_not_block_event_loop():
    """A blocking sync dispatch must not stall other async work."""
    stub = StubMCPServer(
        call_result={"content": [{"type": "text", "text": "ok"}]},
        call_delay=0.5,
    )
    server = build_server(config=None, server_instance=stub)
    handler = _get_handler(server, "tools/call")

    async def scenario() -> tuple[str, str]:
        order: list[str] = []

        async def sleeper() -> None:
            await asyncio.sleep(0.05)
            order.append("sleeper")

        call_task = asyncio.create_task(
            handler(None, types.CallToolRequestParams(name="search", arguments={}))
        )
        await asyncio.gather(asyncio.create_task(sleeper()), call_task)
        order.append("call")
        return tuple(order)  # type: ignore[return-value]

    first, second = asyncio.run(scenario())
    assert (first, second) == ("sleeper", "call"), (
        "sync tool dispatch blocked the event loop; it must run in a worker thread"
    )


# ---- 4. isError pass-through ------------------------------------------------


def test_call_tool_preserves_is_error():
    stub = StubMCPServer(
        call_result={"content": [{"type": "text", "text": "Chunk not found: xyz"}], "isError": True}
    )
    server = build_server(config=None, server_instance=stub)

    result = _call_tool(server, "get_chunk", {"chunk_id": "xyz"})

    assert result.is_error is True
    assert result.content[0].text == "Chunk not found: xyz"


@pytest.mark.parametrize("tool", TOOLS, ids=[t["name"] for t in TOOLS])
def test_every_golden_tool_has_required_keys(tool):
    """Belt-and-braces: each tool still carries name/description/inputSchema."""
    assert tool["name"] and tool.get("description") and tool.get("inputSchema")
