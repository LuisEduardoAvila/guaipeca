"""SDK tool bridge: expose Guaipeca's tool registry through the ``mcp`` SDK.

This module does not change any transport. It proves the ten existing tools work
when driven by the official SDK's low-level ``Server``:

- ``on_list_tools`` maps the existing ``TOOLS`` dicts (post
  ``_tools_with_corpus_enum()``) to ``types.Tool`` with ``inputSchema`` passed
  through verbatim.
- ``on_call_tool`` delegates to the existing, synchronous ``_handle_tool_call``
  on a worker thread (``anyio.to_thread.run_sync``) so a blocking search does
  not stall the event loop.

``_handle_tool_call`` keeps ownership of dispatch, argument validation and
result shaping; this is a bridge, not a reimplementation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anyio
from mcp import types
from mcp.server.lowlevel import Server

from guaipeca import __version__

if TYPE_CHECKING:
    from guaipeca.config import GuaipecaConfig
    from guaipeca.mcp_server import GuaipecaMCPServer

__all__ = ["build_server", "run_stdio"]


def _tool_to_sdk(tool: dict[str, Any]) -> types.Tool:
    """Convert one raw tool dict to ``types.Tool``.

    ``inputSchema`` is passed through exactly as produced by
    ``_tools_with_corpus_enum()`` — the golden-file test depends on it.
    """
    return types.Tool(
        name=tool["name"],
        description=tool.get("description"),
        inputSchema=tool.get("inputSchema") or {"type": "object", "properties": {}},
    )


def _result_to_call_tool_result(result: Any) -> types.CallToolResult:
    """Convert ``_handle_tool_call``'s MCP-shaped dict to ``CallToolResult``.

    Anything that does not look like the expected dict is reported as a tool
    error containing the raw JSON, so a bridge bug is visible to the client
    instead of crashing the session.
    """
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        try:
            raw = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        except Exception:
            raw = repr(result)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=raw)],
            isError=True,
        )

    content: list[types.TextContent] = []
    for item in result["content"]:
        if isinstance(item, dict) and item.get("type") == "text":
            content.append(types.TextContent(type="text", text=str(item.get("text", ""))))
        else:
            content.append(
                types.TextContent(
                    type="text",
                    text=json.dumps(item, ensure_ascii=False, default=str),
                )
            )

    return types.CallToolResult(content=content, isError=bool(result.get("isError", False)))


def build_server(config: GuaipecaConfig, server_instance: GuaipecaMCPServer) -> Server:
    """Build an SDK ``Server`` whose tools are Guaipeca's existing registry."""

    async def on_list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[_tool_to_sdk(tool) for tool in server_instance._tools_with_corpus_enum()]
        )

    async def on_call_tool(_ctx: Any, params: Any) -> types.CallToolResult:
        call_params = {
            "name": params.name,
            "arguments": params.arguments or {},
        }
        result = await anyio.to_thread.run_sync(
            server_instance._handle_tool_call,
            call_params,
        )
        return _result_to_call_tool_result(result)

    return Server(
        "guaipeca",
        version=__version__,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def run_stdio(config: GuaipecaConfig, server_instance: GuaipecaMCPServer) -> None:
    """Serve ``server_instance`` over the SDK's stdio transport until EOF."""
    from mcp.server.stdio import stdio_server

    server = build_server(config, server_instance)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
