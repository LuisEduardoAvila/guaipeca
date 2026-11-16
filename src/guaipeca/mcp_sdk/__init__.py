"""Official MCP SDK bridge for Guaipeca.

Phase 2 of the SDK migration: exposes the existing tool registry
(``guaipeca.mcp_server``) through the ``mcp`` package without changing any
transport. Phase 4 adds :mod:`guaipeca.mcp_sdk.http`, the Streamable HTTP
cutover served under uvicorn.
"""
