"""Tests for the SDK-backed HTTP transport (``guaipeca.mcp_sdk.http``).

Covers the Phase 4 cutover end to end through Starlette's ``TestClient``: the
real app, real SDK server, real routing — only the ``GuaipecaMCPServer``
instance is a stub, so no embedding model is loaded.

``base_url`` is a loopback address on purpose: the SDK's transport-security
middleware validates the Host header, and the default ``testserver`` host is
correctly rejected with 421.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

# Ensure project root is on sys.path (same preamble as the other HTTP tests)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from guaipeca import __version__
from guaipeca.config import CorpusConfig, ServerConfig
from guaipeca.mcp_sdk.http import (
    _allowed_host_patterns,
    _allowed_origin_patterns,
    _check_bearer,
    _HmacTokenVerifier,
    build_app,
)
from guaipeca.mcp_server import TOOLS

MODERN_VERSION = "2026-07-28"
HANDSHAKE_VERSION = "2025-11-25"
EVIL_ORIGIN = "https://evil.example"


# ---- Stub server instance ---------------------------------------------------


class StubMCPServer:
    """Stands in for ``GuaipecaMCPServer`` without the model or searcher."""

    def __init__(self, auth_token: str | None = None, corpus_dir: str | None = None):
        server_config = ServerConfig()
        server_config.auth_token = auth_token
        config = type("Config", (), {})()
        config.server = server_config
        config.corpora = [CorpusConfig(name="docs", path=corpus_dir)] if corpus_dir else []
        config.get_corpus = lambda name: next(
            (c for c in config.corpora if c.name == name), None
        )
        self.config = config
        self.deleted: list[tuple[str, str]] = []

    def _tools_with_corpus_enum(self) -> list[dict]:
        return [dict(tool) for tool in TOOLS]

    def _handle_tool_call(self, params: dict) -> dict:
        name = params.get("name")
        if name == "search":
            return {
                "content": [
                    {"type": "text", "text": json.dumps({"result": "mock search result"})}
                ]
            }
        if name == "get_chunk":
            return {"isError": True, "content": [{"type": "text", "text": "Chunk not found"}]}
        return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {name}"}]}

    def _handle_delete(self, corpus_name: str, filename: str, do_index: bool) -> dict:
        self.deleted.append((corpus_name, filename))
        return {"corpus": corpus_name, "filename": filename, "deleted": True, "indexed": False}


# ---- Harness ----------------------------------------------------------------


def _make_client(server: StubMCPServer | None = None, **app_kwargs) -> tuple[TestClient, StubMCPServer]:
    stub = server or StubMCPServer()
    app = build_app(stub.config, stub, **app_kwargs)
    return TestClient(app, base_url="http://127.0.0.1:8090"), stub


@pytest.fixture
def client():
    """App with auth off, lifespan entered for the test's duration."""
    stub = StubMCPServer()
    with _make_client(stub)[0] as test_client:
        yield test_client


@pytest.fixture
def auth_client():
    """App with a configured Bearer token."""
    stub = StubMCPServer(auth_token="test-secret-token")
    with _make_client(stub)[0] as test_client:
        yield test_client


def _initialize(
    client: TestClient, protocol_version: str = HANDSHAKE_VERSION, token: str | None = None
) -> str:
    """Legacy handshake on /mcp; returns the Mcp-Session-Id."""
    headers = {"Accept": "application/json, text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text
    session_id = response.headers.get("mcp-session-id")
    assert session_id, f"no Mcp-Session-Id in {dict(response.headers)}"
    return session_id


# ---- 1. Legacy handshake + tools/list ---------------------------------------


def test_legacy_initialize_then_tools_list(client):
    session_id = _initialize(client)

    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        headers={"Mcp-Session-Id": session_id},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 2
    tools = body["result"]["tools"]
    assert len(tools) == 10
    assert {tool["name"] for tool in tools} == {tool["name"] for tool in TOOLS}
    # The SDK serialises inputSchema under its wire alias, not input_schema.
    assert all("inputSchema" in tool for tool in tools)


def test_legacy_tools_call_runs_the_existing_dispatcher(client):
    session_id = _initialize(client)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "embedding"}},
        },
        headers={"Mcp-Session-Id": session_id},
    )

    assert response.status_code == 200, response.text
    content = response.json()["result"]["content"]
    assert content[0]["type"] == "text"
    assert json.loads(content[0]["text"]) == {"result": "mock search result"}


# ---- 2. Modern (2026-07-28) stateless request -------------------------------


def test_modern_stateless_tools_list(client):
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/list",
            "params": {
                "_meta": {
                    # The modern envelope carries namespaced keys, not bare ones.
                    "io.modelcontextprotocol/protocolVersion": MODERN_VERSION,
                    "io.modelcontextprotocol/clientCapabilities": {},
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "modern-client",
                        "version": "1.0",
                    },
                }
            },
        },
        headers={
            "MCP-Protocol-Version": MODERN_VERSION,
            "mcp-method": "tools/list",
            "Accept": "application/json, text/event-stream",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == 10
    assert len(body["result"]["tools"]) == 10
    # Stateless: no session is minted for the modern era.
    assert "mcp-session-id" not in response.headers


# ---- 3/4. Origin policy -----------------------------------------------------


def test_evil_origin_rejected(client):
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"Origin": EVIL_ORIGIN},
    )
    assert response.status_code == 403


def test_loopback_origin_allowed(client):
    session_id = _initialize(client)
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
        headers={"Origin": "http://localhost:3000", "Mcp-Session-Id": session_id},
    )
    assert response.status_code == 200


def test_no_origin_allowed(client):
    session_id = _initialize(client)
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
        headers={"Mcp-Session-Id": session_id},
    )
    assert response.status_code == 200
    assert response.json()["result"] == {}


def test_origin_patterns_follow_the_existing_policy():
    loopback = _allowed_origin_patterns(None)
    assert "http://localhost:*" in loopback
    assert "http://127.0.0.1:*" in loopback
    assert "https://[::1]:*" in loopback
    # An allowlist replaces the loopback default, exactly as _origin_allowed().
    assert _allowed_origin_patterns(["https://ui.example.com/"]) == ["https://ui.example.com"]
    # Hosts are a different list: the Host header names the address connected to,
    # so loopback hosts appear without a scheme.
    assert "127.0.0.1:*" in _allowed_host_patterns("127.0.0.1", None)
    assert "ui.example.com" in _allowed_host_patterns("127.0.0.1", ["https://ui.example.com"])


# ---- 5. /health -------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "server": "guaipeca", "version": __version__}


def test_health_requires_auth_when_configured(auth_client):
    assert auth_client.get("/health").status_code == 401
    response = auth_client.get("/health", headers={"Authorization": "Bearer test-secret-token"})
    assert response.status_code == 200


# ---- 6. Legacy SSE transport ----------------------------------------------
#
# GET /sse streams, so these tests never read the stream to exhaustion: an SSE
# response body does not end until the server-side session closes, and a full
# read would hang. The handshake test therefore asserts on status and headers
# first, and only then peeks at the first chunk. The chunk assertion is skipped
# if the client cannot hand one over; a weaker deterministic assertion beats a
# flaky stronger one.


def _first_sse_chunk(response) -> str | None:
    """First chunk of a streamed response, or None if streaming is unavailable.

    Reads exactly one chunk: the stream stays open until the session closes, so
    anything that drains it would hang the test.
    """
    chunks = getattr(response, "iter_bytes", None)
    if chunks is None:
        return None
    iterator = chunks()
    try:
        return next(iterator).decode("utf-8", "replace")
    except StopIteration:
        return ""
    finally:
        iterator.close()


def test_sse_handshake_sends_endpoint_event(client):
    """GET /sse opens a stream and announces the POST /messages endpoint."""
    with client.stream(
        "GET", "/sse", headers={"Accept": "text/event-stream"}
    ) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        chunk = _first_sse_chunk(response)

    if chunk is None:  # streaming helper unavailable: headers-only assertion
        return
    # The SDK writes the endpoint event as `event: endpoint` + `data: <url>`.
    assert "event: endpoint" in chunk, chunk
    # Canonical path carries the trailing slash: the SDK routes this as a
    # Mount, so "/messages" would 307 on POST and most SSE clients drop the
    # body rather than follow. See test_sse_endpoint_does_not_redirect_post.
    assert "/messages/?session_id=" in chunk, chunk


def test_sse_requires_auth_when_configured(auth_client):
    """No token is 401 (and a bad token too); the stream never opens."""
    assert auth_client.get("/sse", headers={"Accept": "text/event-stream"}).status_code == 401
    assert (
        auth_client.get(
            "/sse",
            headers={"Accept": "text/event-stream", "Authorization": "Bearer wrong-token"},
        ).status_code
        == 401
    )


def test_post_messages_unknown_session_is_4xx(client):
    """POST /messages with an unknown session_id is a client error, not a 5xx."""
    response = client.post(
        "/messages?session_id=does-not-exist",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
    )
    assert 400 <= response.status_code < 500, response.text


def test_post_messages_without_session_id_is_4xx(client):
    """POST /messages with no session_id is a client error."""
    response = client.post(
        "/messages",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
    )
    assert 400 <= response.status_code < 500, response.text


def test_post_messages_requires_auth_when_configured(auth_client):
    """POST /messages without a token is 401, before the transport sees it."""
    response = auth_client.post(
        "/messages?session_id=abc",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
    )
    assert response.status_code == 401, response.text


# ---- 7. Auth on /mcp --------------------------------------------------------


def test_mcp_without_token_is_401(auth_client):
    response = auth_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
    )
    assert response.status_code == 401


def test_mcp_with_token_succeeds(auth_client):
    session_id = _initialize(auth_client, token="test-secret-token")  # raises on non-200
    response = auth_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
        headers={"Authorization": "Bearer test-secret-token", "Mcp-Session-Id": session_id},
    )
    assert response.status_code == 200


def test_wrong_token_is_401(auth_client):
    response = auth_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


# ---- Non-MCP endpoints: /tools and DELETE /documents ------------------------


def test_tools_endpoint_matches_tools_list(client):
    response = client.get("/tools")
    assert response.status_code == 200
    assert len(response.json()["tools"]) == 10


def test_delete_documents_delegates_to_handler(client):
    response = client.delete("/documents?corpus=docs&filename=a.md")
    assert response.status_code == 200
    assert response.json()["deleted"] is True


def test_delete_documents_requires_parameters(client):
    response = client.delete("/documents?corpus=docs")
    assert response.status_code == 400


def test_delete_documents_requires_auth(auth_client):
    response = auth_client.delete("/documents?corpus=docs&filename=a.md")
    assert response.status_code == 401


# ---- Download route ---------------------------------------------------------


def test_download_serves_file_from_corpus(tmp_path):
    (tmp_path / "doc.md").write_text("hello", encoding="utf-8")
    stub = StubMCPServer(corpus_dir=str(tmp_path))
    with _make_client(stub)[0] as test_client:
        response = test_client.get("/download/docs/doc.md")

    assert response.status_code == 200
    assert response.text == "hello"
    assert response.headers["content-disposition"] == 'attachment; filename="doc.md"'


def test_download_rejects_unknown_corpus(tmp_path):
    stub = StubMCPServer(corpus_dir=str(tmp_path))
    with _make_client(stub)[0] as test_client:
        response = test_client.get("/download/nope/doc.md")
    assert response.status_code == 404
    assert "corpus not found" in response.json()["error"]


# ---- 7. Ported from the retired hand-rolled transport suite -----------------
#
# These assertions lived in tests/test_streamable_http.py, which drove
# GuaipecaMCPServer.run_http(). That transport no longer serves, but the
# behaviour they pin down is still Guaipeca's contract, so they are ported
# here rather than dropped. Each test names its origin.


def test_unknown_session_id_is_404(client):
    """test_streamable_http.py::test_unknown_session_id."""
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Mcp-Session-Id": "does-not-exist"},
    )
    assert response.status_code == 404


def test_delete_session_lifecycle(client):
    """test_streamable_http.py::test_delete_valid_session / test_delete_unknown_session."""
    session_id = _initialize(client)

    gone = client.delete("/mcp", headers={"Mcp-Session-Id": "nope"})
    assert gone.status_code in (404, 400), gone.text

    deleted = client.delete("/mcp", headers={"Mcp-Session-Id": session_id})
    assert deleted.status_code == 200, deleted.text

    # The session is gone; a later request on it must not be served.
    after = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        headers={"Mcp-Session-Id": session_id},
    )
    assert after.status_code == 404


def test_delete_without_session_header_is_400(client):
    """test_streamable_http.py::test_delete_without_session_header."""
    response = client.delete("/mcp")
    assert response.status_code in (400, 405), response.text


def test_get_mcp_is_405(client):
    """test_streamable_http.py::test_get_mcp_returns_405."""
    response = client.get("/mcp")
    assert response.status_code in (405, 400), response.text


def test_batch_request_returns_array(client):
    """test_streamable_http.py::test_batch_two_requests."""
    session_id = _initialize(client)
    response = client.post(
        "/mcp",
        json=[
            {"jsonrpc": "2.0", "id": 10, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 11, "method": "ping"},
        ],
        headers={"Mcp-Session-Id": session_id},
    )
    # The SDK may answer a batch with an array or reject it outright; either is
    # a decision, not a crash. What must never happen is a 5xx or empty body.
    assert response.status_code < 500, response.text
    if response.status_code == 200:
        body = response.json()
        assert isinstance(body, list), f"expected a JSON array, got {type(body).__name__}"


def test_malformed_json_is_not_a_crash(client):
    """test_streamable_http.py::test_malformed_json."""
    response = client.post(
        "/mcp",
        content=b'{"jsonrpc": "2.0", "id": 1, ',
        headers={"Content-Type": "application/json"},
    )
    assert 400 <= response.status_code < 500, response.text
    assert response.status_code != 500


def test_unknown_path_is_404(client):
    """test_streamable_http.py::test_post_unknown_path."""
    response = client.post("/definitely-not-a-route", json={})
    assert response.status_code == 404


def test_oversized_body_is_413(tmp_path):
    """test_streamable_http.py::test_oversized_body / test_body_at_cap_accepted.

    Pins the body cap to config.server.max_body_size rather than a hardcoded
    constant: the SDK enforces whatever we pass to max_request_body_size.
    """
    stub = StubMCPServer()
    stub.config.server.max_body_size = 2048
    with _make_client(stub)[0] as test_client:
        too_big = test_client.post(
            "/mcp",
            content=b"x" * 8192,
            headers={"Content-Type": "application/json"},
        )
        assert too_big.status_code == 413, too_big.text

        small = test_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": HANDSHAKE_VERSION,
                             "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}},
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert small.status_code == 200, small.text


def test_acao_echoes_validated_origin(client):
    """test_origin_validation.py::test_acao_echoes_validated_origin."""
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": HANDSHAKE_VERSION,
                         "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}},
        headers={"Origin": "http://localhost:3000",
                 "Accept": "application/json, text/event-stream"},
    )
    assert response.status_code == 200, response.text
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_sse_endpoint_does_not_redirect_post(client):
    """The advertised message path must accept POST without a redirect.

    Found only by driving a real SSE client: the SDK routes the message path as
    a Starlette ``Mount``, so ``/messages`` 307s to ``/messages/``. Most SSE
    clients do not follow a redirect on POST - the body is dropped and the
    session never reaches the transport - so the handshake appears to hang
    rather than fail. The transport therefore advertises the canonical
    trailing-slash path.

    The POST happens inside the stream context: once the GET closes, the
    session is torn down and any POST is a 404 regardless of its path, which
    would let this test pass without proving anything about redirects.
    """
    with client.stream(
        "GET", "/sse", headers={"Accept": "text/event-stream"}
    ) as response:
        assert response.status_code == 200

        endpoint = None
        lines = response.iter_lines()
        for line in lines:
            if line.startswith("data:") and "/messages" in line:
                endpoint = line[5:].strip()
                break

        assert endpoint, "no endpoint event was sent"
        assert "/messages/?" in endpoint, (
            f"the advertised message path {endpoint!r} is not the canonical "
            "'/messages/' - POSTing to it will redirect and drop the body"
        )

        post = client.post(
            endpoint,
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            follow_redirects=False,
        )
        assert post.status_code != 307, (
            f"POST {endpoint} redirected to {post.headers.get('location')}"
        )
        assert post.status_code == 202, (
            f"expected 202 Accepted from a live session, got {post.status_code}"
        )


# ---- 8. Transport advertisement ---------------------------------------------


def test_tools_endpoint_advertises_preferred_transport(client):
    """/tools must say which transport is preferred, not just list tools."""
    body = client.get("/tools").json()
    assert "tools" in body
    transports = body["transports"]
    assert transports["preferred"] == "streamable-http"
    assert transports["endpoints"]["streamable-http"] == "/mcp"
    # SSE is advertised as still-served, not as the preferred path.
    assert "sse" in transports["endpoints"]
    assert "deprecated" in transports["note"].lower()


def test_initialize_instructions_prefer_streamable_http(client):
    """The MCP-standard channel: a client reads this before it picks a transport."""
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": HANDSHAKE_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            },
        },
        headers={"Accept": "application/json, text/event-stream"},
    )
    assert response.status_code == 200, response.text
    instructions = response.json()["result"].get("instructions") or ""
    assert "/mcp" in instructions
    assert "prefer" in instructions.lower()
    assert "deprecated" in instructions.lower()


def test_sse_stream_is_marked_deprecated(client):
    """A legacy SSE client never reads `instructions`, so the stream says so.

    RFC 8594 Deprecation + RFC 8288 Link. Only the GET stream carries them:
    that is where a client decides to keep the connection.
    """
    with client.stream(
        "GET", "/sse", headers={"Accept": "text/event-stream"}
    ) as response:
        assert response.status_code == 200
        assert response.headers.get("deprecation") == "true", dict(response.headers)
        link = response.headers.get("link", "")
        assert "/mcp" in link, link


# ---- 9. Protocol-version negotiation (the CHANGELOG makes these claims) ----


def test_modern_unsupported_version_gets_32022(client):
    """An unsupported revision on the modern era must be rejected, not downgraded.

    This is the criterion from #22: a 2026-07-28 client asking for a version we
    do not serve should learn which versions exist, so it can retry with one.
    A silent downgrade would leave it speaking semantics neither side agrees on.

    Verified against the running server before being written: status 400 with
    `{"code": -32022, "message": "Unsupported protocol version",
      "data": {"supported": ["2026-07-28"], "requested": "1900-01-01"}}`.
    """
    meta = {
        "io.modelcontextprotocol/protocolVersion": "1900-01-01",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "1"},
    }
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {"_meta": meta}},
        headers={
            "MCP-Protocol-Version": "1900-01-01",
            "mcp-method": "tools/list",
            "Accept": "application/json, text/event-stream",
        },
    )
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == -32022, error
    assert "1900-01-01" in json.dumps(error)
    assert "supported" in error["data"]
    assert MODERN_VERSION in error["data"]["supported"]


def test_legacy_handshake_selects_a_supported_version(client):
    """The legacy handshake is allowed to pick a version — 200, not -32022.

    Pinned deliberately, because the two eras behave differently and someone
    reading only the test above would reasonably think the old path is broken.
    2025-11-25 and earlier revisions have the server choose from what it
    supports; only the modern per-request envelope rejects the request outright.
    """
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "1900-01-01",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            },
        },
        headers={"Accept": "application/json, text/event-stream"},
    )
    assert response.status_code == 200, response.text
    negotiated = response.json()["result"]["protocolVersion"]
    assert negotiated in {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}, negotiated


def test_server_discover_is_reachable_on_the_modern_era(client):
    """`server/discover` is how a modern client learns versions up front.

    The SDK provides it; nothing in guaipeca implemented or tested it before
    this, which left a documented capability unverified.
    """
    meta = {
        "io.modelcontextprotocol/protocolVersion": MODERN_VERSION,
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "1"},
    }
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 8, "method": "server/discover", "params": {"_meta": meta}},
        headers={
            "MCP-Protocol-Version": MODERN_VERSION,
            "mcp-method": "server/discover",
            "Accept": "application/json, text/event-stream",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "error" not in body, body
    result = body["result"]
    # Wire shape is camelCase (pydantic's alias), not the snake_case attribute
    # name the SDK model exposes. Comparing the attribute name here would look
    # for a key the server never sends and fail for the wrong reason.
    assert MODERN_VERSION in result.get("supportedVersions", []), result


# ---- 10. Non-ASCII Bearer tokens must fail closed, not crash ----------------
#
# A Bearer token containing non-ASCII characters made ``hmac.compare_digest`` raise
# ``TypeError`` (it only accepts ASCII str/bytes), which turned a plain 401 into a
# 500 on every request. This is a fail-closed bug: an invalid/malformed token must
# be rejected, not crash the server. The comparison must stay constant-time, so the
# guard must *protect* the ``hmac.compare_digest`` call rather than swapping it for
# ``==``. These two tests pin that contract on both code paths.


async def _run_asgi(app: Any, scope: dict) -> tuple[int, bytes]:
    """Drive an ASGI callable directly, capturing status + body.

    Bypasses httpx's *client-side* ASCII header enforcement (the Starlette
    ``TestClient`` refuses to encode a non-ASCII header value as ASCII before the
    request ever leaves the client). Real HTTP sends the Authorization header as
    raw UTF-8 bytes; Starlette decodes scope headers as latin-1, so a non-ASCII
    token reaches ``_check_bearer`` as a non-ASCII str. Driving the app at the
    ASGI level reproduces that real path faithfully.
    """
    captured: dict[str, Any] = {}

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            captured["status"] = message["status"]
        elif message["type"] == "http.response.body":
            captured.setdefault("body", b"")
            captured["body"] += message.get("body", b"")

    await app(scope, receive, send)
    return captured.get("status"), captured.get("body", b"")


def test_non_ascii_bearer_token_is_401_not_500():
    """A non-ASCII Bearer token on a ``_check_bearer`` route yields 401, not 500.

    ``_check_bearer`` guards ``/health``, ``/tools``, ``/download``,
    ``DELETE /documents`` and, via ``_guard_asgi``, ``/sse`` + ``/messages``.
    Any one of these routing through the compare is enough to expose the bug;
    here we assert the observable contract: 401, never a 5xx. The request is
    driven at the ASGI level (see ``_run_asgi``) so the non-ASCII header reaches
    the route instead of being rejected by the test client.
    """
    stub = StubMCPServer(auth_token="test-secret-token")
    app = build_app(stub.config, stub)
    # Raw UTF-8 bytes exactly as the Authorization header would arrive on the
    # wire; Starlette decodes it as latin-1, so the token is non-ASCII to
    # ``_check_bearer``. This is the value that makes compare_digest raise.
    auth_value = "Bearer café-ünïcöde-tökën-ñûñ".encode()  # UTF-8, as on the wire

    for path in ("/health", "/tools"):
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "headers": [(b"authorization", auth_value)],
            "server": ("127.0.0.1", 8090),
            "client": ("127.0.0.1", 12345),
        }
        status, _ = asyncio.run(_run_asgi(app, scope))
        assert status == 401, (
            f"{path} returned {status} (expected 401, not a 5xx/crash)"
        )


async def test_hmac_verifier_returns_none_on_non_ascii_token():
    """``_HmacTokenVerifier.verify_token`` returns None (does not raise) on non-ASCII.

    This is the ``/mcp`` path. A non-ASCII token must be rejected with ``None``,
    not raise ``TypeError`` into the SDK's auth flow.
    """
    verifier = _HmacTokenVerifier("test-secret-token")
    result = await verifier.verify_token("café-ünïcöde-tökën-ñûñ")
    assert result is None


def test_check_bearer_returns_false_on_non_ascii_token():
    """``_check_bearer`` fails closed (returns False), does not raise, on non-ASCII.

    Exercises the non-MCP route path directly, bypassing the SDK's
    ``AuthenticationMiddleware`` so we test ``_check_bearer`` itself. The header
    arrives as raw UTF-8 bytes on the wire; Starlette decodes it as latin-1, so
    ``_check_bearer`` sees a non-ASCII string and must compare it without raising.
    """
    scope = {
        "type": "http",
        "headers": [(b"authorization", b"Bearer caf\xc3\xa9-uni-code")],
    }
    request = Request(scope)
    assert _check_bearer(request, "test-secret-token") is False
