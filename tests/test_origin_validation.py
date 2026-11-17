"""Tests for Origin header validation (CORS / DNS-rebinding protection).

A browser on an attacker-controlled page can make a victim's browser POST to
a Guaipeca server bound on 0.0.0.0. The Origin header is the only signal the
server gets; DNS rebinding makes the Host header untrustworthy. So: an
unrecognised Origin is rejected with 403 before any auth check (an evil page
must not be able to distinguish "no token" from "wrong token"), an absent
Origin is always fine (curl, CLI, non-browser MCP clients), and loopback
origins are fine by default.

Reuses the mock-server harness pattern from test_streamable_http.py.
"""



from __future__ import annotations

# SUPERSEDED by Phase 4 of the MCP SDK migration (design/mcp-sdk-migration-plan.md).
# These tests drive GuaipecaMCPServer.run_http(), the hand-rolled ThreadingHTTPServer
# transport, which no longer serves - HTTP is now mcp_sdk/http.py on uvicorn.
# The behaviour they assert has equivalents in tests/test_mcp_sdk_http.py, EXCEPT
# the cases listed in the Phase 5 assertion-mapping table (dead-session 404,
# /mcp trailing slash, batch arrays, malformed request line, 413). Skipped rather
# than deleted so nothing is lost before that mapping is reviewed.
# Phase 5 ports the surviving assertions and then removes this file.
import pytest as _pytest

pytestmark = _pytest.mark.skip(
    reason="hand-rolled HTTP transport removed in Phase 4; see tests/test_mcp_sdk_http.py"
)
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from guaipeca import __version__
from guaipeca.mcp_server import MCP_VERSION, TOOLS, _origin_allowed

EVIL_ORIGIN = "http://evil.example.com"


# ---- Mock server (no embedding model needed) ----

class MockMCPServer:
    """Lightweight MCP server for testing HTTP transport."""

    def __init__(self, config):
        self.config = config

    def handle_request(self, method: str, params: dict) -> dict | None:
        params = params or {}
        if method.startswith("notifications/"):
            return None
        if method == "initialize":
            return {
                "protocolVersion": MCP_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "guaipeca", "version": __version__},
            }
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "ping":
            return {}
        return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

    def _process_jsonrpc(self, raw_body: bytes) -> dict | None:
        request = None
        try:
            request = json.loads(raw_body)
            method = request.get("method", "")
            params = request.get("params", {})
            req_id = request.get("id")
            result = self.handle_request(method, params)
            if result is None or req_id is None:
                return None
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except json.JSONDecodeError:
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
        except Exception as e:
            req_id = request.get("id") if request is not None else None
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(e)},
            }


# ---- Server harness ----

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(auth_token: str | None = None,
                  allowed_origins: list[str] | None = None) -> dict:
    """Start the real run_http handler around a MockMCPServer.

    ``allowed_origins`` mirrors config.ServerConfig.allowed_origins. When it is
    None the attribute is simply omitted, which exercises the defensive
    getattr() path used by duck-typed configs.
    """
    from guaipeca.mcp_server import GuaipecaMCPServer as RealServer

    server_attrs = {"auth_token": auth_token}
    if allowed_origins is not None:
        server_attrs["allowed_origins"] = allowed_origins
    config = type("FakeConfig", (), {"server": type("Srv", (), server_attrs)()})()

    server = MockMCPServer(config)
    run_http = RealServer.run_http.__get__(server, MockMCPServer)

    import signal as signal_module
    original_signal = signal_module.signal

    def patched_signal(sig, handler):
        try:
            return original_signal(sig, handler)
        except (ValueError, OSError):
            pass

    signal_module.signal = patched_signal
    try:
        port = _find_free_port()
        thread = threading.Thread(
            target=run_http, args=("127.0.0.1", port), daemon=True
        )
        thread.start()

        base_url = f"http://127.0.0.1:{port}"
        for _ in range(30):
            try:
                resp = urllib.request.urlopen(f"{base_url}/health", timeout=2)
                if resp.status == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            pytest.fail("Server did not start within ~6 seconds")
    finally:
        signal_module.signal = original_signal

    return {
        "base_url": base_url,
        "mcp_url": f"{base_url}/mcp",
        "sse_url": f"{base_url}/sse",
        "port": port,
    }


# ---- HTTP helpers ----

def _request(url: str, method: str = "POST", body: dict | None = None,
             origin: str | None = None, session_id: str | None = None,
             auth_token: str | None = None) -> tuple[int, dict | bytes | None, dict]:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if origin is not None:
        req.add_header("Origin", origin)
    if session_id:
        req.add_header("Mcp-Session-Id", session_id)
    if auth_token:
        req.add_header("Authorization", f"Bearer {auth_token}")

    try:
        resp = urllib.request.urlopen(req, timeout=5)
        raw = resp.read()
        status = resp.status
        headers = {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
        headers = {k.lower(): v for k, v in e.headers.items()}

    try:
        parsed = json.loads(raw) if raw else None
    except (json.JSONDecodeError, ValueError):
        parsed = raw
    return status, parsed, headers


def _init_body() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }


def _initialize(server: dict, origin: str | None = None,
                auth_token: str | None = None) -> str:
    status, resp, headers = _request(
        server["mcp_url"], body=_init_body(), origin=origin, auth_token=auth_token
    )
    assert status == 200, f"Initialize failed: {status} {resp}"
    session_id = headers.get("mcp-session-id")
    assert session_id
    return session_id


def _get_sse_status(server: dict, origin: str | None) -> int:
    """Open GET /sse and return the status code (reads only the head)."""
    sock = socket.create_connection(("127.0.0.1", server["port"]), timeout=5)
    try:
        head = f"GET /sse HTTP/1.1\r\nHost: 127.0.0.1:{server['port']}\r\n"
        if origin is not None:
            head += f"Origin: {origin}\r\n"
        sock.sendall((head + "\r\n").encode())
        sock.settimeout(5)
        buf = sock.recv(1024).decode("utf-8", errors="replace")
    finally:
        sock.close()
    return int(buf.split(" ", 2)[1])


# ---- Unit tests for the policy function ----

class TestOriginPolicy:
    def test_absent_origin_allowed(self):
        assert _origin_allowed("", None) is True

    def test_loopback_origins_allowed(self):
        for origin in (
            "http://localhost:3000",
            "http://127.0.0.1:8090",
            "https://127.0.0.1",
            "http://[::1]:8080",
            "http://ip6-localhost:3000",
        ):
            assert _origin_allowed(origin, None) is True, origin

    def test_evil_origin_rejected(self):
        for origin in (
            "http://evil.example.com",
            "https://evil.example.com:8443",
            "http://127.0.0.1.evil.example.com",
            "null",
        ):
            assert _origin_allowed(origin, None) is False, origin

    def test_allowlist_exact_match(self):
        allowed = ["https://ui.example.com"]
        assert _origin_allowed("https://ui.example.com", allowed) is True
        assert _origin_allowed("https://ui.example.com/", allowed) is True
        assert _origin_allowed("https://other.example.com", allowed) is False
        # An allowlist replaces the loopback default rather than extending it.
        assert _origin_allowed("http://localhost:3000", allowed) is False


# ---- End-to-end HTTP tests ----

@pytest.fixture(scope="module")
def server():
    info = _start_server()
    yield info


@pytest.fixture(scope="module")
def auth_server():
    info = _start_server(auth_token="test-secret-token")
    yield info


@pytest.fixture(scope="module")
def allowlisted_server():
    info = _start_server(allowed_origins=["https://ui.example.com"])
    yield info


class TestOriginValidationHttp:
    def test_evil_origin_rejected_on_post_mcp(self, server):
        status, _resp, headers = _request(
            server["mcp_url"], body=_init_body(), origin=EVIL_ORIGIN
        )
        assert status == 403
        assert "mcp-session-id" not in headers

    def test_evil_origin_rejected_on_get_sse(self, server):
        assert _get_sse_status(server, EVIL_ORIGIN) == 403

    def test_no_origin_allowed(self, server):
        status, _resp, headers = _request(server["mcp_url"], body=_init_body())
        assert status == 200
        assert "mcp-session-id" in headers

    def test_loopback_origin_allowed(self, server):
        status, _resp, headers = _request(
            server["mcp_url"], body=_init_body(), origin="http://localhost:3000"
        )
        assert status == 200
        assert "mcp-session-id" in headers

    def test_evil_origin_rejected_on_delete(self, server):
        session_id = _initialize(server)
        status, _resp, _headers = _request(
            server["mcp_url"], method="DELETE", origin=EVIL_ORIGIN, session_id=session_id
        )
        assert status == 403

    def test_evil_origin_rejected_on_options(self, server):
        status, _resp, _headers = _request(
            server["mcp_url"], method="OPTIONS", origin=EVIL_ORIGIN
        )
        assert status == 403


class TestOriginBeatsAuth:
    def test_evil_origin_gets_403_not_401(self, auth_server):
        """Origin is checked before auth so a cross-origin page cannot probe
        the token: it must see 403, never 401."""
        status, _resp, _headers = _request(
            auth_server["mcp_url"], body=_init_body(), origin=EVIL_ORIGIN
        )
        assert status == 403

    def test_no_origin_still_401_without_token(self, auth_server):
        status, _resp, _headers = _request(auth_server["mcp_url"], body=_init_body())
        assert status == 401


class TestConfiguredAllowlist:
    def test_allowlisted_origin_allowed(self, allowlisted_server):
        status, _resp, headers = _request(
            allowlisted_server["mcp_url"],
            body=_init_body(),
            origin="https://ui.example.com",
        )
        assert status == 200
        assert "mcp-session-id" in headers

    def test_loopback_rejected_when_allowlist_configured(self, allowlisted_server):
        status, _resp, _headers = _request(
            allowlisted_server["mcp_url"],
            body=_init_body(),
            origin="http://localhost:3000",
        )
        assert status == 403


class TestAccessControlAllowOrigin:
    def test_acao_echoes_validated_origin(self, server):
        status, _resp, headers = _request(
            server["mcp_url"], body=_init_body(), origin="http://127.0.0.1:3000"
        )
        assert status == 200
        assert headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"

    def test_acao_legacy_value_without_origin(self, server):
        """No Origin request → unchanged legacy behaviour ('*' without auth)."""
        status, _resp, headers = _request(server["mcp_url"], body=_init_body())
        assert status == 200
        assert headers.get("access-control-allow-origin") == "*"

    def test_acao_legacy_value_without_origin_when_auth(self, auth_server):
        """No Origin request with auth on → legacy 'null'."""
        status, _resp, headers = _request(
            auth_server["mcp_url"], body=_init_body(), auth_token="test-secret-token"
        )
        assert status == 200
        assert headers.get("access-control-allow-origin") == "null"

    def test_options_echoes_validated_origin(self, server):
        status, _resp, headers = _request(
            server["mcp_url"], method="OPTIONS", origin="http://localhost:3000"
        )
        assert status == 204
        assert headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_unparsable_origin_rejected(self, server):
        """A garbage Origin value must not crash the handler — it is a 403."""
        status, _resp, _headers = _request(
            server["mcp_url"], body=_init_body(), origin="https://[bad"
        )
        assert status == 403


def test_urlparse_import_present():
    """Guard: the module uses urllib.parse at module scope for _origin_allowed."""
    assert urllib.parse.urlparse("http://localhost").hostname == "localhost"
