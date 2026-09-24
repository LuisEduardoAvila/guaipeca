"""Regression tests for HTTP robustness: malformed request lines must not
produce unhandled tracebacks.

Bug: when a client sends a malformed HTTP request line and immediately closes
the connection, BaseHTTPRequestHandler.parse_request() calls send_error(),
which tries to write an HTML error body.  If the socket is already closed,
the write raises BrokenPipeError / ConnectionResetError.  The exception
propagates through ThreadingMixIn.process_request_thread and is printed as
a noisy traceback via handle_error — even though the server itself survives.

Fix: override send_error in the Handler class to catch ConnectionError.

These tests verify:
1. A malformed request line with immediate connection close does not produce
   an unhandled traceback.
2. The server stays up and continues serving valid requests afterwards.
"""

from __future__ import annotations

import socket
import struct
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from guaipeca.mcp_server import TOOLS, MCP_VERSION, KNOWN_PROTOCOL_VERSIONS
from guaipeca import __version__


# ---- Mock MCP Server (same pattern as test_streamable_http.py) ----

class MockMCPServer:
    """Lightweight MCP server for testing — no embedding model."""

    def __init__(self, auth_token: str | None = None):
        self.auth_token = auth_token

    def handle_request(self, method: str, params: dict) -> dict | None:
        params = params or {}
        if method.startswith("notifications/"):
            return None
        if method == "initialize":
            client_version = params.get("protocolVersion", MCP_VERSION)
            if client_version in KNOWN_PROTOCOL_VERSIONS:
                negotiated = client_version
            else:
                negotiated = MCP_VERSION
            return {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "guaipeca", "version": __version__},
            }
        elif method == "tools/list":
            return {"tools": TOOLS}
        elif method == "ping":
            return {}
        else:
            return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

    def _process_jsonrpc(self, raw_body: bytes) -> dict | None:
        import json
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
        except Exception as e:
            import json
            req_id = request.get("id") if request is not None else None
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(e)},
            }


# ---- Helpers ----

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_mock_server(auth_token: str | None = None) -> dict:
    """Start a real GuaipecaMCPServer HTTP server with mocked heavy parts."""
    from guaipeca.mcp_server import GuaipecaMCPServer as RealServer

    original_init = RealServer.__init__
    original_handle_request = RealServer.handle_request

    def patched_init(self, config):
        self.config = config

    def patched_warmup(self):
        pass

    def patched_handle_request(self, method: str, params: dict) -> dict | None:
        mock = MockMCPServer(auth_token=auth_token)
        return mock.handle_request(method, params)

    RealServer.__init__ = patched_init
    RealServer._warmup_model = patched_warmup
    RealServer.handle_request = patched_handle_request

    class FakeConfig:
        def __init__(self):
            self.server = type("server", (), {"auth_token": auth_token})()

    config = FakeConfig()
    server = RealServer(config)

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    import signal as signal_module
    original_signal = signal_module.signal

    def patched_signal(sig, handler):
        try:
            return original_signal(sig, handler)
        except (ValueError, OSError):
            pass

    signal_module.signal = patched_signal

    thread = threading.Thread(
        target=server.run_http,
        args=("127.0.0.1", port),
        daemon=True,
    )
    thread.start()

    health_url = f"{base_url}/health"
    for _ in range(30):
        try:
            req = urllib.request.Request(health_url)
            resp = urllib.request.urlopen(req, timeout=2)
            if resp.status == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        RealServer.__init__ = original_init
        RealServer.handle_request = original_handle_request
        signal_module.signal = original_signal
        pytest.fail("Server did not start within 15 seconds")

    return {
        "base_url": base_url,
        "health_url": health_url,
        "server": server,
        "thread": thread,
        "_original_init": original_init,
        "_original_handle_request": original_handle_request,
        "_original_signal": original_signal,
    }


def _stop_mock_server(server_info: dict):
    from guaipeca.mcp_server import GuaipecaMCPServer as RealServer
    import signal as signal_module
    RealServer.__init__ = server_info["_original_init"]
    RealServer.handle_request = server_info["_original_handle_request"]
    signal_module.signal = server_info["_original_signal"]


def _send_garbage_and_rst(host: str, port: int, payload: bytes = b"GARBAGE\r\n\r\n"):
    """Send a malformed request line and immediately close with RST.

    This triggers the BrokenPipeError/ConnectionResetError path in
    BaseHTTPRequestHandler.send_error when the fix is not in place.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    # Force RST on close so the server gets ConnectionResetError when
    # it tries to write the error response.
    linger = struct.pack("ii", 1, 0)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
    s.sendall(payload)
    s.close()


# ---- Tests ----

class TestMalformedRequestNoCrash:
    """Malformed request lines must not produce unhandled tracebacks."""

    @pytest.fixture
    def server(self):
        info = _start_mock_server()
        yield info
        _stop_mock_server(info)

    def test_garbage_request_line_no_traceback(self, server):
        """Send a garbage request line with immediate RST close.

        Before the fix: this produced a BrokenPipeError/ConnectionResetError
        traceback printed to stderr by ThreadingMixIn.handle_error.

        After the fix: send_error catches ConnectionError silently.
        """
        port = int(server["base_url"].rsplit(":", 1)[1])

        # Capture stderr to check for tracebacks
        import io
        old_stderr = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured

        try:
            _send_garbage_and_rst("127.0.0.1", port)
            time.sleep(0.5)  # Give the server thread time to process
        finally:
            sys.stderr = old_stderr

        stderr_output = captured.getvalue()
        assert "BrokenPipeError" not in stderr_output, (
            f"BrokenPipeError leaked to stderr:\n{stderr_output}"
        )
        assert "ConnectionResetError" not in stderr_output, (
            f"ConnectionResetError leaked to stderr:\n{stderr_output}"
        )
        assert "Traceback" not in stderr_output, (
            f"Unexpected traceback in stderr:\n{stderr_output}"
        )

    def test_server_survives_after_malformed_requests(self, server):
        """After receiving malformed requests, the server must still serve
        valid requests normally."""
        port = int(server["base_url"].rsplit(":", 1)[1])

        # Bomb it with a few malformed requests
        for _ in range(5):
            _send_garbage_and_rst("127.0.0.1", port)
        time.sleep(0.5)

        # Server should still respond to a valid health check
        req = urllib.request.Request(server["health_url"])
        resp = urllib.request.urlopen(req, timeout=5)
        assert resp.status == 200
        import json
        body = json.loads(resp.read())
        assert body["status"] == "ok"
        assert body["server"] == "guaipeca"

    def test_various_malformed_lines(self, server):
        """Test several types of malformed request lines."""
        port = int(server["base_url"].rsplit(":", 1)[1])

        malformed_payloads = [
            b"GARBAGE\r\n\r\n",
            b"\r\n\r\n",
            b"NOTAVALIDMETHOD\r\n\r\n",
            b"GET\r\n\r\n",  # Missing path and version
            b"GET / HTTP/1.1 EXTRA\r\n\r\n",  # Too many tokens
            b"\x00\x01\x02\r\n\r\n",  # Binary garbage
        ]

        import io
        old_stderr = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured

        try:
            for payload in malformed_payloads:
                _send_garbage_and_rst("127.0.0.1", port, payload)
            time.sleep(0.5)
        finally:
            sys.stderr = old_stderr

        stderr_output = captured.getvalue()
        assert "Traceback" not in stderr_output, (
            f"Unexpected traceback in stderr:\n{stderr_output}"
        )

        # Verify server is still healthy
        req = urllib.request.Request(server["health_url"])
        resp = urllib.request.urlopen(req, timeout=5)
        assert resp.status == 200

class TestNonLoopbackHostDetection:
    """Tests for the startup warning heuristic (no auth + public bind).

    The helper only decides whether to log a WARNING — it does not change
    request handling or auth behavior.
    """

    def test_loopback_hosts_are_safe(self):
        from guaipeca.mcp_server import _is_non_loopback_host

        for host in ("127.0.0.1", "::1", "localhost", "LOCALHOST"):
            assert _is_non_loopback_host(host) is False, host

    def test_wildcard_bind_is_exposed(self):
        from guaipeca.mcp_server import _is_non_loopback_host

        for host in ("0.0.0.0", "::"):
            assert _is_non_loopback_host(host) is True, host

    def test_lan_and_hostname_binds_are_exposed(self):
        from guaipeca.mcp_server import _is_non_loopback_host

        for host in ("192.168.1.10", "10.0.0.5", "example.com", ""):
            assert _is_non_loopback_host(host) is True, host
