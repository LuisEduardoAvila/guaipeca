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

# The malformed-request tests drive GuaipecaMCPServer.run_http(), the hand-rolled
# ThreadingHTTPServer transport, which no longer serves after Phase 4 of the MCP
# SDK migration. Robustness against garbage request lines is now uvicorn's
# responsibility, not Guaipeca's - it is not our parser any more.
# Kept (not deleted) so Phase 5's assertion-mapping table can record the decision
# explicitly. The host-warning tests above still run: they test pure helpers.
import pytest as _pytest

_OLD_TRANSPORT = _pytest.mark.skip(
    reason="hand-rolled transport removed in Phase 4; malformed-request parsing is uvicorn's now"
)

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

from guaipeca import __version__
from guaipeca.mcp_server import KNOWN_PROTOCOL_VERSIONS, MCP_VERSION, TOOLS

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
    import signal as signal_module

    from guaipeca.mcp_server import GuaipecaMCPServer as RealServer
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

@_OLD_TRANSPORT
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

    The helper decides whether a bind is loopback-only; :func:`_check_bind_exposure`
    turns that verdict into a refusal, so this predicate is now load-bearing
    rather than advisory.
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


class TestBindExposureRefusal:
    """An unauthenticated public bind is refused, not merely warned about.

    The route set includes ``/download`` (serves stored documents) and the
    ``upload``/``delete`` tools (write and remove files), so this combination
    exposes read, write and delete to anything that can route to the host.
    """

    def test_loopback_binds_start_without_a_token(self):
        from guaipeca.mcp_sdk.http import _check_bind_exposure

        for host in ("127.0.0.1", "::1", "localhost"):
            _check_bind_exposure(host, None)  # must not raise

    def test_public_bind_with_token_is_allowed(self):
        from guaipeca.mcp_sdk.http import _check_bind_exposure

        for host in ("0.0.0.0", "::", "192.168.1.10"):
            _check_bind_exposure(host, "a-token")  # must not raise

    def test_public_bind_without_token_is_refused(self):
        import pytest

        from guaipeca.mcp_sdk.http import _check_bind_exposure

        for host in ("0.0.0.0", "::", "192.168.1.10", "example.com", ""):
            with pytest.raises(ValueError) as excinfo:
                _check_bind_exposure(host, None)
            message = str(excinfo.value)
            # The error must name both remedies and the offending value.
            assert "auth_token" in message
            assert "127.0.0.1" in message
            assert f"server.host={host!r}" in message

    def test_empty_token_counts_as_no_token(self):
        """A blank token is not authentication; config validation rejects it
        earlier, but the guard must not treat '' as configured."""
        import pytest

        from guaipeca.mcp_sdk.http import _check_bind_exposure

        with pytest.raises(ValueError):
            _check_bind_exposure("0.0.0.0", "")


class TestUnauthenticatedLoopbackWarning:
    """A loopback bind with no token is allowed, and announced.

    It is the default and the sensible choice on a single-user host, so it must
    not be refused. It is announced because "this port needs no credentials" is
    something the operator should be told rather than left to infer, and it is
    announced on every start rather than once — a suppressed warning would be
    silent on the day the bind is widened.
    """

    def _stderr(self, *args):
        """Return the WARNING records the guard emits (empty if it refuses)."""
        import logging

        from guaipeca.mcp_sdk.http import _check_bind_exposure

        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Capture()
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            _check_bind_exposure(*args)
        finally:
            root.removeHandler(handler)
        return [r.getMessage() for r in records]

    def test_loopback_without_token_warns(self):
        for host in ("127.0.0.1", "::1", "localhost"):
            messages = self._stderr(host, None)
            assert any("serving without authentication" in m for m in messages), host
            # The remedy must be named, not merely the problem.
            assert any("GUAIPECA_AUTH_TOKEN" in m for m in messages), host

    def test_loopback_with_token_is_silent(self):
        assert self._stderr("127.0.0.1", "a-token") == []

    def test_public_bind_with_token_is_silent(self):
        assert self._stderr("0.0.0.0", "a-token") == []

    def test_warning_is_emitted_once(self):
        """One line per start: the logger already reaches stderr via cli.main,
        so a second explicit print would duplicate it in the service log."""
        messages = self._stderr("127.0.0.1", None)
        assert sum("serving without authentication" in m for m in messages) == 1

    def test_cmd_serve_then_build_app_announces_once(self):
        """Regression: cmd_serve checks early (to refuse before loading the
        model) and build_app checks again. Both announcing printed the line
        twice, 0.6s apart, in a real serve run.

        Drives the real ``build_app`` rather than calling the internals, so the
        test sees what the two call sites actually do.
        """
        import logging
        import tempfile
        from pathlib import Path

        from guaipeca.config import GuaipecaConfig
        from guaipeca.mcp_sdk import http as sdk_http
        from guaipeca.mcp_server import GuaipecaMCPServer

        sdk_http._BIND_ANNOUNCED = False

        d = Path(tempfile.mkdtemp())
        cfg_path = d / "c.yaml"
        cfg_path.write_text(
            f"indexing:\n  data_dir: {d}\n"
            f"server:\n  transport: http\n  host: 127.0.0.1\n  port: 8090\n"
            f"corpora:\n  - name: docs\n    path: {d}\n    extensions: [.md]\n"
        )
        config = GuaipecaConfig.from_yaml(str(cfg_path))

        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Capture()
        logging.getLogger().addHandler(handler)
        try:
            # As cmd_serve does, before constructing the server.
            sdk_http._check_bind_exposure("127.0.0.1", None)
            # Then the real app build, which checks again as a backstop.
            sdk_http.build_app(config, GuaipecaMCPServer(config), host="127.0.0.1")
        finally:
            logging.getLogger().removeHandler(handler)
            sdk_http._BIND_ANNOUNCED = False

        assert (
            sum("serving without authentication" in r.getMessage() for r in records)
            == 1
        )

    def test_public_bind_refusal_does_not_also_warn(self):
        """Refusal and warning are mutually exclusive: one message, one reason."""
        import pytest

        with pytest.raises(ValueError):
            self._stderr("0.0.0.0", None)

    def test_warning_is_also_logged(self, caplog):
        """And it carries a level, so a collector can filter on it."""
        import logging

        from guaipeca.mcp_sdk.http import _check_bind_exposure

        with caplog.at_level(logging.WARNING):
            _check_bind_exposure("127.0.0.1", None)
        assert any(
            r.levelno == logging.WARNING
            and "serving without authentication" in r.getMessage()
            for r in caplog.records
        )


class TestServeRefusalIsClean:
    """A refused startup must be a clean failure, not a traceback with exit 0.

    Regression guard: the first implementation raised through `server.run()`,
    which surfaced a 20-line traceback and still exited 0 — a unit file or
    script checking the status saw success.
    """

    def test_cmd_serve_exits_nonzero_with_one_clean_error(self, tmp_path, capsys):
        import pytest

        from guaipeca import cli

        # Minimal config: public bind, no token. Written directly rather than
        # copied from the sample so this test does not depend on docs.
        config_path = tmp_path / "guaipeca.yaml"
        config_path.write_text(
            f"""indexing:
  data_dir: {tmp_path}
server:
  transport: http
  host: 0.0.0.0
  port: 8090
corpora:
  - name: docs
    path: {tmp_path}
    extensions: [.md]
"""
        )

        args = type(
            "Args", (), {"config": str(config_path), "transport": "http", "port": 18095}
        )()

        with pytest.raises(SystemExit) as excinfo:
            cli.cmd_serve(args)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "refusing to serve" in err
        # The message must name the offending value and both remedies.
        assert "0.0.0.0" in err
        assert "auth_token" in err
        assert "127.0.0.1" in err
        # No traceback: a deliberate refusal is not a crash.
        assert "Traceback" not in err
