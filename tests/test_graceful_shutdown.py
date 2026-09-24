"""Regression test for graceful shutdown on SIGTERM/SIGINT.

Bug: the signal handler called httpd.shutdown() directly.  That call blocks
waiting for serve_forever() to acknowledge the shutdown flag, but the
handler runs in the main thread where serve_forever() also runs — so the
handshake deadlocks.  The process wedges in futex_do_wait, systemd's stop
times out, and the port stays bound until a SIGKILL.

Fix: the handler only sets a threading.Event.  serve_forever() runs in a
daemon thread; the main thread waits on the event, then calls shutdown()
and os._exit(0) to bypass native non-daemon thread pools (fastembed/numpy
BLAS / tokenizers rayon) that block interpreter finalization.

These tests verify:
1. SIGTERM causes the process to exit within a few seconds.
2. The listening port is released after exit.
3. SIGINT behaves identically (ctrl-C in foreground mode).
"""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def _port_is_free(host: str, port: int) -> bool:
    """Return True if the port is NOT accepting connections (i.e. released)."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return False
    except (ConnectionRefusedError, OSError):
        return True


# ---------------------------------------------------------------------------
# Subprocess-based test: starts a real server process, sends SIGTERM, and
# verifies the process exits and the port is released.  This is the
# end-to-end proof that the fix works.
#
# We use proc.wait() / proc.poll() to detect exit (not os.kill(pid, 0),
# which returns success for zombie processes that haven't been reaped).
# ---------------------------------------------------------------------------

_SERVER_SCRIPT = """\
import sys, os, signal, threading, time, socket
from pathlib import Path
sys.path.insert(0, str(Path({src!r}) / "src"))

from guaipeca.mcp_server import GuaipecaMCPServer

def _patched_init(self, config):
    self.config = config
def _patched_warmup(self):
    pass
def _patched_handle_request(self, method, params):
    if method == "ping":
        return {{}}
    return {{"error": {{"code": -32601, "message": "mock"}}}}

GuaipecaMCPServer.__init__ = _patched_init
GuaipecaMCPServer._warmup_model = _patched_warmup
GuaipecaMCPServer.handle_request = _patched_handle_request

class FakeConfig:
    def __init__(self):
        self.server = type("server", (), {{"auth_token": None}})()

server = GuaipecaMCPServer(FakeConfig())
server.run_http("127.0.0.1", {port})
"""


def _start_server_subprocess(port: int) -> subprocess.Popen:
    """Start a real server process on the given port."""
    script = _SERVER_SCRIPT.format(src=str(PROJECT_ROOT), port=port)
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


class TestGracefulShutdown:
    """SIGTERM/SIGINT must cause the server to exit promptly."""

    @pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT],
                             ids=["SIGTERM", "SIGINT"])
    def test_signal_shuts_down_cleanly(self, sig):
        """Send a signal and verify the process exits within 10 seconds
        and the port is released."""
        port = _find_free_port()
        proc = _start_server_subprocess(port)
        try:
            # Wait for the server to start listening
            assert _wait_for_port("127.0.0.1", port, timeout=20), (
                f"Server did not start on port {port} within 20s"
            )

            # Verify it's actually responding
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            resp = urllib.request.urlopen(req, timeout=5)
            assert resp.status == 200

            # Send the signal
            proc.send_signal(sig)
            t0 = time.monotonic()

            # Wait for the process to exit (using proc.wait, not os.kill
            # which returns success for unreaped zombies).
            try:
                rc = proc.wait(timeout=10)
                elapsed = time.monotonic() - t0
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                pytest.fail(
                    f"Process {proc.pid} did not exit within 10s after "
                    f"{signal.Signals(sig).name} (elapsed={elapsed:.1f}s)"
                )

            assert rc == 0, (
                f"Unexpected exit code {rc} (expected 0 from os._exit(0), "
                f"elapsed={elapsed:.1f}s)"
            )

            # Verify the port is released
            time.sleep(0.5)  # Give the OS a moment to free the socket
            assert _port_is_free("127.0.0.1", port), (
                f"Port {port} still in use after process exit "
                f"(elapsed={elapsed:.1f}s)"
            )

        finally:
            # Cleanup: make sure the process is dead
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)