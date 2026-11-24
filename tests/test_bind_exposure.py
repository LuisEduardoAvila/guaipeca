"""Tests for the bind-exposure guard behind `guaipeca serve`.

Covers the loopback-only predicate, the refusal of an unauthenticated public
bind, the one-per-start warning for an unauthenticated loopback bind, and the
clean (non-traceback) failure of a refused startup.

(The malformed-request-line tests that lived here drove the hand-rolled
ThreadingHTTPServer transport, which was removed in Phase 4 of the MCP SDK
migration; garbage request lines are uvicorn's to parse now, not ours.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

class TestNonLoopbackHostDetection:
    """Tests for the loopback-only predicate behind the startup bind refusal.

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

        from guaipeca.mcp_sdk.http import _check_bind_exposure

        with pytest.raises(ValueError):
            _check_bind_exposure("0.0.0.0", "")

    def test_build_app_also_refuses_a_public_bind(self):
        """The guard in ``build_app`` is a backstop for callers that assemble
        the app directly, so it needs its own test.

        ``cmd_serve`` checks before constructing the server, which means every
        CLI-path test keeps passing if this second call site is deleted. The
        check runs before ``build_server(...)``, so a bare object stands in for
        the server instance here.
        """
        import tempfile

        from guaipeca.config import GuaipecaConfig
        from guaipeca.mcp_sdk.http import build_app

        d = Path(tempfile.mkdtemp())
        cfg_path = d / "c.yaml"
        cfg_path.write_text(
            f"indexing:\n  data_dir: {d}\n"
            f"server:\n  transport: http\n  host: 0.0.0.0\n  port: 8090\n"
            f"corpora:\n  - name: test-docs\n    path: {d}\n",
            encoding="utf-8",
        )
        config = GuaipecaConfig.from_yaml(str(cfg_path))
        assert config.server.auth_token is None

        with pytest.raises(ValueError) as excinfo:
            build_app(config, object(), host="0.0.0.0")
        assert "refusing to serve" in str(excinfo.value)


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
