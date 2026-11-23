"""Acceptance tests for the CORS cluster (#73-#79).

Seven low-severity findings, all in ``_CorsMiddleware`` or
``_allowed_origin_patterns`` in ``src/guaipeca/mcp_sdk/http.py``. They share
this file because they share one change surface and one test harness.

Each test names the issue it pins. They fail on unfixed code.
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from guaipeca.config import ServerConfig
from guaipeca.mcp_sdk.http import _CorsMiddleware, build_app

ORIGIN = "http://localhost:3000"
PREFLIGHT = {"Origin": ORIGIN, "Access-Control-Request-Method": "POST"}


class _Stub:
    def __init__(self, config):
        self.config = config

    def _tools_with_corpus_enum(self):
        return []


def _client(allowed_origins=None, **kwargs) -> TestClient:
    config = type("Config", (), {})()
    config.server = ServerConfig()
    if allowed_origins is not None:
        config.server.allowed_origins = allowed_origins
    config.corpora = []
    config.get_corpus = lambda name: None
    app = build_app(config, _Stub(config), **kwargs)
    return TestClient(app, base_url="http://127.0.0.1:8090")


# --------------------------------------------------------------------------
# #73: preflight advertises only the methods each route actually serves
# --------------------------------------------------------------------------

def test_preflight_advertises_only_the_methods_the_path_serves():
    """`/health` serves GET; advertising POST/DELETE invites a request that 405s."""
    client = _client()

    health = client.options("/health", headers=PREFLIGHT)
    methods = health.headers.get("access-control-allow-methods", "").upper()

    assert "GET" in methods
    assert "POST" not in methods, f"/health advertises POST it does not serve: {methods}"
    assert "DELETE" not in methods, f"/health advertises DELETE it does not serve: {methods}"


def test_preflight_on_mcp_still_advertises_post_and_delete():
    """/mcp genuinely serves POST; the narrowed list must not break it."""
    client = _client()

    r = client.options("/mcp", headers=PREFLIGHT)
    methods = r.headers.get("access-control-allow-methods", "").upper()

    assert "POST" in methods


# --------------------------------------------------------------------------
# #74: a '*' in allowed_origins must not produce a raw traceback
# --------------------------------------------------------------------------

def test_star_in_allowed_origins_is_a_clean_error_not_a_traceback(monkeypatch, capsys):
    """The guard exists but nothing catches it, so the CLI prints a traceback."""
    from guaipeca import cli

    config = type("Config", (), {})()
    config.server = ServerConfig()
    config.server.allowed_origins = ["https://ui.example.com/*"]
    config.corpora = []
    config.server.transport = "http"

    called = {"run": False}

    class _FakeServer:
        def __init__(self, *a, **k):
            pass

        def run(self, **kwargs):
            called["run"] = True

    monkeypatch.setattr("guaipeca.mcp_server.GuaipecaMCPServer", _FakeServer)
    monkeypatch.setattr(cli, "_load_config", lambda _p: config)

    args = type("Args", (), {})()
    args.config = "x.yaml"
    args.transport = None
    args.port = None

    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_serve(args)

    out = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "allowed_origins" in out.err
    assert "Traceback" not in out.err, "the operator gets a traceback, not a message"
    assert not called["run"], "the server must not start with a bad origin list"


# --------------------------------------------------------------------------
# #75: preflight on a path that does not exist should not claim success
# --------------------------------------------------------------------------

def test_preflight_on_unknown_path_is_not_a_blanket_204():
    """A preflight that succeeds for a 404 path tells the browser to proceed."""
    client = _client()

    preflight = client.options("/definitely-not-a-route", headers=PREFLIGHT)
    real = client.get("/definitely-not-a-route")

    assert real.status_code == 404
    assert preflight.status_code == 404, (
        f"preflight returned {preflight.status_code} for a path the real "
        f"request 404s; it should not advertise a cross-origin success"
    )


# --------------------------------------------------------------------------
# #76: ACAO must echo the matched allowlist entry, not the raw request origin
# --------------------------------------------------------------------------

def test_acao_echoes_the_matched_entry_not_the_raw_origin():
    """A trailing slash in the request must not survive into ACAO."""
    client = _client(allowed_origins=["https://ui.example.com"])

    r = client.get("/health", headers={"Origin": "https://ui.example.com/"})

    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://ui.example.com", (
        f"ACAO echoed the raw origin: {r.headers.get('access-control-allow-origin')!r}"
    )


# --------------------------------------------------------------------------
# #77: harden the Origin handling
# --------------------------------------------------------------------------

def test_control_characters_in_origin_are_rejected_not_echoed():
    """Raw control bytes do not reach ACAO (defence in depth for ASGI transports)."""
    client = _client()

    r = client.get("/health", headers={"Origin": f"{ORIGIN}\rX-Injected: pwned"})

    acao = r.headers.get("access-control-allow-origin", "")
    assert "\r" not in acao and "\n" not in acao, f"control chars echoed: {acao!r}"


def test_oversized_origin_is_rejected():
    """A 100KB Origin should not pass the check and be echoed back."""
    client = _client()
    huge = "http://localhost:3000/" + "a" * 100_000

    r = client.get("/health", headers={"Origin": huge})

    acao = r.headers.get("access-control-allow-origin", "")
    assert len(acao) < 1000, f"a {len(acao)}-byte origin was accepted and echoed"


def test_empty_allowed_origins_list_is_treated_like_none():
    """`[]` and `None` mean the same thing.

    Not a defect on current code (both return the loopback patterns), kept as a
    pin so a future change to the empty-list path is a visible failure rather
    than a silent behaviour change. Issue #77 listed this as a third finding;
    it does not reproduce - see the PR body.
    """
    from guaipeca.mcp_sdk.http import _allowed_origin_patterns

    assert _allowed_origin_patterns([]) == _allowed_origin_patterns(None)


# --------------------------------------------------------------------------
# #78: one Vary header, not two
# --------------------------------------------------------------------------

def test_vary_is_not_duplicated_when_inner_app_sets_its_own():
    async def endpoint(request):
        return PlainTextResponse("ok", headers={"Vary": "Accept-Encoding"})

    app = Starlette(routes=[Route("/x", endpoint)])
    client = TestClient(_CorsMiddleware(app, auth_token=None, allowed_origins=None))

    r = client.get("/x", headers={"Origin": ORIGIN})

    vary = r.headers.get_list("vary")
    assert len(vary) == 1, f"two Vary headers issued: {vary}"
    combined = vary[0].lower()
    assert "origin" in combined
    assert "accept-encoding" in combined, "the inner app's Vary value was dropped"


# --------------------------------------------------------------------------
# #79: small cleanups
# --------------------------------------------------------------------------

def test_malformed_allowed_origin_entry_is_reported(caplog):
    """An entry that normalises to empty is dropped; it must say so."""
    import logging

    from guaipeca.mcp_sdk.http import _allowed_origin_patterns

    with caplog.at_level(logging.WARNING):
        _allowed_origin_patterns(["", "/", "  ", "https://ui.example.com"])

    assert caplog.text.strip(), "a dropped origin entry was skipped silently"


def test_allowed_origin_patterns_docstring_states_it_raises():
    """The '*' ValueError is part of the contract and should be documented."""
    from guaipeca.mcp_sdk.http import _allowed_origin_patterns

    doc = _allowed_origin_patterns.__doc__ or ""
    assert "raise" in doc.lower() or "valueerror" in doc.lower(), (
        "the docstring does not mention that a '*' entry raises"
    )
