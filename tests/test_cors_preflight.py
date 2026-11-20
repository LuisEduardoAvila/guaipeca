"""CORS behaviour of the SDK-backed HTTP app (issues #36, #37, #39).

#36: ``_CorsMiddleware``'s docstring claimed it answered preflights. It did not
- ``OPTIONS`` fell through to the routes and returned 405 with an ACAO header,
which reads like "CORS is configured" while a browser still aborts the request.
The middleware now answers a preflight for an allowed or absent origin.

#39: the middleware read only the FIRST ``Origin`` header. A request with an
allowed first and a disallowed second proceeded. Browsers never send two, but
the control is all-must-match and the code implemented first-must-match.

#37: documentation only - covered in ``test_mcp_sdk_http.py``'s docstrings and
the module docstring; no behaviour test belongs here.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from guaipeca.mcp_sdk.http import build_app


class _Stub:
    """Minimal server: build_app only needs ``config`` and a tool registry."""

    def __init__(self, config):
        self.config = config

    def _tools_with_corpus_enum(self):
        return []


def _client(*, allowed_origins: list[str] | None = None, **kwargs) -> TestClient:
    # Same shape as StubMCPServer in test_mcp_sdk_http.py: build_app reads
    # config.server and config.get_corpus, nothing else.
    from guaipeca.config import ServerConfig

    config = type("Config", (), {})()
    config.server = ServerConfig()
    if allowed_origins is not None:
        config.server.allowed_origins = allowed_origins
    config.corpora = []
    config.get_corpus = lambda name: None
    app = build_app(config, _Stub(config), **kwargs)
    return TestClient(app, base_url="http://127.0.0.1:8090")


PREFLIGHT = {
    "Origin": "http://localhost:3000",
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "authorization, content-type",
}

ROUTES = ["/mcp", "/sse", "/health", "/tools"]


@pytest.mark.parametrize("path", ROUTES)
def test_preflight_is_answered_on_every_wrapped_route(path):
    """#36: OPTIONS with the preflight headers gets 2xx and the allow headers.

    Parametrised over all four routes because the middleware wraps all of them;
    fixing /mcp alone would leave a browser client broken on /tools.
    """
    response = _client().options(path, headers=PREFLIGHT)

    assert response.status_code in (200, 204), (
        f"{path} returned {response.status_code}; a browser aborts the real "
        f"request unless the preflight is 2xx"
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    allow_methods = response.headers.get("access-control-allow-methods", "")
    assert "POST" in allow_methods.upper()
    assert "DELETE" in allow_methods.upper()
    allow_headers = response.headers.get("access-control-allow-headers", "").lower()
    assert "authorization" in allow_headers
    assert "content-type" in allow_headers


def test_preflight_from_disallowed_origin_is_still_403():
    """#36 AC: the existing rejection path is unchanged by answering preflights."""
    response = _client().options(
        "/mcp", headers={**PREFLIGHT, "Origin": "http://evil.example.com"}
    )
    assert response.status_code == 403


def test_preflight_without_origin_is_answered():
    """curl and non-browser clients send OPTIONS with no Origin; do not 405 them."""
    response = _client().options("/mcp")
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-methods")


def test_allowlisted_origin_is_echoed_in_preflight():
    """The echoed origin comes from the allowlist match, not from the request text."""
    client = _client(allowed_origins=["https://ui.example.com"])
    response = client.options(
        "/mcp", headers={**PREFLIGHT, "Origin": "https://ui.example.com"}
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == "https://ui.example.com"


def test_request_with_one_allowed_and_one_disallowed_origin_is_rejected():
    """#39: all Origin headers must match, not just the first.

    Browsers never send two; a proxy or hand-built client can, and the failure
    mode was asymmetric - a disallowed FIRST header got 403 while a disallowed
    SECOND header was invisible.
    """
    response = _client().get(
        "/health",
        headers=[
            ("origin", "http://localhost:3000"),
            ("origin", "https://evil.example"),
        ],
    )
    assert response.status_code == 403, (
        "a disallowed second Origin header was ignored"
    )


def test_request_with_two_allowed_origins_is_allowed():
    """The other side of the domain: rejecting on any duplicate would break this."""
    response = _client().get(
        "/health",
        headers=[
            ("origin", "http://localhost:3000"),
            ("origin", "http://localhost:3000"),
        ],
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_disallowed_first_origin_still_rejected():
    """Regression guard: the original single-header behaviour must not regress."""
    response = _client().get("/health", headers={"origin": "http://evil.example.com"})
    assert response.status_code == 403


# ---- #37: the "*" comment described a guard that did not exist ------------


def test_wildcard_origin_entry_is_rejected_at_startup():
    """#37: an operator writing '*' means a wildcard; matching is exact, so it
    could only ever fail to match. Fail loudly instead of silently 403ing every
    browser client.
    """
    from guaipeca.mcp_sdk.http import _allowed_origin_patterns

    with pytest.raises(ValueError, match="exact"):
        _allowed_origin_patterns(["https://ui.example.com/*"])

    with pytest.raises(ValueError):
        _allowed_origin_patterns(["*"])


def test_plain_entries_survive_the_wildcard_guard():
    """The other side: a normal allowlist is untouched, trailing slash included."""
    from guaipeca.mcp_sdk.http import _allowed_origin_patterns

    assert _allowed_origin_patterns(["https://ui.example.com/"]) == [
        "https://ui.example.com"
    ]
    assert _allowed_origin_patterns(["https://a.example", ""]) == ["https://a.example"]


def test_health_docstring_no_longer_claims_unauthenticated():
    """#37: the module docstring must not say /health is always open.

    An operator wired an unauthenticated monitor to a token-protected server on
    the strength of that line and read the 401 as an outage.
    """
    import inspect

    from guaipeca.mcp_sdk import http as http_module

    doc = inspect.getdoc(http_module) or ""
    health_line = next(line for line in doc.splitlines() if "/health" in line)
    assert "unauthenticated only when" in health_line.lower() or (
        "no" in health_line.lower() and "token" in health_line.lower()
    ), f"/health docstring still claims unconditional openness: {health_line!r}"
