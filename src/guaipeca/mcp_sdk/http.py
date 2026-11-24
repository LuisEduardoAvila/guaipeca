"""HTTP transport for the SDK-backed MCP server (Phase 4 of the migration).

Serves one Starlette app under uvicorn::

    /mcp                    Streamable HTTP, both protocol eras
    /health                 liveness probe (unauthenticated only when no
                            server.auth_token is set - it returns 401 once one
                            is, see ``_health`` and
                            tests/test_mcp_sdk_http.py::test_health_requires_auth_when_configured)
    /tools                  tool listing (convenience endpoint)
    /download/{c}/{f}       file download
    DELETE /documents       delete an uploaded file
    /sse, /messages         legacy SSE transport (GET stream + POST messages)

The SDK owns everything MCP-shaped — session handling, protocol-version
routing (handshake vs 2026-07-28 stateless requests), JSON parsing, the
request-body limit and DNS-rebinding protection. This module wires the
existing ``GuaipecaMCPServer`` into it and re-implements only what the SDK has
no equivalent for. The legacy SSE transport is served by the SDK's own
``MCPServer.sse_app()``, built around this module's low-level server (see
:func:`_build_sse_app`); its two routes are merged into the app below so the
same CORS wrapper applies.

Security notes
--------------
* Origin policy keeps the PR #23 behaviour: no Origin is allowed (curl, CLI,
  non-browser clients), otherwise only loopback origins unless
  ``server.allowed_origins`` is a non-empty list, which then becomes an exact
  allowlist. See :func:`_security_settings`.
* Auth needs BOTH ``auth=AuthSettings(...)`` and ``token_verifier``: the SDK
  wraps the route in RequireAuthMiddleware whenever a verifier is present, but
  only installs the BearerAuthBackend that can validate it when ``auth`` is
  also set - a verifier on its own 401s every request. OAuth discovery routes
  appear only when ``auth_server_provider`` is passed, which we do not.
* The SSE routes are built by ``MCPServer.sse_app()``, outside the SDK's auth
  configuration (which covers ``/mcp`` only), so they bypass that wrapper.
  :func:`_guard_asgi` re-applies the same Bearer check as the custom routes
  before merging them; without it a token-protected server would expose
  ``/sse`` and ``/messages`` unauthenticated.
"""

from __future__ import annotations

import asyncio
import errno
import hmac
import logging
import mimetypes
import sys
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from guaipeca import __version__
from guaipeca.mcp_server import _is_non_loopback_host, _origin_allowed

from .tools import build_server

if TYPE_CHECKING:  # pragma: no cover - typing only

    from guaipeca.config import GuaipecaConfig
    from guaipeca.mcp_server import GuaipecaMCPServer

__all__ = [
    "DEFAULT_MAX_BODY_SIZE",
    "TransportSecuritySettings",
    "_HmacTokenVerifier",
    "_allowed_host_patterns",
    "_allowed_origin_patterns",
    # Private by naming convention, listed here because the test suite imports
    # them directly to exercise the Bearer comparison in isolation (#43).
    "_check_bearer",
    "_security_settings",
    "build_app",
    "serve",
]

logger = logging.getLogger(__name__)

# Default request-body cap, mirrors config.ServerConfig.max_body_size (70MB).
# Used only when a duck-typed config omits the attribute.
DEFAULT_MAX_BODY_SIZE = 73400320

# Origin patterns understood by the SDK's TransportSecurityMiddleware: an entry
# ending in ``:*`` matches ``<prefix><any port>``. A bare loopback host with no
# port is therefore not covered by the port patterns, so it is listed too.
_LOOPBACK_ORIGIN_PATTERNS = (
    "http://localhost:*",
    "https://localhost:*",
    "http://127.0.0.1:*",
    "https://127.0.0.1:*",
    "http://[::1]:*",
    "https://[::1]:*",
    # Port-less loopback origins: the port pattern would otherwise miss
    # ``Origin: http://localhost`` (browsers omit the default port).
    "http://localhost",
    "https://localhost",
    "http://127.0.0.1",
    "https://127.0.0.1",
    "http://[::1]",
    "https://[::1]",
)

# Host patterns for the loopback default. The ``:*`` arm matches any port; the
# bare form is the Host header a client sends for a default-port origin.
_LOOPBACK_HOST_PATTERNS = (
    "localhost:*",
    "localhost",
    "127.0.0.1:*",
    "127.0.0.1",
    "[::1]:*",
    "[::1]",
)


def _allowed_origin_patterns(allowed_origins: list[str] | None) -> list[str]:
    """Translate ``server.allowed_origins`` into the SDK's match patterns.

    ``None`` keeps today's loopback default (see ``_origin_allowed``); a
    non-empty list is an exact-match allowlist and also replaces the loopback
    default, exactly as ``_origin_allowed`` does.

    Raises:
        ValueError: when an entry contains ``*``: matching is exact string
            comparison, so a wildcard would never match and the
            misconfiguration would be silent (#37).
    """
    if not allowed_origins:
        return list(_LOOPBACK_ORIGIN_PATTERNS)
    # The old policy ignored a trailing "/". ``_origin_allowed`` compares raw
    # strings, so normalise here.
    patterns: list[str] = []
    for entry in allowed_origins:
        normalised = entry.strip().rstrip("/")
        if not normalised:
            # #79: an entry that normalises to empty is a configuration
            # mistake; name it instead of dropping it silently.
            logger.warning(f"ignoring empty server.allowed_origins entry {entry!r}")
            continue
        if "*" in normalised:
            raise ValueError(
                f"server.allowed_origins entry {entry!r} contains '*'. Origin "
                f"matching is exact, so a wildcard would never match; list each "
                f"origin explicitly (including any port)."
            )
        patterns.append(normalised)
    return patterns


# #77: a serialised origin is scheme://host[:port]; a browser emits at most a
# few hundred bytes, so 2048 is a generous cap for anything legitimate.
_MAX_ORIGIN_BYTES = 2048


def _origin_is_well_formed(origin: str) -> bool:
    """Whether an Origin value could have come from a browser (#77).

    Control bytes below 0x21 (and DEL, 0x7F) would let a hand-built request
    smuggle header bytes into an echoed ACAO, and a serialised origin never
    approaches 2048 bytes. The header arrived latin-1-decoded (one character
    per wire byte), so ``ord`` and ``len`` are byte checks.
    """
    if len(origin) > _MAX_ORIGIN_BYTES:
        return False
    return not any(ord(char) < 0x21 or ord(char) == 0x7F for char in origin)


def _matched_origin(origin: str, allowed_origins: list[str] | None) -> str | None:
    """The allowlist entry ``origin`` matched, normalised to no trailing "/".

    Returns None when there is no allowlist (the loopback default has no entry
    to echo) or nothing matched. The normalisation mirrors ``_origin_allowed``
    exactly, so an origin the policy allowed always yields its matched entry
    here (#76).
    """
    if not allowed_origins:
        return None
    normalised = origin.rstrip("/")
    for entry in allowed_origins:
        if normalised == entry.rstrip("/"):
            return entry.rstrip("/")
    return None


def _allowed_host_patterns(host: str, allowed_origins: list[str] | None) -> list[str]:
    """Host-header patterns paired with :func:`_allowed_origin_patterns`.

    The Host header names the address the client connected to, so with the
    loopback default it must be loopback *plus the bind host* (``0.0.0.0`` and
    friends have no Host form of their own). An explicit origin allowlist
    means the operator fronts the server with a real hostname, so that
    hostname's hosts are allowed alongside loopback.
    """
    patterns = list(_LOOPBACK_HOST_PATTERNS)
    bind = (host or "").strip()
    if bind and not bind.startswith(("0.0.0.0", "::")) and bind not in {
        "localhost",
        "127.0.0.1",
        "[::1]",
    }:
        # Non-wildcard bind (a LAN address): clients address it directly.
        patterns.append(bind)
        patterns.append(f"{bind}:*")
    if allowed_origins:
        for origin in allowed_origins:
            try:
                hostname = urllib.parse.urlparse(origin).hostname
            except ValueError:
                continue
            if hostname:
                patterns.append(hostname)
                patterns.append(f"{hostname}:*")
    # dict.fromkeys: de-duplicate while keeping a deterministic order.
    return list(dict.fromkeys(patterns))


def _security_settings(
    host: str, allowed_origins: list[str] | None
) -> TransportSecuritySettings:
    """Build the DNS-rebinding settings from the existing origin policy."""
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_origins=_allowed_origin_patterns(allowed_origins),
        allowed_hosts=_allowed_host_patterns(host, allowed_origins),
    )


class _HmacTokenVerifier:
    """``TokenVerifier`` protocol for a single shared Bearer secret.

    The SDK's OAuth-shaped ``AuthSettings`` is not what this server has: the
    protocol is one method, and ``verify_token`` compares the presented Bearer
    token against the single shared secret with ``hmac.compare_digest``. There
    is no issuer, no audience and no scope list to consult.
    """

    def __init__(self, expected_token: str) -> None:
        self._expected_token = expected_token
        # Encode once here rather than per request. The auth-token environment
        # variable is read from the environment, and Python decodes env vars
        # with ``surrogateescape``, so a value set under a non-UTF-8 locale can
        # contain lone surrogates that ``encode("utf-8")`` refuses -- encoding
        # lazily in the request path would turn that into a 500 on every
        # authenticated route, the exact failure mode this module exists to
        # prevent. A secret that cannot be encoded can never match a token on
        # the wire, so record ``None`` and fail closed.
        try:
            self._expected_bytes: bytes | None = expected_token.encode("utf-8")
        except UnicodeEncodeError:
            self._expected_bytes = None

    async def verify_token(self, token: str) -> AccessToken | None:
        if self._expected_bytes is None:
            return None
        # ``hmac.compare_digest`` only accepts ASCII *str*; a non-ASCII token
        # (e.g. a UTF-8 Bearer secret sent on the wire) would raise
        # ``TypeError`` and surface as a 500. Compare bytes instead: the
        # compare stays constant-time, and a bad token now fails closed
        # with ``None`` instead of crashing. Never swap this for ``==``.
        #
        # The token arrives latin-1-decoded (starlette decodes raw header
        # bytes with ``decode("latin-1")``), so encode it back with latin-1
        # to recover the wire bytes; encoding it as UTF-8 would double-encode
        # and reject a legitimately configured non-ASCII secret.
        # ``surrogateescape`` keeps a token that cannot round-trip through
        # latin-1 from raising -- it simply fails to match.
        if not hmac.compare_digest(
            token.encode("latin-1", "surrogateescape"),
            self._expected_bytes,
        ):
            return None
        return AccessToken(token=token, client_id="guaipeca", scopes=[])


# ---- Non-MCP route handlers -------------------------------------------------


def _unauthorized() -> JSONResponse:
    """Body the hand-rolled server sent from ``_send_json(401, {...})``."""
    return JSONResponse({"error": "unauthorized"}, status_code=401)


def _check_bearer(request: Any, auth_token: str | None) -> bool:
    """Bearer check for the non-MCP routes (``/health``, ``/tools``, ...).

    The SDK guards ``/mcp`` itself through the token verifier; these routes
    bypass that wrapper, so they repeat the constant-time comparison.
    """
    if not auth_token:
        return True
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return False
    # ``hmac.compare_digest`` rejects non-ASCII *str* with ``TypeError`` (a
    # malformed/foreign Bearer token used to turn a 401 into a 500 here).
    # Compare bytes so the compare is still constant-time while a non-ASCII
    # token fails closed with ``False``. Do not swap for ``==``.
    #
    # starlette decoded the header as latin-1, so encode the presented value
    # back with latin-1 to recover the exact wire bytes (starlette's decode is
    # lossless, so this round-trips). The secret side is UTF-8: that is what a
    # client sends on the wire. Encoding the header with UTF-8 here would
    # double-encode and reject a correct non-ASCII secret.
    #
    # The secret is encoded per call rather than cached because this is a plain
    # function with no constructor. The environment variable it comes from is
    # decoded by Python with ``surrogateescape``, so a value set under a
    # non-UTF-8 locale can hold lone surrogates that ``encode`` refuses; fail
    # closed (False) instead of raising a 500 on every route.
    try:
        expected = auth_token.encode("utf-8")
    except UnicodeEncodeError:
        return False
    # ``surrogateescape`` here rescues lone surrogates only - the ones a token
    # set under a non-UTF-8 locale can carry. It does NOT make an arbitrary
    # non-latin-1 character safe: "don\u2019t".encode("latin-1",
    # "surrogateescape") raises. In practice the presented side never exceeds
    # U+00FF, because Starlette decodes header bytes as latin-1, so this cannot
    # raise from a real request - but the protection is the decode, not this
    # error handler.
    #
    # Residual, documented rather than "fixed" (#43): ``compare_digest`` returns
    # early when the lengths differ, so timing could in principle distinguish a
    # wrong-length token from a wrong-value one. Measured here at ~25ns
    # (0.110us equal-length vs 0.084us empty, ~1/400,000th of a LAN RTT), masked
    # by network jitter.
    #
    # Hashing both sides to a fixed 32 bytes WOULD close that gate - two SHA-256
    # digests are always the same length, so the early return never fires. It
    # costs one digest per request. Deliberately not done here: the measured
    # magnitude does not justify it, and this should fold into a change that
    # rewrites the comparison wholesale rather than sitting alone.
    return hmac.compare_digest(
        header[7:].encode("latin-1", "surrogateescape"),
        expected,
    )


# #73/#75: the methods each route actually serves, keyed by the path a request
# arrives with. The preflight must advertise only these (#73), and a path that
# is not in this table is unknown to the app, so its preflight must not claim
# a 204 the real request would 404 (#75). One table feeds both so the two
# checks cannot disagree.
#
# Derived from the route table in build_app() and the SDK's own routes: /mcp is
# the SDK's Streamable HTTP route (POST, plus GET/DELETE for session teardown),
# /mcp/ is the explicit redirect route, /sse is the legacy SSE stream (GET),
# /messages is the SSE message Mount (POST), and /download/... and /messages/...
# are prefix matches because the download filename keeps embedded slashes
# (``:path`` converter) and the Mount answers under its trailing-slash prefix.
_ROUTE_METHODS: dict[str, str] = {
    "/health": "GET, OPTIONS",
    "/tools": "GET, OPTIONS",
    "/documents": "DELETE, OPTIONS",
    "/mcp": "POST, GET, DELETE, OPTIONS",
    "/mcp/": "POST, GET, DELETE, OPTIONS",
    "/sse": "GET, OPTIONS",
    "/messages": "POST, OPTIONS",
}


def _allowed_methods_for_path(path: str) -> str | None:
    """The methods the app serves at ``path``, or None for an unknown path.

    See :data:`_ROUTE_METHODS` for the source of the table. ``/download/`` and
    ``/messages/`` are prefix matches: the download filename keeps embedded
    slashes and the SSE message Mount answers under its trailing-slash prefix.
    """
    if path.startswith("/download/"):
        return "GET, OPTIONS"
    if path.startswith("/messages/"):
        return "POST, OPTIONS"
    return _ROUTE_METHODS.get(path)


class _CorsMiddleware:
    """Add ``Access-Control-Allow-Origin`` and answer CORS preflights.

    The SDK validates the Origin header (DNS-rebinding protection) but does not
    emit CORS response headers, so without this a browser client passes the
    origin check and then has its response withheld.

    Origin policy: a non-empty Origin that fails ``_origin_allowed`` is answered
    with 403 here, before any route runs. A request with no Origin gets ``*``
    when no auth token is configured and ``null`` when one is; a validated
    Origin is echoed back - with an allowlist, the entry that matched
    (normalised, no trailing "/"), not the raw request text (issue #76).

    A lone empty ``Origin`` header is filtered as absent: it carries no origin
    to match, so the no-Origin policy above applies (issue #79). An Origin
    containing control bytes or longer than 2048 bytes is answered 400 before
    matching or echoing (issue #77, see ``_origin_is_well_formed``).

    Every ``Origin`` header must match: a request carrying two, one allowed and
    one not, is rejected (issue #39). Browsers send at most one, but a proxy or
    a hand-built request can send several, and the control is all-must-match.

    Preflights are answered here with ``204`` (issue #36). The routes never
    handle ``OPTIONS``: without this they answered ``405`` carrying an ACAO
    header, which reads like working CORS while the browser still aborts. The
    advertised methods are the ones the path actually serves (issue #73, see
    ``_allowed_methods_for_path``), and a preflight for a path the app does not
    serve is delegated to the inner app so it gets the same 404 the real
    request would (issue #75).
    """

    def __init__(self, app: Any, auth_token: str | None, allowed_origins: list[str] | None) -> None:
        self.app = app
        self.auth_token = auth_token
        self.allowed_origins = allowed_origins

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # #39: read EVERY Origin header. Reading only the first made the check
        # first-must-match; the policy is all-must-match, and a disallowed
        # second header used to be invisible.
        origins = [
            value.decode("latin-1").strip()
            for key, value in scope.get("headers", [])
            if key == b"origin"
        ]
        origins = [value for value in origins if value]

        for candidate in origins:
            # #77: reject origins no browser could have produced BEFORE
            # matching or echoing (see _origin_is_well_formed). The check
            # lives in this same loop so the all-must-match pass covers
            # every Origin header in one pass.
            if not _origin_is_well_formed(candidate):
                response = JSONResponse({"error": "invalid origin"}, status_code=400)
                await response(scope, receive, send)
                return
            if not _origin_allowed(candidate, self.allowed_origins):
                # The SDK's transport security would reject this too; answering
                # here keeps the 403 and stops a disallowed origin reaching any
                # route.
                response = JSONResponse({"error": "origin not allowed"}, status_code=403)
                await response(scope, receive, send)
                return

        origin = origins[0] if origins else ""
        # #76: with an allowlist, echo the entry that matched (normalised,
        # no trailing "/"), not the raw request origin - otherwise a trailing
        # "/" in the request would survive into ACAO. Without an allowlist
        # there is no entry to echo, so the request origin is echoed as
        # before; an absent origin keeps the "*" / "null" rule.
        if origin:
            matched = _matched_origin(origin, self.allowed_origins)
            allow = matched if matched is not None else origin
        else:
            allow = "*" if not self.auth_token else "null"

        async def send_with_cors(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"access-control-allow-origin", allow.encode("latin-1")))
                if origin:
                    # #78: combine with a Vary the inner app already sent
                    # instead of appending a second header: the browser treats
                    # them as one union, but two headers where one suffices is
                    # a bug, and the inner app's value must not be dropped.
                    existing = [
                        part.strip()
                        for key, value in headers
                        if key == b"vary"
                        for part in value.decode("latin-1").split(",")
                    ]
                    # "*" already means "vary on everything" — don't append Origin.
                    if "*" not in existing and not any(part.lower() == "origin" for part in existing):
                        existing.append("Origin")
                    headers = [(key, value) for key, value in headers if key != b"vary"]
                    headers.append((b"vary", ", ".join(existing).encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        if scope["method"] == "OPTIONS":
            # #36: answer the preflight. Every Origin has already been vetted
            # above, so this request is from an allowed (or absent) origin.
            # #75: a path the app does not serve must not get a blanket 204 -
            # delegate to the inner app so the preflight gets the same 404
            # the real request would.
            methods = _allowed_methods_for_path(scope["path"])
            if methods is None:
                await self.app(scope, receive, send_with_cors)
                return
            # #73: advertise only the methods the path actually serves; the
            # old hardcoded list invited requests that would 405.
            response = Response(
                status_code=204,
                headers={
                    "access-control-allow-origin": allow,
                    "access-control-allow-methods": methods,
                    "access-control-allow-headers": (
                        "Authorization, Content-Type, Mcp-Session-Id"
                    ),
                    "access-control-max-age": "600",
                    **({"vary": "Origin"} if origin else {}),
                },
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send_with_cors)


def _server_config(config: GuaipecaConfig) -> Any:
    """``config.server`` if present, else None.

    No validation happens here: a duck-typed test config with a partial
    ``server`` section is returned as-is, and each caller applies its own
    default for a missing attribute (``auth_token``, ``max_body_size``, ...).
    An object without a ``server`` section means "no server settings", so the
    documented defaults apply.
    """
    return getattr(config, "server", None)


def _build_sse_app(
    lowlevel_server: Any,
    *,
    max_body_size: int,
    allowed_origins: list[str] | None,
    host: str,
) -> Any:
    """Build the legacy SSE Starlette app around an existing low-level server.

    ``MCPServer.sse_app()`` is the SDK's ready-made transport: on ``GET /sse``
    it opens an SSE stream and emits an ``endpoint`` event pointing at
    ``POST /messages?session_id=...``. It drives whatever server sits on
    ``_lowlevel_server``, and the SDK exposes no public way to swap that — so
    the assignment below is deliberate and is the only way to keep our
    existing tool registry (:func:`guaipeca.mcp_sdk.tools.build_server`).
    ``tests/test_mcp_sdk_tools.py::test_mcpserver_still_exposes_lowlevel_attribute``
    guards the private attribute so an SDK upgrade fails loudly instead of
    silently losing ``/sse``.

    ``message_path`` is ``/messages/`` *with* the trailing slash: the SDK routes
    it as a Starlette ``Mount``, so ``/messages`` answers with a 307 and most
    SSE clients drop the request body rather than follow a redirect on POST.
    See :func:`_build_sse_app`'s inline comment and
    ``test_sse_endpoint_does_not_redirect_post``.

    The returned app carries no Bearer check of its own; ``build_app`` passes it
    through :func:`_guard_asgi` before merging its routes.
    """
    from mcp.server.mcpserver.server import MCPServer

    wrapper = MCPServer("guaipeca")
    # Intentional: reuse the Server built by build_server(), with Guaipeca's
    # tools, instead of a fresh one with only the SDK's defaults.
    #
    # Check BEFORE assigning. ``MCPServer.__init__`` sets this attribute itself,
    # and the class has no ``__slots__``, so on a release that renamed it our
    # assignment would silently create a new unused attribute: ``sse_app()``
    # would drive the SDK's own default server and /sse would serve ZERO tools,
    # with no error at startup or at request time. Testing for the attribute's
    # existence up front catches a rename whether or not assignment succeeds.
    if not hasattr(wrapper, "_lowlevel_server"):
        raise RuntimeError(
            "MCPServer no longer sets '_lowlevel_server' in __init__, so the "
            "SDK's private hook has been renamed or removed. Guaipeca's SSE "
            "transport would serve an empty tool list rather than fail. Update "
            "_build_sse_app() to use the SDK's new hook (see issue #25); do not "
            "drop this check."
        )
    wrapper._lowlevel_server = lowlevel_server
    return wrapper.sse_app(
        sse_path="/sse",
        # "/messages/" with the trailing slash, deliberately. The SDK routes this
        # as a Mount, so the canonical path is "/messages/"; advertising
        # "/messages" makes Starlette's redirect_slashes answer a legacy
        # client's POST with a 307, which most SSE clients do not follow for a
        # POST (the request body is dropped and the session never reaches the
        # transport). The old hand-rolled server accepted both.
        message_path="/messages/",
        max_request_body_size=max_body_size,
        transport_security=_security_settings(host, allowed_origins),
        host=host,
    )


def _guard_asgi(
    endpoint: Any, auth_token: str | None, *, deprecation_headers: bool = False
) -> Any:
    """Require our Bearer token in front of a single ASGI callable.

    ``sse_app()`` is built by the SDK outside ``build_app``'s auth
    configuration (that only covers ``/mcp``), so without this ``/sse`` and
    ``/messages`` would be the one unauthenticated way in when a token is set.
    A request without a valid token is answered with our usual 401 body before
    the SSE transport sees it. No token configured means no wrapper, matching
    the other custom routes.

    ``deprecation_headers`` marks the response as deprecated and points at the
    preferred endpoint (RFC 8594 ``Deprecation`` + RFC 8288 ``Link``). A legacy
    SSE client never reads the MCP ``initialize`` instructions, so this response
    is the only place it can be told. Headers are added when the SSE stream
    starts, not on every POST, because that is where a client decides to keep
    the connection.
    """

    async def _send_deprecated(send: Any) -> Any:
        async def _wrapped(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if not any(k.lower() == b"deprecation" for k, _ in headers):
                    headers.append((b"deprecation", b"true"))
                    headers.append((b"link", b'</mcp>; rel="alternate"; type="application/json"'))
                    message = {**message, "headers": headers}
            await send(message)

        return _wrapped

    if not auth_token:
        if not deprecation_headers:
            return endpoint

        async def _plain(scope: Any, receive: Any, send: Any) -> None:
            await endpoint(scope, receive, await _send_deprecated(send))

        return _plain

    async def _guarded(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await endpoint(scope, receive, send)
            return
        request = Request(scope, receive)
        if not _check_bearer(request, auth_token):
            await _unauthorized()(scope, receive, send)
            return
        if deprecation_headers:
            send = await _send_deprecated(send)
        await endpoint(scope, receive, send)

    return _guarded


def _health(server_instance: GuaipecaMCPServer):
    async def _handler(request: Any) -> JSONResponse:
        if not _check_bearer(request, getattr(_server_config(server_instance.config), "auth_token", None)):
            return _unauthorized()
        return JSONResponse({"status": "ok", "server": "guaipeca", "version": __version__})

    return _handler


def _tools(server_instance: GuaipecaMCPServer):
    async def _handler(request: Any) -> JSONResponse:
        auth_token = getattr(_server_config(server_instance.config), "auth_token", None)
        if not _check_bearer(request, auth_token):
            return _unauthorized()
        # Same list the SDK advertises: _tools_with_corpus_enum() is what
        # build_server()'s tools/list handler returns.
        return JSONResponse(
            {
                "tools": server_instance._tools_with_corpus_enum(),
                # Transport facts for a human or a client that probes here
                # before connecting. The same preference is also in the MCP
                # initialize `instructions` for clients that never read this.
                "transports": {
                    "preferred": "streamable-http",
                    "endpoints": {
                        "streamable-http": "/mcp",
                        "sse": "/sse",
                    },
                    "note": (
                        "Use POST /mcp. It serves both protocol eras on one "
                        "endpoint. The legacy SSE transport is served for "
                        "compatibility and is deprecated upstream."
                    ),
                },
            }
        )

    return _handler


def _download(server_instance: GuaipecaMCPServer, auth_token: str | None):
    """``GET /download/{corpus}/{filename}`` with the same checks and errors."""

    async def _handler(request: Any) -> Any:
        if not _check_bearer(request, auth_token):
            return _unauthorized()

        # Starlette decodes the path; ``:path`` keeps embedded slashes so the
        # traversal checks below see the same string the old handler did.
        corpus_name = request.path_params["corpus"]
        filename = request.path_params["filename"]

        if ".." in filename or filename.startswith("/"):
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        if ".." in corpus_name or corpus_name.startswith("/"):
            return JSONResponse({"error": "invalid corpus"}, status_code=400)

        corpus_cfg = server_instance.config.get_corpus(corpus_name)
        if corpus_cfg is None:
            return JSONResponse({"error": f"corpus not found: {corpus_name}"}, status_code=404)

        corpus_path = Path(corpus_cfg.path)
        file_path = corpus_path / filename
        try:
            file_path.resolve().relative_to(corpus_path.resolve())
        except ValueError:
            return JSONResponse({"error": "path outside corpus directory"}, status_code=400)

        if not file_path.exists() or not file_path.is_file():
            return JSONResponse({"error": f"file not found: {filename}"}, status_code=404)

        content_type, _ = mimetypes.guess_type(str(file_path))
        return FileResponse(
            file_path,
            media_type=content_type or "application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return _handler


def _delete_documents(server_instance: GuaipecaMCPServer, auth_token: str | None):
    """``DELETE /documents?corpus=&filename=`` — delegates to ``_handle_delete``."""

    async def _handler(request: Any) -> JSONResponse:
        if not _check_bearer(request, auth_token):
            return _unauthorized()

        corpus_name = request.query_params.get("corpus")
        filename = request.query_params.get("filename")
        if not corpus_name or not filename:
            return JSONResponse({"error": "corpus and filename required"}, status_code=400)

        # delete_upload() does file I/O and may re-index the corpus (a search
        # write lock, CPU-bound embedding). Keep it off the event loop.
        result = await asyncio.to_thread(
            server_instance._handle_delete, corpus_name, filename, True
        )
        status = 400 if result.get("error") else 200
        return JSONResponse(result, status_code=status)

    return _handler


def _mcp_trailing_slash(request: Any) -> RedirectResponse:
    """``/mcp/`` -> ``/mcp`` (308, method-preserving)."""
    return RedirectResponse("/mcp", status_code=308)


# Set once the unauthenticated-loopback warning has been emitted, so the two
# call sites (cmd_serve and build_app) do not each print it.
_BIND_ANNOUNCED = False


def _check_bind_exposure(bind_host: str, auth_token: str | None) -> None:
    """Refuse to start a public bind that has no authentication.

    A bind address that is not loopback-only makes every route reachable by any
    client that can route to this host: ``/download`` serves stored documents
    and the ``upload``/``delete`` tools write and remove files. A log warning
    is not a control on a machine reachable from more than one network, so the
    combination is refused outright rather than advised against.

    There is deliberately no override flag. An operator who wants the service
    reachable sets an ``auth_token``; one who does not want auth keeps the bind
    on loopback and reaches it through a tunnel.

    A loopback bind with no token is a legitimate configuration (it is the
    default, and the sensible one on a single-user host), so it is allowed — but
    it is announced, because "this port needs no credentials" is something the
    operator should know rather than infer.

    Raises:
        ValueError: when ``bind_host`` is not loopback-only and no
            ``auth_token`` is configured.
    """
    _check_bind_exposure_and_announce(bind_host, auth_token, announce=True)


def _check_bind_exposure_and_announce(
    bind_host: str, auth_token: str | None, *, announce: bool
) -> None:
    """Body of :func:`_check_bind_exposure`, with announcement made optional.

    ``cmd_serve`` runs the check before constructing the server (so a refusal
    does not pay for the embedding model), and ``build_app`` runs it again as the
    last line of defence for callers that build the app directly. The
    *announcement* must happen once per process, so only the first caller asks
    for it; the refusal is raised by whichever runs.
    """
    if auth_token:
        return
    if not _is_non_loopback_host(bind_host):
        if announce:
            globals()["_BIND_ANNOUNCED"] = True
            _announce_unauthenticated_loopback(bind_host)
        return
    raise ValueError(
        f"refusing to serve an unauthenticated public bind: server.host={bind_host!r} "
        "is not loopback-only and no server.auth_token is configured.\n"
        "Every route would be reachable without authentication, including "
        "/download and the upload/delete tools.\n"
        "Set GUAIPECA_AUTH_TOKEN in the service environment (recommended - "
        "it keeps the secret out of the config file), or set server.auth_token "
        "in the config, to require auth. Or set server.host to 127.0.0.1 to "
        "serve this machine only."
    )


def _announce_unauthenticated_loopback(bind_host: str) -> None:
    """Announce that a loopback bind is serving without authentication.

    Not a refusal: loopback-only with no token is the default and is fine on a
    single-user host. It is stated on every start rather than once, and there is
    no way to silence it — the day someone widens the bind is the day this line
    stops being true, and a suppressed warning would not be there to say so.

    Emitted through the logger only. ``cli.main`` installs a stream handler, so
    the record already reaches stderr (where the operator and journald see it)
    with the timestamp and level; adding a second ``print`` would say it twice.
    """
    message = (
        f"serving without authentication on {bind_host!r} (loopback only): "
        "every route is open to anything running on this machine, including "
        "/download and the upload/delete tools. Set GUAIPECA_AUTH_TOKEN to require "
        "a Bearer token."
    )
    logger.warning(message)


def build_app(
    config: GuaipecaConfig, server_instance: GuaipecaMCPServer, *, host: str | None = None
) -> Any:
    """Assemble the HTTP app: SDK Streamable HTTP plus Guaipeca's plain endpoints.

    Returns the SDK app wrapped in :class:`_CorsMiddleware` (an ASGI callable),
    so the annotation is ``Any`` rather than ``Starlette``.

    ``host`` is the bind address, used to refuse an unauthenticated public
    bind and to add its Host header form to the allowlist. Omitted, the
    loopback defaults apply.

    Raises:
        ValueError: when ``host`` binds beyond loopback and no ``auth_token``
            is configured. See :func:`_check_bind_exposure`.
    """
    server_config = _server_config(config)
    auth_token = getattr(server_config, "auth_token", None)
    max_body_size = getattr(server_config, "max_body_size", DEFAULT_MAX_BODY_SIZE)
    allowed_origins = getattr(server_config, "allowed_origins", None)
    bind_host = host or getattr(server_config, "host", "127.0.0.1")

    # Announce only if nobody has yet: cmd_serve checks before constructing the
    # server, so in the normal serve path the warning has already been emitted
    # and this call is the enforcement backstop. Direct build_app() callers
    # (tests, embedders) still get the announcement.
    _check_bind_exposure_and_announce(
        bind_host, auth_token, announce=not _BIND_ANNOUNCED
    )
    globals()["_BIND_ANNOUNCED"] = True

    srv = build_server(server_instance)

    custom_routes = [
        Route("/health", _health(server_instance), methods=["GET"]),
        Route("/tools", _tools(server_instance), methods=["GET"]),
        # Starlette's router only redirects on the trailing-slash mismatch for
        # a route it already matched. Registered explicitly so POST /mcp/ does
        # not fall through to a 404 (and GET /mcp/ does not answer /mcp's 405).
        Route("/mcp/", _mcp_trailing_slash, methods=["GET", "POST", "DELETE"]),
        Route(
            "/download/{corpus}/{filename:path}",
            _download(server_instance, auth_token),
            methods=["GET"],
        ),
        Route("/documents", _delete_documents(server_instance, auth_token), methods=["DELETE"]),
    ]

    kwargs: dict[str, Any] = {
        "streamable_http_path": "/mcp",
        # Inline JSON responses (the hand-rolled POST /mcp did not stream).
        # Response mode is deliberately not changed in this phase.
        "json_response": True,
        "max_request_body_size": max_body_size,
        "transport_security": _security_settings(bind_host, allowed_origins),
        "custom_starlette_routes": custom_routes,
    }
    if auth_token:
        # ``token_verifier`` alone is NOT enough: streamable_http_app() wraps the
        # route in RequireAuthMiddleware whenever a verifier is passed, but only
        # installs the BearerAuthBackend that can validate it when ``auth`` is
        # also set. Without an AuthSettings every request 401s regardless of
        # the token. Issuer/resource point at this server because Guaipeca is
        # its own resource server with a single shared secret, not an
        # OAuth-delegating one.
        base_url = f"http://{bind_host}:{getattr(server_config, 'port', 8090)}"
        kwargs["auth"] = AuthSettings(
            issuer_url=base_url,
            resource_server_url=base_url,
            required_scopes=None,
            # Our verifier does not set AccessToken.resource, and a
            # single-shared-secret server has no audience to enforce.
            validate_token_resource=False,
        )
        kwargs["token_verifier"] = _HmacTokenVerifier(auth_token)

    app = srv.streamable_http_app(**kwargs)

    # Legacy SSE transport. Its two routes are absolute (``/sse``,
    # ``/messages``), so they are appended to the SDK app's route table rather
    # than mounted under a prefix; wrapping the whole thing in
    # _CorsMiddleware() then applies CORS to them as well. The routes share one
    # wrapper around the SSE app, which re-applies our Bearer check because the
    # SDK built that app without our auth configuration (see _guard_asgi).
    sse_app = _build_sse_app(
        srv,
        max_body_size=max_body_size,
        allowed_origins=allowed_origins,
        host=bind_host,
    )
    # Wrapped, not mounted: Mount strips the matched prefix from
    # ``scope["path"]``, so the SDK's inner routes (which expect the absolute
    # "/sse") would stop matching, and ``route.app`` is the endpoint itself, not
    # the Starlette. Guarding per route keeps each endpoint's own semantics and
    # applies the Bearer check that sse_app() does not put on these two paths.
    for route in getattr(sse_app, "routes", ()):
        # Only the GET stream is marked deprecated: that is where a client
        # chooses to keep the connection. /messages carries the per-request
        # POSTs of a session already opened here, and repeated headers on
        # those would say nothing new.
        is_stream = getattr(route, "path", None) == "/sse"
        route.app = _guard_asgi(route.app, auth_token, deprecation_headers=is_stream)
        app.routes.append(route)

    return _CorsMiddleware(app, auth_token, allowed_origins)


# ---- uvicorn run loop -------------------------------------------------------

_PORT_ATTEMPTS = 5


def _probe_bind(host: str, port: int) -> None:
    """Bind and immediately release ``(host, port)`` to test availability.

    Raises OSError (EADDRINUSE) when the address is taken. There is an
    inherent race between this probe and uvicorn's own bind; a second process
    grabbing the port in between surfaces as uvicorn's ``sys.exit(3)``.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))


def _pick_port(host: str, port: int) -> int:
    """First free port at ``port`` or the four above it."""
    actual_port = port
    for _ in range(_PORT_ATTEMPTS):
        try:
            _probe_bind(host, actual_port)
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
            logger.warning(f"Port {actual_port} in use, trying {actual_port + 1}")
            actual_port += 1
            continue
        return actual_port
    raise RuntimeError(f"Could not find available port starting at {port}")


def serve(
    config: GuaipecaConfig,
    server_instance: GuaipecaMCPServer,
    host: str = "127.0.0.1",
    port: int = 8090,
) -> None:
    """Run :func:`build_app` under uvicorn until SIGTERM/SIGINT.

    Port-retry replaces the old hand-rolled loop; uvicorn's own bind failure calls
    ``sys.exit(3)``, which a retry cannot intercept, so the port is chosen
    before uvicorn starts. Shutdown is uvicorn's graceful path; see the module
    docstring and the migration plan for the native-thread caveat.
    """
    import uvicorn

    actual_port = _pick_port(host, port)
    try:
        app = build_app(config, server_instance, host=host)
    except ValueError as exc:
        # #74: build_app validates the configuration (a '*' in
        # allowed_origins, an unauthenticated public bind, ...); a raw
        # traceback is not a startup message, so print the error and exit 1,
        # the same shape cmd_serve uses for _check_bind_exposure. Only
        # ValueError: any other exception is a real bug and must not be
        # swallowed.
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Starting Guaipeca MCP server (HTTP/Streamable) on {host}:{actual_port}")
    print(f"Guaipeca MCP server listening on http://{host}:{actual_port}", file=sys.stderr)

    uvicorn_server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=actual_port, log_level="info")
    )
    uvicorn_server.run()

    # uvicorn has exited after its graceful shutdown and closed the listener.
    # Native pools (ONNX/OpenMP in fastembed, OpenBLAS in numpy/faiss) park
    # threads with no stop API and are non-daemon; on Linux CPython does not
    # join them at interpreter shutdown, so falling off the end is expected to
    # be enough. Do NOT reintroduce os._exit(0): it hides a stuck shutdown
    # instead of fixing it. If the live SIGTERM check hangs, find the owning
    # thread first.
    logger.info("HTTP server stopped, exiting process")
