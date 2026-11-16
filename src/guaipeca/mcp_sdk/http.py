"""HTTP transport for the SDK-backed MCP server (Phase 4 of the migration).

Serves one Starlette app under uvicorn::

    /mcp                    Streamable HTTP, both protocol eras
    /health                 liveness probe (unauthenticated, as before)
    /tools                  tool listing (convenience endpoint)
    /download/{c}/{f}       file download
    DELETE /documents       delete an uploaded file
    /sse, /messages         410 with an explanation (legacy transport removed)

The SDK owns everything MCP-shaped — session handling, protocol-version
routing (handshake vs 2026-07-28 stateless requests), JSON parsing, the
request-body limit and DNS-rebinding protection. This module wires the
existing ``GuaipecaMCPServer`` into it and re-implements only what the SDK has
no equivalent for.

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
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
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
    "_allowed_host_patterns",
    "_allowed_origin_patterns",
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
    """
    if not allowed_origins:
        return list(_LOOPBACK_ORIGIN_PATTERNS)
    # The old policy ignored a trailing "/". The SDK compares raw strings, so
    # normalise here; neither form may contain "*", which the SDK would treat
    # as a port wildcard.
    return [origin.rstrip("/") for origin in allowed_origins if origin.rstrip("/")]


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

    The SDK's OAuth-shaped ``AuthSettings`` is not what this server has; the
    protocol is one method and the whole check is a constant-time compare, the
    same comparison ``run_http``'s ``_check_auth`` performed.
    """

    def __init__(self, expected_token: str) -> None:
        self._expected_token = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if not hmac.compare_digest(token, self._expected_token):
            return None
        return AccessToken(token=token, client_id="guaipeca", scopes=[])


# ---- Non-MCP route handlers -------------------------------------------------


def _unauthorized() -> JSONResponse:
    """Body the hand-rolled server sent from ``_send_json(401, {...})``."""
    return JSONResponse({"error": "unauthorized"}, status_code=401)


def _check_bearer(request: Any, auth_token: str | None) -> bool:
    """Same check as ``run_http``'s ``_check_auth`` for the non-MCP routes.

    The SDK guards ``/mcp`` itself through the token verifier; these routes
    bypass that wrapper, so they repeat the constant-time comparison.
    """
    if not auth_token:
        return True
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[7:], auth_token)


def _cors_origin(request: Any, auth_token: str | None, allowed_origins: list[str] | None) -> str:
    """Access-Control-Allow-Origin value, matching ``_cors_value``.

    Echo a validated request Origin; otherwise keep the legacy value
    ('*' without auth, 'null' with auth).
    """
    origin = request.headers.get("origin", "").strip()
    if origin and _origin_allowed(origin, allowed_origins):
        return origin
    return "*" if not auth_token else "null"


class _CorsMiddleware:
    """Add ``Access-Control-Allow-Origin`` and answer CORS preflights.

    The SDK validates the Origin header (DNS-rebinding protection) but does not
    emit CORS response headers, so without this a browser client passes the
    origin check and then has its response withheld. Pre-PR-23 behaviour:
    echo a validated Origin, else '*' without auth and 'null' with.
    """

    def __init__(self, app: Any, auth_token: str | None, allowed_origins: list[str] | None) -> None:
        self.app = app
        self.auth_token = auth_token
        self.allowed_origins = allowed_origins

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        origin = ""
        for key, value in scope.get("headers", []):
            if key == b"origin":
                origin = value.decode("latin-1").strip()
                break

        if origin and not _origin_allowed(origin, self.allowed_origins):
            # The SDK's transport security would reject this too; answering here
            # keeps the 403 and stops a disallowed origin reaching any route.
            response = JSONResponse({"error": "origin not allowed"}, status_code=403)
            await response(scope, receive, send)
            return

        allow = origin if origin else ("*" if not self.auth_token else "null")

        async def send_with_cors(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"access-control-allow-origin", allow.encode("latin-1")))
                if origin:
                    headers.append((b"vary", b"Origin"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_cors)


def _server_config(config: GuaipecaConfig) -> Any:
    """``config.server`` if it looks usable, else None.

    ``server_instance.config`` may be duck-typed in tests; anything without a
    ``server`` section is treated as "no server settings" and the documented
    defaults apply.
    """
    return getattr(config, "server", None)


def _legacy_removed():
    """Handler for the retired ``/sse`` and ``/messages`` endpoints.

    Returns 410 Gone, not 404: the routes were removed on purpose, and a stale
    SSE client should get a body it can act on rather than a silent hang.
    """

    async def _handler(request: Any) -> JSONResponse:
        return JSONResponse(
            {
                "error": "legacy SSE transport removed",
                "message": (
                    "The legacy SSE transport (GET /sse + POST /messages) was removed in the "
                    "MCP SDK migration. Use POST /mcp instead: it serves both the session-based "
                    "handshake and the 2026-07-28 stateless protocol."
                ),
                "use": "/mcp",
                "protocol_versions": ["2025-03-26", "2025-06-18", "2025-11-25", "2026-07-28"],
            },
            status_code=410,
        )

    return _handler


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
        return JSONResponse({"tools": server_instance._tools_with_corpus_enum()})

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


def build_app(
    config: GuaipecaConfig, server_instance: GuaipecaMCPServer, *, host: str | None = None
) -> Any:
    """Assemble the HTTP app: SDK Streamable HTTP plus Guaipeca's plain endpoints.

    Returns the SDK app wrapped in :class:`_CorsMiddleware` (an ASGI callable),
    so the annotation is ``Any`` rather than ``Starlette``.

    ``host`` is the bind address, used to warn about an unauthenticated public
    bind and to add its Host header form to the allowlist. Omitted, the
    loopback defaults apply.
    """
    server_config = _server_config(config)
    auth_token = getattr(server_config, "auth_token", None)
    max_body_size = getattr(server_config, "max_body_size", DEFAULT_MAX_BODY_SIZE)
    allowed_origins = getattr(server_config, "allowed_origins", None)
    bind_host = host or getattr(server_config, "host", "127.0.0.1")

    if not auth_token and _is_non_loopback_host(bind_host):
        logger.warning(
            "=" * 60 + "\n"
            f"⚠️  UNAUTHENTICATED PUBLIC BIND: host={bind_host!r}\n"
            "   No auth_token is configured, so ALL endpoints are\n"
            "   reachable WITHOUT authentication from any client that\n"
            "   can reach this address.\n"
            "   To require auth: set server.auth_token in the config.\n"
            "   To restrict to this machine: bind host=127.0.0.1.\n"
            + "=" * 60
        )

    srv = build_server(config, server_instance)

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
        Route("/sse", _legacy_removed(), methods=["GET", "POST", "DELETE"]),
        Route("/messages", _legacy_removed(), methods=["GET", "POST", "DELETE"]),
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
    """First free port at ``port`` or the four above it (``run_http`` parity)."""
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

    Port-retry replaces ``run_http``'s loop; uvicorn's own bind failure calls
    ``sys.exit(3)``, which a retry cannot intercept, so the port is chosen
    before uvicorn starts. Shutdown is uvicorn's graceful path; see the module
    docstring and the migration plan for the native-thread caveat.
    """
    import uvicorn

    actual_port = _pick_port(host, port)
    app = build_app(config, server_instance, host=host)

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
