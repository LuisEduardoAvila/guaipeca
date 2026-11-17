# Changelog

Notable changes to Guaipeca. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are semver.

## [0.5.0] — MCP SDK migration

The MCP transport layer now runs on the official [`mcp`](https://pypi.org/project/mcp/)
Python SDK instead of a hand-written JSON-RPC/HTTP implementation.

### Breaking

- **The legacy SSE transport is deprecated, not removed.** `GET /sse` and
  `POST /messages` are still served, but `POST /mcp` (Streamable HTTP) is the
  preferred transport and upstream has deprecated SSE. The server says so
  wherever a client can see it: `initialize` instructions, the `/tools`
  response, and a `Deprecation: true` + `Link: </mcp>` header on the SSE
  stream. Expect removal in a future release.
- **`transport: "sse"` is rejected** by config validation. Valid values are
  `stdio`, `http`, and `both`.
- **`GET /mcp` / `DELETE /mcp` behaviour is now the SDK's**, not ours. Responses
  keep their shape, but the routing decisions (405, 404 on an unknown session)
  come from the SDK's session manager.
- **Protocol version negotiation errors are visible on the modern era.** A
  2026-07-28 request carrying an unsupported `protocolVersion` now gets JSON-RPC
  `-32022` ("Unsupported protocol version") listing the versions the server
  supports, instead of the previous silent downgrade. On the legacy handshake
  the server still selects a version it supports and returns 200, which is what
  that revision specifies.

### Added

- **Both protocol eras on one endpoint.** `POST /mcp` serves the handshake era
  (`2025-11-25` and earlier) and the stateless era (`2026-07-28`) from the same
  port, dispatching on how the client opens.
- **`server.max_body_size` is enforced by the SDK** and configured from your
  existing config key rather than a hardcoded constant.
- Origin / DNS-rebinding validation is now `TransportSecuritySettings`, driven by
  `server.allowed_origins` with the same policy as before (no `Origin` allowed;
  loopback only by default; an explicit allowlist replaces that).

### Changed

- **HTTP is served by uvicorn.** Graceful shutdown is uvicorn's; the
  `os._exit(0)` workaround is gone. Verified: `SIGTERM` exits in ~0.2s for both
  `http` and `both`, with stdin held open.
- `mcp_server.py` dropped ~730 lines of hand-rolled transport code (1578 → 850).
- Blocking retrieval runs on a worker thread via `anyio.to_thread`, so a long
  search does not stall the event loop. Guarded by a test that fails if any SDK
  handler is a bare blocking `def`.

### Fixed

- **CORS response headers.** The SDK validates `Origin` but emits no
  `Access-Control-Allow-Origin`, so a browser client would pass the origin check
  and then have its response withheld. A small ASGI middleware restores the
  previous behaviour (echo a validated origin; `*` without auth, `null` with).
- **Auth wiring.** `token_verifier` alone does not work with the SDK's
  `streamable_http_app`: it wraps the route in auth middleware without installing
  the backend that validates tokens, so every request 401s. Auth now passes both
  `auth=AuthSettings(...)` and `token_verifier`.

### Removed

- Hand-rolled `ThreadingHTTPServer` transport, session registry, and message
  queues.
- `tests/test_streamable_http.py` and `tests/test_origin_validation.py` as
  active suites. Their still-relevant assertions were **ported** to
  `tests/test_mcp_sdk_http.py`; the rest were dropped with a stated reason
  (`/sse` removal, uvicorn owning HTTP parsing). Both files remain, skipped, so
  the mapping is auditable rather than implied.

### Notes

- `both` still means "stdio when launched as a subprocess, HTTP when launched as
  a daemon" — it does not run two listeners, because the SDK runs one server per
  transport.
- The embedding model is still warmed eagerly at construction; see issue #24 for
  lazy warmup.
