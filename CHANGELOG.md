# Changelog

Notable changes to Guaipeca. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are semver.

## [Unreleased]

### Changed

- **Malformed or empty uploads are now refused instead of silently saved.**
  The MCP `upload` tool decodes its base64 payload strictly: content with
  characters outside the base64 alphabet is rejected as invalid base64
  instead of being silently truncated, and empty or whitespace-only content
  is refused by the shared upload core (so `guaipeca upload` refuses empty
  and whitespace-only files the same way). A refused upload writes nothing —
  no file, no temp file, no manifest entry — so the delete-then-upload
  workflow can no longer replace a real document with an empty one.
- **Overwriting and deleting files the server did not upload is now refused.**
  The `upload` and `delete` tools (and the `upload`/`delete`/
  `DELETE /documents` endpoints) track every server-uploaded file in a
  per-corpus manifest (`uploaded.json` in the corpus directory). Uploading
  over an existing filename is now refused unless the server uploaded that
  file itself or the corpus sets `upload.replace: {<corpus>: true}` in the
  config; deleting a file is refused unless it is tracked in the manifest,
  has an allowed extension, and is not a reserved name (`uploaded.json`,
  `.guaipeca*`, `*.tmp`). Deleting a tracked file whose content was edited
  outside the API is refused as drift. The delete-then-upload cycle still
  works: deleting removes the manifest entry, freeing the name for the
  replacement. Two new commands/tools inspect and repair the manifest:
  `guaipeca uploads-status --corpus C [--json]` (MCP `uploads_status`)
  reports tracked, drifted, orphan and missing files, and
  `guaipeca uploads-prune --corpus C [--apply] [--json]` (MCP
  `uploads_prune`) removes stale manifest entries (dry run without
  `--apply`; never deletes files).
- **A public bind without auth is refused, not warned about.** Binding
  `server.host` to anything other than loopback (`0.0.0.0`, `::`, a LAN address,
  a hostname) while no `server.auth_token` is configured now raises at startup
  instead of logging a warning and serving. Every route was reachable — read
  via `/download`, write and delete via the `upload`/`delete` tools — by any
  client that could route to the host (#53). There is no override flag; set a
  token, or keep the bind on `127.0.0.1`. The default bind was already
  `127.0.0.1`, so only configurations that opt into a wider bind are affected.
- **The container token comes from the environment.** `config/guaipeca.container.yaml`
  binds `0.0.0.0` for port mapping, which requires auth, so the token is supplied
  as `GUAIPECA_AUTH_TOKEN` rather than written into the config file. The sample
  carries no token key at all — a placeholder that works as a real credential is
  worse than none, because it looks configured — and `docker-compose.yml`
  declares the variable as required, so compose refuses to start with a message
  naming it. The README quick start exports a generated token before starting.

### Fixed

- **CORS preflights advertise only the methods each route actually serves.**
  The preflight answered every path with a hardcoded `GET, POST, DELETE,
  OPTIONS`, inviting cross-origin requests that the route would then 405
  (`/health` and `/tools` serve GET only, `/documents` DELETE only, `/sse` GET
  only). The list now comes from the app's route table (#73).
- **A `*` in `server.allowed_origins` is a clean startup error, not a
  traceback.** The wildcard guard raised `ValueError` but nothing caught it, so
  `guaipeca serve` printed a raw traceback. The CLI now validates the allowlist
  before constructing the server (same pre-construction pass as the public-bind
  refusal) and `serve()` catches it for direct callers: the operator gets
  `Error: server.allowed_origins entry ... contains '*'` on stderr and exit 1
  (#74).
- **A preflight for a path the app does not serve no longer claims success.**
  The middleware answered every `OPTIONS` with a blanket `204`, telling the
  browser to proceed with a request that would 404. Unknown paths are now
  delegated to the app, so the preflight gets the same 404 the real request
  would (#75).
- **`Access-Control-Allow-Origin` echoes the matched allowlist entry, not the
  raw request origin.** A request origin with a trailing slash
  (`https://ui.example.com/`) matched the allowlist but was echoed verbatim,
  so the browser saw an ACAO value it did not send and refused the response.
  The header now carries the allowlist entry, normalised to no trailing slash;
  with no allowlist the request origin is echoed as before (#76).
- **Malformed `Origin` headers are rejected before anything is echoed.**
  An Origin containing control bytes (below 0x21 or 0x7F) or longer than 2048
  bytes is answered `400` instead of being matched and echoed — a hand-built
  request could no longer smuggle header bytes into an echoed
  `Access-Control-Allow-Origin` (#77).
- **The response carries one `Vary: Origin` header, not two.** When the inner
  app sent its own `Vary` (e.g. `Accept-Encoding`), the middleware appended a
  second `Vary` header instead of combining. The values are now merged into a
  single header, and the inner app's value is preserved (#78).
- **A modified file no longer leaves stale chunks in the index.** Incremental
  reindexing overwrote a file's stored hash before computing the set of
  deleted files, so a file whose content changed was never cleaned up: its old
  chunks stayed in the index and in FAISS while the new chunks were appended on
  top. The index grew on every edit, superseded content stayed retrievable,
  and one chunk ID could sit at more than one FAISS position. A changed file is
  now treated like a deleted file for cleanup, and any deletion or change
  triggers a full rebuild of the FAISS index from the surviving chunks, so
  reindexing after an edit returns only the current content (#62).
- **A corpus with chunks but zero FAISS vectors is no longer reported as
  "No indexed content found".** The search-time guard counted only the FAISS
  vector count; a populated BM25 index now also counts as indexed content.
- **`config/guaipeca.yaml` and README no longer describe `0.0.0.0` as a plain
  "network access" toggle.** Both now state that it requires auth and that the
  server refuses to start otherwise.

## [0.6.0] — 2026-11-20

Fixes and hardening since 0.5.0. No breaking changes.

### Added

- **`guaipeca --version`.** Prints the bare package version, the same string
  `/health` and the MCP `initialize` response report, and exits 0. Previously
  the flag did not exist, so a shell check of a deployed host had no CLI
  surface at all (#45).
- **Container images are tagged by version.** `ghcr.io/.../guaipeca:<version>`
  joins `latest` and the short sha, so an image can be pinned to a release
  rather than looked up by commit (#45).

### Changed

- **Faster CLI startup.** `numpy` was imported at module scope by
  `embedding.py` and `index.py`, so every command — including `status`,
  `--help` and `list-documents`, which never compute an array — paid for it.
  On this Pi 5 `import guaipeca.mcp_server` drops from ~950ms to ~72ms; the
  figure is host-specific, the direction is not. `fastembed` and `bm25s` were
  already deferred. Commands that embed or search now pay that import on their
  first call instead of at startup — the cost moves, it does not vanish. In a
  broken install (numpy missing or ABI-mismatched) the failure therefore
  surfaces at first use rather than at startup (#32).
- **A `v*` tag must match the declared version.** CI now runs
  `tools/release_check.py` on tag builds and fails when `v0.6.1` is pushed on a
  tree declaring `0.6.0`. The two version files were already checked against
  each other; the release tag was not (#45).

### Fixed

- **A Bearer token containing non-ASCII characters returned 500, not 401.**
  `hmac.compare_digest` raised `TypeError` on str input, so every authenticated
  route errored instead of refusing. Both Bearer paths now compare bytes,
  keeping the comparison constant-time (fail closed) (#41).
- **A legitimately configured non-ASCII secret rejected every real client.**
  Re-encoding starlette's latin-1-decoded header as UTF-8 double-encoded it; the
  presented token is now re-encoded with latin-1 so the compare is against the
  actual wire bytes (#41).
- **A secret containing lone surrogates (non-UTF-8 locale) was a 500 on every
  authenticated route.** The token is now encoded once at construction, and a
  secret that cannot be encoded fails closed (#41).
- **Two threads could each build an embedding model.** The lazy `model` property
  had no lock, so concurrent first use constructed two ONNX sessions. Now
  thread-safe and fail-fast (#31).
- **A fresh clone silently indexed nothing.** Copying the sample config to a
  missing `~/.guaipeca/` fell back to the bundled sample, whose corpora point at
  `/path/to/your/docs`, and `index` printed `Files indexed: 0` with exit 0. Three
  guards now distinguish an unedited sample (exit 2), a nonexistent corpus path
  (exit 1) and a genuinely empty folder (warning, exit 0) (#33).

### Removed

- Dead code left by the MCP SDK migration: an uncalled `_cors_origin` duplicating
  the origin policy, a docstring promising a check the body never performed, and
  references to functions deleted with the hand-rolled server (#38).

### Notes

- **Reporting a version is not the same as running it.** An installed copy keeps
  executing its old code until it is reinstalled and the service restarted —
  `sudo systemctl restart guaipeca`. `/health` will report the version of the
  binary it is actually running, not the newest release (#45).

## [0.5.0] — MCP SDK migration

The MCP transport layer now runs on the official [`mcp`](https://pypi.org/project/mcp/)
Python SDK instead of a hand-written JSON-RPC/HTTP implementation.

### Breaking

- **BM25 no longer returns zero-score padding.** `bm25s` is asked for `top_k * 3` rows regardless
  of relevance, so non-matches arrived with score `0.0` and were fused as if they were weak hits.
  They are now dropped. Ranking of real hits is unchanged (the sparse list is already
  rank-ordered); only padding disappears.
- **Index lock failures no longer fail open.** The lock file is opened without truncation, and a
  failure to open it now refuses the index run instead of proceeding unlocked. Contention keeps
  the existing "indexing in progress" message. Platforms with no locking primitive still index
  with a warning.
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

- **Relevance floor for search (`max_distance`).** Dense hits below the floor are dropped, so an
  off-topic query can return an empty result set instead of the least-bad chunk. Configurable per
  corpus and globally, plus `--max-distance` on the CLI. Distance is `1 - cosine similarity`, so
  lower is stricter. BM25-only hits are never filtered (they have no dense score, and exact-token
  matches are often low-similarity and high-value). An all-filtered search is a success with
  `results: []` and a `filtered_by_distance` count, not an error. **No default is shipped** — the
  value depends on corpus and model; measure with `tools/eval_retrieval.py --calibrate`.
- **Results report `distance`** (`1 - raw_score`) on dense hits, `null` for BM25-only hits.
- **`status` reports the active relevance floor**, so an empty result can be attributed to
  filtering rather than a failed index.
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

- **`mcp` is pinned to `==2.3.0` in `pyproject.toml`, not given a range.** The SSE
  transport reaches through `MCPServer._lowlevel_server`, which is private API
  with no public equivalent (issue #25). `MCPServer` has no `__slots__`, so on a
  release that renamed that attribute our assignment would still succeed —
  `sse_app()` would then drive the SDK's own empty server and `/sse` would serve
  **zero tools**, with no exception and no log line. A range would let a rebuild
  pull such a release in. Bump the pin deliberately: change the line, run the
  suite, and the runtime guard in `_build_sse_app` plus its tests will report a
  moved hook immediately.
- `both` still means "stdio when launched as a subprocess, HTTP when launched as
  a daemon" — it does not run two listeners, because the SDK runs one server per
  transport.
- The embedding model is still warmed eagerly at construction; see issue #24 for
  lazy warmup.
