# MCP SDK Migration — Implementation Plan

**Decision:** Option B (official `mcp` SDK).
**Issue:** #22 · **Baseline:** `main` @ `81fa0f3` · merged as `3a2b5ea`.

> **Amended after delivery.** This plan says `/sse` would be **dropped**. It was
> dropped, then restored: the SDK ships `MCPServer.sse_app()`, so keeping the
> legacy transport costs little, and existing clients were not confirmed gone.
> SSE is now served and marked deprecated in favour of `POST /mcp`. Phase 4's
> "delete `/sse`" and Phase 5's "retire `/sse` tests" should be read in that
> light; the shipped state is described in README and `CHANGELOG.md`.

---

## Ground rules for the whole effort

1. **One PR per phase.** Each phase must end with a green suite and a working `guaipeca serve`. No phase leaves `main` broken.
2. **PR #23 merges first.** The migration builds on the Origin/version work; rebase onto it, don't duplicate it.
3. **Behaviour is frozen during migration.** No feature work, no refactors of `search.py`/`index.py`/`chunking.py`. The only thing changing is the transport layer.
4. **Every phase has an observable proof.** A passing unit test is not enough on its own — each phase states what you can `curl` or watch in a log to confirm it.
5. **The old server stays until the new one is proven.** Phase 4 lands the new stack behind the same `serve` command; the hand-rolled `ThreadingHTTPServer` path is deleted only after Phase 5 passes.

---

## Agent model

| Role | Who | Does |
|---|---|---|
| **Orchestrator** | main (me) | Plans phases, spawns workers, validates every result against real code, runs the suite and live probes, opens PRs. Owns the final call on "done". |
| **Implementation** | sub-agents (docs/coding tier) | Write code and tests for one phase at a time from a precise brief. **No shell** — they cannot run pytest/ruff/git, so their reports are always treated as unverified claims. |
| **Reasoning** | `expert-flash` | Called only for genuine unknowns: SDK API semantics not obvious from source, design forks, "why is this failing" when the cause isn't mechanical. Not for writing code. |

**Handoff protocol** (learned from the previous round, where the docs agent could not run tests):
- Give each worker exact file:line anchors, not prose descriptions.
- Workers return `git diff --stat` **only if** they have shell; otherwise they return a self-report of changes, which I verify by reading the diff.
- **I always run** `ruff` and `pytest` and the live probe myself. A worker saying "tests pass" with no shell is not evidence.
- One phase = one worker session, `cleanup:"keep"` so I can re-task it for fixes without re-briefing.

---

## Phase 0 — Spike (gate for everything else) · 1 day

**Question:** does `mcp` 2.3.0 install and work on this Pi 5 (aarch64, Python 3.13.5) with a sync tool path?

Already verified from the published wheel (so this is not a blank cheque):
- `mcp/server/_streamable_http_modern.py` — 2026-07-28 support, with the legacy path explicitly retained
- `StreamableHTTPSessionManager.handle_request` is the ASGI entry
- `Server.streamable_http_app(...)` takes `max_request_body_size`, `transport_security`, `auth`, `token_verifier`, `custom_starlette_routes` — i.e. **body-size limits and DNS-rebinding protection are native SDK features**, not hand-rolled middleware
- `Server._handle_discover` exists

**Work:**
1. `python -m venv /tmp/gsdk && pip install mcp==2.3.0` on the Pi. Confirm every dep resolves to a wheel (watch `pydantic-core`, `uvloop`, `httptools`). **Record the resolved versions.**
2. Build a throwaway 2-tool server (one sync tool wrapped in `anyio.to_thread.run_sync`, one async) on `Server.streamable_http_app()`, behind uvicorn.
3. Drive it with the SDK's own client: initialize → `tools/list` → `tools/call`. Then drive it with **curl** as a legacy 2025-11-25 client and once as a raw 2026-07-28 stateless request.
4. Measure: RSS idle, RSS after 20 searches, cold start → first tool response, and median search latency versus today's server on the same corpus.

**Exit criteria — RESULTS (run 2026-11-16 on the Pi 5, aarch64, Python 3.13.5):**
- [x] All deps install as wheels on aarch64, no source builds
      `pip install mcp==2.3.0` -> 29 packages, zero source builds, 7.2s.
      pydantic 2.14.3, pydantic-core 2.50.4, starlette 1.10.0, uvicorn 0.54.0,
      anyio 4.16.0, sse-starlette 3.7.0, mcp-types 2.3.0.
- [x] Legacy initialize works through the SDK; modern stateless request works
      Both negotiated on one endpoint: legacy `initialize` -> 200 + session;
      2026-07-28 stateless `tools/list` -> 200, no session.
- [x] `to_thread`-wrapped sync tool returns correct output under concurrent load
      8 parallel calls of a 20 ms blocking function finished in 30 ms
      (serial would be 160 ms).
- [x] RSS acceptable: 63 MB peak after importing the full server stack
      (host is a 7.9 GB Pi shared with other services).
- [x] Clean SIGTERM: `http` and `both` both exit in ~0.2 s with stdin held open,
      which is why `os._exit(0)` could be removed.

Latency versus the pre-migration server was **not** measured in the spike (the
tool was synthetic). It was checked later on the live service: a real
`epm-docs` search returned in ~425 ms end to end through `/mcp`.
- [ ] Search latency within ~15% of the current server; RSS acceptable on a 7.9 GB box
- [ ] `guaipeca serve` equivalent runs under uvicorn and shuts down cleanly on SIGTERM

**Kill / re-plan triggers:** any dep needs a compiler; latency regresses materially; the SDK's shutdown fights `os._exit` in a way that leaks native threads. If a trigger fires → escalate to `expert-flash` with the evidence, then decide A vs B.

**Owner:** main does the measurement (needs shell). `expert-flash` only if a trigger fires.

---

## Phase 1 — Dependency + config + packaging · 0.5 day

**PR 1.**
- `pyproject.toml`: add `mcp>=2.3.0` to `dependencies`; keep it a *core* dep (not optional) so `guaipeca serve` always works.
- `config.py`: `ServerConfig` gains nothing new for transport. `transport: "both"` becomes `"stdio" | "http"` — **`both` cannot survive**, because the SDK runs one server per transport. Decide: keep `both` meaning "run stdio when launched as a subprocess, HTTP when launched as a daemon" (today's practical behaviour) and document it, rather than running both simultaneously.
- Document the dropped `/sse` in `config.py` docstring and README: `transport: "sse"` is no longer valid.
- CI (`test.yml`) unchanged; add a job step that asserts `pip install -e ".[all]"` succeeds on the runner.

**Proof:** `python -c "import mcp; print(mcp.__version__)"`; config validation rejects `transport: sse` with a clear error; full suite green (no behaviour yet changed).

**Owner:** implementation sub-agent; main verifies.

---

## Phase 2 — Tool registry + schema bridge (no transport change) · 1 day

The highest-value phase to isolate: prove the tools work through the SDK **before** touching HTTP.

**PR 2.**
- New module `src/guaipeca/mcp_sdk/tools.py`:
  - Build a `Server(...)` instance.
  - Register a `list_tools` handler returning `types.Tool(name=..., description=..., inputSchema=...)` **reusing the existing `TOOLS` dicts verbatim** — they are already JSON Schema, and the SDK accepts raw JSON Schema. `_tools_with_corpus_enum()` keeps working as a pre-serialisation step.
  - Register a `call_tool` handler that delegates to the existing `_handle_tool_call` unchanged, wrapped in `anyio.to_thread.run_sync` (or `asyncio.to_thread`).
- **Golden-file test:** capture today's `tools/list` JSON for all 10 tools and assert the SDK's output is byte-identical (modulo `serverInfo`). This is the regression net for the whole migration.
- **Parity test:** every tool in `MCP_TOOL_TO_CLI` in `cli.py` still resolves; the existing `test_cli_mcp_parity.py` keeps passing untouched.

**Proof:** SDK server over **stdio only** answers `tools/list` and `tools/call` for all 10 tools, output identical to today. Run the real corpus through it and diff results for 5 known queries.

**Owner:** implementation sub-agent writes; main runs the golden-file comparison. `expert-flash` only if the SDK rejects a schema shape.

---

## Phase 3 — stdio cutover · 0.5 day

**PR 3.** Replace `run_stdio()` (the 12-line stdin loop) with the SDK's stdio session. `cli.py cmd_serve` becomes an `asyncio.run(...)` entry.
- **The risk is the stray sync handler:** every handler must be wrapped. Add a lint-style unit test that introspects registered handlers and fails if any non-tool handler is a bare `def` that does blocking work. Cheap insurance against the exact failure mode the estimate flagged.

**Proof:** `guaipeca serve --transport stdio` driven by a real MCP client (Claude Code / the SDK client) runs a search end-to-end. `os._exit` hack deleted if no longer needed.

**Owner:** implementation sub-agent; main verifies with a real client.

---

## Phase 4 — HTTP cutover + non-MCP endpoints · 2 days

**PR 4. The big one.**

- `mcp_sdk/http.py`: build `Server.streamable_http_app(...)` with:
  - `max_request_body_size` from `config.server.max_body_size` — **replaces** the hand-rolled 413 logic
  - `transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_origins=..., allowed_hosts=...)` from `config.server.allowed_origins` — **replaces** `_origin_allowed`/`_check_origin`, keeping the same policy (loopback default, no-Origin allowed). Keep the existing unit tests as a spec for the mapping.
  - `token_verifier` / `auth` for the optional Bearer path, or a thin ASGI middleware if the SDK's OAuth-shaped auth doesn't fit a single shared secret — **this is the one genuine fork; if it fights, escalate to `expert-flash` rather than hacking.**
  - `custom_starlette_routes` for the non-MCP endpoints: `/health`, `/tools`, `GET /download/{corpus}/{filename}`, `DELETE /documents`.
- **Delete the legacy `/sse` + `/messages` handlers and their tests.** Add a `404`/`405`-with-message response pointing at `/mcp` so a stale SSE client gets an actionable error rather than a silent hang.
- Uvicorn run loop replacing `_Server`/`run_http`, preserving: the port-retry behaviour, SIGTERM graceful shutdown, and the log lines ops greps for.

**Proof (live, by me, not by the worker):**
- Legacy client (`initialize` → `tools/list` → `tools/call`) over HTTP
- Modern 2026-07-28 stateless request
- `Origin: https://evil.example` → 403
- No `Origin` → 200
- Oversized body → 413
- `DELETE /documents` and `GET /download/...` still work
- SIGTERM → clean exit, no orphan threads, `systemctl` sees a clean stop

**Owner:** implementation sub-agent for code; `expert-flash` on the auth fork if needed; main owns all live verification.

---

## Phase 5 — Test migration + hardening · 1.5 days

**PR 5.**
- Port **assertions, not harnesses**. Build a checklist mapping every assertion in `test_streamable_http.py` (1,044 lines) to either "covered in the new suite" or "deliberately dropped, because…". Nothing may silently vanish.
- Must survive: dead-session 404, Origin 403, 413, malformed request line, batch arrays, notification 202, port-retry, graceful shutdown.
- New: modern-era tests (stateless request, `server/discover`, `UnsupportedProtocolVersionError` -32022 for an unknown version).
- Retire `/sse` tests with a link to the removal decision.

**Proof:** suite green, and the assertion-mapping table is in the PR description. Coverage of `mcp_sdk/` at least equal to today's `mcp_server.py`.

**Owner:** implementation sub-agent; main reviews the mapping table line by line.

---

## Phase 6 — Docs, cleanup, release · 0.5 day

**PR 6.**
- README: transports, the `/sse` removal, `transport` config change, a migration note for anyone still on `/sse`.
- `docs/ARCHITECTURE.md`: new transport section, diagram of SDK layering.
- `openspec/`: the decided rationale from issue #22, so the next "which MCP version" question has an answer.
- Delete `mcp_server.py`'s transport half; keep `_handle_tool_call`/`TOOLS` where they are if that's cleaner than moving them.
- Version bump + `CHANGELOG`.

**Proof:** `grep` finds no stale `2025-06-18`, no `/sse` references outside the removal note; Docker image builds and runs on `3.13-slim`.

---

## Sequencing and dependencies

```
Phase 0 (spike)  ──gate──►  Phase 1  ──►  Phase 2  ──►  Phase 3
                                                │
                                                └──►  Phase 4  ──►  Phase 5  ──►  Phase 6
```

Phase 2 and Phase 3 are strictly ordered (registry before stdio). Phase 4 depends on 2 but not 3. Phase 5 depends on 4. Nothing parallelises usefully except Phase 6 docs prose, which I can write while 5 runs.

## Total

| Phase | Days (likely) |
|---|---|
| 0 Spike | 1.0 |
| 1 Deps/config | 0.5 |
| 2 Tools bridge | 1.0 |
| 3 stdio | 0.5 |
| 4 HTTP | 2.0 |
| 5 Tests | 1.5 |
| 6 Docs/release | 0.5 |
| **Total** | **7.0** |

Down from 8 because `/sse` is dropped (~1 day saved, and it removes the `SseServerTransport` port entirely).

## Deliberately not in scope

- Any change to retrieval, chunking, caching, or embedding
- OAuth (the SDK's auth story is OAuth-shaped; we only need a shared Bearer secret — if that proves awkward it becomes its own issue, not a blocker here)
- The 133 pre-existing ruff findings
- Reinstalling the editable install to fix the 0.3.5/0.4.0 dist-info mismatch
