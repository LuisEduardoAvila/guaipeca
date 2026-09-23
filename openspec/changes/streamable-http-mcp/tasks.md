# Tasks: MCP Streamable HTTP Transport

## Phase 1: Failing Tests First

Write `tests/test_streamable_http.py` covering all spec scenarios. These tests should fail until Phase 2 implementation lands.

- [ ] 1.1 Create `tests/test_streamable_http.py` with a test fixture that starts the HTTP server on a random port (reuse existing `conftest.py` patterns)
- [ ] 1.2 Test: `POST /mcp` with `initialize` → 200, `Mcp-Session-Id` header present, `protocolVersion` in response body (REQ-002 SC-001)
- [ ] 1.3 Test: `POST /mcp` with `tools/list` after initialize → 200, `Content-Type: application/json`, tools array in response (REQ-001 SC-001)
- [ ] 1.4 Test: `POST /mcp` with `tools/call` (search) after initialize → 200, MCP content format response (REQ-001 SC-002)
- [ ] 1.5 Test: `POST /mcp` with `ping` after initialize → 200, `{"jsonrpc":"2.0","id":...,"result":{}}` (REQ-001 SC-003)
- [ ] 1.6 Test: `POST /mcp` with `notifications/initialized` → 202, empty body (REQ-008 SC-001)
- [ ] 1.7 Test: `POST /mcp` with request missing `id` field → 202, empty body (REQ-008 SC-002)
- [ ] 1.8 Test: `POST /mcp` with `tools/list` before initialize (no session) → 400 (REQ-003 SC-002, REQ-013 SC-001)
- [ ] 1.9 Test: `POST /mcp` with unknown `Mcp-Session-Id` → 400 (REQ-003 SC-003)
- [ ] 1.10 Test: `DELETE /mcp` with valid session → 200, session removed (REQ-007 SC-001)
- [ ] 1.11 Test: `DELETE /mcp` with unknown session → 404 (REQ-007 SC-002)
- [ ] 1.12 Test: `DELETE /mcp` without session header → 400 (REQ-007 SC-003)
- [ ] 1.13 Test: `GET /mcp` → 405 (REQ-006 SC-001)
- [ ] 1.14 Test: Batch request (two requests) → 200, JSON array response (REQ-009 SC-001)
- [ ] 1.15 Test: Batch with notification → 200, array with only request responses (REQ-009 SC-002)
- [ ] 1.16 Test: Empty batch `[]` → 200, `[]` (REQ-009 SC-003)
- [ ] 1.17 Test: Malformed JSON body → 200, parse error -32700 (REQ-001 edge case)
- [ ] 1.18 Test: Oversized body (>10MB) → 413 (REQ-001 edge case)
- [ ] 1.19 Test: Auth required (valid token) → request processed (REQ-005 SC-001)
- [ ] 1.20 Test: Auth required (missing token) → 401 (REQ-005 SC-002)
- [ ] 1.21 Test: `OPTIONS /mcp` → 204, `Mcp-Session-Id` in `Access-Control-Allow-Headers` (REQ-012 SC-001)
- [ ] 1.22 Test: Legacy `GET /sse` still works (opens stream, sends endpoint event) (REQ-004 SC-001)
- [ ] 1.23 Test: Legacy and streamable sessions coexist (REQ-004 SC-002)
- [ ] 1.24 Test: `POST /mcp/` (trailing slash) → 404 (REQ-010 edge case)
- [ ] 1.25 Test: `POST /unknown` → 404 (REQ-010 SC-001)
- [ ] 1.26 Test: Initialize with older protocol version `2024-11-05` on `/mcp` → server rejects with JSON-RPC error -32602 (R3 ruling: /mcp requires >= 2025-03-26; minimum version rule applies to ALL versions below 2025-03-26, not just 2024-11-05; legacy /sse is unaffected)
- [ ] 1.26a Test: Initialize with older protocol version `2024-11-05` on legacy `/sse` → server echoes it back (REQ-002 SC-002a, R5: pinned legacy clients unaffected)
- [ ] 1.27 Test: Initialize with unknown protocol version `2099-01-01` → server responds with its default (REQ-002 SC-003)

## Phase 2: Implement /mcp in mcp_server.py

All changes in `src/guaipeca/mcp_server.py` unless noted.

- [ ] 2.1 Bump `MCP_VERSION` constant from `"2024-11-05"` to `"2025-06-18"` (line 42)
- [ ] 2.2 Add known-versions set `KNOWN_PROTOCOL_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18"}` near `MCP_VERSION`
- [ ] 2.3 Update `handle_request` for `initialize` method: read `params.get("protocolVersion")`, negotiate version — echo client's version if known (in `KNOWN_PROTOCOL_VERSIONS`), else return `MCP_VERSION`. The R3 rejection (min version 2025-03-26 for `/mcp`) is enforced in the POST `/mcp` HTTP handler BEFORE calling `_process_jsonrpc`, NOT in `handle_request` (which is shared with legacy `/sse` that still honours all known versions)
- [ ] 2.4 Rename `SSESession` to `MCPSession` (or add new `MCPSession` class with `transport` field); update legacy SSE code to use the unified class with `transport="sse"`
- [ ] 2.5 Add `do_GET` handler for `GET /mcp` → 405 with JSON error body
- [ ] 2.6 Add `do_POST` handler for `POST /mcp`:
  - [ ] 2.6.1 Auth check (reuse `_check_auth`)
  - [ ] 2.6.2 Content-Length + 10MB cap (reuse existing `max_body_size` constant)
  - [ ] 2.6.3 Parse JSON; handle parse error → -32700
  - [ ] 2.6.4 Detect batch (list) vs single request
  - [ ] 2.6.5 Session validation: check `Mcp-Session-Id` header (missing/unknown → 400); `initialize` without session → create new session; `initialize` with existing session → 400
  - [ ] 2.6.6 Call `_process_jsonrpc` for single; batch wrapper for list
  - [ ] 2.6.7 Notification (None result) → 202, empty body, `Content-Length: 0`
  - [ ] 2.6.8 Batch all-notifications → 202, empty body
  - [ ] 2.6.9 Request (dict result) → 200, `Content-Type: application/json`, JSON body
  - [ ] 2.6.10 Set `Mcp-Session-Id` response header on initialize response only
  - [ ] 2.6.11 Set `Access-Control-Allow-Origin` header (reuse `cors_origin`)
- [ ] 2.7 Add `do_DELETE` handler for `DELETE /mcp`:
  - [ ] 2.7.1 Auth check
  - [ ] 2.7.2 Check `Mcp-Session-Id` header (missing → 400, unknown → 404)
  - [ ] 2.7.3 Remove session from `sessions` dict, set `active=False`
  - [ ] 2.7.4 Return 200 `{"status":"session terminated"}`
- [ ] 2.8 Update `do_OPTIONS` to add `Mcp-Session-Id` to `Access-Control-Allow-Headers`
- [ ] 2.9 Add `Mcp-Session-Id` to CORS allowed headers string in `do_OPTIONS`

## Phase 3: Docs Updates

- [ ] 3.1 `README.md`: Add `POST /mcp` to the HTTP endpoints section (after line 513); mention Streamable HTTP transport, protocol version `2025-06-18`, `Mcp-Session-Id` header; update the "Dual transport" feature bullet to "Triple transport" or "Multi-transport" mentioning stdio + legacy SSE + Streamable HTTP
- [ ] 3.2 `docs/container-deployment.md`: Update verification section with a `curl` example for `POST /mcp` initialize handshake; note that `/mcp` is the recommended endpoint for modern clients
- [ ] 3.3 `docs/ARCHITECTURE.md`: Update the "Transports" section (around lines 253–265) to add Streamable HTTP: `POST /mcp` (inline JSON-RPC), `DELETE /mcp` (session termination), `GET /mcp → 405`; note shared session store
- [ ] 3.4 `config/guaipeca.container.yaml`: Update comment on `transport: http` line to mention Streamable HTTP support on `POST /mcp`

## Phase 4: Version Bump

- [ ] 4.1 Verify current version: `grep '__version__' src/guaipeca/__init__.py` (expected: `0.3.4`) and `grep '^version' pyproject.toml` (expected: `0.3.4`)
- [ ] 4.2 Bump `pyproject.toml`: `version = "0.3.4"` → `version = "0.3.5"`
- [ ] 4.3 Bump `src/guaipeca/__init__.py`: `__version__ = "0.3.4"` → `__version__ = "0.3.5"`

## Phase 5: Verification

**Checkpoint: Code lands on a feature branch. Do NOT push or merge without Luis's explicit approval.**

- [ ] 5.1 Run full test suite: `cd /home/luis/.openclaw/workspace/projects/guaipeca && python -m pytest tests/ -v`
- [ ] 5.2 Run new streamable HTTP tests only: `python -m pytest tests/test_streamable_http.py -v`
- [ ] 5.3 Manual curl: initialize handshake
  ```bash
  curl -s -D - -X POST http://localhost:8090/mcp \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl-test","version":"1.0"}}}'
  # Expect: HTTP/1.1 200, Mcp-Session-Id header, protocolVersion in body
  ```
- [ ] 5.4 Manual curl: tools/list inline response (use session ID from 5.3)
  ```bash
  curl -s -X POST http://localhost:8090/mcp \
    -H "Content-Type: application/json" \
    -H "Mcp-Session-Id: <session-id-from-step-5.3>" \
    -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
  # Expect: JSON response with tools array
  ```
- [ ] 5.5 Manual curl: legacy SSE still works
  ```bash
  curl -N http://localhost:8090/sse
  # Expect: event: endpoint / data: /messages?session_id=<uuid>
  ```
- [ ] 5.6 Manual curl: health endpoint unaffected
  ```bash
  curl -s http://localhost:8090/health
  # Expect: {"status":"ok","server":"guaipeca","version":"0.3.5"}
  ```
- [ ] 5.7 Manual curl: GET /mcp returns 405
  ```bash
  curl -s -o /dev/null -w "%{http_code}" http://localhost:8090/mcp
  # Expect: 405
  ```
- [ ] 5.8 Manual curl: DELETE /mcp terminates session
  ```bash
  curl -s -X DELETE http://localhost:8090/mcp -H "Mcp-Session-Id: <session-id>"
  # Expect: {"status":"session terminated"}
  ```
- [ ] 5.9 Confirm no new dependencies in `pyproject.toml` (diff check)
- [ ] 5.10 Confirm `git status` shows changes only in expected files; no accidental modifications to source outside `mcp_server.py`