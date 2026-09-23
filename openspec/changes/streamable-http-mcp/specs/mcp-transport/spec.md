# Specification: MCP Streamable HTTP Transport

## Requirements

### REQ-001: POST /mcp Accepts JSON-RPC and Responds Inline
**As a** modern MCP client
**I want** to send a JSON-RPC request to `POST /mcp` and receive the response inline in the HTTP response body
**So that** I can use the Streamable HTTP transport without maintaining a separate SSE connection

#### Scenarios

##### SC-001: tools/list via POST /mcp
**Given** the server is running with an established session (Mcp-Session-Id header)
**When** the client sends `POST /mcp` with `Content-Type: application/json` and body `{"jsonrpc":"2.0","id":1,"method":"tools/list"}`
**Then** the server responds with HTTP 200, `Content-Type: application/json`
**And** the response body is `{"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}`

##### SC-002: tools/call via POST /mcp
**Given** an established session with indexed corpora
**When** the client sends `POST /mcp` with body `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"search","arguments":{"query":"HFM audit"}}}`
**Then** the server responds with HTTP 200, `Content-Type: application/json`
**And** the response body contains the search results in MCP content format

##### SC-003: ping via POST /mcp
**Given** an established session
**When** the client sends `POST /mcp` with body `{"jsonrpc":"2.0","id":3,"method":"ping"}`
**Then** the server responds with HTTP 200 and body `{"jsonrpc":"2.0","id":3,"result":{}}`

**Edge Cases:**
- Malformed JSON body → HTTP 200 with `{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}`
- Valid JSON but not a JSON-RPC object (e.g. bare array of strings) → HTTP 200 with error code -32600 (Invalid Request)
- Oversized body (>10MB) → HTTP 413 `{"error":"request body too large"}`
- `Content-Type` is not `application/json` → server still attempts to parse the body (lenient); if parsing fails, returns parse error as above

---

### REQ-002: Initialize Handshake with Protocol Version Negotiation
**As a** MCP client
**I want** to send an `initialize` request via `POST /mcp` and receive the server's protocol version and capabilities
**So that** both sides agree on the protocol version before further communication

#### Scenarios

##### SC-001: Successful initialize
**Given** the server is running
**When** the client sends `POST /mcp` with body:
```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test-client","version":"1.0"}}}
```
**Then** the server responds with HTTP 200
**And** the response includes header `Mcp-Session-Id: <uuid>`
**And** the response body is:
```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"<negotiated>","capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"guaipeca","version":"<version>"}}}
```
Where `<negotiated>` is the highest protocol version the server supports that is <= the client's requested version, falling back to the server's default if the client's version is unknown.

##### SC-002: Client requests older protocol version (2024-11-05) on /mcp
**Given** the server supports `2025-06-18` as its advertised version
**When** the client sends `initialize` with `protocolVersion: "2024-11-05"` on `POST /mcp`
**Then** the server rejects the request with HTTP 200 and a JSON-RPC error:
`{"jsonrpc":"2.0","id":<id>,"error":{"code":-32602,"message":"Unsupported protocol version: 2024-11-05. Minimum supported version for Streamable HTTP transport is 2025-03-26."}}`
**And** no `Mcp-Session-Id` header is returned
**And** no session is created

> **R3 Ruling (Luis):** `/mcp` requires protocol version >= `2025-03-26`. The Streamable HTTP transport did not exist before `2025-03-26`, so `2024-11-05` is rejected on `/mcp`. Legacy `/sse` continues to accept `2024-11-05` with full backwards compat.
>
> **Minimum version rule:** Any `protocolVersion` older than `2025-03-26` sent to `POST /mcp` is rejected with JSON-RPC error `-32602` (Invalid params). The error message names the minimum supported version. This applies to all protocol versions below `2025-03-26`, including `2024-11-05` and any other legacy version. The legacy `/sse` transport is unaffected and continues to honour all known protocol versions with full backwards compatibility.

##### SC-002a: Client requests older protocol version on legacy /sse
**Given** the server supports `2025-06-18` as its advertised version
**When** the client sends `initialize` with `protocolVersion: "2024-11-05"` on `GET /sse` + `POST /messages`
**Then** the server responds with `protocolVersion: "2024-11-05"` (honours the client's requested version for backwards compatibility)
**And** the session is established normally

##### SC-003: Client requests unknown protocol version
**Given** the server does not recognize the client's requested protocol version
**When** the client sends `initialize` with `protocolVersion: "2099-01-01"`
**Then** the server responds with its own default `protocolVersion` (the latest it supports)
**And** the session is established normally

**Edge Cases:**
- Missing `protocolVersion` in initialize params → server uses its default version (`2025-06-18`), session still established
- `protocolVersion` < `2025-03-26` (e.g. `2024-11-05`) on `POST /mcp` → rejected with JSON-RPC error -32602 (per R3 ruling); legacy `/sse` still honours it
- Missing `clientInfo` → server still responds with its info; session established
- `initialize` sent with a `Mcp-Session-Id` header referencing an existing session → server returns `400` with error `{"error":"session already initialized"}`

---

### REQ-003: Mcp-Session-Id Validation
**As a** server
**I want** to validate the `Mcp-Session-Id` header on non-initialize requests to `/mcp`
**So that** only requests from established sessions are processed

#### Scenarios

##### SC-001: Valid session ID
**Given** a session was established via `initialize` and the server returned `Mcp-Session-Id: abc-123`
**When** the client sends `POST /mcp` with header `Mcp-Session-Id: abc-123` and body `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`
**Then** the server processes the request and returns the tools list

##### SC-002: Missing session ID on non-initialize request
**Given** no prior `initialize` was sent
**When** the client sends `POST /mcp` (no `Mcp-Session-Id` header) with body `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`
**Then** the server responds with HTTP 400 and body `{"jsonrpc":"2.0","id":2,"error":{"code":-32000,"message":"Missing or invalid Mcp-Session-Id header"}}`

##### SC-003: Unknown session ID
**Given** no session with ID `xyz-999` exists
**When** the client sends `POST /mcp` with header `Mcp-Session-Id: xyz-999` and a JSON-RPC request
**Then** the server responds with HTTP 400 with the same error as SC-002

##### SC-004: initialize does not require session ID
**Given** no session exists
**When** the client sends `POST /mcp` with body `{"jsonrpc":"2.0","id":1,"method":"initialize",...}` (no `Mcp-Session-Id` header)
**Then** the server processes the initialize and returns a new session ID in the response header

**Edge Cases:**
- Empty `Mcp-Session-Id` header value (`""`) → treated as missing, returns 400
- Whitespace-only session ID → treated as invalid, returns 400

---

### REQ-004: Legacy /sse + /messages Remain Functional (Backwards Compat)
**As a** legacy MCP client
**I want** to connect via `GET /sse` and `POST /messages` as before
**So that** existing integrations continue to work after the Streamable HTTP transport is added

#### Scenarios

##### SC-001: Legacy SSE handshake still works
**Given** the server is running
**When** a client connects to `GET /sse`
**Then** the server opens an SSE stream and sends an `endpoint` event with the POST URL
**And** the client can subsequently `POST /messages?session_id=<id>` and receive responses via the SSE stream

##### SC-002: Legacy and streamable HTTP sessions coexist
**Given** a legacy SSE session is active
**When** a different client establishes a streamable HTTP session via `POST /mcp`
**Then** both sessions operate independently without interference
**And** the legacy SSE session's response queue is unaffected by the streamable HTTP session

**Edge Cases:**
- Legacy session and streamable session have separate ID spaces (both are UUIDs, collision is negligible); they share the same `sessions` dict but are keyed by unique IDs
- Server shutdown cleans up both session types

---

### REQ-005: Auth Enforced Identically on /mcp
**As a** server operator
**I want** Bearer token authentication applied to `/mcp` the same way it is applied to `/sse` and `/messages`
**So that** the streamable HTTP transport is not an auth bypass

#### Scenarios

##### SC-001: Auth required, valid token
**Given** `server.auth_token` is set to `"secret"`
**When** the client sends `POST /mcp` with header `Authorization: Bearer secret`
**Then** the request is processed normally

##### SC-002: Auth required, missing or invalid token
**Given** `server.auth_token` is set to `"secret"`
**When** the client sends `POST /mcp` without an `Authorization` header or with a wrong token
**Then** the server responds with HTTP 401 `{"error":"unauthorized"}`

##### SC-003: No auth configured
**Given** `server.auth_token` is `None`
**When** any client sends `POST /mcp` without an `Authorization` header
**Then** the request is processed normally

**Edge Cases:**
- Auth check happens before session validation — a 401 is returned even if the session ID is invalid
- CORS `Access-Control-Allow-Origin` follows existing logic: `"*"` when no auth, `"null"` when auth is enabled

---

### REQ-006: GET /mcp Returns 405 Method Not Allowed
**As a** server
**I want** `GET /mcp` to return `405 Method Not Allowed`
**So that** clients know server-initiated SSE streaming is not supported on this endpoint

#### Scenarios

##### SC-001: GET /mcp without Accept header
**Given** the server is running
**When** the client sends `GET /mcp`
**Then** the server responds with HTTP 405 and `Content-Type: application/json`
**And** the body is `{"error":"GET /mcp not supported; use POST /mcp for JSON-RPC requests"}`

##### SC-002: GET /mcp with Accept: text/event-stream
**Given** the client sends `Accept: text/event-stream`
**When** the client sends `GET /mcp`
**Then** the server still responds with HTTP 405 (SSE upgrade is not implemented in this change)

**Justification:** Implementing `GET /mcp` as an SSE stream for server-initiated messages adds complexity (persistent connection management, event ID tracking, resumability) that is not needed for Guaipeca's use case — it is a request/response RAG server with no server-initiated messages. Returning `405` is spec-compliant and signals to clients that only `POST` is supported. This can be revisited if a future need arises.

**Edge Cases:**
- `HEAD /mcp` → same `405` (BaseHTTPRequestHandler does not define `do_HEAD`, so the default handler returns `501 Not Implemented`; this is acceptable)

---

### REQ-007: DELETE /mcp Session Termination
**As a** MCP client
**I want** to terminate a streamable HTTP session by sending `DELETE /mcp`
**So that** the server cleans up session resources promptly

#### Scenarios

##### SC-001: Valid session termination
**Given** a streamable HTTP session exists with `Mcp-Session-Id: abc-123`
**When** the client sends `DELETE /mcp` with header `Mcp-Session-Id: abc-123`
**Then** the server removes the session from its session store
**And** responds with HTTP 200 `{"status":"session terminated"}`

##### SC-002: DELETE with unknown session ID
**Given** no session with ID `xyz-999` exists
**When** the client sends `DELETE /mcp` with header `Mcp-Session-Id: xyz-999`
**Then** the server responds with HTTP 404 `{"error":"session not found"}`

##### SC-003: DELETE without session ID header
**Given** the client sends no `Mcp-Session-Id` header
**When** the client sends `DELETE /mcp`
**Then** the server responds with HTTP 400 `{"error":"Missing Mcp-Session-Id header"}`

**Edge Cases:**
- Deleting a legacy SSE session ID via `DELETE /mcp` → the session is found in the shared `sessions` dict and removed; the SSE stream will close on its next keepalive cycle. This is acceptable (cross-transport cleanup).
- Auth is enforced: `DELETE /mcp` without valid Bearer token (when auth is configured) returns `401` before session lookup

---

### REQ-008: Notification Handling (202 with Empty Body)
**As a** MCP client
**I want** notifications (requests with no `id` or methods starting with `notifications/`) to receive `202 Accepted` with an empty body
**So that** the server acknowledges the notification without sending a JSON-RPC response

#### Scenarios

##### SC-001: notifications/initialized
**Given** an established session
**When** the client sends `POST /mcp` with body `{"jsonrpc":"2.0","method":"notifications/initialized"}`
**Then** the server responds with HTTP 202, `Content-Length: 0`, and an empty body

##### SC-002: Request with no id field
**Given** an established session
**When** the client sends `POST /mcp` with body `{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}` (no `id` field)
**Then** the server responds with HTTP 202 with empty body

**Edge Cases:**
- Notification with unknown method (not `notifications/*` and no `id`) → still `202` with empty body; the server logs the unknown method but does not return an error (notifications are fire-and-forget)
- Notification before initialize → `400` (session not established); the session ID check happens before method dispatch

---

### REQ-009: Batch Requests
**As a** MCP client
**I want** to send a JSON array of JSON-RPC requests to `POST /mcp` and receive a JSON array of responses
**So that** I can batch multiple operations in a single HTTP round-trip

#### Scenarios

##### SC-001: Batch of two requests
**Given** an established session
**When** the client sends `POST /mcp` with body:
```json
[
  {"jsonrpc":"2.0","id":1,"method":"tools/list"},
  {"jsonrpc":"2.0","id":2,"method":"ping"}
]
```
**Then** the server responds with HTTP 200, `Content-Type: application/json`
**And** the response body is a JSON array:
```json
[
  {"jsonrpc":"2.0","id":1,"result":{"tools":[...]}},
  {"jsonrpc":"2.0","id":2,"result":{}}
]
```

##### SC-002: Batch with a notification
**Given** an established session
**When** the client sends `POST /mcp` with body:
```json
[
  {"jsonrpc":"2.0","method":"notifications/initialized"},
  {"jsonrpc":"2.0","id":1,"method":"ping"}
]
```
**Then** the server responds with HTTP 200
**And** the response body is a JSON array with one entry (the ping response); the notification produces no array entry

##### SC-003: Empty batch array
**Given** an established session
**When** the client sends `POST /mcp` with body `[]`
**Then** the server responds with HTTP 200 and body `[]`

**Edge Cases:**
- Batch with all notifications → HTTP 202 with empty body (not a JSON array), matching the single-notification behaviour
- Batch with mix of requests and notifications → HTTP 200 with a JSON array containing only the responses to requests (non-notification entries)
- Malformed entry in batch (not valid JSON-RPC object) → that entry gets an error response in the array; other entries are processed normally

---

### REQ-010: 404 for Unknown Paths Preserved
**As a** client
**I want** unknown HTTP paths to continue returning `404`
**So that** the server's existing behaviour is unchanged

#### Scenarios

##### SC-001: Unknown path on POST
**Given** the server is running
**When** the client sends `POST /unknown`
**Then** the server responds with HTTP 404 `{"error":"not found"}`

##### SC-002: Unknown path on GET
**Given** the server is running
**When** the client sends `GET /unknown`
**Then** the server responds with HTTP 404 `{"error":"not found"}`

##### SC-003: Unknown path on DELETE
**Given** the server is running
**When** the client sends `DELETE /unknown`
**Then** the server responds with HTTP 404 `{"error":"not found"}`

**Edge Cases:**
- `POST /mcp/` (trailing slash) → `404` (exact path match only, consistent with existing `/sse` and `/messages` behaviour)
- `POST /MCP` (uppercase) → `404` (case-sensitive, consistent with existing paths)

---

### REQ-011: Accept Header Validation
**As a** server
**I want** to handle the `Accept` header on `POST /mcp` requests
**So that** the response format aligns with client expectations

#### Scenarios

##### SC-001: Accept: application/json
**Given** an established session
**When** the client sends `POST /mcp` with `Accept: application/json` and a JSON-RPC request body
**Then** the server responds with HTTP 200 and `Content-Type: application/json`

##### SC-002: Accept: text/event-stream
**Given** an established session
**When** the client sends `POST /mcp` with `Accept: text/event-stream` and a JSON-RPC request body
**Then** the server responds with HTTP 200 and `Content-Type: application/json` (SSE upgrade on POST is not implemented; the server always responds inline for now)
**And** a `Warning` header is not required — the spec allows returning JSON when the server does not support SSE upgrade

##### SC-003: No Accept header
**Given** an established session
**When** the client sends `POST /mcp` without an `Accept` header
**Then** the server responds with HTTP 200 and `Content-Type: application/json`

**Edge Cases:**
- `Accept: */*` → server responds with `application/json` (wildcard accepted)
- `Accept: application/json, text/event-stream` → server responds with `application/json` (first accepted type)

---

### REQ-012: CORS and OPTIONS Handling for /mcp
**As a** browser-based client
**I want** CORS preflight responses to include `/mcp` methods
**So that** I can make cross-origin requests to the streamable HTTP endpoint

#### Scenarios

##### SC-001: OPTIONS /mcp
**Given** the server is running
**When** the client sends `OPTIONS /mcp`
**Then** the server responds with `204 No Content`
**And** includes `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS`
**And** includes `Access-Control-Allow-Headers: Content-Type, Authorization, Mcp-Session-Id`

##### SC-002: CORS headers on POST /mcp response
**Given** auth is not configured (CORS origin is `*`)
**When** the client sends `POST /mcp` with a valid JSON-RPC request
**Then** the response includes `Access-Control-Allow-Origin: *`

**Edge Cases:**
- When auth is configured, `Access-Control-Allow-Origin` is `null` (matching existing behaviour)
- `Mcp-Session-Id` must be in `Access-Control-Allow-Headers` so browser clients can send it cross-origin

---

### REQ-013: Request Before Initialize
**As a** server
**I want** to reject non-initialize requests sent before a session is established
**So that** clients follow the required handshake sequence

#### Scenarios

##### SC-001: tools/list before initialize
**Given** no session has been established
**When** the client sends `POST /mcp` (no `Mcp-Session-Id` header) with body `{"jsonrpc":"2.0","id":1,"method":"tools/list"}`
**Then** the server responds with HTTP 400 and error `{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"Missing or invalid Mcp-Session-Id header"}}`

##### SC-002: initialize is the only method that does not require a session
**Given** no session exists
**When** the client sends `POST /mcp` (no `Mcp-Session-Id` header) with `method: "initialize"`
**Then** the server processes the initialize and returns a new session ID

**Edge Cases:**
- `ping` before initialize → `400` (ping requires an established session, consistent with the spec)
- A client that sends `initialize` with a stale `Mcp-Session-Id` from a previous (now dead) session → `400` with `{"error":"session already initialized"}` (the client should start a new session without the header)