# Design: Document-level Search Results (`return_mode`)

## Overview

Guaipeca's MCP `search` tool currently returns **chunks** (fragments of documents). This feature adds a `return_mode` parameter that lets the MCP client choose the granularity of results:

- `chunks` (default) — current behavior, returns chunks with scores
- `documents` — returns full source documents that matched, deduplicated by source path, with best chunk score as the document score
- `auto` — chunks if total result size < `max_chars` threshold (default 8000), otherwise collapses to documents

This is **client-decided**, not folder-enforced — different MCP clients want different things from the same corpus.

## API Changes

### New parameter: `return_mode`

Added to the `search` MCP tool and `Searcher.search()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `return_mode` | string | `"chunks"` | Result granularity: `chunks`, `documents`, or `auto` |

### New config: `search.max_chars`

| Config path | Type | Default | Description |
|------------|------|---------|-------------|
| `search.max_chars` | int | `8000` | Threshold for `auto` mode (total result chars) |

### Return shapes

#### `chunks` mode (existing, unchanged)

```json
{
  "results": [
    {
      "chunk_id": "corpus:abc123",
      "summary": "Heading or first lines",
      "location": "/path/to/doc.md#heading",
      "corpus": "my-corpus",
      "topic": "finance",
      "score": 0.95
    }
  ],
  "total": 5,
  "query": "embedding",
  "return_mode": "chunks"
}
```

#### `documents` mode

```json
{
  "results": [
    {
      "source_path": "/path/to/doc.md",
      "corpus": "my-corpus",
      "topic": "finance",
      "score": 0.95,
      "text": "# Full document text...",
      "char_count": 12000,
      "matched_chunks": 3
    }
  ],
  "total": 2,
  "query": "embedding",
  "return_mode": "documents"
}
```

#### `auto` mode

Returns `chunks` shape if total result text size < `max_chars`, otherwise returns `documents` shape. The `return_mode` field in the response indicates which mode was actually used.

## Implementation Plan

### 1. `src/guaipeca/config.py`

- Add `max_chars: int = 8000` to `SearchConfig`
- Parse `max_chars` from YAML `search` section
- Add validation in `_validate_types()`

### 2. `src/guaipeca/search.py`

- Add `return_mode` parameter to `Searcher.search()`
- Add `max_chars` parameter to `Searcher.search()` (defaults to config value)
- Add `_collapse_to_documents()` method that:
  - Groups chunk results by `source_path`
  - For each unique source path, reads the full document text from disk
  - Sets document score = best (highest) chunk score from that document
  - Counts matched chunks per document
  - Returns list of document-level result dicts
- Add `_estimate_chunks_size()` helper to sum text lengths of chunk results
- In `auto` mode: compute chunks, check total size vs `max_chars`, collapse if needed
- Include `return_mode` in the returned dict

### 3. `src/guaipeca/mcp_server.py`

- Add `return_mode` to the search tool definition's `inputSchema`
- Pass `return_mode` from MCP arguments to `Searcher.search()`

### 4. `tests/test_return_mode.py` (new file)

- Test `chunks` mode (backward compat — existing behavior unchanged)
- Test `documents` mode (deduplication, full text, best score)
- Test `auto` mode (below threshold → chunks, above → documents)
- Test edge cases: no results, single result, mixed corpora
- Test MCP tool schema includes `return_mode`

### 5. `README.md`

- Document the `return_mode` parameter in the MCP Tools section
- Document `search.max_chars` in the Configuration section

## Edge Cases

### Document too large
Documents are returned in full — the LLM/client decides what to do with large text. No truncation is applied. This is by design: the client requested `documents` mode and should handle large payloads.

### No chunks found
If search returns no results, `documents` mode returns an empty results list (same as `chunks` mode). No error.

### Mixed corpora
Documents from different corpora are deduplicated independently (by `source_path`, which is unique per corpus path). Each document result includes its `corpus` and `topic` fields.

### Source file deleted between indexing and search
If a source file no longer exists when trying to read full text for `documents` mode, that document is skipped (graceful degradation). Chunk results from deleted files are already filtered in the search pipeline.

### `auto` mode threshold
`auto` compares the total character count of all chunk **texts** (retrieved via `get_chunk`) against `max_chars`. If below, returns chunks. If at or above, collapses to documents.

## Backward Compatibility

- `return_mode` defaults to `"chunks"` — existing clients see no change
- `Searcher.search()` signature adds `return_mode` and `max_chars` as keyword-only params with defaults
- Existing tests continue to pass (no changes to chunk-mode behavior)
- The `return_mode` field in the response is additive (new key in the dict)