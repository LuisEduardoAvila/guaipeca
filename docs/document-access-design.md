# Document Access: get_document, list_documents, and /download Endpoint

**Date:** 2026-09-14
**Status:** Implemented

## Overview

Three new capabilities to give MCP clients and HTTP users better access to
indexed documents:

1. **`get_document`** MCP tool — retrieve full converted text of a document
2. **`list_documents`** MCP tool — enumerate indexed documents in a corpus
3. **`/download/{corpus}/{filename}`** HTTP endpoint — download original binary files

Additionally, search results are updated to include `source_path` and
`filename` fields so clients can discover documents and construct download
URLs.

## Design Decisions

### get_document

- **Returns converted text (markdown), not raw bytes.** MCP clients are LLMs
  that consume text. The existing `Converter` class is reused.
- **Lookup by `corpus` + `source_path`.** This matches how documents are
  tracked in the index (file_hashes keyed by source_path).
- **Returns metadata:** source_path, corpus, filename, file size, chunk
  count, and the converted text.
- **Error handling:** returns a clear MCP error result if the document is
  not found or the corpus doesn't exist.

### list_documents

- **Lists indexed documents** — uses file_hashes (the index's record of
  tracked files) as the source of truth for what's indexed.
- **Optional corpus filter.** If no corpus specified, lists across all
  corpora.
- **Returns:** source_path, filename, corpus, chunk count, file size,
  last_indexed (from file_hashes mtime).
- **Chunk count per file** is computed from the loaded metadata (chunks
  whose source_path matches).

### /download/{corpus}/{filename} HTTP Endpoint

- **Serves the original binary file** — PDF, DOCX, etc. as-is.
- **Proper headers:** `Content-Type` from mimetypes, `Content-Disposition`
  with `attachment; filename="..."`.
- **Auth required:** same Bearer token check as other HTTP endpoints.
- **Path traversal protection:** reject paths containing `..` or absolute
  paths. Only the filename (basename) is used to construct the path.
- **Corpus must exist** in config. The filename is resolved within the
  corpus's configured path.

### Search Result Document References

- **chunks mode results** gain `source_path` and `filename` fields.
- **documents mode results** already include `source_path`; add
  `filename` too.
- These fields let clients call `get_document` or construct `/download`
  URLs from search results.

## Implementation

### mcp_server.py

- Add `get_document` and `list_documents` to `TOOLS` list
- Add handlers in `_handle_tool_call`
- Add `/download/{corpus}/{filename}` route in `do_GET`
- Path traversal check in download handler

### index.py

- Add `list_documents()` method to `CorpusIndex` — returns list of indexed
  files with metadata (chunk count, file size, last_indexed timestamp)
- Add `get_document_text()` method to `CorpusIndex` — returns converted
  text for a source_path within the corpus

### search.py

- Add `source_path` and `filename` to chunk-level search results
- Add `filename` to document-level search results
- Add `list_documents()` and `get_document()` methods to `Searcher`
  (delegates to corpus indexes)

## Test Plan

- `test_get_document` — returns text, metadata; error on not found
- `test_list_documents` — lists files, filters by corpus, chunk counts
- `test_download_endpoint` — serves file, auth, path traversal protection
- `test_search_result_references` — chunks and documents include
  source_path + filename