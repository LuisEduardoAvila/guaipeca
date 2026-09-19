# Guaipeca Architecture

This document describes the internal architecture of Guaipeca, covering the data pipeline, concurrency model, crash safety, and MCP transport.

## Pipeline Overview

```
guaipeca.yaml (config)
       │
  Config Loader (config.py)
       │
  ┌────────────────────────────────────────┐
  │                                        │
  ▼                                        ▼
Indexer (index.py)                   Searcher (search.py)
  │                                        │
  ├─ Converter (converter.py)              ├─ Embed query
  │   ├─ PDF → pymupdf (heading detect)    │
  │   └─ Other → markitdown                │
  ├─ Chunker (chunking.py)                 ├─ FAISS search per corpus
  │   ├─ Split on ## headings              │   (read lock, over-fetch ×3)
  │   ├─ Table-aware (atomic)             │
  │   ├─ Code-block-aware (never split)   ├─ BM25 search per corpus (hybrid mode)
  │   └─ Overlap (200 chars default)       │   (read lock, over-fetch ×3)
  ├─ Embedder (embedding.py)              │
  │   ├─ Content preprocessing (heading+500)  │
  │   ├─ LRU+TTL cache (10K entries, 1hr)    │
  │   └─ fastembed (L2 norm)              ├─ RRF Fusion (k=60) when hybrid
  ├─ FAISS IndexFlatIP (per corpus)       │
  │   └─ .fai.bak backup before writes       │
  ├─ BM25 Index (bm25s, per corpus)       ├─ Weighted merge by corpus weight
  ├─ Atomic writes (temp + rename)       │
  └─ File locking (fcntl.flock)           └─ Top-k results
       │
  ▼
MCP Server (mcp_server.py)
  ├─ 10 tools: search, get_chunk, get_document, list_documents, index, status, list_folders, upload, delete, get_toc
  ├─ stdio transport (newline JSON-RPC)
  ├─ HTTP/SSE transport (ThreadingHTTPServer)
  ├─ Optional Bearer token auth
  └─ Error sanitization (path scrubbing, truncation)
```

## PDF Conversion Pipeline

Guaipeca uses a two-tier conversion strategy:

### PDFs → pymupdf with heading detection

Location: `converter.py`, `_convert_pdf_with_headings()` method (lines 73–170).

**Two-pass approach:**

1. **Font-size histogram** — Iterates all pages, collecting font sizes via `page.get_text("dict")`. Builds a histogram of font sizes weighted by character count. The most common size is identified as body text.

2. **Heading detection & markdown emission** — Sizes significantly larger than body text (threshold: 2pt above body) are mapped to heading levels (H1–H6, largest → H1). Re-iterates pages, emitting `#` markers for heading-sized text and plain text for body content.

**Fallback:** If pymupdf is unavailable, extracts no text, or no heading sizes are detected (uniform-font document), falls back to markitdown.

**Caching:** Converted output is cached by SHA256 file hash in `data/converted/`.

### Other formats → markitdown

DOCX, PPTX, XLSX, HTML, CSV, XML, and other supported formats are converted by markitdown. `.md` and `.txt` files pass through directly (no conversion).

## Chunking

Location: `chunking.py`

### Structure-aware splitting

- **Split on `##` (level 2) headings** — Level 1 (`#`) is treated as document title (included in first chunk). Level 3+ (`###`, `####`) stay with their parent `##` section.
- **Table-aware** — Pipe-delimited markdown tables are extracted and replaced with placeholders before chunking, then restored. Tables are never split across chunks.
- **Code-block-aware** — Fenced code blocks (``` or ~~~) are never split mid-block. A code block exceeding `max_size` is kept as one chunk (even if oversized). Closing fence must have ≥ opening fence length (CommonMark spec compliance).
- **Paragraph fallback** — When a section exceeds `max_size`, it is split by paragraph boundaries (respecting code blocks).

### Overlap

The last N characters (default 200) from the end of one chunk are carried into the start of the next chunk within a section. This prevents concepts from being split across chunk boundaries. Overlap is extracted only from original (non-overlap) content to prevent compounding across multiple splits (P2-4 fix).

### Stable chunk IDs

Chunk IDs are SHA256-based: `SHA256(source_path:heading:offset)[:16]`. This makes them stable across index rebuilds — the same file/heading/offset always produces the same ID. A lookup map (`_stable_id_to_pos`) maps stable IDs to FAISS positional indices for retrieval.

Location: `chunking.py` `_make_chunk_id()` (lines 82–85), `index.py` `_rebuild_stable_id_map()` (lines 206–208).

## Embedding

Location: `embedding.py`

- **Model:** fastembed (ONNX Runtime, default: `all-MiniLM-L6-v2`, 384 dimensions)
- **Normalization:** L2-normalized embeddings for cosine similarity via FAISS inner product (`IndexFlatIP`). fastembed does not normalize by default, so embeddings are L2-normalized after generation.
- **Batch encoding:** `embed()` processes batches for indexing efficiency
- **Query encoding:** `embed_query()` embeds a single query string
- **Model warmup:** The model is loaded at server startup with a warmup `embed(["warmup"])` call (`mcp_server.py` `_warmup_model()`), ensuring the first search/index request isn't slow
- **Cache directory:** `cache_dir` is expanded via `os.path.expanduser()` in both `EmbeddingConfig.__post_init__` and `EmbeddingService.__init__`
- **Content preprocessing:** See [Embedding Pipeline](#embedding-pipeline) section below
- **Embedding cache:** See [Embedding Pipeline](#embedding-pipeline) section below

## Data Integrity

Location: `index.py` `_save()` (lines 267–296), `_load_faiss_with_recovery()` (lines 253–278).

### FAISS backup + corruption recovery

Every time the FAISS index is saved, the existing `.fai` file is copied to `.fai.bak` **before** the new index is written. This creates a one-generation backup that can be recovered if the primary file is corrupted.

**Write sequence** (`_save()`):
1. Metadata JSON written (atomic: temp + rename)
2. Existing `faiss_index.fai` → copied to `faiss_index.fai.bak` (via `shutil.copy2`)
3. New FAISS index written atomically (temp + rename)
4. File hashes written (atomic)
5. BM25 index saved (best-effort)

If the process crashes during step 3, the `.fai.bak` from step 2 still holds the previous valid index.

**Load-time fallback chain** (`_load_faiss_with_recovery()`):
1. Try reading `faiss_index.fai` — if valid, use it
2. If missing or corrupted, try `faiss_index.fai.bak` — log warning, recover
3. If both fail, create a fresh empty `IndexFlatIP` — log warning, start from scratch

All fallback paths log warnings so operators can detect corruption events. The recovery is fully automatic — no config or manual intervention needed.

### Crash-safe write ordering

The write order in `_save()` is designed so that a crash at any point leaves the system in a consistent state:

1. **Metadata first** — If the process crashes after metadata is written but before FAISS, the old FAISS index still matches old metadata (indices align).
2. **FAISS second** — Once FAISS is written, it matches the new metadata. The backup from step 2 holds the previous generation.
3. **File hashes last** — Hashes are the least critical; if lost, files are simply re-indexed on next run.

## Embedding Pipeline

Location: `embedding.py`

### Content preprocessing

Before embedding, chunk text is preprocessed to improve semantic signal: the heading is prepended to the first 500 characters of the chunk body. This gives the embedding model a concise, heading-anchored representation rather than a potentially long, diluted chunk.

- **Preprocessing function:** `EmbeddingService.preprocess_text(text, heading, max_chars=500)` — static method
- **Applied during:** indexing (`embed()` when `headings` are provided) and stale-vector rebuilds (`_rebuild_index()`)
- **Not applied to:** query embeddings (`embed_query()`) — queries are already short user input
- **Full text preserved:** The complete chunk text is stored in `metadata.json` for retrieval via `get_chunk()` — only the embedding input is preprocessed
- **Config:** `embedding.preprocess: true` (default). Set to `false` to send raw chunk text to the embedder.

### Embedding cache (LRU + TTL)

An in-process LRU cache prevents redundant embedding computations on repeated content (e.g. re-indexing unchanged files after a force reindex of a subset).

**Cache implementation:** `_EmbeddingCache` class (OrderedDict-based).

| Property | Value |
|----------|-------|
| Max entries | 10,000 (configurable: `embedding.cache_max_entries`) |
| TTL | 3,600 seconds / 1 hour (configurable: `embedding.cache_ttl_seconds`) |
| Cache key | `SHA256(model_name:text)` — includes model name so switching models doesn't return stale vectors |
| Eviction | LRU (oldest accessed entry evicted when full) + TTL (expired entries evicted on access) |
| Memory pressure | When system RAM usage exceeds 85% (via `psutil`), cache aggressively evicts down to 70% of max size |
| psutil fallback | If `psutil` is not installed, falls back to `/proc/meminfo` on Linux; if neither available, memory-pressure eviction is skipped |

**API:**
- `embedder.clear_cache()` — clears all entries (called on `force=True` reindex in `_do_index()`)
- `embedder.cache_stats` — returns dict with `hits`, `misses`, `evictions`, `size`, `ttl_seconds`

**Stats are cumulative** across the process lifetime (not reset on `clear()`).

## Indexing

Location: `index.py`

### Incremental indexing

- **SHA256 file hashing** — Each file's content is hashed. If the hash matches the stored value in `file_hashes.json`, the file is skipped.
- **Force re-index** — When `force=True`, the index is reset (FAISS, chunks, hashes, stable ID map) before re-processing all files. This ensures idempotent re-indexing without duplicate vectors.

### Atomic writes

All persistent state is written atomically using the temp-file-then-rename pattern:

- **Metadata JSON** (`metadata.json`) — `_atomic_write_json()` (lines 167–183): writes to temp file, `fsync`, then `os.rename`.
- **FAISS index** (`faiss_index.fai`) — `_atomic_write_faiss()` (lines 184–200): `faiss.write_index` to temp file, then `os.rename`.
- **File hashes** (`file_hashes.json`) — same atomic JSON pattern.

This ensures a crash during write doesn't corrupt files — the old version remains intact until the rename succeeds.

### FAISS backup before writes

Before overwriting `faiss_index.fai`, the existing file is copied to `faiss_index.fai.bak` (see Data Integrity section above). This provides a one-generation backup for corruption recovery.

### Load-time validation and fallback

On load (`_load_faiss_with_recovery()`), the primary `.fai` file is read first. If it's missing or corrupted (FAISS throws), the `.fai.bak` backup is tried. If both fail, a fresh empty index is created. All fallback paths log warnings.

### Cache clearing on force reindex

When `force=True` is passed to `index()`, the embedding cache is cleared via `embedder.clear_cache()` to ensure all embeddings are recomputed fresh (no stale cached vectors from a previous model or configuration).

### Crash-safe write ordering

When saving after indexing (`_save()` method, line 265):

1. **Metadata first** — If the process crashes after metadata is written but before FAISS, the old FAISS index still matches old metadata (indices align).
2. **FAISS second** — Once FAISS is written, it matches the new metadata.
3. **File hashes last** — Hashes are the least critical; if lost, files are simply re-indexed on next run.

### File locking

Location: `index.py` `_acquire_lock()` / `_release_lock()` (lines 163–188).

Uses `fcntl.flock(LOCK_EX | LOCK_NB)` on a per-corpus lock file (`corpus.lock`). This prevents concurrent indexing of the same corpus by different processes. The lock is non-blocking — if another process holds it, indexing returns an error immediately rather than waiting.

### Stale vector cleanup

When files are deleted from a corpus, their vectors become "stale" (orphaned in the FAISS index).

- **Search-time filtering** — Search results check `os.path.exists(chunk.source_path)` and skip chunks whose source file no longer exists (lines 415–431).
- **Rebuild threshold** — When the stale vector ratio exceeds 10% (`STALE_REBUILD_THRESHOLD = 0.10`), the entire index is rebuilt from scratch, keeping only chunks from existing files (`_rebuild_index()`, lines 222–250).

### Delete and reindex

The `delete` MCP tool and `DELETE /documents` HTTP endpoint remove files from upload-enabled corpora (those listed in `upload.allow`). After deletion, an incremental reindex is triggered by default (same as upload). The incremental reindex detects the removed file via `_scan_files()` → `deleted = set(self._file_hashes.keys()) - current_files`, removes its hash, and relies on the stale vector cleanup mechanisms above (search-time filtering + rebuild threshold). A `force=True` reindex is not needed for correctness — users can request one separately via the `index` tool if desired.

## Concurrency Model

Location: `index.py` `_ReadWriteLock` class (lines 30–77).

Guaipeca uses a custom read-write lock per `CorpusIndex`:

- **Multiple concurrent readers** — Search operations acquire read locks, allowing multiple simultaneous searches on the same corpus.
- **Exclusive writer** — Indexing operations acquire write locks, blocking all searches on that corpus during the write phase.
- **Writer priority** — Waiting writers are prioritized over new readers (tracked via `_waiting_writers` counter), preventing writer starvation.

The lock is implemented using `threading.Condition` with reader/writer counters. Read locks are released quickly (after copying index state), and FAISS search runs outside the lock to minimize contention.

## MCP Server

Location: `mcp_server.py`

### Tools

Seven tools are exposed:

| Tool | Purpose |
|------|---------|
| `search` | Semantic search across corpora (returns summary + location + score) |
| `get_chunk` | Retrieve full chunk text by chunk_id |
| `index` | Trigger indexing for a corpus or all corpora |
| `status` | Get corpus statistics (chunk counts, indexed files) |
| `list_folders` | List corpora that accept file uploads |
| `upload` | Upload a base64-encoded file to a corpus |
| `delete` | Delete a file from an upload-enabled corpus (triggers incremental reindex) |

### Transports

- **stdio** — Newline-delimited JSON-RPC over stdin/stdout. Used for local LLM integration.
- **HTTP/SSE** — MCP-compliant Server-Sent Events transport:
  - `GET /sse` — Opens a persistent SSE stream; server sends an `endpoint` event with a POST URL
  - `POST /messages?session_id=<id>` — Client sends JSON-RPC messages via POST; server pushes responses through the SSE stream
  - `GET /health` — Health check endpoint
  - `GET /tools` — Convenience endpoint listing available tools
  - `DELETE /documents?corpus=<name>&filename=<name>` — Delete a file from an upload-enabled corpus
  - Uses `ThreadingHTTPServer` for concurrent request handling (SSE connections are long-lived)
  - Per-session message queues (`Queue`) for SSE response delivery
  - Thread-safe session dict protected by `threading.Lock`
- **both** — stdio runs in a daemon thread, HTTP in the main thread

### Authentication

- Optional Bearer token auth (`server.auth_token` config field)
- When auth is enabled, CORS is restricted (`null` origin instead of `*`)
- Error messages are sanitized: absolute file paths replaced with `[path]`, messages truncated to 300 characters

### Error sanitization

Location: `mcp_server.py` `_sanitize_error_message()` (lines 24–34).

Replaces absolute file paths with `[path]` placeholder and truncates overly long messages to 300 characters. Prevents information leakage through error responses.

## Search

Location: `search.py`

- **Multi-corpus weighted merge** — Searches all configured corpora (or a specified subset), applies corpus weights to scores, merges and sorts results.
- **Over-fetch** — Fetches `top_k * 2` results per corpus to compensate for stale entries that will be filtered out, improving merge quality.
- **Stale filtering** — Results from deleted files are filtered out at query time via `os.path.exists()` check.
- **Query length limit** — 10,000 characters max (`MAX_QUERY_LENGTH = 10_000`) to prevent DoS via memory exhaustion.

### Hybrid BM25 + FAISS Search

When `search.hybrid` is enabled (or `hybrid=True` is passed to the search tool), the Searcher runs both FAISS dense search and BM25 sparse keyword search per corpus, then fuses results using Reciprocal Rank Fusion (RRF).

**RRF Formula:** `score = weight / (k + rank)`

- `k` (default 60) — smoothing constant from the RRF paper. Higher values reduce the influence of top ranks.
- `dense_weight` / `bm25_weight` — tune the contribution of each method.
- Results appearing in both dense and sparse get higher RRF scores (sum of both contributions).
- No score normalization needed — RRF uses rank positions, not raw scores.

**BM25 Index:**
- Built during `index()` using `bm25s` library (pure Python, ARM64 compatible)
- Full rebuild on every index() call (cheap — no model inference, just token counting)
- Saved to `bm25_index/` directory alongside FAISS index
- Uses same read-write lock as FAISS (concurrent reads, exclusive writes)
- Reset on `force=True` (same as FAISS index)

## Configuration

Location: `config.py`

All config is loaded from a single YAML file. Sections:

| Section | Purpose |
|---------|---------|
| `corpora` | List of document folders with name, path, topic, weight, extensions |
| `embedding` | Model name, dimensions, cache directory, preprocessing, cache TTL/max entries |
| `chunking` | Max chunk size, overlap, table-aware, split-on-headings |
| `indexing` | Data directory, incremental flag, max file size, excluded dirs |
| `server` | Transport, port, host, auth token |
| `upload` | Allowed corpora, max file size, allowed extensions, auto-index |
| `search` | Hybrid mode, BM25/dense weights, RRF constant |

Type validation is performed in `_validate_types()` — numeric fields are checked for correct types, corpus weights default to 1.0 if non-numeric. Search config validates that `hybrid` is boolean, weights are non-negative numbers, and `rrf_k` is a positive integer.

## Container Architecture

Guaipeca can run in a Docker container for VM deployment (ARM64 or x86_64). The Pi 5 stays bare-metal; containers are for VMs.

### Container Layout

```
┌─────────────────────────────────────────────┐
│              Docker Container               │
│                                             │
│  /app/guaipeca/          # application code │
│  /data/                  # mounted volume   │
│    ├── config/           # guaipeca.yaml    │
│    ├── corpora/          # source documents │
│    ├── index/            # FAISS + BM25     │
│    └── models/           # embedding cache  │
│                                             │
│  Port 8090 → MCP HTTP/SSE                  │
│  User: guaipeca (non-root)                  │
│  Entrypoint: guaipeca serve --config ...    │
└─────────────────────────────────────────────┘
         │
         ▼
   Volume: guaipeca-data → /data
```

### Multi-stage Docker Build

**Builder stage:** Python 3.12-slim + build deps → installs all pip dependencies into a venv.

**Runtime stage:** Python 3.12-slim + runtime system libs → copies venv from builder, creates non-root user, exposes port 8090, sets health check.

This keeps the image small (~643MB before model download) and separates build-only dependencies from runtime. Uses fastembed (ONNX Runtime ~46MB) instead of sentence-transformers/torch (~959MB), reducing image size by ~63%.

### Volume Mapping

All persistent state lives in a single volume mounted at `/data`:

| Host Path | Container Path | Contents |
|-----------|---------------|----------|
| `data/config/` | `/data/config/` | `guaipeca.yaml` |
| `data/corpora/` | `/data/corpora/` | Source documents per corpus |
| `data/index/` | `/data/index/` | FAISS indexes, BM25 indexes, metadata |
| `data/models/` | `/data/models/` | Embedding model cache |

### Model Caching

The embedding model is NOT baked into the image. On first run, `_warmup_model()` in `mcp_server.py` downloads the model to `/data/models/`. Subsequent starts load from the volume — no network needed.

### CI/CD Pipeline

GitHub Actions builds multi-arch images (arm64 + amd64) on push to main:

- `build-container.yml` — Buildx multi-arch, push to ghcr.io, tag with `latest` + commit SHA
- `test.yml` — Run pytest + ruff on bare-metal runner

Docker layer caching via `cache-from/cache-to` speeds up rebuilds.

### No Code Changes Required

The existing codebase supports container deployment without modifications:
- `os.path.expanduser()` is a no-op on absolute paths (`/data/...`)
- `--config` CLI flag accepts any path
- `host: 0.0.0.0` in config binds to all interfaces
- `--transport http` selects HTTP-only mode
- `GET /health` endpoint already exists for health checks

See `docs/containerization-design.md` for the full design document and `docs/container-deployment.md` for deployment instructions.

## ToC-Aware Features

Guaipeca implements five STAIR-inspired features that leverage document structure (table of contents) for improved retrieval:

### Feature A: Heading Path Metadata

Each chunk stores a `heading_path: list[str]` tracking the hierarchy of headings from the document title down to the chunk's section. This is computed during chunking in `_split_on_headings()` by maintaining a heading stack as lines are processed.

- Level 1 (`#`) headings are document titles, included in the path
- Level 2 (`##`) headings are section boundaries, included in the path
- Level 3+ (`###`, `####`) stay with their parent section but are included in the path
- Non-structured documents (no headings): `heading_path = []`

Each chunk also stores a `section_id: str` computed as `SHA256(source_path + ":" + top_heading)[:16]`, enabling section-level grouping. For non-structured docs: `section_id = "unstructured:{hash}"`.

Location: `chunking.py` `_split_on_headings()`, `_make_section_id()`.

### Feature B: ToC Keyword Re-Ranking

When `search.toc_rerank: true` is enabled in config, search results are re-ranked by keyword overlap between query terms and heading_path terms.

- Score boost = (overlapping_terms / total_query_terms) * toc_rerank_weight
- Applied after RRF fusion (if hybrid) or after dense search, before final sort
- Results with empty heading_path (non-structured docs) are not re-ranked
- Config: `toc_rerank: false` (default), `toc_rerank_weight: 0.3`

Location: `search.py` `_rerank_by_toc()`.

### Feature C: Section Aggregation

When `section_mode=true` is passed to search, results are grouped by `section_id` and each group is concatenated into a full section text result.

- Fetches all chunks in each matched section via `section_map`
- Concatenates chunks sorted by `char_offset`
- Returns section text, heading, heading_path, and chunk_count
- Non-structured docs: all chunks in the "unstructured" section are returned as one block

Location: `search.py` `_aggregate_sections()`, `index.py` `get_section_chunks()`.

### Feature D: ToC Navigation

The `get_toc` MCP tool (8th tool) returns the table of contents for a corpus or specific document.

- If `document` specified: returns nested tree `{title, heading, level, children, chunk_count, structured}`
- If no document: returns list of `{filename, title, top_level_headings, chunk_count, structured}`
- Non-structured docs: returns `{structured: false, children: []}`
- ToC trees are built during indexing and stored in `toc_metadata.json`

Location: `index.py` `_build_toc_tree_for_doc()`, `get_toc_tree()`, `search.py` `get_toc()`, `mcp_server.py` get_toc tool.

### Feature E: Section Filter

When `section_filter` is passed to search, results are restricted to chunks within a specific section.

- Direct section_id lookup via `section_to_positions` map
- Heading path prefix matching: `section_filter="Chapter 3"` matches any section where a heading equals "Chapter 3" or the joined heading_path starts with "Chapter 3"
- Uses brute-force cosine similarity on the filtered subset (FAISS reconstruct + dot product)
- If section not found: returns empty results with error message
- Non-structured docs: filtering by "unstructured:{hash}" returns all chunks from that document

Location: `index.py` `search_section()`, `search.py` `_search_with_section_filter()`.

### Data Storage

ToC metadata is persisted in `toc_metadata.json` alongside each corpus index:
- `section_map`: `dict[str, list[str]]` mapping section_id to chunk_ids
- `toc_trees`: `dict[str, dict]` mapping source_path to ToC tree
- `section_to_positions`: `dict[str, list[int]]` mapping section_id to FAISS positions

### Migration / Backward Compatibility

Old indexes without ToC fields are handled gracefully:
- `heading_path` missing in metadata.json: defaults to `[]`
- `section_id` missing: defaults to `""`, recomputed by `_rebuild_toc_metadata()`
- `toc_metadata.json` missing: all ToC metadata is rebuilt from chunk data on load
- Chunk dataclass new fields have safe defaults: `heading_path=[]`, `section_id=""`