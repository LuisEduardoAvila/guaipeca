# Guaipeca 🧠

**Self-hosted RAG MCP server that runs where others can't**

> **Why "Guaipeca"?** — _Guaipeca_ is southern Brazilian slang (Tupi-Guarani origin) for a scrappy mutt — a _cusco_, no pedigree, no frills, but loyal and gets the job done. Seemed fitting for a lightweight RAG server with no API keys, no GPU, and no cloud dependencies.

Guaipeca is a self-hosted retrieval-augmented generation (RAG) server that exposes semantic search over your document corpora via the Model Context Protocol (MCP). It uses a lightweight tech stack — FAISS indexes, fastembed (ONNX Runtime) embeddings, structure-aware chunking — and is fully configurable and decoupled from any specific memory system.

## Why Guaipeca?

Most RAG solutions assume you have a GPU, a cloud budget, or at least a beefy server. Guaipeca doesn't.

- **No API keys** — runs entirely locally, no OpenAI/Anthropic/Google calls
- **No GPU** — ONNX Runtime on CPU, works on ARM64, x86_64, anything Linux
- **No cloud** — your documents never leave your machine
- **No torch** — fastembed (~46MB) instead of sentence-transformers (~959MB)
- **Under 500MB total** — embedding model + FAISS index + dependencies
- **Deploy in minutes** — `pip install`, one YAML config, `guaipeca serve`

Proven on a Raspberry Pi 5 (8GB RAM, ARM64) alongside other services. If it runs there, it runs anywhere.

## Features

- **Multi-corpus** — define multiple document folders, each with its own weight and topic label
- **MCP-native** — exposes `search`, `get_chunk`, `get_document`, `list_documents`, `index`, `status`, `list_folders`, `upload`, `delete`, and `get_toc` tools via MCP
- **Multi-transport** — stdio (for local LLM integration) + Streamable HTTP (`POST /mcp`) for remote/network access. One endpoint serves both the handshake (2025-11-25 and earlier) and stateless (2026-07-28) protocol eras.
- **Hybrid search** — optional BM25 sparse keyword search fused with FAISS dense vectors via Reciprocal Rank Fusion (RRF)
- **Any document format** — pymupdf (PDFs with font-based heading detection + line-wrap stitching) + markitdown (DOCX, PPTX, XLSX, HTML → markdown)
- **Structure-aware chunking** — splits on headings, treats tables as atomic units, zero LLM calls
- **Local embeddings** — fastembed (ONNX Runtime) on CPU, no GPU or API keys needed
- **Incremental indexing** — SHA256 file hashing, only re-indexes changed files
- **FAISS backup + recovery** — automatic `.fai.bak` backup before writes, fallback chain on load (primary → backup → fresh)
- **Embedding cache (LRU + TTL)** — 10K-entry LRU cache with 1hr TTL, memory-pressure aware eviction via psutil
- **Content preprocessing** — heading + first 500 chars sent to embedder for better semantic signal; full text stored in metadata
- **Concurrent** — read-write lock allows multiple simultaneous searches, exclusive indexing
- **Runs anywhere** — ARM64 (Pi 5, ARM servers), x86_64 (laptops, VMs, dedicated servers), any Linux box with Python 3.12+ (3.13 recommended, and what the Docker image and Pi deployment run). Proven on Pi 5 with Python 3.13, ~90MB model + ~20MB FAISS
- **Optional auth** — Bearer token authentication for HTTP transport

## Quick Start

**Which one?** Docker for a fresh box, a VM, or anything you want to reach from another
machine; the venv install for an existing single host — a laptop, or a Pi already running
other services.

### Quick Start (venv)

```bash
git clone https://github.com/LuisEduardoAvila/guaipeca.git
cd guaipeca
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

mkdir -p ~/.guaipeca
cp config/guaipeca.yaml ~/.guaipeca/guaipeca.yaml
```

Then edit `~/.guaipeca/guaipeca.yaml` and point the one corpus it defines at a folder that
exists on this machine:

```yaml
# change this
- name: docs
  path: /path/to/your/docs

# to this (any folder you already have notes in)
- name: notes
  path: /home/you/projects/my-notes
```

The sample ships a single corpus on purpose. Add more by copying that block — each needs its
own folder, and `guaipeca index` exits 1 naming any corpus whose folder does not exist. While
the config is still byte-identical to the bundled sample, `guaipeca index` refuses with exit 2.

```bash
guaipeca --check-model      # downloads the embedding model (~90MB, CPU only)
guaipeca index
guaipeca search "something you know is in your documents"
guaipeca serve              # MCP server on port 8090
```

**PEP 668:** a bare `pip install` fails on Debian 12+ and Ubuntu 23.04+ with
`externally-managed-environment` — that is why the venv step exists.

### Quick Start (Docker)

```bash
# 1. Prepare data directory (run from the cloned repo: docker-compose.yml lives there)
mkdir -p data/{config,corpora,index,models}
cp config/guaipeca.container.yaml data/config/guaipeca.yaml
# Optionally add a server.auth_token to data/config/guaipeca.yaml first

# 2. Put your documents in data/corpora/<corpus-name>/
#    The container config defines one corpus named epm-docs, so:
#    data/corpora/epm-docs/yourfile.pdf
#    NOT directly in data/corpora/ - files one level up match no corpus.

# 3. Start the container
docker compose up -d

# 4. Check health
curl http://localhost:8090/health

# 5. Index corpora (first time)
docker compose exec guaipeca guaipeca --config /data/config/guaipeca.yaml index
```

If you rename or add a corpus in `data/config/guaipeca.yaml`, create the matching
`data/corpora/<name>/` directory first.

## Configuration

Create `~/.guaipeca/guaipeca.yaml`:

```yaml
corpora:
  - name: epm-docs
    path: /path/to/your/docs
    topic: "EPM & Oracle HFM/ARCS"
    weight: 1.5
    extensions: [.md, .txt, .pdf, .docx]

  - name: finance-reference
    path: /path/to/your/finance-docs
    topic: "Finance & Accounting Standards"
    weight: 1.3
    extensions: [.md, .txt, .pdf]

embedding:
  model: all-MiniLM-L6-v2
  dimensions: 384
  preprocess: true              # heading + first 500 chars for better embeddings
  cache_ttl_seconds: 3600      # embedding cache entry TTL (1 hour)
  cache_max_entries: 10000     # max cached embeddings (LRU eviction)

chunking:
  max_size: 2000
  overlap: 200
  table_aware: true
  split_on_headings: true

server:
  transport: both  # stdio | http | both (see Transports below)
  port: 8090
  host: 127.0.0.1  # Use 0.0.0.0 for network access
  # auth_token: "your-secret-token"  # Optional: require Bearer token auth

upload:
  allow: ["epm-docs"]  # corpora that accept file uploads
  max_file_size: 52428800  # 50MB
  allowed_extensions: [.md, .txt, .pdf, .docx]
  auto_index: true  # index after upload

# Search configuration (optional — all values have safe defaults)
search:
  hybrid: true               # Enable BM25 + FAISS hybrid search (default: true; see tools/eval_retrieval.py)
  bm25_weight: 1.0           # BM25 contribution weight in RRF fusion
  dense_weight: 1.0          # Dense (FAISS) contribution weight in RRF fusion
  rrf_k: 60                  # RRF constant (higher = smoother ranking)
  max_chars: 8000            # Threshold for auto return_mode (total result chars)
```

### Transports

`server.transport` accepts three values:

| Value | Behaviour |
|-------|-----------|
| `stdio` | MCP over stdin/stdout. For local clients that launch Guaipeca as a subprocess. |
| `http` | MCP over Streamable HTTP (`POST /mcp`) plus the non-MCP endpoints (`/health`, `/tools`, `/download/...`, `DELETE /documents`). For a daemon serving the network. |
| `both` (default) | stdio when launched as a subprocess, HTTP when launched as a daemon. This is **not** two simultaneous listeners — the MCP SDK runs one server per transport, so `both` picks one based on how the process was started. |

`sse` is not a valid value. The legacy SSE transport is served by `transport: http`
alongside Streamable HTTP — it is not selectable on its own, because there is
nothing to select: the SSE routes are always mounted on the same port.
`transport: http` serves `POST /mcp` for both protocol eras (the older
session-based handshake and the newer stateless requests) **and** `GET /sse` +
`POST /messages`. A config still set to `transport: sse` fails validation with
an error pointing at `http`.

`POST /mcp` is the preferred transport. The server says so in three places, so
a client finds it whichever path it takes: the MCP `initialize` `instructions`,
the `/tools` response (`transports.preferred`), and a `Deprecation: true` +
`Link: </mcp>` header on the `GET /sse` stream. SSE is kept for clients that
cannot use Streamable HTTP; it is deprecated upstream and will be removed in a
future release.

## Embedding Models

Guaipeca uses [FastEmbed](https://github.com/qdrant/fastembed) (ONNX Runtime) for local embeddings — no torch, no GPU, no API keys. Models are downloaded automatically on first use and cached locally.

**Content preprocessing** is enabled by default (`embedding.preprocess: true`): before embedding, each chunk is preprocessed to its heading + first 500 characters, giving the model better semantic signal. The full text is still stored in metadata for retrieval. Disable with `preprocess: false` if you want raw chunk text sent to the embedder.

### Choosing a Model

Pick based on your use case and hardware:

#### Ultra-light (384-dim, <100MB) — edge devices, SBCs, 2GB RAM
| Model | Dimensions | Size | Best for |
|-------|-----------|------|----------|
| `all-MiniLM-L6-v2` | 384 | 90MB | General use, fast (default) |
| `BAAI/bge-small-en-v1.5` | 384 | 67MB | Better quality, same speed |
| `snowflake/arctic-embed-xs` | 384 | 90MB | General use, compact |

#### Balanced (512–768 dim, 120–520MB) — desktops, small VMs, laptops
| Model | Dimensions | Size | Best for |
|-------|-----------|------|----------|
| `jinaai/jina-embeddings-v2-small-en` | 512 | 120MB | Long documents (8192 tokens) |
| `BAAI/bge-base-en-v1.5` | 768 | 210MB | Higher quality, moderate speed |
| `nomic-ai/nomic-embed-text-v1.5-Q` | 768 | 130MB | Quantized, good balance |
| `snowflake/arctic-embed-s` | 384 | 130MB | Good quality, compact |

#### Quality-optimized (768–1024 dim, 420MB+) — dedicated servers, workstations
| Model | Dimensions | Size | Best for |
|-------|-----------|------|----------|
| `BAAI/bge-large-en-v1.5` | 1024 | 1.2GB | Best quality, slowest |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 640MB | High quality |
| `jinaai/jina-embeddings-v2-base-en` | 768 | 520MB | Long context (8192 tokens) |

#### Multilingual
| Model | Dimensions | Size | Best for |
|-------|-----------|------|----------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 220MB | ~50 languages, fast |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 1.0GB | ~50 languages, better quality |
| `intfloat/multilingual-e5-large` | 1024 | 2.2GB | ~100 languages, best quality |

#### Code
| Model | Dimensions | Size | Best for |
|-------|-----------|------|----------|
| `jinaai/jina-embeddings-v2-base-code` | 768 | 640MB | Code + docs, 30+ programming languages |

### Switching Models

1. Update `embedding.model` and `embedding.dimensions` in your config
2. Reindex with `--force` (FAISS vectors must match the new dimensions):

```bash
guaipeca index --force
```

> ⚠️ Changing dimensions (e.g. 384 → 768) requires a full reindex. The old index is incompatible with the new model's vectors.

### Recommendations

| Use case | Recommended model | Why |
|----------|-------------------|-----|
| **Edge / SBC (Pi 5, ARM64, 2GB RAM)** | `all-MiniLM-L6-v2` | Fast, 90MB, good enough quality |
| **Small VM / laptop (4GB RAM)** | `BAAI/bge-small-en-v1.5` | Better quality, still 384-dim |
| **Dedicated server / workstation** | `BAAI/bge-base-en-v1.5` | 768-dim, noticeable quality gain |
| **Long documents** | `jinaai/jina-embeddings-v2-base-en` | 8192 token context |
| **Multilingual** | `paraphrase-multilingual-MiniLM-L12-v2` | 50+ languages, 384-dim |
| **Code search** | `jinaai/jina-embeddings-v2-base-code` | Code + docs, 30+ languages |

## MCP Tools

### search
Search across indexed corpora. Returns summary + location + score.
```
search(query="HFM data audit", top_k=5, corpora=["epm-docs"], hybrid=true, return_mode="chunks")
search(query="consolidation", section_mode=true)
search(query="revenue", section_filter="Financial Statements")
```

**section_mode** groups results by document section, returning full section text instead of individual chunks.
**section_filter** restricts search to a specific section (by section ID or heading path prefix).

**return_mode** controls result granularity:
- `chunks` (default) — returns individual chunks with summary, location, and score
- `documents` — returns full source documents, deduplicated by path, with best chunk score as document score
- `auto` — returns chunks if total result size < `search.max_chars` (default 8000), otherwise collapses to documents

**max_distance** sets a relevance floor for dense hits, so an off-topic query can return an
empty result set instead of the least-bad chunk:

```python
search(query="HFM data audit", top_k=5, corpora=["epm-docs"], max_distance=0.55)
```

Distance is `1 - cosine similarity`, so **lower is stricter**: `0.55` keeps hits whose cosine is
at least `0.45`. It is configurable per corpus and globally (`search.max_distance`), and as
`--max-distance` on the CLI. Omitting it preserves the previous behaviour exactly.

Behaviour worth knowing:

- **BM25-only hits are never filtered.** They have no dense score to judge, and hybrid exists to
  catch exact-token matches the embedder misses — an error code or product name is often a
  low-similarity, high-value hit. They report `"distance": null`.
- **All hits filtered is a success, not an error**: `{"results": [], "total": 0,
  "max_distance_applied": true, "filtered_by_distance": N}`. A partial filter reports the same
  counter, so a shrunk result set can be told apart from a query that simply matched less.
- **No default is shipped.** The usable value depends on corpus, embedding model, and query mix.
  Measure it with `tools/eval_retrieval.py --calibrate`, which reports the on-topic and off-topic
  distance distributions per corpus; pick the separation point. Corpora small enough to have no
  distribution (single-digit chunk counts) cannot be calibrated and should be left unset.
- `status` reports the active floor per corpus, so an empty result can be attributed to filtering
  rather than a broken index.

> **Note:** `dedup_sections` is **not** an MCP tool argument. It is an option on the internal `Searcher.search()` method (`dedup_sections=true`), available when using Guaipeca as a Python library — it collapses results sharing a `section_id`, keeping only the best-scoring chunk per section. It is not exposed over MCP, and the `search` tool does not accept it.

> **Note on table formatting:** Raw chunk output (`return_mode="chunks"`) may show markdown tables with rows run-together (no blank lines between rows). This is expected — the chunker treats tables as atomic units and preserves row adjacency. For properly formatted tables, use `get_document` or `get_chunk` to retrieve the full text with original formatting.

### get_chunk
Retrieve full chunk text by chunk_id from search results.
```
get_chunk(chunk_id="epm-docs:a1b2c3d4e5f67890")
```

### get_document
Retrieve the full converted text (markdown) of a document by corpus and source_path. Returns text content plus metadata (filename, file size, chunk count).
```
get_document(corpus="epm-docs", source_path="/path/to/docs/report.pdf")
```

### list_documents
List all indexed documents in a corpus (or all corpora if no corpus specified). Returns source_path, filename, chunk count, file size, and last_indexed timestamp.
```
list_documents(corpus="epm-docs")
list_documents()  # all corpora
```

### index
Trigger indexing for a corpus or all corpora.
```
index(corpus="all", force=false)
```

### status
Get corpus statistics: chunk counts, indexed files.
```
status()
```

### list_folders
List corpora that accept file uploads via the upload tool. Returns corpus name, path, topic, and allowed extensions.
```
list_folders()
```

### upload
Upload a file to a corpus for conversion and indexing. File content must be base64-encoded. Only corpora listed in `upload.allow` can receive files.
```
upload(corpus="epm-docs", filename="report.pdf", content="<base64>", index=true)
```

### get_toc
Get the table of contents for a corpus or a specific document. Returns a nested tree of headings with chunk counts, or a list of all documents with their top-level headings. Non-structured documents return a `structured: false` flag.
```
get_toc(corpus="epm-docs")
get_toc(corpus="epm-docs", document="/path/to/report.md")
```

### delete
Delete a file from a corpus that accepts uploads. Only corpora listed in `upload.allow` can have files deleted. The corpus is re-indexed after deletion to remove stale chunks.
```
delete(corpus="epm-docs", filename="report.pdf", index=true)
```

### ToC-Aware Search Features

Guaipeca supports three STAIR-inspired features that leverage document structure:

- **Section mode** (`section_mode=true`): Groups search results by their document section, concatenating all chunks in each section. Useful for retrieving complete sections rather than fragments.
- **Section filter** (`section_filter="heading text"`): Restricts search to chunks within a specific section. Useful for searching within a chapter or section.
- **ToC re-ranking** (`search.toc_rerank: true` in config): Boosts results whose heading path contains query terms. Non-structured documents are not re-ranked.

**Non-structured documents** (no heading markers) degrade gracefully: `heading_path` is empty, `section_id` uses `"unstructured:{hash}"`, `get_toc` returns `structured: false`, `section_filter` returns all chunks from that document, and `toc_rerank` skips re-ranking.

**`heading_path` semantics** (v0.3.4+): `heading_path` contains only the document title (`#`, level 1) and the enclosing section heading (`##`, level 2). Deep headings (`###`, `####`, etc.) are part of the section content but do NOT appear in `heading_path`. This ensures `section_id` is always derived from the `##` section heading, so chunks within the same section share the same `section_id`, and chunks from different sections never collide.

## CLI

```bash
guaipeca --check-model                 # Check/download embedding model
guaipeca index [corpus] [--force]      # Index corpora
guaipeca search "query" [--top-k N]    # Search
guaipeca search "query" --hybrid       # Hybrid search (BM25 + FAISS)
guaipeca search "query" --return-mode auto --max-chars 12000   # Auto mode
guaipeca search "query" --section-mode --section-filter "Chapter 3"
guaipeca status                        # Show stats
guaipeca get-chunk <chunk_id>          # Retrieve full chunk text
guaipeca get-document --corpus C --path /abs/file.md   # Retrieve document text
guaipeca list-documents [corpus]       # List indexed documents
guaipeca list-folders                  # List upload-enabled corpora
guaipeca get-toc --corpus C [--document /abs/file.md]  # Table of contents
guaipeca upload --corpus C <file>      # Upload a file and index it
guaipeca delete --corpus C <filename>  # Delete a file and re-index
guaipeca serve [--transport both] [--port 8090]  # Start MCP server
```

The CLI mirrors the MCP tools one-for-one (except `serve`, which is
CLI-only). Read/upload/delete commands print the same JSON payloads the MCP
server returns. `guaipeca search` accepts the same knobs as the MCP `search`
tool: `--return-mode`, `--max-chars`, `--section-mode`, and `--section-filter`
(in addition to `--top-k`, `--corpora`, and `--hybrid`). A parity test
(`tests/test_cli_mcp_parity.py`) fails if the two surfaces drift apart.

## Running as a systemd Service

For always-on deployment on any Linux system (Pi 5, VM, laptop, server), install Guaipeca as a systemd service.

### 1. Create the service file

`guaipeca.service` is gitignored and absent from a fresh clone, so the tracked template is
the source: it carries the `CHANGEME` user placeholder and `%h`-relative paths, which is why
the real unit is not tracked.

```bash
sed -e 's|CHANGEME|'"$USER"'|g' guaipeca.service.template > /tmp/guaipeca.service
sudo cp /tmp/guaipeca.service /etc/systemd/system/guaipeca.service
sudo systemctl daemon-reload
sudo systemctl enable --now guaipeca
```

The template's `ExecStart` assumes `~/.local/bin/guaipeca`, where a `pipx install` or
`pip install --user` puts the executable — not the repo's `.venv/bin/` from Quick Start.
Install with pipx (recommended for a daemon), or edit that line to the venv path.

### 2. Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable guaipeca   # start on boot
sudo systemctl start guaipeca    # start now
```

This is a no-op if you already ran `enable --now` in step 1; it is the equivalent when you
edited the unit or copied it in by hand.

### 3. Verify

```bash
systemctl status guaipeca
curl http://127.0.0.1:8090/health
# → {"status": "ok", "server": "guaipeca", "version": "0.5.0"}
```

### 4. Check logs

```bash
journalctl -u guaipeca -f          # follow logs
journalctl -u guaipeca --since "1 hour ago"
```

### Management

```bash
sudo systemctl stop guaipeca       # stop
sudo systemctl restart guaipeca    # restart
sudo systemctl disable guaipeca    # stop starting on boot
```

### Notes

- The service runs `guaipeca serve` with `--transport http` — stdio is not usable under
  systemd, where the unit has no stdin
- Port 8090 is bound to `127.0.0.1` by default (config `server.host`). Use `0.0.0.0` for network access
- Resource limits (`MemoryMax`, `CPUQuota`) prevent Guaipeca from starving other services on shared or low-resource hosts
- The embedding model loads at startup (~1s with fastembed/ONNX). First search/index may take slightly longer if the model needs to download
- Config changes require a restart: `sudo systemctl restart guaipeca`
- Re-index after adding files: `guaipeca --config ~/.guaipeca/guaipeca.yaml index`

## Architecture

```
guaipeca.yaml (config)
       │
  Config Loader
       │
  ┌────────────┬────────────┐
  │            │            │
Indexer     Searcher     MCP Server
  │            │            │
  │            │            ├── stdio transport
  │            │            └── HTTP: Streamable (/mcp, port 8090)
  │            │
  │            └── FAISS per-corpus + weighted merge
  │            
  └── pymupdf (PDF) ─┐
  └── markitdown (other) ─┤
          ↓                 │
       chunk → embed → FAISS
       (read-write lock: concurrent readers, exclusive writer)
```

## Data Layout

```
~/.guaipeca/
├── guaipeca.yaml          # Config
├── data/
│   ├── corpora/
│   │   ├── epm-docs/
│   │   │   ├── faiss_index.fai
│   │   │   ├── faiss_index.fai.bak  # backup (auto-generated)
│   │   │   ├── metadata.json
│   │   │   ├── file_hashes.json
│   │   │   └── bm25_index/       # BM25 sparse keyword index
│   │   └── finance-reference/
│   │       ├── faiss_index.fai
│   │       ├── faiss_index.fai.bak  # backup (auto-generated)
│   │       ├── metadata.json
│   │       ├── file_hashes.json
│   │       └── bm25_index/       # BM25 sparse keyword index
│   └── converted/          # conversion cache (pymupdf + markitdown)
└── models/
    └── all-MiniLM-L6-v2/   # cached embedding model
```

## Dependencies

| Package | Purpose | Size |
|---------|---------|------|
| fastembed | Local embeddings (ONNX Runtime) | ~46MB (onnxruntime) + ~90MB (model) |
| faiss-cpu | Vector index | ~20MB |
| pymupdf | PDF heading detection | ~25MB |
| markitdown | Document conversion (DOCX, PPTX, XLSX, HTML) | ~30MB |
| bm25s | BM25 sparse keyword search | <1MB |
| pyyaml | Config parsing | <1MB |
| numpy | Array operations | ~15MB |

## ARM64 / Low-Resource System Dependencies

markitdown relies on libraries that require system-level packages on ARM64 and some minimal Linux installations.
Install these before `pip install`:

```bash
# Debian/Ubuntu/Raspberry Pi OS
sudo apt install libxml2-dev libxslt-dev libffi-dev libjpeg-dev

# If using PDF conversion:
sudo apt install poppler-utils

# If using DOCX conversion:
sudo apt install antiword
```

If you encounter build errors with `lxml` or `charset-normalizer` on ARM64,
ensure you have `build-essential` and `python3-dev` installed:

```bash
sudo apt install build-essential python3-dev
```

## Container Deployment (Docker)

Guaipeca ships with a Docker container for easy deployment on any platform (ARM64 or x86_64). Run it on a VM, a cloud instance, or alongside other containers — the image is multi-arch and ~643MB before model download. The commands are in [Quick Start (Docker)](#quick-start-docker); this section covers the deployment details around them.

### Building from Source

```bash
docker build -t guaipeca .
docker run -d --name guaipeca \
  -p 8090:8090 \
  -v ./data:/data \
  --restart unless-stopped \
  guaipeca
```

> **Note:** Pre-built images will be available on ghcr.io after the first
> push to `main` triggers the CI/CD pipeline. Until then, build from source
> using the command above.

### Container Data Layout

```
data/              → /data (mounted volume)
├── config/        → /data/config     (guaipeca.yaml)
├── corpora/       → /data/corpora    (source documents)
├── index/         → /data/index      (FAISS + BM25 + metadata)
└── models/        → /data/models     (embedding model cache)
```

The embedding model (~90MB) downloads to the volume on first run. Subsequent starts load from cache — no network needed.

Documents live in `data/corpora/<corpus-name>/`, one directory per corpus name defined in
`data/config/guaipeca.yaml` — never directly in `data/corpora/`, which belongs to no corpus.

### Automated Rebuilds

Push to `main` triggers a GitHub Actions workflow that:
- Builds multi-arch images (arm64 + amd64)
- Tags with `latest` and commit SHA
- Pushes to `ghcr.io/luis-eduardo-avila/guaipeca`

## Security

- **Default binding:** The HTTP server binds to `127.0.0.1` (localhost) by default.
  Change `host: 0.0.0.0` in config only if you need network access.
- **Authentication:** Set `auth_token` in the server config to require
  `Authorization: Bearer <token>` headers on HTTP requests. When auth is enabled,
  CORS is restricted (no wildcard origin).
- **Origin validation (DNS-rebinding protection):** browser requests carry an
  `Origin` header, and incoming requests with an unrecognised origin are
  rejected with `403 Forbidden` before the auth check. By default only loopback
  origins are accepted (`http://localhost:<port>`, `http://127.0.0.1:<port>`,
  `http://[::1]:<port>`, any port). Requests with **no** `Origin` header (curl,
  CLI tools, non-browser MCP clients) are always allowed. If you serve a browser
  UI from any other host — including over a LAN address — you must add it
  explicitly, or its requests will be rejected:
  ```yaml
  server:
    allowed_origins:
      - https://guaipeca.example.com   # exact match, scheme included
      - http://192.168.1.20:3000
  ```
  Setting `allowed_origins` replaces the loopback default (it does not add to
  it), so list every origin you need. Allowed origins are echoed back in
  `Access-Control-Allow-Origin`.
- **HTTP endpoints:**
  - `POST /mcp` — Streamable HTTP transport (MCP protocol version `2025-11-25`); JSON-RPC request in, JSON-RPC response inline. `initialize` issues an `Mcp-Session-Id` header used on subsequent requests. Modern clients should prefer this endpoint. A client may send `MCP-Protocol-Version` on subsequent requests; an absent header is treated as the `2025-03-26` baseline, an unknown value gets `400`.
  - `DELETE /mcp` — terminate a Streamable HTTP session (send the `Mcp-Session-Id` header)
  - `GET /mcp` — returns `405 Method Not Allowed` (Guaipeca has no server-initiated messages; use `POST /mcp`)
  - `GET /sse`, `POST /messages` — legacy SSE transport, still served but **deprecated**. Use `POST /mcp` instead; the stream carries `Deprecation: true` and a `Link` header pointing at `/mcp`.
  - `GET /health` — health check
  - `GET /tools` — list available tools
  - `GET /download/{corpus}/{filename}` — download original file (requires auth, path traversal protected)
  - `DELETE /documents?corpus=<name>&filename=<name>` — delete a file from an upload-enabled corpus (requires auth)
- **Error messages:** Client error responses are sanitized to avoid leaking
  internal file paths or library details.

## License

MIT