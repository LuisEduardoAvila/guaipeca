"""MCP server for Guaipeca.

Exposes ten tools via Model Context Protocol:
- search: semantic search across corpora (returns summary + location)
- get_chunk: retrieve full chunk text by chunk_id
- get_document: retrieve full document text by corpus and source_path
- list_documents: list all indexed documents in a corpus
- index: trigger indexing for a corpus or all
- status: get corpus statistics
- list_folders: list corpora that accept file uploads
- upload: upload a file to a corpus for conversion and indexing
- delete: delete a file from an upload-enabled corpus
- get_toc: get table of contents for a corpus or document

Transports:
- stdio: newline-delimited JSON-RPC over stdin/stdout
- http: MCP-compliant SSE-based transport (GET /sse + POST /messages)
- both: stdio in a daemon thread + HTTP in main thread
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from . import __version__
from .config import GuaipecaConfig
from .converter import Converter
from .embedding import EmbeddingService
from .search import Searcher

logger = logging.getLogger(__name__)

# MCP protocol version
MCP_VERSION = "2024-11-05"

# Tool definitions
TOOLS = [
    {
        "name": "search",
        "description": (
            "Search across indexed document corpora. Returns results with "
            "summary, location, corpus name, topic, and relevance score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5).",
                    "default": 5,
                },
                "corpora": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of corpus names to search. Default: all.",
                },
                "hybrid": {
                    "type": "boolean",
                    "description": "Enable BM25 hybrid search (overrides config). Default: false.",
                    "default": False,
                },
                "return_mode": {
                    "type": "string",
                    "description": (
                        "Result granularity: 'chunks' returns individual chunks (default), "
                        "'documents' returns full source documents deduplicated by path, "
                        "'auto' returns chunks if total size < max_chars threshold, "
                        "otherwise documents."
                    ),
                    "default": "chunks",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Threshold for auto return_mode (total result chars). Default: 8000.",
                    "default": 8000,
                },
                "section_mode": {
                    "type": "boolean",
                    "description": "If true, group results by section and return full section text. Default: false.",
                    "default": False,
                },
                "section_filter": {
                    "type": "string",
                    "description": "Restrict search to a specific section ID or heading path prefix. Example: 'Chapter 3'.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_chunk",
        "description": (
            "Retrieve the full text of a specific chunk by its chunk_id. "
            "Use after search to get complete content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "string",
                    "description": "Chunk ID from search results (format: corpus:stable_id).",
                },
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "get_document",
        "description": (
            "Retrieve the full converted text (markdown) of a document by "
            "corpus name and source_path. Returns text content plus metadata "
            "(filename, file size, chunk count)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "corpus": {
                    "type": "string",
                    "description": "Corpus name containing the document.",
                },
                "source_path": {
                    "type": "string",
                    "description": "Absolute path to the source file (from list_documents or search results).",
                },
            },
            "required": ["corpus", "source_path"],
        },
    },
    {
        "name": "list_documents",
        "description": (
            "List all indexed documents in a corpus (or all corpora if no "
            "corpus specified). Returns source_path, filename, chunk count, "
            "file size, and last_indexed timestamp for each document."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "corpus": {
                    "type": "string",
                    "description": "Corpus name to list documents for, or omit for all corpora.",
                },
            },
        },
    },
    {
        "name": "index",
        "description": (
            "Trigger indexing for a specific corpus or all corpora. "
            "Incremental by default (only indexes changed files)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "corpus": {
                    "type": "string",
                    "description": "Corpus name to index, or 'all' for all corpora.",
                    "default": "all",
                },
                "force": {
                    "type": "boolean",
                    "description": "Force re-index all files (ignore hash cache).",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "status",
        "description": "Get status of all corpora: chunk counts, indexed files.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "list_folders",
        "description": (
            "List corpora that accept file uploads via the upload tool. "
            "Returns corpus name, path, topic, and allowed extensions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "upload",
        "description": (
            "Upload a file to a corpus for conversion and indexing. "
            "The file content must be base64-encoded. "
            "Only corpora configured in upload.allow can receive files."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "corpus": {
                    "type": "string",
                    "description": "Target corpus name (must be in upload.allow list).",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename including extension (e.g. 'report.pdf').",
                },
                "content": {
                    "type": "string",
                    "description": "Base64-encoded file content.",
                },
                "index": {
                    "type": "boolean",
                    "description": "If true (default), index the corpus after upload.",
                    "default": True,
                },
            },
            "required": ["corpus", "filename", "content"],
        },
    },
    {
        "name": "get_toc",
        "description": (
            "Get the table of contents for a corpus or a specific document. "
            "Returns a nested tree of headings with chunk counts, or a list of "
            "documents with their top-level headings. Non-structured documents "
            "return a structured=false flag."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "corpus": {
                    "type": "string",
                    "description": "Corpus name to get ToC for.",
                },
                "document": {
                    "type": "string",
                    "description": "Optional: specific document source path to get ToC for. If omitted, lists all documents.",
                },
            },
            "required": ["corpus"],
        },
    },
    {
        "name": "delete",
        "description": (
            "Delete a file from a corpus that accepts uploads. "
            "Only corpora configured in upload.allow can have files deleted. "
            "The corpus is re-indexed after deletion to remove stale chunks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "corpus": {
                    "type": "string",
                    "description": "Target corpus name (must be in upload.allow list).",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename including extension (e.g. 'report.pdf').",
                },
                "index": {
                    "type": "boolean",
                    "description": "If true (default), re-index the corpus after deletion.",
                    "default": True,
                },
            },
            "required": ["corpus", "filename"],
        },
    },
]


def _sanitize_error_message(exc: Exception) -> str:
    """Sanitize error messages for client responses.

    Removes internal file paths and library details that could
    leak system information. Full exception is logged internally.
    """
    msg = str(exc)
    # Replace absolute file paths with generic placeholder.
    # Only match paths that look like real filesystem paths (3+ segments,
    # or 2 segments with a file extension) to avoid sanitizing short
    # slash-prefixed tokens like "/api/v1" in error messages.
    import re
    _path_re = re.compile(r"(?:^|\s)/(?:[\w.-]+/){2,}[\w.-]+|(?:^|\s)/[\w.-]+/[\w-]+\.[\w]+")
    msg = _path_re.sub(" [path]", msg)
    # Truncate overly long messages
    if len(msg) > 300:
        msg = msg[:300] + "..."
    return msg


def _check_model_cached(embedder: EmbeddingService) -> bool:
    """Check if the embedding model is cached locally.

    Returns True if the model appears to be cached, False otherwise.
    """
    try:
        cache_dir = embedder.cache_dir
        if cache_dir and os.path.isdir(cache_dir):
            # Check for any subdirectories (fastembed caches models as dirs)
            entries = os.listdir(cache_dir)
            if entries:
                return True
        # Also check default fastembed cache
        default_cache = os.path.expanduser("~/.cache/fastembed")
        if os.path.isdir(default_cache):
            return True
    except Exception:
        pass
    return False


class GuaipecaMCPServer:
    """MCP server exposing Guaipeca retrieval tools."""

    def __init__(self, config: GuaipecaConfig):
        self.config = config
        self.converter = Converter(
            cache_dir=str(Path(config.indexing.data_dir) / "converted")
        )
        self.embedder = EmbeddingService(
            model_name=config.embedding.model,
            cache_dir=config.embedding.cache_dir,
            dimensions=config.embedding.dimensions,
        )

        # Check if model is cached before starting
        if not _check_model_cached(self.embedder):
            logger.warning(
                "=" * 60 + "\n"
                "⚠️  MODEL NOT CACHED LOCALLY\n"
                f"   Model: {config.embedding.model}\n"
                f"   Cache: {config.embedding.cache_dir}\n"
                "   The server will start but the first search/index\n"
                "   operation will require network access to download\n"
                "   the model (~90MB).\n"
                "   To pre-download: run 'guaipeca --check-model'\n"
                "   Or run once with network access.\n"
                + "=" * 60
            )
        else:
            logger.info(f"Model {config.embedding.model} found in cache")

        # Eager model loading: trigger model download/load at startup so
        # the first search/index request isn't slow.
        self._warmup_model()

        self.searcher = Searcher(
            config=config,
            embedding_service=self.embedder,
            converter=self.converter,
        )

    def _warmup_model(self) -> None:
        """Load the embedding model at startup with a warmup call."""
        logger.info("Loading embedding model...")
        try:
            self.embedder.embed(["warmup"])
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(
                "Failed to load embedding model. If this is a network issue, "
                "ensure the model is pre-downloaded. Run 'guaipeca --check-model' "
                f"to verify. Error: {e}"
            )
            # Don't block startup — model will be loaded lazily on first use

    def handle_request(self, method: str, params: dict) -> dict | None:
        """
        Handle a single MCP JSON-RPC request.

        Returns the result dict, or None for notifications (which require no response).
        """
        params = params or {}

        # Handle notifications — these are one-way messages from the client
        # that should NOT receive a response (no error, no result)
        if method.startswith("notifications/"):
            logger.debug(f"Received notification: {method}")
            return None

        if method == "initialize":
            return {
                "protocolVersion": MCP_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "guaipeca",
                    "version": __version__,
                },
            }

        elif method == "tools/list":
            return {"tools": TOOLS}

        elif method == "tools/call":
            return self._handle_tool_call(params)

        elif method == "ping":
            return {}

        else:
            return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

    def _handle_tool_call(self, params: dict) -> dict:
        """Handle a tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "search":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)
            corpora = arguments.get("corpora")
            hybrid = arguments.get("hybrid")
            return_mode = arguments.get("return_mode", "chunks")
            max_chars = arguments.get("max_chars")
            section_mode = arguments.get("section_mode", False)
            section_filter = arguments.get("section_filter")
            result = self.searcher.search(
                query, top_k=top_k, corpora=corpora, hybrid=hybrid,
                return_mode=return_mode, max_chars=max_chars,
                section_mode=section_mode, section_filter=section_filter,
            )
            return self._result_to_mcp(result)

        elif tool_name == "get_chunk":
            chunk_id = arguments.get("chunk_id", "")
            result = self.searcher.get_chunk(chunk_id)
            if result is None:
                return self._error_result(f"Chunk not found: {chunk_id}")
            return self._result_to_mcp(result)

        elif tool_name == "get_document":
            corpus = arguments.get("corpus", "")
            source_path = arguments.get("source_path", "")
            result = self.searcher.get_document(corpus, source_path)
            if result is None:
                return self._error_result(
                    f"Document not found: {source_path} in corpus {corpus}"
                )
            return self._result_to_mcp(result)

        elif tool_name == "list_documents":
            corpus = arguments.get("corpus")
            result = self.searcher.list_documents(corpus=corpus)
            if "error" in result:
                return self._error_result(result["error"])
            return self._result_to_mcp(result)

        elif tool_name == "index":
            corpus = arguments.get("corpus", "all")
            force = arguments.get("force", False)
            result = self.searcher.index_corpus(corpus_name=corpus, force=force)
            return self._result_to_mcp(result)

        elif tool_name == "status":
            return self._result_to_mcp(self.searcher.status)

        elif tool_name == "list_folders":
            return self._result_to_mcp(self._list_upload_folders())

        elif tool_name == "upload":
            corpus = arguments.get("corpus", "")
            filename = arguments.get("filename", "")
            content_b64 = arguments.get("content", "")
            do_index = arguments.get("index", self.config.upload.auto_index)
            result = self._handle_upload(corpus, filename, content_b64, do_index)
            if result.get("error"):
                return self._error_result(result["error"])
            return self._result_to_mcp(result)

        elif tool_name == "get_toc":
            corpus = arguments.get("corpus", "")
            document = arguments.get("document")
            result = self.searcher.get_toc(corpus=corpus, document=document)
            if "error" in result:
                return self._error_result(result["error"])
            return self._result_to_mcp(result)

        elif tool_name == "delete":
            corpus = arguments.get("corpus", "")
            filename = arguments.get("filename", "")
            do_index = arguments.get("index", self.config.upload.auto_index)
            result = self._handle_delete(corpus, filename, do_index)
            if result.get("error"):
                return self._error_result(result["error"])
            return self._result_to_mcp(result)

        else:
            return self._error_result(f"Unknown tool: {tool_name}")

    @staticmethod
    def _result_to_mcp(result: Any) -> dict:
        """Convert a result dict to MCP content format."""
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2, ensure_ascii=False),
                }
            ]
        }

    @staticmethod
    def _error_result(message: str) -> dict:
        """Create an MCP error result."""
        return {
            "isError": True,
            "content": [
                {"type": "text", "text": message},
            ],
        }

    def _list_upload_folders(self) -> dict:
        """List corpora that accept file uploads."""
        allowed = self.config.upload.allow
        folders = []
        for corpus in self.config.corpora:
            if corpus.name in allowed:
                folders.append({
                    "name": corpus.name,
                    "path": corpus.path,
                    "topic": corpus.topic,
                    "extensions": corpus.extensions,
                })
        return {
            "folders": folders,
            "total": len(folders),
            "allowed_extensions": self.config.upload.allowed_extensions,
            "max_file_size": self.config.upload.max_file_size,
        }

    def _handle_upload(
        self, corpus_name: str, filename: str, content_b64: str, do_index: bool
    ) -> dict:
        """Handle a file upload to a corpus.

        Validates the corpus is upload-enabled, checks extension and size,
        saves the file, and optionally triggers indexing.

        Args:
            corpus_name: Target corpus name.
            filename: Filename with extension.
            content_b64: Base64-encoded file content.
            do_index: Whether to index the corpus after upload.

        Returns:
            Result dict with saved path and optional index stats.
        """
        import base64

        # Validate corpus exists and is upload-enabled
        if corpus_name not in self.config.upload.allow:
            return {"error": f"Corpus '{corpus_name}' does not accept uploads"}

        corpus = self.config.get_corpus(corpus_name)
        if corpus is None:
            return {"error": f"Corpus not found: {corpus_name}"}

        # Validate filename
        filename = os.path.basename(filename)  # strip any path components
        if not filename:
            return {"error": "Invalid filename"}

        ext = Path(filename).suffix.lower()
        if ext not in self.config.upload.allowed_extensions:
            return {
                "error": f"Extension '{ext}' not allowed. Permitted: {self.config.upload.allowed_extensions}"
            }

        # Decode content
        try:
            content_bytes = base64.b64decode(content_b64)
        except Exception:
            return {"error": "Invalid base64 content"}

        # Check size limit
        if len(content_bytes) > self.config.upload.max_file_size:
            max_mb = self.config.upload.max_file_size / (1024 * 1024)
            return {"error": f"File too large ({len(content_bytes)} bytes, max {max_mb:.0f}MB)"}

        # Save to corpus directory
        corpus_path = Path(corpus.path)
        corpus_path.mkdir(parents=True, exist_ok=True)
        dest = corpus_path / filename

        # Write atomically (temp + rename)
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(corpus_path), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(content_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, str(dest))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return {"error": f"Failed to save file: {filename}"}

        logger.info(f"Uploaded {filename} to corpus '{corpus_name}' ({len(content_bytes)} bytes)")

        result = {
            "corpus": corpus_name,
            "filename": filename,
            "path": str(dest),
            "size_bytes": len(content_bytes),
            "saved": True,
        }

        # Optionally index the corpus
        if do_index:
            index_result = self.searcher.index_corpus(corpus_name=corpus_name)
            result["index"] = index_result
            result["indexed"] = True
        else:
            result["indexed"] = False

        return result

    def _handle_delete(
        self, corpus_name: str, filename: str, do_index: bool
    ) -> dict:
        """Handle a file deletion from a corpus.

        Validates the corpus is upload-enabled, checks the file exists,
        deletes it, and optionally triggers re-indexing.
        """
        if corpus_name not in self.config.upload.allow:
            return {"error": f"Corpus '{corpus_name}' does not accept uploads (delete not allowed)"}

        corpus = self.config.get_corpus(corpus_name)
        if corpus is None:
            return {"error": f"Corpus not found: {corpus_name}"}

        filename = os.path.basename(filename)
        if not filename:
            return {"error": "Invalid filename"}

        corpus_path = Path(corpus.path)
        file_path = corpus_path / filename

        try:
            real_path = file_path.resolve()
            real_corpus = corpus_path.resolve()
            if not str(real_path).startswith(str(real_corpus)):
                return {"error": "Invalid filename (path traversal detected)"}
        except Exception:
            return {"error": "Invalid filename"}

        if not file_path.exists():
            return {"error": f"File not found: {filename}"}

        if not file_path.is_file():
            return {"error": f"Not a file: {filename}"}

        try:
            file_size = file_path.stat().st_size
            os.unlink(str(file_path))
        except OSError as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return {"error": f"Failed to delete file: {filename}"}

        logger.info(f"Deleted {filename} from corpus '{corpus_name}' ({file_size} bytes)")

        result = {
            "corpus": corpus_name,
            "filename": filename,
            "path": str(file_path),
            "size_bytes": file_size,
            "deleted": True,
        }

        if do_index:
            index_result = self.searcher.index_corpus(corpus_name=corpus_name)
            result["index"] = index_result
            result["indexed"] = True
        else:
            result["indexed"] = False

        return result

    # ---- shared JSON-RPC processing ----

    def _process_jsonrpc(self, raw_body: bytes) -> dict | None:
        """Process a raw JSON-RPC request body and return a response dict.

        Shared between stdio and HTTP POST handlers to avoid duplication.

        Args:
            raw_body: Raw bytes (or string) containing a JSON-RPC request.

        Returns:
            Response dict (jsonrpc, id, result/error) or None for notifications.
            Returns an error response dict if parsing fails.
        """
        request = None
        try:
            if isinstance(raw_body, bytes):
                request = json.loads(raw_body)
            else:
                request = json.loads(raw_body)
            method = request.get("method", "")
            params = request.get("params", {})
            req_id = request.get("id")
            result = self.handle_request(method, params)

            # Notifications have no id and should not receive a response
            if result is None or req_id is None:
                return None

            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except json.JSONDecodeError:
            logger.error("Invalid JSON in request body")
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
        except Exception as e:
            logger.error(f"Error handling JSON-RPC request: {e}", exc_info=True)
            req_id = request.get("id") if request is not None else None
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": _sanitize_error_message(e)},
            }

    # ---- stdio transport ----

    def run_stdio(self):
        """Run MCP server over stdio transport."""
        logger.info("Starting Guaipeca MCP server (stdio)")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            response = self._process_jsonrpc(line.encode("utf-8"))
            if response is None:
                continue

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    # ---- HTTP transport (MCP-compliant SSE) ----

    def run_http(self, host: str = "127.0.0.1", port: int = 8090):
        """
        Run MCP server over HTTP transport using Server-Sent Events.

        MCP HTTP transport:
        - GET /sse — opens an SSE stream, server sends an `endpoint` event with a POST URL
        - POST /messages?session_id=<id> — client sends JSON-RPC messages via POST
        - Server pushes responses back through the SSE stream
        - GET /health — health check endpoint
        - GET /tools — list available tools (convenience endpoint)

        If auth_token is configured, validates Authorization: Bearer <token> header.
        """
        import http.server
        import urllib.parse
        from queue import Empty, Queue

        server_instance = self
        auth_token = self.config.server.auth_token
        # Restrict CORS to non-wildcard when auth is enabled
        cors_origin = "*" if not auth_token else "null"

        # Session management: each SSE connection gets a session with a message queue
        sessions: dict[str, SSESession] = {}
        sessions_lock = threading.Lock()

        class SSESession:
            """Represents a single SSE client session."""
            def __init__(self, client_address: str):
                self.session_id = str(uuid.uuid4())
                self.client_address = client_address
                self.response_queue: Queue = Queue()
                self.active = True
                # The POST endpoint URL for this session
                self.post_endpoint = f"/messages?session_id={self.session_id}"

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _check_auth(self) -> bool:
                """Check if the request has valid auth. Returns True if authorized."""
                if not auth_token:
                    return True
                auth_header = self.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:]
                    return hmac.compare_digest(token, auth_token)
                return False

            def _send_json(self, code: int, data: dict):
                body = json.dumps(data).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_sse_event(self, event: str, data: str):
                """Send a single SSE event to the client."""
                self.wfile.write(f"event: {event}\n".encode())
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)

                # SSE endpoint — opens a persistent event stream
                if parsed.path == "/sse":
                    if not self._check_auth():
                        self._send_json(401, {"error": "unauthorized"})
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", cors_origin)
                    self.end_headers()

                    # Create a new session
                    session = SSESession(self.client_address[0])
                    with sessions_lock:
                        sessions[session.session_id] = session
                    logger.info(f"SSE session started: {session.session_id} from {self.client_address[0]}")

                    # Send the endpoint event so the client knows where to POST messages
                    self._send_sse_event("endpoint", session.post_endpoint)

                    # Keep the SSE stream open and push responses as they arrive
                    try:
                        while session.active:
                            try:
                                response_data = session.response_queue.get(timeout=15)
                                # Check if it's a close signal
                                if response_data is None:
                                    break
                                self._send_sse_event("message", response_data)
                            except Empty:
                                # Send a comment as keepalive ping
                                self.wfile.write(b": ping\n\n")
                                self.wfile.flush()
                    except (ConnectionError, BrokenPipeError):
                        logger.info(f"SSE connection closed for session {session.session_id}")
                    finally:
                        session.active = False
                        with sessions_lock:
                            sessions.pop(session.session_id, None)
                        logger.info(f"SSE session ended: {session.session_id}")
                    return

                # Health check
                if parsed.path == "/health":
                    self._send_json(200, {"status": "ok", "server": "guaipeca", "version": __version__})
                    return

                # Tools listing (convenience endpoint)
                if parsed.path == "/tools":
                    if not self._check_auth():
                        self._send_json(401, {"error": "unauthorized"})
                        return
                    self._send_json(200, {"tools": TOOLS})
                    return

                # File download endpoint: /download/{corpus}/{filename}
                if parsed.path.startswith("/download/"):
                    if not self._check_auth():
                        self._send_json(401, {"error": "unauthorized"})
                        return

                    # Parse corpus and filename from path
                    # URL path: /download/{corpus}/{filename}
                    parts = parsed.path.split("/", 3)  # ['', 'download', 'corpus', 'filename']
                    if len(parts) < 4:
                        self._send_json(400, {"error": "path must be /download/{corpus}/{filename}"})
                        return

                    corpus_name = parts[2]
                    filename = urllib.parse.unquote(parts[3])

                    # Path traversal protection: reject .. and absolute paths
                    if ".." in filename or filename.startswith("/"):
                        self._send_json(400, {"error": "invalid filename"})
                        return
                    if ".." in corpus_name or corpus_name.startswith("/"):
                        self._send_json(400, {"error": "invalid corpus"})
                        return

                    # Look up corpus in config
                    corpus_cfg = server_instance.config.get_corpus(corpus_name)
                    if corpus_cfg is None:
                        self._send_json(404, {"error": f"corpus not found: {corpus_name}"})
                        return

                    # Resolve file path within corpus directory
                    corpus_path = Path(corpus_cfg.path)
                    file_path = corpus_path / filename

                    # Final safety check: ensure resolved path is within corpus
                    try:
                        file_path.resolve().relative_to(corpus_path.resolve())
                    except ValueError:
                        self._send_json(400, {"error": "path outside corpus directory"})
                        return

                    if not file_path.exists() or not file_path.is_file():
                        self._send_json(404, {"error": f"file not found: {filename}"})
                        return

                    # Serve the file with proper headers
                    import mimetypes
                    content_type, _ = mimetypes.guess_type(str(file_path))
                    if content_type is None:
                        content_type = "application/octet-stream"

                    file_size = file_path.stat().st_size
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(file_size))
                    self.send_header(
                        "Content-Disposition",
                        f'attachment; filename="{filename}"'
                    )
                    self.end_headers()
                    with open(file_path, "rb") as f:
                        self.wfile.write(f.read())
                    return

                # Unknown path
                self._send_json(404, {"error": "not found"})

            def do_POST(self):
                parsed = urllib.parse.urlparse(self.path)
                query_params = urllib.parse.parse_qs(parsed.query)

                # Message endpoint — client sends JSON-RPC messages here
                if parsed.path == "/messages":
                    if not self._check_auth():
                        self._send_json(401, {"error": "unauthorized"})
                        return

                    session_id = query_params.get("session_id", [None])[0]
                    with sessions_lock:
                        if not session_id or session_id not in sessions:
                            self._send_json(400, {
                                "jsonrpc": "2.0",
                                "id": None,
                                "error": {"code": -32000, "message": "invalid or missing session_id"},
                            })
                            return
                        session = sessions[session_id]

                    content_length = int(self.headers.get("Content-Length", 0))
                    # Enforce max body size to prevent memory exhaustion
                    max_body_size = 10 * 1024 * 1024  # 10MB limit
                    if content_length > max_body_size:
                        self._send_json(413, {"error": "request body too large"})
                        return
                    body = self.rfile.read(content_length) if content_length > 0 else b"{}"

                    # Accept the request immediately (202 Accepted)
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

                    # Process the request and push the response through SSE
                    response = server_instance._process_jsonrpc(body)
                    if response is not None:
                        session.response_queue.put(json.dumps(response))
                    return

                # Unknown POST path
                self._send_json(404, {"error": "not found"})

            def do_DELETE(self):
                """Handle DELETE requests (file deletion)."""
                parsed = urllib.parse.urlparse(self.path)
                query_params = urllib.parse.parse_qs(parsed.query)

                # DELETE /documents?corpus=<name>&filename=<name>
                if parsed.path == "/documents":
                    if not self._check_auth():
                        self._send_json(401, {"error": "unauthorized"})
                        return

                    corpus_name = query_params.get("corpus", [None])[0]
                    filename = query_params.get("filename", [None])[0]

                    if not corpus_name or not filename:
                        self._send_json(400, {"error": "corpus and filename required"})
                        return

                    result = server_instance._handle_delete(corpus_name, filename, do_index=True)
                    if result.get("error"):
                        self._send_json(400, result)
                    else:
                        self._send_json(200, result)
                    return

                self._send_json(404, {"error": "not found"})

            def do_OPTIONS(self):
                """Handle CORS preflight requests."""
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", cors_origin)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.end_headers()

            def log_message(self, format, *args):
                logger.debug(f"HTTP {format % args}")

        # Use ThreadingHTTPServer for concurrent request handling
        # (SSE connections are long-lived, so single-threaded would block)
        actual_port = port
        for attempt in range(5):
            try:
                httpd = http.server.ThreadingHTTPServer((host, actual_port), Handler)
                break
            except OSError:
                actual_port += 1
                logger.warning(f"Port {actual_port - 1} in use, trying {actual_port}")
        else:
            raise RuntimeError(f"Could not find available port starting at {port}")

        logger.info(f"Starting Guaipeca MCP server (HTTP/SSE) on {host}:{actual_port}")
        print(f"Guaipeca MCP server listening on http://{host}:{actual_port}", file=sys.stderr)

        import signal
        def _handle_term(signum, frame):
            logger.info("Received SIGTERM, shutting down HTTP server")
            httpd.shutdown()
        signal.signal(signal.SIGTERM, _handle_term)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("HTTP server shutting down")
            httpd.shutdown()

    def run(self, transport: str = "both", host: str = "127.0.0.1", port: int = 8090):
        """Run the server with the specified transport(s)."""
        if transport == "stdio":
            self.run_stdio()
        elif transport == "http":
            self.run_http(host, port)
        elif transport == "both":
            # Run stdio in a daemon thread, HTTP in main
            stdio_thread = threading.Thread(target=self.run_stdio, daemon=True)
            stdio_thread.start()
            self.run_http(host, port)
        else:
            raise ValueError(f"Unknown transport: {transport}")