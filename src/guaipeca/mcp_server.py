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
        + Streamable HTTP transport (POST /mcp, DELETE /mcp)
- both: stdio in a daemon thread + HTTP in main thread
"""

from __future__ import annotations

import difflib
import hmac
import ipaddress
import json
import logging
import os
import sys
import threading
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

from . import __version__
from .config import GuaipecaConfig
from .converter import Converter
from .embedding import EmbeddingService
from .search import Searcher
from .uploads import delete_upload, list_upload_folders, save_upload

logger = logging.getLogger(__name__)


def _is_non_loopback_host(host: str) -> bool:
    """Return True if ``host`` is a bind address that is not loopback-only.

    Loopback addresses (127.0.0.0/8, ::1) and the hostname ``localhost`` are
    treated as safe (not exposing). The wildcard/unspecified addresses
    (0.0.0.0, ::) bind ALL interfaces and therefore DO expose the service and
    return True. Non-IP hostnames (other than ``localhost``) are assumed to
    expose the service.

    Used only to decide whether to emit a startup security warning.
    """
    import ipaddress
    h = (host or "").strip().lower()
    if not h:
        return True  # bind-all fallback — treat as exposed
    if h in ("localhost", "ip6-localhost", "ip6-loopback"):
        return False
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return True  # non-IP hostname — assume routable/exposed
    # Wildcard/unspecified addresses (0.0.0.0, ::) bind ALL interfaces, so
    # they expose the service. Only genuine loopback is considered safe.
    return not ip.is_loopback


def _origin_allowed(origin: str, allowed_origins: list[str] | None) -> bool:
    """Decide whether an incoming Origin header is acceptable.

    - No Origin header: always allowed (curl, CLI, non-browser clients).
    - Explicit allowlist: exact match, trailing "/" ignored.
    - Otherwise: allow only http/https on loopback hosts, any port.
      DNS rebinding always targets a public hostname, so this closes it.
    """
    if not origin:
        return True
    if allowed_origins:
        norm = origin.rstrip("/")
        return norm in {a.rstrip("/") for a in allowed_origins}
    try:
        parsed = urllib.parse.urlparse(origin)
        if parsed.scheme not in ("http", "https"):
            return False
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost", "ip6-localhost", "ip6-loopback")


# MCP protocol version (latest revision this server implements)
MCP_VERSION = "2025-11-25"

# Known protocol versions for negotiation
KNOWN_PROTOCOL_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}

# Minimum protocol version required for Streamable HTTP transport (/mcp)
MIN_STREAMABLE_VERSION = "2025-03-26"

# Tool definitions
TOOLS = [
    {
        "name": "search",
        "description": (
            "Search indexed document corpora. Pass the scope as corpora=[\"name\"] "
            "(a LIST). Omitting corpora searches EVERY corpus, so scope explicitly "
            "when the user names one. Call status first if you do not know the "
            "corpus names. Returns results with chunk_id, summary, location, "
            "corpus, topic and score; follow up with get_chunk for full text."
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
                    "description": (
                        "LIST of corpus names to search, e.g. [\"epm-docs\"]. "
                        "Other tools take corpus=\"name\" (singular); this one is "
                        "plural and takes an array. Omit to search all corpora."
                    ),
                },
                "hybrid": {
                    "type": "boolean",
                    "description": (
                        "Override BM25+dense hybrid retrieval. Omit to use the "
                        "deployment default from config (search.hybrid), which is "
                        "what you almost always want."
                    ),
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
                    "description": (
                        "Restrict search to a section ID or heading path prefix, "
                        "e.g. 'Chapter 3'. Only has effect when section_mode is "
                        "true; ignored otherwise."
                    ),
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
                    "description": "Single corpus name (not a list), e.g. 'epm-docs'.",
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
                    "description": "Single corpus name (not a list). Omit to list every corpus.",
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
                    "description": "Single corpus name (not a list), or 'all'.",
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
        "description": (
            "List the configured corpora with chunk counts, indexed files and "
            "BM25 availability. Call this first to learn valid corpus names "
            "before searching or scoping other tools."
        ),
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
                    "description": "Single corpus name (not a list); must be in upload.allow.",
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
                    "description": "Single corpus name (not a list).",
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
                    "description": (
                        "Single corpus name (not a list); must be in upload.allow."
                    ),
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
            preprocess=config.embedding.preprocess,
            cache_ttl_seconds=config.embedding.cache_ttl_seconds,
            cache_max_entries=config.embedding.cache_max_entries,
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
            # Protocol version negotiation: echo client's version if known,
            # otherwise fall back to server's default. The R3 minimum-version
            # check for /mcp is enforced in the HTTP handler, not here.
            client_version = params.get("protocolVersion", MCP_VERSION)
            if client_version in KNOWN_PROTOCOL_VERSIONS:
                negotiated = client_version
            else:
                logger.info(
                    "Unknown MCP protocol version %r requested; "
                    "falling back to %s",
                    client_version,
                    MCP_VERSION,
                )
                negotiated = MCP_VERSION
            return {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "guaipeca",
                    "version": __version__,
                },
            }

        elif method == "tools/list":
            return {"tools": self._tools_with_corpus_enum()}

        elif method == "tools/call":
            return self._handle_tool_call(params)

        elif method == "ping":
            return {}

        else:
            return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

    def _tools_with_corpus_enum(self) -> list[dict]:
        """Return TOOLS with the live corpus names advertised on corpus parameters.

        A caller that does not know the valid names has to invent one, and an
        invented name is either rejected (good) or, before validation existed,
        silently treated as "search everything" (bad). Advertising the names in
        the schema removes the guess.

        Declaring an ``enum`` also lets JSON Schema validation reject an
        unknown name before the call reaches this server.
        """
        try:
            names = sorted(self.searcher.corpora.keys())
        except Exception:
            # Status must keep working even if corpora cannot be enumerated.
            return TOOLS
        if not names:
            return TOOLS

        tools = []
        for tool in TOOLS:
            properties = dict((tool.get("inputSchema") or {}).get("properties") or {})
            changed = False
            for param in ("corpus", "corpora"):
                if param not in properties:
                    continue
                spec = dict(properties[param])
                if param == "corpora":
                    spec["items"] = {**spec.get("items", {}), "enum": names}
                else:
                    # `index` accepts the sentinel "all"; keep it selectable.
                    values = [*names, "all"] if tool["name"] == "index" else names
                    spec["enum"] = values
                properties[param] = spec
                changed = True
            if changed:
                tool = {
                    **tool,
                    "inputSchema": {**tool["inputSchema"], "properties": properties},
                }
            tools.append(tool)
        return tools

    def _handle_tool_call(self, params: dict) -> dict:
        """Handle a tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        rejection = self._reject_unknown_arguments(tool_name, arguments)
        if rejection is not None:
            return rejection

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
            # search() reports validation failures as {"results": [], "error": ...}.
            # Returning that as a success made a wrong corpus name look like a
            # valid zero-hit search, which a client cannot distinguish.
            if isinstance(result, dict) and result.get("error"):
                return self._error_result(result["error"])
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

    @staticmethod
    def _reject_unknown_arguments(tool_name: str, arguments: dict) -> dict | None:
        """Return an error result if ``arguments`` carries keys the tool does not declare.

        JSON Schema does not forbid extra properties, and this server read
        arguments with ``arguments.get(...)``, so a misspelled or wrong-typed
        key was silently dropped. That mattered: ``search`` interpreted a
        missing ``corpora`` as "every corpus", so a caller passing
        ``corpus`` (singular) got results from all corpora while believing
        the search was scoped. Silent scope widening is indistinguishable from
        a correct answer, so it must be an error.
        """
        tool = next((t for t in TOOLS if t["name"] == tool_name), None)
        if tool is None:
            return None

        schema = tool.get("inputSchema") or {}
        allowed = set((schema.get("properties") or {}).keys())
        # A tool declaring no properties (status, list_folders) still accepts no
        # arguments: an empty schema means "none", not "anything". Those calls
        # pass {} and are unaffected, while stray keys are caught instead of
        # silently ignored.

        unknown = sorted(set(arguments or {}) - allowed)
        if not unknown:
            return None

        # Suggest the closest declared name when the caller typo'd a known one.
        suggestions = []
        for key in unknown:
            close = difflib.get_close_matches(key, allowed, n=1, cutoff=0.6)
            suggestions.append(f"{key!r} (did you mean {close[0]!r}?)" if close else repr(key))

        return GuaipecaMCPServer._error_result(
            f"Unknown argument(s) for tool {tool_name!r}: {', '.join(suggestions)}. "
            f"Accepted: {', '.join(sorted(allowed)) or '(none)'}."
        )

    def _list_upload_folders(self) -> dict:
        """List corpora that accept file uploads (delegates to shared core)."""
        return list_upload_folders(self.config)

    def _handle_upload(
        self, corpus_name: str, filename: str, content_b64: str, do_index: bool
    ) -> dict:
        """Handle a file upload to a corpus.

        Decodes the base64 payload (MCP wire format) and delegates validation,
        persistence and optional indexing to the shared uploads core.

        Args:
            corpus_name: Target corpus name.
            filename: Filename with extension.
            content_b64: Base64-encoded file content.
            do_index: Whether to index the corpus after upload.

        Returns:
            Result dict with saved path and optional index stats.
        """
        import base64

        # Decode content
        try:
            content_bytes = base64.b64decode(content_b64)
        except Exception:
            return {"error": "Invalid base64 content"}

        return save_upload(
            self.config, corpus_name, filename, content_bytes, do_index, self.searcher
        )

    def _handle_delete(
        self, corpus_name: str, filename: str, do_index: bool
    ) -> dict:
        """Handle a file deletion from a corpus (delegates to shared core)."""
        return delete_upload(
            self.config, corpus_name, filename, do_index, self.searcher
        )

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
        """Run MCP server over HTTP transport using Server-Sent Events.

        RETIRED in the MCP SDK migration (Phase 4). ``run()`` now uses
        ``run_http_sdk`` instead, so this method is unreachable from the CLI.
        It is kept only until Phase 6 deletes the hand-rolled transport; the
        ``raise`` below makes an accidental call fail loudly instead of
        silently serving a transport with no SSE routes.

        Original MCP HTTP transport:
        - GET /sse — opens an SSE stream, server sends an `endpoint` event with a POST URL
        - POST /messages?session_id=<id> — client sends JSON-RPC messages via POST
        - Server pushes responses back through the SSE stream
        - GET /health — health check endpoint
        - GET /tools — list available tools (convenience endpoint)

        If auth_token is configured, validates Authorization: Bearer <token> header.
        """
        raise RuntimeError(
            "run_http() was retired in the MCP SDK migration; use run_http_sdk() "
            "(or run(transport='http')), which serves POST /mcp via the official SDK."
        )
        import http.server
        import urllib.parse
        from queue import Queue

        server_instance = self
        auth_token = self.config.server.auth_token
        # Restrict CORS to non-wildcard when auth is enabled
        cors_origin = "*" if not auth_token else "null"
        # Configurable request-body cap (see ServerConfig.max_body_size).
        # Read defensively: tests and embedding callers may pass a duck-typed
        # config whose .server only exposes auth_token.
        max_body_size = getattr(self.config.server, "max_body_size", 73400320)
        # Browser Origin allowlist for CORS / DNS-rebinding protection.
        # Read defensively for the same reason as max_body_size above.
        allowed_origins = getattr(self.config.server, "allowed_origins", None)

        # Security nice-to-have: warn when exposed on a non-loopback interface
        # without auth. Optional auth is intentional by design, so this is a
        # warning only — it does not alter request handling.
        if not auth_token and _is_non_loopback_host(host):
            logger.warning(
                "=" * 60 + "\n"
                f"⚠️  UNAUTHENTICATED PUBLIC BIND: host={host!r}\n"
                "   No auth_token is configured, so ALL endpoints are\n"
                "   reachable WITHOUT authentication from any client that\n"
                "   can reach this address.\n"
                "   To require auth: set server.auth_token in the config.\n"
                "   To restrict to this machine: bind host=127.0.0.1.\n"
                + "=" * 60
            )

        # Security nice-to-have: warn when exposed on a non-loopback interface
        # without auth. Optional auth is intentional by design, so this is a
        # warning only — it does not alter request handling.
        if not auth_token and _is_non_loopback_host(host):
            logger.warning(
                "=" * 60 + "\n"
                f"⚠️  UNAUTHENTICATED PUBLIC BIND: host={host!r}\n"
                "   No auth_token is configured, so ALL endpoints are\n"
                "   reachable WITHOUT authentication from any client that\n"
                "   can reach this address.\n"
                "   To require auth: set server.auth_token in the config.\n"
                "   To restrict to this machine: bind host=127.0.0.1.\n"
                + "=" * 60
            )

        # Session management: each SSE connection gets a session with a message queue
        sessions: dict[str, MCPSession] = {}
        sessions_lock = threading.Lock()

        class MCPSession:
            """Represents a single MCP session (SSE or Streamable HTTP)."""
            def __init__(self, client_address: str, transport: str = "sse"):
                self.session_id = str(uuid.uuid4())
                self.client_address = client_address
                self.transport = transport  # "sse" or "streamable"
                self.response_queue: Queue = Queue()  # used by SSE transport
                self.active = True
                # The POST endpoint URL for this session (SSE only)
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

            def _check_origin(self) -> bool:
                """Validate the Origin header. Returns True if the request may proceed.

                Must run before auth so a cross-origin browser request is
                rejected with 403 rather than prompting for a 401.
                """
                origin = self.headers.get("Origin", "").strip()
                if _origin_allowed(origin, allowed_origins):
                    return True
                logger.warning(
                    "Rejected request with disallowed Origin %r from %s",
                    origin,
                    self.client_address[0],
                )
                self._send_json(403, {})
                return False

            def _cors_value(self) -> str:
                """Access-Control-Allow-Origin value for this response.

                Echo a validated request Origin so cross-origin browser
                clients get an exact match; otherwise keep the legacy value
                ('*' without auth, 'null' with auth).
                """
                origin = self.headers.get("Origin", "").strip()
                if origin and _origin_allowed(origin, allowed_origins):
                    return origin
                return cors_origin

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

                if not self._check_origin():
                    return

                # Legacy SSE endpoints were removed in the MCP SDK migration.
                # They used to be the SSE stream (/sse) and its message sink
                # (/messages); both now answer 410 with a pointer to /mcp, as
                # mcp_sdk.http's routes do on the live server.
                if parsed.path in ("/sse", "/messages"):
                    self._send_json(410, {
                        "error": "legacy SSE transport removed",
                        "message": (
                            "The legacy SSE transport (GET /sse + POST /messages) was removed "
                            "in the MCP SDK migration. Use POST /mcp instead: it serves both "
                            "the session-based handshake and the 2026-07-28 stateless protocol."
                        ),
                        "use": "/mcp",
                    })
                    return

                # /mcp with a trailing slash: clients that build URLs by
                # appending a path segment land here. Starlette's router
                # redirects to /mcp (308); this route does the same.
                if parsed.path == "/mcp/":
                    self.send_response(308)
                    self.send_header("Location", "/mcp")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
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

                # GET /mcp → 405 Method Not Allowed (Streamable HTTP transport
                # does not support server-initiated SSE streaming)
                if parsed.path == "/mcp":
                    self._send_json(405, {
                        "error": "GET /mcp not supported; use POST /mcp for JSON-RPC requests"
                    })
                    return

                # Unknown path
                self._send_json(404, {"error": "not found"})

            def do_POST(self):
                parsed = urllib.parse.urlparse(self.path)

                if not self._check_origin():
                    return

                # Streamable HTTP transport — POST /mcp
                if parsed.path == "/mcp":
                    if not self._check_auth():
                        self._send_json(401, {"error": "unauthorized"})
                        return

                    content_length = int(self.headers.get("Content-Length", 0))
                    # Enforce configurable max body size to prevent memory exhaustion
                    if content_length > max_body_size:
                        self._send_json(413, {"error": "request body too large"})
                        return
                    body = self.rfile.read(content_length) if content_length > 0 else b"{}"

                    # Parse JSON to determine request type
                    try:
                        parsed_body = json.loads(body)
                    except (json.JSONDecodeError, ValueError):
                        # Malformed JSON → parse error
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", self._cors_value())
                        resp_body = json.dumps({
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": "Parse error"},
                        }).encode("utf-8")
                        self.send_header("Content-Length", str(len(resp_body)))
                        self.end_headers()
                        self.wfile.write(resp_body)
                        return

                    # Batch request (JSON array)
                    if isinstance(parsed_body, list):
                        # Empty batch → 200 with empty array (per JSON-RPC 2.0)
                        if len(parsed_body) == 0:
                            resp_body = b"[]"
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Content-Length", str(len(resp_body)))
                            self.send_header("Access-Control-Allow-Origin", self._cors_value())
                            self.end_headers()
                            self.wfile.write(resp_body)
                            return

                        responses = []
                        for entry in parsed_body:
                            entry_bytes = json.dumps(entry).encode("utf-8")
                            resp = server_instance._process_jsonrpc(entry_bytes)
                            if resp is not None:
                                responses.append(resp)

                        if not responses:
                            # All entries were notifications → 202 empty body
                            self.send_response(202)
                            self.send_header("Content-Length", "0")
                            self.send_header("Access-Control-Allow-Origin", self._cors_value())
                            self.end_headers()
                            return

                        resp_body = json.dumps(responses).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(resp_body)))
                        self.send_header("Access-Control-Allow-Origin", self._cors_value())
                        self.end_headers()
                        self.wfile.write(resp_body)
                        return

                    # Single request
                    if not isinstance(parsed_body, dict):
                        # Not a valid JSON-RPC object → invalid request error
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", self._cors_value())
                        resp_body = json.dumps({
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32600, "message": "Invalid Request"},
                        }).encode("utf-8")
                        self.send_header("Content-Length", str(len(resp_body)))
                        self.end_headers()
                        self.wfile.write(resp_body)
                        return

                    method = parsed_body.get("method", "")
                    req_id = parsed_body.get("id")
                    params = parsed_body.get("params", {})
                    session_id_header = self.headers.get("Mcp-Session-Id", "").strip()

                    # Handle initialize
                    if method == "initialize":
                        # If session header references existing session → reject
                        if session_id_header:
                            with sessions_lock:
                                if session_id_header in sessions:
                                    self._send_json(400, {"error": "session already initialized"})
                                    return

                        # R3: Reject protocolVersion < 2025-03-26 on /mcp.
                        # Only *known* versions below the minimum are rejected here;
                        # unknown versions fall through and get the server default.
                        client_version = params.get("protocolVersion", "")
                        if client_version in KNOWN_PROTOCOL_VERSIONS and client_version < MIN_STREAMABLE_VERSION:
                            # Known but too old for /mcp
                            err_resp = {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {
                                    "code": -32602,
                                    "message": (
                                        f"Unsupported protocol version: {client_version}. "
                                        f"Minimum supported version for Streamable HTTP transport is "
                                        f"{MIN_STREAMABLE_VERSION}."
                                    ),
                                },
                            }
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Access-Control-Allow-Origin", self._cors_value())
                            resp_body = json.dumps(err_resp).encode("utf-8")
                            self.send_header("Content-Length", str(len(resp_body)))
                            self.end_headers()
                            self.wfile.write(resp_body)
                            return

                        # Create new session
                        session = MCPSession(self.client_address[0], transport="streamable")
                        with sessions_lock:
                            sessions[session.session_id] = session
                        logger.info(f"Streamable HTTP session started: {session.session_id} from {self.client_address[0]}")

                        # Process the initialize request
                        result = server_instance.handle_request(method, params)

                        # Notifications have no id → 202
                        if result is None or req_id is None:
                            self.send_response(202)
                            self.send_header("Content-Length", "0")
                            self.send_header("Access-Control-Allow-Origin", self._cors_value())
                            self.end_headers()
                            return

                        resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
                        resp_body = json.dumps(resp).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(resp_body)))
                        self.send_header("Mcp-Session-Id", session.session_id)
                        self.send_header("Access-Control-Allow-Origin", self._cors_value())
                        self.end_headers()
                        self.wfile.write(resp_body)
                        return

                    # Non-initialize request: require valid session
                    if not session_id_header:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", self._cors_value())
                        resp_body = json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32000, "message": "Missing or invalid Mcp-Session-Id header"},
                        }).encode("utf-8")
                        self.send_header("Content-Length", str(len(resp_body)))
                        self.end_headers()
                        self.wfile.write(resp_body)
                        return

                    with sessions_lock:
                        if session_id_header not in sessions:
                            # A present-but-unrecognized session id means the
                            # session is gone (e.g. the server restarted). Per
                            # the MCP Streamable HTTP spec the server MUST reply
                            # 404 so clients can detect expiry and re-initialize.
                            # Returning 400 here left clients (e.g. opencode)
                            # retrying a dead session forever, since they key
                            # their reconnect on the 404 status, not on a
                            # -32000 JSON-RPC body.
                            self.send_response(404)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Access-Control-Allow-Origin", self._cors_value())
                            resp_body = json.dumps({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32000, "message": "Session not found"},
                            }).encode("utf-8")
                            self.send_header("Content-Length", str(len(resp_body)))
                            self.end_headers()
                            self.wfile.write(resp_body)
                            return

                    # Streamable HTTP: a non-empty, unknown MCP-Protocol-Version
                    # header is a client error (400). An ABSENT header is always
                    # accepted — older clients predate the header and must keep
                    # working (absent => treat as the 2025-03-26 baseline).
                    protocol_version_header = self.headers.get(
                        "MCP-Protocol-Version", ""
                    ).strip()
                    if (
                        protocol_version_header
                        and protocol_version_header not in KNOWN_PROTOCOL_VERSIONS
                    ):
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", self._cors_value())
                        resp_body = json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {
                                "code": -32000,
                                "message": (
                                    "Unsupported MCP-Protocol-Version: "
                                    f"{protocol_version_header}"
                                ),
                            },
                        }).encode("utf-8")
                        self.send_header("Content-Length", str(len(resp_body)))
                        self.end_headers()
                        self.wfile.write(resp_body)
                        return

                    # Process the request
                    result = server_instance.handle_request(method, params)

                    # Notification → 202 empty body
                    if result is None or req_id is None:
                        self.send_response(202)
                        self.send_header("Content-Length", "0")
                        self.send_header("Access-Control-Allow-Origin", self._cors_value())
                        self.end_headers()
                        return

                    resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
                    resp_body = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.send_header("Access-Control-Allow-Origin", self._cors_value())
                    self.end_headers()
                    self.wfile.write(resp_body)
                    return

                # Legacy SSE message sink was removed with GET /sse (MCP SDK
                # migration). Answer with the same 410 so an old client sees a
                # pointer to /mcp instead of a 404 that looks like a typo.
                if parsed.path in ("/sse", "/messages"):
                    self._send_json(410, {
                        "error": "legacy SSE transport removed",
                        "message": (
                            "The legacy SSE transport (GET /sse + POST /messages) was removed "
                            "in the MCP SDK migration. Use POST /mcp instead: it serves both "
                            "the session-based handshake and the 2026-07-28 stateless protocol."
                        ),
                        "use": "/mcp",
                    })
                    return

                # Unknown POST path
                self._send_json(404, {"error": "not found"})

            def do_DELETE(self):
                """Handle DELETE requests (session termination and file deletion)."""
                parsed = urllib.parse.urlparse(self.path)
                query_params = urllib.parse.parse_qs(parsed.query)

                if not self._check_origin():
                    return

                # DELETE /mcp — terminate a streamable HTTP (or legacy SSE) session
                if parsed.path == "/mcp":
                    if not self._check_auth():
                        self._send_json(401, {"error": "unauthorized"})
                        return

                    session_id_header = self.headers.get("Mcp-Session-Id", "").strip()
                    if not session_id_header:
                        self._send_json(400, {"error": "Missing Mcp-Session-Id header"})
                        return

                    # Same Streamable HTTP rule as POST /mcp: unknown non-empty
                    # MCP-Protocol-Version is a 400; absent is always fine.
                    protocol_version_header = self.headers.get(
                        "MCP-Protocol-Version", ""
                    ).strip()
                    if (
                        protocol_version_header
                        and protocol_version_header not in KNOWN_PROTOCOL_VERSIONS
                    ):
                        self._send_json(400, {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {
                                "code": -32000,
                                "message": (
                                    "Unsupported MCP-Protocol-Version: "
                                    f"{protocol_version_header}"
                                ),
                            },
                        })
                        return

                    with sessions_lock:
                        if session_id_header not in sessions:
                            self._send_json(404, {"error": "session not found"})
                            return
                        session = sessions.pop(session_id_header)
                        session.active = False

                    logger.info(f"Session terminated via DELETE /mcp: {session_id_header}")
                    self._send_json(200, {"status": "session terminated"})
                    return

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
                if not self._check_origin():
                    return
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", self._cors_value())
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Mcp-Session-Id, MCP-Protocol-Version")
                self.end_headers()

            def send_error(self, code, message=None, explain=None):
                """Override send_error to catch ConnectionError.

                When a client sends a malformed request line and immediately
                closes the connection (e.g. a port scanner or a healthcheck
                that sends garbage), BaseHTTPRequestHandler.parse_request()
                calls send_error() which tries to write an HTML error body.
                If the socket is already closed, the write raises
                BrokenPipeError / ConnectionResetError, producing a noisy
                traceback in the logs even though the server survives.

                This override delegates to the parent implementation and
                silently swallows ConnectionError so the handler exits
                cleanly without an unhandled-exception traceback.
                """
                try:
                    super().send_error(code, message, explain)
                except (ConnectionError, BrokenPipeError):
                    # Client gone — nothing we can do. Close the connection
                    # and move on without a traceback.
                    self.close_connection = True

            def log_message(self, format, *args):
                logger.debug(f"HTTP {format % args}")

        # Use ThreadingHTTPServer for concurrent request handling.
        # (SSE connections are long-lived, so single-threaded would block.)
        #
        # daemon_threads + block_on_close are essential for graceful shutdown:
        # ThreadingHTTPServer spawns one non-daemon thread per connection by
        # default and server_close() joins them.  Long-lived SSE streams keep
        # those threads parked, so even after serve_forever() stops the
        # interpreter blocks at exit waiting on the workers.  Making them
        # daemon threads and skipping the join lets the process exit promptly.
        class _Server(http.server.ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False

        actual_port = port
        for attempt in range(5):
            try:
                httpd = _Server((host, actual_port), Handler)
                break
            except OSError:
                actual_port += 1
                logger.warning(f"Port {actual_port - 1} in use, trying {actual_port}")
        else:
            raise RuntimeError(f"Could not find available port starting at {port}")

        logger.info(f"Starting Guaipeca MCP server (HTTP/SSE) on {host}:{actual_port}")
        print(f"Guaipeca MCP server listening on http://{host}:{actual_port}", file=sys.stderr)

        # Graceful shutdown: the signal handler must NOT call httpd.shutdown()
        # directly — that call blocks waiting for serve_forever() to acknowledge
        # the shutdown flag, and the handshake deadlocks when the handler runs
        # in the same thread as serve_forever().  Instead the handler only sets
        # a threading.Event; a background thread calls httpd.shutdown() after
        # the event fires, unblocking serve_forever() cleanly.
        import signal

        _shutdown_event = threading.Event()

        def _handle_term(signum, frame):
            logger.info("Received %s, initiating shutdown", signal.Signals(signum).name)
            _shutdown_event.set()
        signal.signal(signal.SIGTERM, _handle_term)
        signal.signal(signal.SIGINT, _handle_term)

        # Run serve_forever() in a daemon thread so the main thread can wait
        # on the shutdown event without blocking the signal handler.
        _serve_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        _serve_thread.start()

        # Block until a signal sets the event.
        _shutdown_event.wait()

        logger.info("Shutdown flag set, stopping HTTP server")
        httpd.shutdown()
        _serve_thread.join(timeout=5)
        try:
            httpd.server_close()
        except Exception:
            pass
        logger.info("HTTP server stopped, exiting process")

        # Force-exit to bypass native non-daemon thread pools spawned by
        # fastembed/numpy (rayon in tokenizers.abi3.so, BLAS in
        # libscipy_openblas64).  Those threads park on condvar waits for
        # the lifetime of the process and have no stop API.  If we return
        # normally the interpreter blocks at finalization waiting for them,
        # wedging the process in futex_do_wait — systemd's stop times out
        # and the port stays bound until a SIGKILL.
        for _stream in (sys.stdout, sys.stderr):
            try:
                _stream.flush()
            except Exception:
                pass
        os._exit(0)

    def run_sdk_stdio(self):
        """Run the stdio transport through the official SDK (async stdio session).

        Replaces ``run_stdio()``'s hand-rolled stdin line loop. ``run_stdio`` is
        kept until the HTTP cutover (Phase 4) and cleanup (Phase 6).

        The SDK's ``Server`` is fully async, so it needs an event loop. When
        called from a thread without a running loop (the ``both`` transport),
        ``asyncio.run`` creates that loop in the calling thread. Do not call
        this from inside a running loop: ``asyncio.run`` would raise, and a
        second loop on the same thread is not what the SDK expects.
        """
        import asyncio

        from .mcp_sdk.tools import run_stdio as _run_sdk_stdio

        logger.info("Starting Guaipeca MCP server (stdio, mcp SDK)")
        asyncio.run(_run_sdk_stdio(self.config, self))

    def run_http_sdk(self, host: str = "127.0.0.1", port: int = 8090):
        """Run the HTTP transport through the official SDK (Phase 4 cutover).

        ``mcp_sdk.http.serve`` owns port-retry, uvicorn and shutdown; the
        non-MCP endpoints (/health, /tools, downloads, DELETE /documents) are
        Starlette routes on the same app. ``run_http`` is retained but
        unreachable until Phase 6 removes it.
        """
        from .mcp_sdk.http import serve as _serve_http

        logger.info("Starting Guaipeca MCP server (HTTP, mcp SDK)")
        _serve_http(self.config, self, host, port)

    def run(self, transport: str = "both", host: str = "127.0.0.1", port: int = 8090):
        """Run the server with the specified transport(s)."""
        if transport == "stdio":
            self.run_sdk_stdio()
        elif transport == "http":
            self.run_http_sdk(host, port)
        elif transport == "both":
            # Run stdio in a daemon thread, HTTP in main. The stdio side is an
            # asyncio loop owned entirely by that worker thread (see
            # run_sdk_stdio), so it cannot interact with the HTTP thread's loop
            # or signal handling.
            stdio_thread = threading.Thread(target=self.run_sdk_stdio, daemon=True)
            stdio_thread.start()
            self.run_http_sdk(host, port)
        else:
            raise ValueError(f"Unknown transport: {transport}")
