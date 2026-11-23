"""MCP server for Guaipeca.

Exposes twelve tools via Model Context Protocol:
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
- uploads_status: show the upload-manifest state of a corpus
- uploads_prune: remove stale upload-manifest entries (dry run by default)

Transports:
- stdio: newline-delimited JSON-RPC over stdin/stdout
- http: MCP-compliant SSE-based transport (GET /sse + POST /messages)
        + Streamable HTTP transport (POST /mcp, DELETE /mcp)
- both: stdio in a daemon thread + HTTP in main thread
"""

from __future__ import annotations

import difflib
import ipaddress
import json
import logging
import os
import re
import threading
import urllib.parse
from pathlib import Path
from typing import Any

from . import __version__
from .config import GuaipecaConfig
from .converter import Converter
from .embedding import EmbeddingService
from .search import Searcher
from .uploads import (
    delete_upload,
    list_upload_folders,
    manifest_status,
    prune_orphans,
    save_upload,
)

logger = logging.getLogger(__name__)


def _is_non_loopback_host(host: str) -> bool:
    """Return True if ``host`` is a bind address that is not loopback-only.

    Loopback addresses (127.0.0.0/8, ::1) and the hostname ``localhost`` are
    treated as safe (not exposing). The wildcard/unspecified addresses
    (0.0.0.0, ::) bind ALL interfaces and therefore DO expose the service and
    return True. Non-IP hostnames (other than ``localhost``) are assumed to
    expose the service: names are not resolved here, so a hostname that would
    happen to resolve to loopback is still treated as exposing. That is the
    conservative direction on purpose.

    The verdict is load-bearing: ``mcp_sdk.http._check_bind_exposure`` turns a
    True result into a refusal to start when no ``auth_token`` is configured,
    so this is no longer only a warning heuristic.
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
#
# NOTE: these constants describe the pre-migration hand-rolled negotiation.
# They are NOT what the serving transports use any more - the SDK owns version
# negotiation on both `/mcp` and the SSE transport, and it knows about
# 2026-07-28 as well. They survive because `handle_request()` still reads them
# and is still exercised by unit tests; `_process_jsonrpc()` (their only other
# caller) is no longer reachable from any transport.
#
# Do not extend this set to fix a negotiation bug: the fix belongs in the SDK.
MCP_VERSION = "2025-11-25"

# Known protocol versions for negotiation (hand-rolled path only - see above)
KNOWN_PROTOCOL_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}

# Minimum protocol version for the retired hand-rolled `/mcp` handler. The
# enforcement it refers to no longer exists: the SDK's transport decides which
# eras it serves. Kept only because `handle_request()` still checks it.
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
                    "maximum": 200,
                },
                "corpora": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 20,
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
                    "enum": ["chunks", "documents", "auto"],
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
                "max_distance": {
                    "type": "number",
                    "description": (
                        "Relevance floor for dense hits, expressed as 1 - cosine "
                        "similarity (0 = identical, 2 = opposite). Hits below the "
                        "floor are dropped before fusion; a corpus's own setting "
                        "wins over this. Omit to use the config value (usually no "
                        "floor). Lower = stricter."
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
            "List the configured corpora with chunk counts, indexed files, "
            "BM25 availability and the active relevance floor (max_distance). "
            "Call this first to learn valid corpus names before searching or "
            "scoping other tools."
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
            "Only corpora configured in upload.allow can receive files. "
            "Overwriting an existing filename is refused unless the server "
            "uploaded that file itself (it is tracked in the upload manifest) "
            "or the corpus sets upload.replace to true."
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
            "Only corpora configured in upload.allow can have files deleted, "
            "and only files the server uploaded itself (tracked in the upload "
            "manifest) may be deleted. A file whose content was edited outside "
            "the API is refused as drift. The corpus is re-indexed after "
            "deletion to remove stale chunks."
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
    {
        "name": "uploads_status",
        "description": (
            "Show the upload-manifest state of a corpus: how many server-"
            "uploaded files are tracked, which tracked files drifted (edited "
            "outside the API), which files on disk were never uploaded through "
            "the server (orphans), and which manifest entries have no file "
            "left. Read-only; changes nothing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "corpus": {
                    "type": "string",
                    "description": "Single corpus name (not a list); must be in upload.allow.",
                },
            },
            "required": ["corpus"],
        },
    },
    {
        "name": "uploads_prune",
        "description": (
            "Remove stale upload-manifest entries (entries whose file no "
            "longer exists on disk). Never deletes files: orphan files on "
            "disk are left alone. Dry run by default - pass apply=true to "
            "actually rewrite the manifest."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "corpus": {
                    "type": "string",
                    "description": "Single corpus name (not a list); must be in upload.allow.",
                },
                "apply": {
                    "type": "boolean",
                    "description": "If true, remove the stale entries. Default: false (dry run).",
                    "default": False,
                },
            },
            "required": ["corpus"],
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


# Path-like runs in an exception message: 3+ segments, or 2 segments ending
# in a file extension. Shorter slash-prefixed tokens ("/mcp", "/api/v1") are
# not filesystem paths and are left alone.
_CLIENT_PATH_RE = re.compile(
    r"(?:^|[\s(:=])/(?:[\w.-]+/){2,}[\w.-]+|(?:^|[\s(:=])/[\w.-]+/[\w-]+\.[A-Za-z0-9]+"
)


def _client_error_message(exc: Exception) -> str:
    """Build a client-safe message for an exception raised inside a tool call.

    The full exception (with traceback) is logged server-side; what reaches
    the client must not leak filesystem layout or internal type names. The
    message is taken from ``str(exc)`` only (the exception class name is
    never included), absolute file paths are replaced with a placeholder, and
    the result is truncated.
    """
    msg = str(exc) or "internal error"
    msg = _CLIENT_PATH_RE.sub(" [path]", msg)
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

        try:
            if tool_name == "search":
                query = arguments.get("query", "")
                top_k = arguments.get("top_k", 5)
                if isinstance(top_k, bool) or not isinstance(top_k, (int, str)):
                    return self._error_result(
                        f"top_k must be an integer, got {type(top_k).__name__}"
                    )
                if isinstance(top_k, str):
                    # A numeric string ("5") is a common client slip; coerce
                    # it instead of letting int math raise a TypeError later.
                    try:
                        top_k = int(top_k)
                    except ValueError:
                        return self._error_result(
                            f"top_k must be an integer, got {top_k!r}"
                        )
                top_k = max(1, min(200, top_k))
                corpora = arguments.get("corpora")
                if isinstance(corpora, str):
                    # A bare string would otherwise be iterated character by
                    # character, searching one-character "corpora".
                    corpora = [corpora]
                elif corpora is not None and (
                    not isinstance(corpora, list)
                    or not all(isinstance(c, str) for c in corpora)
                ):
                    return self._error_result(
                        f"corpora must be a list of corpus names, got {corpora!r}"
                    )
                hybrid = arguments.get("hybrid")
                return_mode = arguments.get("return_mode", "chunks")
                if return_mode not in ("chunks", "documents", "auto"):
                    return self._error_result(
                        "return_mode must be one of 'chunks', 'documents' or "
                        f"'auto', got {return_mode!r}"
                    )
                max_chars = arguments.get("max_chars")
                section_mode = arguments.get("section_mode", False)
                section_filter = arguments.get("section_filter")
                max_distance = arguments.get("max_distance")
                result = self.searcher.search(
                    query, top_k=top_k, corpora=corpora, hybrid=hybrid,
                    return_mode=return_mode, max_chars=max_chars,
                    section_mode=section_mode, section_filter=section_filter,
                    max_distance=max_distance,
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

            elif tool_name == "uploads_status":
                corpus = arguments.get("corpus", "")
                result = manifest_status(self.config, corpus)
                if result.get("error"):
                    return self._error_result(result["error"])
                return self._result_to_mcp(result)

            elif tool_name == "uploads_prune":
                corpus = arguments.get("corpus", "")
                apply = arguments.get("apply", False)
                result = prune_orphans(self.config, corpus, apply=apply)
                if result.get("error"):
                    return self._error_result(result["error"])
                return self._result_to_mcp(result)

            else:
                return self._error_result(f"Unknown tool: {tool_name}")
        except Exception as exc:
            # The searcher and the upload adapters raise freely (OSError,
            # KeyError, model errors). A raw exception would leak filesystem
            # paths and internals to the client, so log the full detail here
            # and return a sanitised message instead.
            logger.error(
                "Error handling tool call %r: %s", tool_name, exc, exc_info=True
            )
            return self._error_result(_client_error_message(exc))

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

        # Decode content. validate=True rejects anything outside the base64
        # alphabet instead of silently dropping it: a lenient decode turns
        # junk into a shorter or empty payload that would be saved as a
        # corrupted (or blank) document.
        #
        # Whitespace is stripped first, and only whitespace. validate=True
        # rejects newlines, so MIME-style payloads wrapped at 76 columns (the
        # base64 CLI default, and what several SDKs emit) would be refused - in
        # the documented delete-then-upload flow that refusal lands AFTER the
        # original was deleted, turning non-atomicity into real document loss.
        # Stripping whitespace keeps the strictness that matters: a character
        # that is not in the alphabet and not whitespace is still an error.
        try:
            payload = "".join(content_b64.split()) if isinstance(content_b64, str) else content_b64
            content_bytes = base64.b64decode(payload, validate=True)
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
    def run_sdk_stdio(self):
        """Run the stdio transport through the official SDK (async stdio session).

        Replaces the hand-rolled stdin line loop that served this transport
        before the MCP SDK migration.

        The SDK's ``Server`` is fully async, so it needs an event loop. When
        called from a thread without a running loop (the ``both`` transport),
        ``asyncio.run`` creates that loop in the calling thread. Do not call
        this from inside a running loop: ``asyncio.run`` would raise, and a
        second loop on the same thread is not what the SDK expects.
        """
        import asyncio

        from .mcp_sdk.tools import run_stdio as _run_sdk_stdio

        logger.info("Starting Guaipeca MCP server (stdio, mcp SDK)")
        asyncio.run(_run_sdk_stdio(self))

    def run_http_sdk(self, host: str = "127.0.0.1", port: int = 8090):
        """Run the HTTP transport through the official SDK.

        ``mcp_sdk.http.serve`` owns port-retry, uvicorn and shutdown; the
        non-MCP endpoints (/health, /tools, downloads, DELETE /documents) are
        Starlette routes on the same app.
        """
        from .mcp_sdk.http import serve as _serve_http

        logger.info("Starting Guaipeca MCP server (HTTP, mcp SDK)")
        _serve_http(self.config, self, host, port)

    def run(self, transport: str = "both", host: str = "127.0.0.1", port: int = 8090):
        """Run the server with the specified transport(s)."""
        # Import the SDK stack once, here, on one thread.
        #
        # ``mcp`` has a genuine import cycle (mcp.server -> mcp -> mcp.client ->
        # mcp.server._memory -> mcp.server). Importing two SDK entry points
        # concurrently from different threads can observe a half-initialised
        # ``mcp.server`` and raise "cannot import name 'Server' from partially
        # initialized module". Under 'both' the stdio thread and the main
        # thread would race exactly like that, failing roughly one start in
        # three. Importing up front makes both transports reuse the cached
        # modules instead of racing to populate them.
        from .mcp_sdk import http as _sdk_http  # noqa: F401
        from .mcp_sdk import tools as _sdk_tools  # noqa: F401

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
