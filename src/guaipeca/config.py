"""Configuration loader for Guaipeca.

Loads and validates YAML config defining corpora, embedding, chunking, and server settings.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# Encodings tried when reading a config file, in order. UTF-8 is the only
# encoding this project writes; the others exist so a file produced by another
# tool (Notepad, a Windows editor, an agent on a cp1252 default) still loads
# instead of failing with a traceback from inside PyYAML.
_CONFIG_ENCODINGS = ("utf-8-sig", "cp1252")


def _read_config_text(path: str) -> str:
    """Read a YAML config as text, tolerating common Windows encodings.

    UTF-8 (with or without a BOM) is read silently. Any other encoding that
    decodes cleanly produces a warning naming the encoding, the file, and how
    to fix it, because a mis-encoded config will still break other tools that
    touch the same file.

    Raises:
        ValueError: if the file cannot be decoded by any known encoding.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise ValueError(f"Cannot read config file {path}: {e}") from e

    for encoding in _CONFIG_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding != "utf-8-sig":
            logger.warning(
                "Config file %s is not UTF-8; decoded as %s. Re-save it as UTF-8 "
                "to avoid problems in other tools.",
                path,
                encoding,
            )
        return text

    # Report the offset from a strict UTF-8 attempt: that is the error a user
    # is most likely to hit, and the one whose position they can act on.
    try:
        raw.decode("utf-8")
        # Unreachable: non-UTF-8 input failed above.
        raise AssertionError("decode should have failed")
    except UnicodeDecodeError as e:
        byte = raw[e.start]
        line = raw[: e.start].count(b"\n") + 1
        raise ValueError(
            f"Config file {path} is not valid UTF-8: byte 0x{byte:02x} at offset "
            f"{e.start} (line {line}). This usually means "
            "the file was saved as Windows-1252 with non-ASCII characters "
            "(smart quotes, em dashes, accented letters). Re-save it as UTF-8."
        ) from e


@dataclass
class CorpusConfig:
    """A single document corpus definition."""
    name: str
    path: str
    topic: str = ""
    weight: float = 1.0
    extensions: list[str] = field(default_factory=lambda: [".md", ".txt"])
    # Relevance floor for dense (cosine) hits in this corpus: keep only chunks
    # with cosine similarity >= 1 - max_distance. None (default) = no floor,
    # preserving the pre-#12 behaviour. BM25-only hits are never filtered.
    max_distance: float | None = None

    def __post_init__(self):
        # Expand ~ in path
        self.path = os.path.expanduser(self.path)
        # Normalize extensions to lowercase with leading dot.
        # Consumers compare Path(filename).suffix.lower().
        self.extensions = [
            (ext if ext.startswith(".") else f".{ext}").lower()
            for ext in self.extensions
        ]


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    cache_dir: str = "~/.guaipeca/models"
    preprocess: bool = True  # Preprocess text before embedding (heading + first 500 chars)
    cache_ttl_seconds: int = 3600  # Embedding cache TTL (1 hour)
    cache_max_entries: int = 10000  # Max embedding cache entries

    def __post_init__(self):
        self.cache_dir = os.path.expanduser(self.cache_dir)


@dataclass
class ChunkingConfig:
    """Chunking parameters."""
    max_size: int = 2000
    overlap: int = 200  # ~10% of max_size; carried between consecutive chunks
    table_aware: bool = True
    split_on_headings: bool = True


@dataclass
class IndexingConfig:
    """Indexing settings."""
    data_dir: str = "~/.guaipeca/data"
    incremental: bool = True
    max_file_size: int = 52428800  # 50MB
    exclude_dirs: list[str] = field(
        default_factory=lambda: ["node_modules", ".git", "venv", "__pycache__"]
    )

    def __post_init__(self):
        self.data_dir = os.path.expanduser(self.data_dir)


@dataclass
class SearchConfig:
    """Search configuration for hybrid BM25 + dense search."""
    # Enable BM25 hybrid search. Default true: measured better or equal on every
    # metric (recall@5, hit@1, MRR) across a 25-query golden set — see
    # tools/eval_retrieval.py. Set false to force dense-only.
    hybrid: bool = True
    bm25_weight: float = 1.0     # Weight for BM25 scores in RRF fusion
    dense_weight: float = 1.0   # Weight for dense (FAISS) scores in RRF fusion
    rrf_k: int = 60              # RRF constant (standard value from literature)
    max_chars: int = 8000        # Threshold for auto return_mode (chars)
    toc_rerank: bool = False     # Enable ToC keyword re-ranking (Phase 5)
    toc_rerank_weight: float = 0.3  # Weight for ToC re-ranking boost
    fusion_alpha: float = 0.3    # Score-vs-rank blend in normalized RRF (0=pure rank, 1=pure score)
    # Global relevance floor for dense hits. None (default) = disabled. A corpus
    # may override it with its own corpus.max_distance.
    max_distance: float | None = None


@dataclass
class UploadConfig:
    """Upload configuration for MCP file ingestion.

    Controls which corpora accept uploaded files and sets safety limits.
    Only corpora explicitly listed in allow can receive uploads.
    """
    allow: list[str] = field(default_factory=list)  # corpus names that accept uploads
    max_file_size: int = 52428800  # 50MB — same as indexing default
    allowed_extensions: list[str] = field(
        default_factory=lambda: [".md", ".txt", ".markdown", ".pdf", ".docx", ".doc",
                                 ".pptx", ".ppt", ".xlsx", ".xls", ".html", ".htm", ".csv"]
    )
    auto_index: bool = True  # index the corpus after uploading a file
    # Per-corpus replacement permission: corpus name -> bool. True allows
    # overwriting a file the server did not upload itself; absent/false
    # refuses it. Re-uploading a file the server uploaded is always allowed.
    replace: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize extensions to have a leading dot and be lowercase,
        # mirroring CorpusConfig. Consumers compare Path(filename).suffix.lower().
        self.allowed_extensions = [
            (ext if ext.startswith(".") else f".{ext}").lower()
            for ext in self.allowed_extensions
        ]


@dataclass
class ServerConfig:
    """MCP server configuration."""
    # "both" does NOT run two listeners: it means stdio when launched as a
    # subprocess, HTTP when launched as a daemon (the SDK runs one server per
    # transport). "sse" was removed in the MCP SDK migration — use "http".
    transport: str = "both"  # stdio | http | both
    port: int = 8090
    host: str = "127.0.0.1"  # P1-5: default to localhost for security
    auth_token: str | None = None  # P1-5: optional Bearer token auth
    # Max HTTP request body size (bytes). Caps the streamable-HTTP/SSE request
    # bodies so a single request cannot exhaust memory. Note upload file content
    # is base64-encoded in the JSON body (~4/3 of raw size), so this must exceed
    # upload.max_file_size by ~33% or large uploads will 413 before the upload
    # tool's own size check runs. Default 70MB leaves headroom for a 50MB file.
    max_body_size: int = 73400320  # 70MB
    # Allowed browser Origin values for CORS / DNS-rebinding protection.
    # None (default) => allow only loopback origins (http/https on
    # localhost, 127.x, [::1], any port). A non-empty list => exact-match
    # allowlist. Requests with NO Origin header (curl, CLI, non-browser
    # MCP clients) are always allowed.
    allowed_origins: list[str] | None = None


def _build(cls, data: dict | None, *, section: str, **explicit):
    """Build a config dataclass from a YAML mapping.

    Keys that match dataclass fields are passed through; keys the YAML omits
    take the dataclass defaults, so the field declaration is the single
    source of truth for every default. Keys that match no field warn (naming
    the key and the section) instead of being dropped silently. Values in
    ``explicit`` win over the YAML and cover required fields and
    pre-validated values.
    """
    fields = {f.name for f in dataclasses.fields(cls)}
    data = data or {}
    for key in data:
        if key not in fields:
            logger.warning(
                "Unknown key %r in config section %r; ignoring it.", key, section
            )
    kwargs = {k: v for k, v in data.items() if k in fields}
    kwargs.update(explicit)
    return cls(**kwargs)


@dataclass
class GuaipecaConfig:
    """Top-level Guaipeca configuration."""
    corpora: list[CorpusConfig] = field(default_factory=list)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    config_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> GuaipecaConfig:
        """Load config from a YAML file."""
        path = os.path.expanduser(path)
        text = _read_config_text(path)
        data = yaml.safe_load(text)

        if not data:
            raise ValueError(f"Empty config file: {path}")

        # Top-level keys that match no section are a typo or a key for another
        # version; warn so they are not dropped silently.
        known_sections = {"corpora", "embedding", "chunking", "indexing",
                          "server", "upload", "search"}
        for key in data:
            if key not in known_sections:
                logger.warning(
                    "Unknown top-level config key %r; ignoring it.", key
                )

        # Parse corpora
        corpora = []
        for c in data.get("corpora", []):
            corpora.append(_build(
                CorpusConfig,
                c,
                section=f"corpora[{c.get('name')!r}]",
                name=c["name"],
                path=c["path"],
            ))

        # Parse embedding
        embedding = _build(EmbeddingConfig, data.get("embedding"), section="embedding")

        # Parse chunking
        chunking = _build(ChunkingConfig, data.get("chunking"), section="chunking")

        # Parse indexing
        indexing = _build(IndexingConfig, data.get("indexing"), section="indexing")

        # Parse server
        server = _build(ServerConfig, data.get("server"), section="server")
        # Allow env var override for auth token (security best practice)
        env_token = os.environ.get("GUAIPECA_AUTH_TOKEN")
        if env_token:
            server.auth_token = env_token

        # Parse upload
        upl = data.get("upload") or {}
        replace = upl.get("replace")
        if replace is None:
            replace = {}
        if not isinstance(replace, dict):
            raise ValueError(
                f"upload.replace must be a mapping of corpus name to boolean, "
                f"got: {replace!r}"
            )
        # Require real booleans. bool() would turn the string "false" and the
        # string "0" into True, silently granting overwrite authority the
        # operator wrote as false - so a coerced value is an error, not a
        # convenience.
        for name, flag in replace.items():
            if not isinstance(flag, bool):
                raise ValueError(
                    f"upload.replace[{name!r}] must be true or false, "
                    f"got: {flag!r}. Quote-free booleans only - the string "
                    f"\"false\" reads as true and would enable overwriting."
                )
        upload = _build(
            UploadConfig,
            {**upl, "replace": {str(name): flag for name, flag in replace.items()}},
            section="upload",
        )

        # Parse search
        search = _build(SearchConfig, data.get("search"), section="search")

        config = cls(
            corpora=corpora,
            embedding=embedding,
            chunking=chunking,
            indexing=indexing,
            server=server,
            upload=upload,
            search=search,
            config_path=path,
        )

        # P2-6: Validate config value types
        config._validate_types()

        return config

    def _validate_types(self) -> None:
        """Validate that numeric and boolean config values have correct types.

        Logs warnings for type mismatches and coerces where possible.
        Raises ValueError for unfixable type errors.
        """
        # Corpus weights must be numeric
        for c in self.corpora:
            if not isinstance(c.weight, (int, float)):
                logger.warning(f"Corpus '{c.name}' weight is not numeric ({type(c.weight).__name__}), defaulting to 1.0")
                c.weight = 1.0
            if c.max_distance is not None and (
                isinstance(c.max_distance, bool)
                or not isinstance(c.max_distance, (int, float))
                or c.max_distance < 0
            ):
                raise ValueError(
                    f"corpora[{c.name}].max_distance must be null or a number >= 0 "
                    f"(1 - cosine similarity), got: {c.max_distance!r}"
                )

        # Chunking config
        if not isinstance(self.chunking.max_size, int) or self.chunking.max_size < 1:
            raise ValueError(f"chunking.max_size must be a positive integer, got: {self.chunking.max_size!r}")
        if not isinstance(self.chunking.overlap, int) or self.chunking.overlap < 0:
            raise ValueError(f"chunking.overlap must be a non-negative integer, got: {self.chunking.overlap!r}")

        # Indexing config
        if not isinstance(self.indexing.max_file_size, int) or self.indexing.max_file_size < 1:
            raise ValueError(f"indexing.max_file_size must be a positive integer, got: {self.indexing.max_file_size!r}")

        # Upload config
        if not isinstance(self.upload.max_file_size, int) or isinstance(self.upload.max_file_size, bool) or self.upload.max_file_size < 1:
            raise ValueError(
                f"upload.max_file_size must be a positive integer, "
                f"got: {self.upload.max_file_size!r}"
            )
        if not isinstance(self.upload.auto_index, bool):
            raise ValueError(f"upload.auto_index must be a boolean, got: {type(self.upload.auto_index).__name__}")
        if not isinstance(self.upload.allow, list) or not all(isinstance(a, str) for a in self.upload.allow):
            raise ValueError(
                f"upload.allow must be a list of strings, got: "
                f"{self.upload.allow!r}"
            )

        # Embedding config
        if not isinstance(self.embedding.dimensions, int) or self.embedding.dimensions < 1:
            raise ValueError(f"embedding.dimensions must be a positive integer, got: {self.embedding.dimensions!r}")

        # Server config
        if not isinstance(self.server.port, int) or self.server.port < 1 or self.server.port > 65535:
            raise ValueError(f"server.port must be an integer 1-65535, got: {self.server.port!r}")
        if self.server.auth_token is not None and not isinstance(self.server.auth_token, str):
            raise ValueError(f"server.auth_token must be a string or null, got: {type(self.server.auth_token).__name__}")
        # An explicitly configured but empty token is falsy, so the charset check
        # below is skipped AND _check_bearer's `if not auth_token: return True`
        # short-circuits before reading the header at all. Measured: with
        # auth_token: "" every request authenticated, including one with no
        # Authorization header. That is worse than an open server, because a
        # configured token makes it look protected. Absent/null stays the way to
        # disable auth; an empty string is a typo, not a decision.
        if self.server.auth_token is not None and not self.server.auth_token.strip():
            raise ValueError(
                "server.auth_token is empty (or whitespace only). A configured "
                "token must have a value: an empty one would silently disable "
                "authentication while appearing to enable it. Remove the "
                "auth_token key entirely to run without authentication."
            )
        # Non-ASCII token carriage is client-dependent, and that is the reason
        # for this gate (#43). The comparison encodes the secret as UTF-8, so a
        # client that sends UTF-8 header bytes CAN authenticate with a secret
        # above U+00FF - this is not a hard protocol limit. But clients that
        # encode the header as latin-1 (Python's http.client, some proxies) send
        # different bytes and get a permanent silent 401, and the operator has
        # no way to tell which kind their client is. Restricting to ASCII makes
        # the token portable; anything else is a trap waiting for a client swap.
        #
        # Control characters are rejected for the same reason from the other
        # direction: they encode fine but cannot legally appear in an HTTP field
        # value, so the token could never be presented at all.
        if self.server.auth_token:
            try:
                self.server.auth_token.encode("latin-1")
            except UnicodeEncodeError:
                raise ValueError(
                    "server.auth_token contains characters above U+00FF. Those "
                    "only authenticate with clients that send UTF-8 header "
                    "bytes; latin-1 clients would get a permanent silent 401. "
                    "Use printable ASCII."
                ) from None
            if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in self.server.auth_token):
                raise ValueError(
                    "server.auth_token contains control characters, which cannot "
                    "appear in an HTTP header value; such a token could never be "
                    "presented. Use printable ASCII."
                )
        if not isinstance(self.server.max_body_size, int) or self.server.max_body_size < 1:
            raise ValueError(f"server.max_body_size must be a positive integer, got: {self.server.max_body_size!r}")
        if self.server.transport == "sse":
            raise ValueError(
                "server.transport is 'sse', but the legacy SSE transport was removed in the "
                "MCP SDK migration. Use 'http' instead: it serves POST /mcp for both protocol "
                "eras. Valid values: 'stdio', 'http', 'both'."
            )
        if self.server.transport not in ("stdio", "http", "both"):
            raise ValueError(
                f"server.transport must be one of 'stdio', 'http', 'both', got: "
                f"{self.server.transport!r}"
            )
        if self.server.allowed_origins is not None and (
            not isinstance(self.server.allowed_origins, list)
            or not all(isinstance(o, str) for o in self.server.allowed_origins)
        ):
            raise ValueError(
                f"server.allowed_origins must be a list of strings or null, got: "
                f"{self.server.allowed_origins!r}"
            )

        # Search config
        if not isinstance(self.search.hybrid, bool):
            raise ValueError(f"search.hybrid must be a boolean, got: {type(self.search.hybrid).__name__}")
        if not isinstance(self.search.bm25_weight, (int, float)) or self.search.bm25_weight < 0:
            raise ValueError(f"search.bm25_weight must be a non-negative number, got: {self.search.bm25_weight!r}")
        if not isinstance(self.search.dense_weight, (int, float)) or self.search.dense_weight < 0:
            raise ValueError(f"search.dense_weight must be a non-negative number, got: {self.search.dense_weight!r}")
        if not isinstance(self.search.rrf_k, int) or self.search.rrf_k < 1:
            raise ValueError(f"search.rrf_k must be a positive integer, got: {self.search.rrf_k!r}")
        if not isinstance(self.search.max_chars, int) or self.search.max_chars < 1:
            raise ValueError(f"search.max_chars must be a positive integer, got: {self.search.max_chars!r}")
        if self.search.max_distance is not None and (
            isinstance(self.search.max_distance, bool)
            or not isinstance(self.search.max_distance, (int, float))
            or self.search.max_distance < 0
        ):
            raise ValueError(
                "search.max_distance must be null or a number >= 0 "
                f"(1 - cosine similarity), got: {self.search.max_distance!r}"
            )
        if not isinstance(self.search.toc_rerank_weight, (int, float)) or isinstance(self.search.toc_rerank_weight, bool):
            raise ValueError(
                f"search.toc_rerank_weight must be a number, "
                f"got: {self.search.toc_rerank_weight!r}"
            )
        if not isinstance(self.search.fusion_alpha, (int, float)) or isinstance(self.search.fusion_alpha, bool):
            raise ValueError(
                f"search.fusion_alpha must be a number in [0, 1], "
                f"got: {self.search.fusion_alpha!r}"
            )
        if not isinstance(self.search.toc_rerank, bool):
            raise ValueError(f"search.toc_rerank must be a boolean, got: {type(self.search.toc_rerank).__name__}")

    def get_corpus(self, name: str) -> CorpusConfig | None:
        """Find a corpus by name."""
        for c in self.corpora:
            if c.name == name:
                return c
        return None

    def validate(self) -> list[str]:
        """Validate config and return list of warnings."""
        warnings = []
        seen_names = set()
        for c in self.corpora:
            if c.name in seen_names:
                warnings.append(f"Duplicate corpus name: {c.name}")
            seen_names.add(c.name)
            if not os.path.isdir(c.path):
                warnings.append(f"Corpus path does not exist: {c.path} ({c.name})")
        return warnings

    @property
    def data_dir(self) -> Path:
        """Corpora data directory."""
        return Path(self.indexing.data_dir) / "corpora"