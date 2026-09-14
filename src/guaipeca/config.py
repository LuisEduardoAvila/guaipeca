"""Configuration loader for Guaipeca.

Loads and validates YAML config defining corpora, embedding, chunking, and server settings.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CorpusConfig:
    """A single document corpus definition."""
    name: str
    path: str
    topic: str = ""
    weight: float = 1.0
    extensions: list[str] = field(default_factory=lambda: [".md", ".txt"])

    def __post_init__(self):
        # Expand ~ in path
        self.path = os.path.expanduser(self.path)
        # Normalize extensions to lowercase with leading dot
        self.extensions = [
            ext if ext.startswith(".") else f".{ext}"
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
    hybrid: bool = False          # Enable BM25 hybrid search (default: false for backward compat)
    bm25_weight: float = 1.0     # Weight for BM25 scores in RRF fusion
    dense_weight: float = 1.0   # Weight for dense (FAISS) scores in RRF fusion
    rrf_k: int = 60              # RRF constant (standard value from literature)
    max_chars: int = 8000       # Threshold for auto return_mode (total result chars)


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


@dataclass
class ServerConfig:
    """MCP server configuration."""
    transport: str = "both"  # stdio | http | both
    port: int = 8090
    host: str = "127.0.0.1"  # P1-5: default to localhost for security
    auth_token: str | None = None  # P1-5: optional Bearer token auth


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
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty config file: {path}")

        # Parse corpora
        corpora = []
        for c in data.get("corpora", []):
            corpora.append(CorpusConfig(
                name=c["name"],
                path=c["path"],
                topic=c.get("topic", ""),
                weight=c.get("weight", 1.0),
                extensions=c.get("extensions", [".md", ".txt"]),
            ))

        # Parse embedding
        emb = data.get("embedding", {})
        embedding = EmbeddingConfig(
            model=emb.get("model", "all-MiniLM-L6-v2"),
            dimensions=emb.get("dimensions", 384),
            cache_dir=emb.get("cache_dir", "~/.guaipeca/models"),
            preprocess=emb.get("preprocess", True),
            cache_ttl_seconds=emb.get("cache_ttl_seconds", 3600),
            cache_max_entries=emb.get("cache_max_entries", 10000),
        )

        # Parse chunking
        chk = data.get("chunking", {})
        chunking = ChunkingConfig(
            max_size=chk.get("max_size", 2000),
            overlap=chk.get("overlap", 200),
            table_aware=chk.get("table_aware", True),
            split_on_headings=chk.get("split_on_headings", True),
        )

        # Parse indexing
        idx = data.get("indexing", {})
        indexing = IndexingConfig(
            data_dir=idx.get("data_dir", "~/.guaipeca/data"),
            incremental=idx.get("incremental", True),
            max_file_size=idx.get("max_file_size", 52428800),
            exclude_dirs=idx.get("exclude_dirs", ["node_modules", ".git", "venv", "__pycache__"]),
        )

        # Parse server
        srv = data.get("server", {})
        server = ServerConfig(
            transport=srv.get("transport", "both"),
            port=srv.get("port", 8090),
            host=srv.get("host", "127.0.0.1"),
            auth_token=srv.get("auth_token"),
        )
        # Allow env var override for auth token (security best practice)
        env_token = os.environ.get("GUAIPECA_AUTH_TOKEN")
        if env_token:
            server.auth_token = env_token

        # Parse upload
        upl = data.get("upload", {})
        upload = UploadConfig(
            allow=upl.get("allow", []),
            max_file_size=upl.get("max_file_size", 52428800),
            allowed_extensions=[
                ext if ext.startswith(".") else f".{ext}"
                for ext in upl.get("allowed_extensions",
                                   [".md", ".txt", ".markdown", ".pdf", ".docx", ".doc",
                                    ".pptx", ".ppt", ".xlsx", ".xls", ".html", ".htm", ".csv"])
            ],
            auto_index=upl.get("auto_index", True),
        )

        # Parse search
        srch = data.get("search", {})
        search = SearchConfig(
            hybrid=srch.get("hybrid", False),
            bm25_weight=srch.get("bm25_weight", 1.0),
            dense_weight=srch.get("dense_weight", 1.0),
            rrf_k=srch.get("rrf_k", 60),
            max_chars=srch.get("max_chars", 8000),
        )

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

        # Chunking config
        if not isinstance(self.chunking.max_size, int) or self.chunking.max_size < 1:
            raise ValueError(f"chunking.max_size must be a positive integer, got: {self.chunking.max_size!r}")
        if not isinstance(self.chunking.overlap, int) or self.chunking.overlap < 0:
            raise ValueError(f"chunking.overlap must be a non-negative integer, got: {self.chunking.overlap!r}")

        # Indexing config
        if not isinstance(self.indexing.max_file_size, int) or self.indexing.max_file_size < 1:
            raise ValueError(f"indexing.max_file_size must be a positive integer, got: {self.indexing.max_file_size!r}")

        # Embedding config
        if not isinstance(self.embedding.dimensions, int) or self.embedding.dimensions < 1:
            raise ValueError(f"embedding.dimensions must be a positive integer, got: {self.embedding.dimensions!r}")

        # Server config
        if not isinstance(self.server.port, int) or self.server.port < 1 or self.server.port > 65535:
            raise ValueError(f"server.port must be an integer 1-65535, got: {self.server.port!r}")
        if self.server.auth_token is not None and not isinstance(self.server.auth_token, str):
            raise ValueError(f"server.auth_token must be a string or null, got: {type(self.server.auth_token).__name__}")

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