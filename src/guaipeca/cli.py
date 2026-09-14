"""CLI for Guaipeca.

Commands:
  guaipeca index [corpus] [--force]   Index a corpus or all
  guaipeca search "query" [--top-k N] [--corpora name1,name2]
  guaipeca status                      Show index stats
  guaipeca serve [--transport stdio|http|both] [--port N]
  guaipeca --check-model              Check/download embedding model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import GuaipecaConfig

DEFAULT_CONFIG_PATHS = [
    "~/.guaipeca/guaipeca.yaml",
    "./guaipeca.yaml",
    "./config/guaipeca.yaml",
]


def _find_config(explicit: str | None = None) -> str:
    """Find the config file to use."""
    import os

    if explicit:
        path = os.path.expanduser(explicit)
        if not os.path.isfile(path):
            print(f"Error: config file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return path

    for candidate in DEFAULT_CONFIG_PATHS:
        path = os.path.expanduser(candidate)
        if os.path.isfile(path):
            return path

    print("Error: no config file found. Create one at ~/.guaipeca/guaipeca.yaml", file=sys.stderr)
    sys.exit(1)


def _load_config(explicit: str | None = None) -> GuaipecaConfig:
    """Load config and print warnings."""
    path = _find_config(explicit)
    config = GuaipecaConfig.from_yaml(path)
    warnings = config.validate()
    for w in warnings:
        print(f"Warning: {w}", file=sys.stderr)
    return config


def _create_services(config: GuaipecaConfig):
    """Create converter, embedder, and searcher from config.

    Shared helper to avoid duplicated service initialization across CLI commands.
    """
    from .converter import Converter
    from .embedding import EmbeddingService
    from .search import Searcher

    converter = Converter(
        cache_dir=str(Path(config.indexing.data_dir) / "converted")
    )
    embedder = EmbeddingService(
        model_name=config.embedding.model,
        cache_dir=config.embedding.cache_dir,
        dimensions=config.embedding.dimensions,
    )
    searcher = Searcher(config=config, embedding_service=embedder, converter=converter)
    return converter, embedder, searcher


def cmd_index(args):
    """Index corpora."""
    config = _load_config(args.config)
    _, _, searcher = _create_services(config)

    corpus = args.corpus or "all"
    result = searcher.index_corpus(corpus_name=corpus, force=args.force)

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    elif "corpora" in result:
        for stats in result["corpora"]:
            print(f"\n[{stats['corpus']}]")
            print(f"  Files indexed: {stats['files_indexed']}")
            print(f"  Chunks created: {stats['chunks_created']}")
            print(f"  Files skipped: {stats['files_skipped']}")
            print(f"  Duration: {stats['duration_ms']}ms")
            if stats["errors"]:
                print(f"  Errors: {len(stats['errors'])}")
                for err in stats["errors"]:
                    print(f"    - {err['file']}: {err['error']}")
    else:
        print(f"\n[{result['corpus']}]")
        print(f"  Files indexed: {result['files_indexed']}")
        print(f"  Chunks created: {result['chunks_created']}")
        print(f"  Files skipped: {result['files_skipped']}")
        print(f"  Duration: {result['duration_ms']}ms")
        if result["errors"]:
            print(f"  Errors: {len(result['errors'])}")
            for err in result["errors"]:
                print(f"    - {err['file']}: {err['error']}")


def cmd_search(args):
    """Search corpora."""
    config = _load_config(args.config)
    _, _, searcher = _create_services(config)

    corpora = args.corpora.split(",") if args.corpora else None
    result = searcher.search(args.query, top_k=args.top_k, corpora=corpora, hybrid=args.hybrid if args.hybrid else None)

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"\nQuery: {result['query']}")
    print(f"Results: {result['total']}")
    if result.get('hybrid'):
        print("Mode: hybrid (BM25 + FAISS)")
    print()

    for i, r in enumerate(result["results"], 1):
        print(f"  {i}. [{r['corpus']}] {r['summary']}")
        print(f"     Location: {r['location']}")
        print(f"     Topic: {r['topic']}")
        print(f"     Score: {r['score']:.4f}")
        print(f"     Chunk ID: {r['chunk_id']}")
        print()


def cmd_status(args):
    """Show corpus status."""
    config = _load_config(args.config)
    from .embedding import EmbeddingService
    from .search import Searcher

    embedder = EmbeddingService(
        model_name=config.embedding.model,
        cache_dir=config.embedding.cache_dir,
        dimensions=config.embedding.dimensions,
    )
    searcher = Searcher(config=config, embedding_service=embedder)

    status = searcher.status
    print("\nGuaipeca Status")
    print(f"  Embedding model: {status['embedding_model']}")
    print(f"  Dimensions: {status['dimensions']}")
    print("\n  Corpora:")
    for c in status["corpora"]:
        print(f"    [{c['corpus']}]")
        print(f"      Total chunks: {c['total_chunks']}")
        print(f"      Total vectors: {c['total_vectors']}")
        print(f"      Files tracked: {c['files_tracked']}")
    print()


def cmd_serve(args):
    """Start MCP server."""
    config = _load_config(args.config)

    # CLI overrides config
    transport = args.transport or config.server.transport
    port = args.port or config.server.port
    host = config.server.host

    from .mcp_server import GuaipecaMCPServer
    server = GuaipecaMCPServer(config)
    server.run(transport=transport, host=host, port=port)


def cmd_check_model(args):
    """Check if the embedding model is cached locally, download if needed."""
    config = _load_config(args.config)
    from .embedding import EmbeddingService

    embedder = EmbeddingService(
        model_name=config.embedding.model,
        cache_dir=config.embedding.cache_dir,
        dimensions=config.embedding.dimensions,
    )

    # Check if model is cached
    import os
    cache_dir = embedder.cache_dir
    if cache_dir and os.path.isdir(cache_dir):
        entries = os.listdir(cache_dir)
        if entries:
            print(f"✓ Model '{config.embedding.model}' is cached at {cache_dir}")
            print(f"  Cached entries: {', '.join(entries[:5])}")
        else:
            print(f"⚠ Model cache directory exists but is empty: {cache_dir}")
            print("  Attempting to download model...")
            _download_model(embedder)
    else:
        print(f"⚠ Model not cached. Cache dir: {cache_dir}")
        print("  Attempting to download model...")
        _download_model(embedder)


def _download_model(embedder):
    """Attempt to download/load the embedding model."""
    try:
        print(f"  Loading model '{embedder.model_name}'...")
        embedder.embed(["warmup"])
        print("✓ Model downloaded and loaded successfully!")
        print(f"  Cache: {embedder.cache_dir}")
    except Exception as e:
        print(f"✗ Failed to download model: {e}", file=sys.stderr)
        print("  Ensure you have network access and fastembed installed.", file=sys.stderr)
        sys.exit(1)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="guaipeca",
        description="Lightweight configurable RAG MCP server",
    )
    parser.add_argument("--config", "-c", help="Path to config YAML file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    # P1-6: --check-model flag to check/download model without starting server
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Check if embedding model is cached locally, download if needed",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # index
    p_index = subparsers.add_parser("index", help="Index corpora")
    p_index.add_argument("corpus", nargs="?", default="all", help="Corpus name or 'all'")
    p_index.add_argument("--force", action="store_true", help="Force re-index all files")
    p_index.set_defaults(func=cmd_index)

    # search
    p_search = subparsers.add_parser("search", help="Search indexed corpora")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_search.add_argument("--corpora", help="Comma-separated corpus names to search")
    p_search.add_argument("--hybrid", action="store_true", help="Enable BM25 hybrid search")
    p_search.set_defaults(func=cmd_search)

    # status
    p_status = subparsers.add_parser("status", help="Show index statistics")
    p_status.set_defaults(func=cmd_status)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start MCP server")
    p_serve.add_argument("--transport", choices=["stdio", "http", "both"], help="Transport type")
    p_serve.add_argument("--port", type=int, help="HTTP port")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    # Handle --check-model as a top-level flag
    if args.check_model:
        cmd_check_model(args)
        return

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)