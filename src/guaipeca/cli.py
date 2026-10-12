"""CLI for Guaipeca.

Commands:
  guaipeca index [corpus] [--force]   Index a corpus or all
  guaipeca search "query" [--top-k N] [--corpora name1,name2]
  guaipeca status                      Show index stats
  guaipeca serve [--transport stdio|http|both] [--port N]
  guaipeca get-chunk <chunk_id>        Retrieve full chunk text
  guaipeca get-document --corpus C --path P   Retrieve full document text
  guaipeca list-documents [corpus]     List indexed documents
  guaipeca list-folders                List upload-enabled corpora
  guaipeca get-toc --corpus C [--document P]  Show table of contents
  guaipeca upload --corpus C <file>    Upload a file for indexing
  guaipeca delete --corpus C <filename>  Delete a file and re-index
  guaipeca --check-model              Check/download embedding model

The MCP server exposes the same read/write capabilities as tools; the CLI
subcommands intentionally mirror them one-for-one (except ``serve``, which is
CLI-only). See ``MCP_TOOL_TO_CLI`` below for the mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import GuaipecaConfig

DEFAULT_CONFIG_PATHS = [
    "~/.guaipeca/guaipeca.yaml",
    "./guaipeca.yaml",
    "./config/guaipeca.yaml",
]

# ---- CLI/MCP parity registry ------------------------------------------------
# Maps every MCP tool name to its CLI subcommand. ``tests/test_cli_mcp_parity.py``
# asserts these stay in sync, so adding an MCP tool (or renaming one) without a
# matching CLI command fails CI.
MCP_TOOL_TO_CLI = {
    "search": "search",
    "index": "index",
    "status": "status",
    "get_chunk": "get-chunk",
    "get_document": "get-document",
    "list_documents": "list-documents",
    "list_folders": "list-folders",
    "get_toc": "get-toc",
    "upload": "upload",
    "delete": "delete",
}

# Maps every property of the MCP ``search`` inputSchema to the argparse dest it
# corresponds to on the CLI ``search`` subcommand.
SEARCH_PROP_TO_CLI_DEST = {
    "query": "query",
    "top_k": "top_k",
    "corpora": "corpora",
    "hybrid": "hybrid",
    "return_mode": "return_mode",
    "max_chars": "max_chars",
    "section_mode": "section_mode",
    "section_filter": "section_filter",
}


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
        preprocess=config.embedding.preprocess,
        cache_ttl_seconds=config.embedding.cache_ttl_seconds,
        cache_max_entries=config.embedding.cache_max_entries,
    )
    searcher = Searcher(config=config, embedding_service=embedder, converter=converter)
    return converter, embedder, searcher


def _print_json(result: dict) -> None:
    """Print a result dict as pretty JSON (matches MCP tool content)."""
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _emit(result: dict) -> None:
    """Print a result, exiting non-zero on an error payload."""
    if isinstance(result, dict) and result.get("error"):
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    _print_json(result)


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
    result = searcher.search(
        args.query,
        top_k=args.top_k,
        corpora=corpora,
        hybrid=args.hybrid if args.hybrid else None,
        return_mode=args.return_mode,
        max_chars=args.max_chars,
        section_mode=args.section_mode,
        section_filter=args.section_filter,
    )

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"\nQuery: {result['query']}")
    print(f"Results: {result['total']}")
    if result.get('hybrid'):
        print("Mode: hybrid (BM25 + FAISS)")
    if result.get('section_mode'):
        print("Mode: sections")
    elif result.get('section_filter'):
        print(f"Section filter: {result['section_filter']}")
    elif result.get('return_mode'):
        print(f"Return mode: {result['return_mode']}")
    print()

    if result.get("section_mode"):
        # Results are aggregated sections (section_mode=True): each has full text.
        for i, r in enumerate(result["results"], 1):
            label = r.get("heading") or " ".join(r.get("text", "")[:80].split())
            print(f"  {i}. [{r['corpus']}] {label}")
            print(f"     Location: {r.get('location', '')}")
            print(f"     Chunks: {r.get('chunk_count', 0)}")
            print(f"     Score: {r['score']:.4f}")
            print()
            print(r.get("text", ""))
            print()
    elif result.get("return_mode") == "documents":
        # Results are whole documents: each carries full text.
        for i, r in enumerate(result["results"], 1):
            print(
                f"  {i}. [{r['corpus']}] {r.get('filename', '')}"
                f"  ({r.get('char_count', 0)} chars, {r.get('matched_chunks', 0)} matching chunks)"
            )
            print(f"     Path: {r.get('source_path', '')}")
            print(f"     Topic: {r.get('topic', '')}")
            print(f"     Score: {r['score']:.4f}")
            print()
            print(r.get("text", ""))
            print()
    else:
        # Chunk-shaped results (chunks mode, or section_filter).
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
        preprocess=config.embedding.preprocess,
        cache_ttl_seconds=config.embedding.cache_ttl_seconds,
        cache_max_entries=config.embedding.cache_max_entries,
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


def cmd_get_chunk(args):
    """Retrieve the full text of a chunk by chunk_id (mirrors MCP get_chunk)."""
    config = _load_config(args.config)
    _, _, searcher = _create_services(config)

    result = searcher.get_chunk(args.chunk_id)
    if result is None:
        print(f"Error: Chunk not found: {args.chunk_id}", file=sys.stderr)
        sys.exit(1)
    _print_json(result)


def cmd_get_document(args):
    """Retrieve the full converted text of a document (mirrors MCP get_document)."""
    config = _load_config(args.config)
    _, _, searcher = _create_services(config)

    result = searcher.get_document(args.corpus, args.path)
    if result is None:
        print(
            f"Error: Document not found: {args.path} in corpus {args.corpus}",
            file=sys.stderr,
        )
        sys.exit(1)
    _print_json(result)


def cmd_list_documents(args):
    """List indexed documents in a corpus, or all corpora (mirrors MCP list_documents)."""
    config = _load_config(args.config)
    _, _, searcher = _create_services(config)

    _emit(searcher.list_documents(corpus=args.corpus))


def cmd_list_folders(args):
    """List upload-enabled corpora (mirrors MCP list_folders)."""
    from .uploads import list_upload_folders

    config = _load_config(args.config)
    _print_json(list_upload_folders(config))


def cmd_get_toc(args):
    """Show the table of contents for a corpus or document (mirrors MCP get_toc)."""
    config = _load_config(args.config)
    _, _, searcher = _create_services(config)

    _emit(searcher.get_toc(corpus=args.corpus, document=args.document))


def cmd_upload(args):
    """Upload a local file to a corpus and index it (mirrors MCP upload)."""
    import os

    from .uploads import save_upload

    config = _load_config(args.config)
    _, _, searcher = _create_services(config)

    file_path = Path(os.path.expanduser(args.file))
    if not file_path.is_file():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_bytes()
    result = save_upload(
        config,
        args.corpus,
        file_path.name,
        content,
        do_index=not args.no_index,
        searcher=searcher,
    )
    _emit(result)


def cmd_delete(args):
    """Delete a file from a corpus and re-index (mirrors MCP delete)."""
    from .uploads import delete_upload

    config = _load_config(args.config)
    _, _, searcher = _create_services(config)

    result = delete_upload(
        config,
        args.corpus,
        args.filename,
        do_index=not args.no_index,
        searcher=searcher,
    )
    _emit(result)


def cmd_check_model(args):
    """Check if the embedding model is cached locally, download if needed."""
    config = _load_config(args.config)
    from .embedding import EmbeddingService

    embedder = EmbeddingService(
        model_name=config.embedding.model,
        cache_dir=config.embedding.cache_dir,
        dimensions=config.embedding.dimensions,
        preprocess=config.embedding.preprocess,
        cache_ttl_seconds=config.embedding.cache_ttl_seconds,
        cache_max_entries=config.embedding.cache_max_entries,
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


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Exposed separately from :func:`main` so tests can introspect the registered
    subcommands and their options (CLI/MCP parity checks).
    """
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
    p_search.add_argument(
        "--return-mode",
        choices=["chunks", "documents", "auto"],
        default="chunks",
        help="Result granularity: chunks (default), documents, or auto",
    )
    p_search.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Threshold for auto return-mode (total result chars)",
    )
    p_search.add_argument(
        "--section-mode",
        action="store_true",
        help="Group results by section and return full section text",
    )
    p_search.add_argument(
        "--section-filter",
        default=None,
        help="Restrict search to a section ID or heading path prefix",
    )
    p_search.set_defaults(func=cmd_search)

    # status
    p_status = subparsers.add_parser("status", help="Show index statistics")
    p_status.set_defaults(func=cmd_status)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start MCP server")
    p_serve.add_argument("--transport", choices=["stdio", "http", "both"], help="Transport type")
    p_serve.add_argument("--port", type=int, help="HTTP port")
    p_serve.set_defaults(func=cmd_serve)

    # get-chunk
    p_get_chunk = subparsers.add_parser("get-chunk", help="Retrieve full chunk text by chunk ID")
    p_get_chunk.add_argument("chunk_id", help="Chunk ID (format: corpus:stable_id)")
    p_get_chunk.set_defaults(func=cmd_get_chunk)

    # get-document
    p_get_doc = subparsers.add_parser(
        "get-document", help="Retrieve full converted text of a document"
    )
    p_get_doc.add_argument("--corpus", required=True, help="Corpus name")
    p_get_doc.add_argument(
        "--path", "--source-path", dest="path", required=True,
        help="Absolute path to the source file",
    )
    p_get_doc.set_defaults(func=cmd_get_document)

    # list-documents
    p_list_docs = subparsers.add_parser(
        "list-documents", help="List indexed documents in a corpus (or all)"
    )
    p_list_docs.add_argument("corpus", nargs="?", default=None, help="Corpus name, or omit for all")
    p_list_docs.set_defaults(func=cmd_list_documents)

    # list-folders
    p_list_folders = subparsers.add_parser(
        "list-folders", help="List corpora that accept file uploads"
    )
    p_list_folders.set_defaults(func=cmd_list_folders)

    # get-toc
    p_get_toc = subparsers.add_parser("get-toc", help="Show table of contents for a corpus/document")
    p_get_toc.add_argument("--corpus", required=True, help="Corpus name")
    p_get_toc.add_argument(
        "--document", default=None, help="Optional specific document source path"
    )
    p_get_toc.set_defaults(func=cmd_get_toc)

    # upload
    p_upload = subparsers.add_parser("upload", help="Upload a file to a corpus for indexing")
    p_upload.add_argument("file", help="Local file to upload")
    p_upload.add_argument("--corpus", required=True, help="Target corpus (must allow uploads)")
    p_upload.add_argument(
        "--no-index", action="store_true", help="Skip indexing after upload"
    )
    p_upload.set_defaults(func=cmd_upload)

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete a file from a corpus and re-index")
    p_delete.add_argument("filename", help="Filename to delete (including extension)")
    p_delete.add_argument("--corpus", required=True, help="Target corpus (must allow uploads)")
    p_delete.add_argument(
        "--no-index", action="store_true", help="Skip re-indexing after deletion"
    )
    p_delete.set_defaults(func=cmd_delete)

    return parser


def main():
    """Entry point."""
    parser = build_parser()
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
