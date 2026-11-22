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
import os
import sys
from pathlib import Path

from . import __version__
from .config import GuaipecaConfig

DEFAULT_CONFIG_PATHS = [
    "~/.guaipeca/guaipeca.yaml",
    "./guaipeca.yaml",
    "./config/guaipeca.yaml",
]


# The bundled sample ships with placeholder corpus paths (/path/to/your/docs).
# A user who copies it but does not edit it gets an index run that can only ever
# produce nothing. Detected by comparing file CONTENT against the bundled copy,
# not by which DEFAULT_CONFIG_PATHS entry matched: a path check would also fire
# for a user who deliberately keeps ./guaipeca.yaml, and would MISS a user who
# copied the sample to ~/.guaipeca/ and never edited it - the actual reported case.
def _is_unedited_sample(path: str) -> bool:
    """True if *path* is byte-identical to the bundled sample config.

    This is the *accidental fallback* case: no config at ``~/.guaipeca/`` and no
    ``./guaipeca.yaml``, so ``_find_config`` landed on the sample shipped in
    ``config/``. Its corpora point at ``/path/to/your/...``, so an index run
    against it can only ever produce nothing.

    A user who *passes* the sample explicitly with ``--config`` knows what they
    are doing and is not warned about — the caller decides that, by only asking
    this function when it did not receive an explicit path.

    Returns False whenever the comparison target cannot be found (pip-installed
    package without config/, unusual layouts): a missing reference must never
    turn into a crash on every command.
    """
    try:
        sample = Path(__file__).resolve().parents[2] / "config" / "guaipeca.yaml"
        with open(path, "rb") as f:
            actual = f.read()
        with open(sample, "rb") as f:
            reference = f.read()
        return actual == reference
    except (OSError, ValueError):
        return False


def _find_missing_corpora(config: GuaipecaConfig, names: list[str]) -> list:
    """Corpus configs (from *names*) whose configured path is not a directory.

    Checked before any indexing work so a typo in the config is reported as a
    configuration error naming the fix, rather than as an index that "produced
    0 files" - which looks like an empty folder and not like a mistake.
    """
    missing = []
    for name in names:
        corpus = config.get_corpus(name)
        if corpus is not None and not os.path.isdir(corpus.path):
            missing.append(corpus)
    return missing


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
    "max_distance": "max_distance",
}


def _find_config(explicit: str | None = None) -> str:
    """Find the config file to use."""
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
    # The verdict travels on the config object because cmd_index needs to
    # escalate this warning to a refusal, and main() never sees the resolved
    # path. from_yaml() cannot know it: only the caller knows which file won.
    #
    # Only "we fell through to the bundled sample" counts. An explicit --config
    # pointing at the same file is a deliberate choice, not the mishap that
    # issue #30 describes, so it is excluded here rather than inside the helper
    # (which cannot tell the two apart once it has only a path).
    config._is_sample_config = explicit is None and _is_unedited_sample(path)
    if config._is_sample_config:
        # Point at the file the user is *supposed* to create, not at the bundled
        # sample they accidentally landed on: telling them to edit the repo's own
        # config/ sample is advice that silently does nothing after the next
        # `git pull`.
        print(
            "Warning: using the bundled sample config - corpus paths are "
            "placeholders. Copy it to ~/.guaipeca/guaipeca.yaml and set real paths.",
            file=sys.stderr,
        )
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


def _corpus_errors(result: dict) -> list[str]:
    """Per-corpus error strings inside an aggregated index result.

    ``index_corpus("all")`` returns ``{"corpora": [...]}`` and each entry may
    carry ``error`` (lock contention, unusable lock file). Those are real
    failures, but they sit below the top-level key ``_emit`` inspects, so a run
    whose corpus was skipped for a lock reported success in both output modes.
    Returned rather than printed so the JSON and human paths cannot drift apart.
    """
    if not isinstance(result, dict):
        return []
    return [
        str(entry["error"])
        for entry in result.get("corpora") or []
        if isinstance(entry, dict) and entry.get("error")
    ]


def _print_index_stats(stats: dict) -> None:
    """Print one corpus's index stats (per-corpus dict shape)."""
    print(f"\n[{stats['corpus']}]")
    print(f"  Files indexed: {stats['files_indexed']}")
    print(f"  Chunks created: {stats['chunks_created']}")
    print(f"  Files skipped: {stats['files_skipped']}")
    print(f"  Duration: {stats['duration_ms']}ms")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
        for err in stats["errors"]:
            print(f"    - {err['file']}: {err['error']}")


def cmd_index(args):
    """Index corpora."""
    config = _load_config(args.config)
    config_path = _find_config(args.config)

    # exit 2, not 1: 1 already means "an index run returned an error result".
    # A distinct code lets a script tell "misconfigured" from "index failed".
    if getattr(config, "_is_sample_config", False):
        print(
            "Error: refusing to index with the unmodified sample config - corpus paths "
            "are placeholders (/path/to/your/docs). Edit the config to point at your real "
            "folders, then retry.",
            file=sys.stderr,
        )
        sys.exit(2)

    corpus = args.corpus or "all"
    if corpus == "all":
        requested = [c.name for c in config.corpora]
    else:
        requested = [corpus]

    # A nonexistent path also reports files_indexed == 0, so it has to be ruled
    # out BEFORE "0 files" is interpreted: otherwise a typo is reported as an
    # empty folder with exit 0, which is exactly the silent failure of issue #30.
    # Every missing path is collected first: one bad entry should not hide the
    # rest while the user is already editing the file.
    missing = _find_missing_corpora(config, requested)
    if missing:
        for c in missing:
            print(f"Error: corpus '{c.name}': path does not exist: {c.path}", file=sys.stderr)
        print(f"Fix: edit paths in {config_path}", file=sys.stderr)
        sys.exit(1)

    _, _, searcher = _create_services(config)
    result = searcher.index_corpus(corpus_name=corpus, force=args.force)

    # A per-corpus failure inside an aggregated result is a failure, not a
    # partial success: check it before either output path so they agree.
    per_corpus_errors = _corpus_errors(result)
    if per_corpus_errors:
        for message in per_corpus_errors:
            print(f"Error: {message}", file=sys.stderr)
        print(f"Error: {len(per_corpus_errors)} corpus(es) were not indexed", file=sys.stderr)
        sys.exit(1)

    if args.json:
        # The sample-config (exit 2) and missing-corpus (exit 1) guards above
        # already ran as plain-text stderr; only a real index result reaches
        # here. _emit exits 1 on an error payload and prints the stats as JSON.
        _emit(result)
        return

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    elif "corpora" in result:
        for stats in result["corpora"]:
            # A per-corpus failure dict has no files_indexed/chunks_created
            # keys; printing it unguarded raised KeyError and tracebacked
            # exactly when the error was meant to be made visible.
            if stats.get("error"):
                print(f"Error: {stats['error']}", file=sys.stderr)
                continue
            _print_index_stats(stats)
            _warn_if_nothing_matched(stats)
    else:
        _print_index_stats(result)
        _warn_if_nothing_matched(result)


def _warn_if_nothing_matched(stats: dict) -> None:
    """Warn when an existing corpus produced no work at all.

    files_skipped > 0 is an incremental re-index with nothing changed: a
    success, and deliberately silent. Only 0 indexed AND 0 skipped means the
    extension filter matched nothing - the folder exists but the files in it
    are not ones this corpus is configured to read. Exit stays 0: nothing is
    broken, but the user must not be told a corpus is indexed when it is empty.

    Errors take precedence. A run where every file failed to convert also
    reports 0 indexed and 0 skipped, and calling that "0 files matched" blames
    the extension filter for what is actually a broken dependency or an
    unreadable file. The per-file errors are already printed by the caller;
    this stays quiet so the message does not contradict them.
    """
    if stats.get("errors"):
        return
    if stats["files_indexed"] == 0 and stats["files_skipped"] == 0:
        print(
            f"Warning: corpus '{stats['corpus']}': 0 files matched - nothing indexed",
            file=sys.stderr,
        )


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
        max_distance=args.max_distance,
    )

    if args.json:
        # _emit exits 1 on an error payload (stderr, empty stdout) and prints
        # the same JSON document the MCP search tool returns on success.
        _emit(result)
        return

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"\nQuery: {result['query']}")
    print(f"Results: {result['total']}")
    if result.get("filtered_by_distance") is not None:
        print(f"Filtered by max_distance: {result['filtered_by_distance']}")
    if result.get("max_distance_applied"):
        # Empty results under an active floor are a success, not a failure:
        # say so, or the user will look for a broken index.
        print("Relevance floor applied: no chunk met max_distance")
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
    # Imported from the module rather than inside the function so tests can
    # replace them wholesale (tests/test_install_guards.py pins the read-side
    # warning this way without loading a model).
    from .embedding import EmbeddingService
    from .search import Searcher

    config = _load_config(args.config)

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
    if args.json:
        _emit(status)
        return
    print("\nGuaipeca Status")
    print(f"  Embedding model: {status['embedding_model']}")
    print(f"  Dimensions: {status['dimensions']}")
    print(f"  Hybrid search: {'enabled' if status.get('hybrid_enabled') else 'disabled'}")
    global_floor = status.get("max_distance")
    print(
        "  Global max distance: "
        + ("disabled" if global_floor is None else f"{global_floor:g} (1 - cosine)")
    )
    print("\n  Corpora:")
    for c in status["corpora"]:
        print(f"    [{c['corpus']}]")
        print(f"      Total chunks: {c['total_chunks']}")
        print(f"      Total vectors: {c['total_vectors']}")
        print(f"      Files tracked: {c['files_tracked']}")
        # BM25 state is what actually distinguishes a healthy hybrid corpus from
        # one silently running dense-only. Reported by searcher.status and by the
        # MCP status tool; omitting it here made the CLI and MCP disagree.
        bm25 = "available" if c.get("bm25_available") else "unavailable"
        print(f"      BM25 index: {bm25}")
        # The active relevance floor: a missing result should be attributable to
        # filtering here rather than blamed on a broken index.
        floor = c.get("max_distance")
        floor_text = "disabled" if floor is None else f"{floor:g} (1 - cosine)"
        print(f"      Max distance: {floor_text}")
    print()


def cmd_serve(args):
    """Start MCP server."""
    config = _load_config(args.config)

    # CLI overrides config
    transport = args.transport or config.server.transport
    port = args.port or config.server.port
    host = config.server.host

    # Refuse a public bind with no auth before doing any expensive setup: the
    # server constructor loads the embedding model (~3s), which is wasted work
    # when the process is about to exit.
    from .mcp_sdk.http import _check_bind_exposure
    from .mcp_server import GuaipecaMCPServer

    try:
        _check_bind_exposure(host, config.server.auth_token)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    server = GuaipecaMCPServer(config)
    try:
        server.run(transport=transport, host=host, port=port)
    except ValueError as exc:
        # A refused startup is a decision, not a crash: report it on one line
        # with a non-zero status so a unit file or script sees the failure.
        if "refusing to serve" not in str(exc):
            raise
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


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
    # Bare version, matching what /health and the MCP initialize response
    # report, so a script can compare the two without stripping a prefix (#45).
    parser.add_argument("--version", action="version", version=__version__)
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
    p_index.add_argument(
        "--json", action="store_true", help="Print the result as JSON instead of text"
    )
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
    p_search.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help=(
            "Relevance floor for dense hits, as 1 - cosine similarity "
            "(omit to use config; lower = stricter)"
        ),
    )
    p_search.add_argument(
        "--json", action="store_true", help="Print the result as JSON instead of text"
    )
    p_search.set_defaults(func=cmd_search)

    # status
    p_status = subparsers.add_parser("status", help="Show index statistics")
    p_status.add_argument(
        "--json", action="store_true", help="Print the result as JSON instead of text"
    )
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
