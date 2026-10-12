"""Shared file-upload and deletion core for Guaipeca.

These helpers are the single source of truth for the write-side operations
exposed by both the CLI (``guaipeca upload`` / ``guaipeca delete`` /
``guaipeca list-folders``) and the MCP server (``upload`` / ``delete`` /
``list_folders`` tools). Keeping them here prevents the two adapters from
drifting apart.

The functions operate on raw bytes; base64 decoding (required by the MCP
protocol) is the caller's responsibility.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .config import GuaipecaConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .search import Searcher

logger = logging.getLogger(__name__)


def list_upload_folders(config: GuaipecaConfig) -> dict:
    """List corpora that accept file uploads.

    Returns:
        Dict with the allowed folders, their extensions, and the global
        extension/size limits.
    """
    allowed = config.upload.allow
    folders = []
    for corpus in config.corpora:
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
        "allowed_extensions": config.upload.allowed_extensions,
        "max_file_size": config.upload.max_file_size,
    }


def save_upload(
    config: GuaipecaConfig,
    corpus_name: str,
    filename: str,
    content_bytes: bytes,
    do_index: bool,
    searcher: Searcher | None = None,
) -> dict:
    """Validate and save an uploaded file to a corpus.

    Validates the corpus is upload-enabled, checks the extension and size,
    saves the file atomically, and optionally triggers indexing.

    Args:
        config: Loaded config with an ``upload`` section.
        corpus_name: Target corpus name (must be in ``upload.allow``).
        filename: Filename with extension.
        content_bytes: Raw file content.
        do_index: Whether to index the corpus after upload.
        searcher: Searcher used to re-index when ``do_index`` is True.

    Returns:
        Result dict with the saved path and optional index stats, or an
        ``{"error": ...}`` dict.
    """
    # Validate corpus exists and is upload-enabled
    if corpus_name not in config.upload.allow:
        return {"error": f"Corpus '{corpus_name}' does not accept uploads"}

    corpus = config.get_corpus(corpus_name)
    if corpus is None:
        return {"error": f"Corpus not found: {corpus_name}"}

    # Validate filename
    filename = os.path.basename(filename)  # strip any path components
    if not filename:
        return {"error": "Invalid filename"}

    ext = Path(filename).suffix.lower()
    if ext not in config.upload.allowed_extensions:
        return {
            "error": f"Extension '{ext}' not allowed. Permitted: {config.upload.allowed_extensions}"
        }

    # Check size limit
    if len(content_bytes) > config.upload.max_file_size:
        max_mb = config.upload.max_file_size / (1024 * 1024)
        return {"error": f"File too large ({len(content_bytes)} bytes, max {max_mb:.0f}MB)"}

    # Save to corpus directory
    corpus_path = Path(corpus.path)
    corpus_path.mkdir(parents=True, exist_ok=True)
    dest = corpus_path / filename

    # Write atomically (temp + rename)
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
    if do_index and searcher is not None:
        index_result = searcher.index_corpus(corpus_name=corpus_name)
        result["index"] = index_result
        result["indexed"] = True
    else:
        result["indexed"] = False

    return result


def delete_upload(
    config: GuaipecaConfig,
    corpus_name: str,
    filename: str,
    do_index: bool,
    searcher: Searcher | None = None,
) -> dict:
    """Delete a file from an upload-enabled corpus.

    Validates the corpus accepts uploads, guards against path traversal,
    deletes the file, and optionally triggers re-indexing.

    Args:
        config: Loaded config with an ``upload`` section.
        corpus_name: Target corpus name (must be in ``upload.allow``).
        filename: Filename with extension.
        do_index: Whether to re-index the corpus after deletion.
        searcher: Searcher used to re-index when ``do_index`` is True.

    Returns:
        Result dict describing the deletion and optional index stats, or an
        ``{"error": ...}`` dict.
    """
    if corpus_name not in config.upload.allow:
        return {"error": f"Corpus '{corpus_name}' does not accept uploads (delete not allowed)"}

    corpus = config.get_corpus(corpus_name)
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

    if do_index and searcher is not None:
        index_result = searcher.index_corpus(corpus_name=corpus_name)
        result["index"] = index_result
        result["indexed"] = True
    else:
        result["indexed"] = False

    return result
