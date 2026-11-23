"""Shared file-upload and deletion core for Guaipeca.

These helpers are the single source of truth for the write-side operations
exposed by both the CLI (``guaipeca upload`` / ``guaipeca delete`` /
``guaipeca list-folders`` / ``guaipeca uploads-status`` /
``guaipeca uploads-prune``) and the MCP server (``upload`` / ``delete`` /
``list_folders`` / ``uploads_status`` / ``uploads_prune`` tools). Keeping
them here prevents the two adapters from drifting apart.

The server tracks every file it uploads in a per-corpus manifest
(``uploaded.json`` next to the corpus files). Overwriting or deleting a file
the server did not upload is refused, so the API can never clobber or remove
documents that predate it.

The functions operate on raw bytes; base64 decoding (required by the MCP
protocol) is the caller's responsibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .config import GuaipecaConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .search import Searcher

logger = logging.getLogger(__name__)

# Manifest filename inside each corpus directory. Maps filename ->
# {"sha256": <hex>, "size_bytes": <int>} for every file the server uploaded.
MANIFEST_FILENAME = "uploaded.json"


def _is_reserved_name(filename: str) -> bool:
    """True for names the API must never write or delete.

    Reserved: the manifest itself, anything under the ``.guaipeca`` prefix,
    and any temporary file (``*.tmp``). Checked regardless of manifest state
    and before any filesystem write.
    """
    return (
        filename == MANIFEST_FILENAME
        or filename.startswith(".guaipeca")
        or filename.endswith(".tmp")
    )


def _read_manifest(corpus_path: Path) -> dict:
    """Read the corpus upload manifest.

    A missing or corrupt manifest is treated as empty: the manifest is
    bookkeeping, and a broken one must never take the API down.
    """
    try:
        with open(corpus_path / MANIFEST_FILENAME, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_manifest(corpus_path: Path, manifest: dict) -> None:
    """Write the corpus upload manifest atomically (temp + rename).

    Same pattern as ``Searcher._atomic_write_json``: a crash mid-write leaves
    the previous manifest intact instead of a truncated one.
    """
    path = corpus_path / MANIFEST_FILENAME
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(corpus_path), suffix=".tmp", prefix=MANIFEST_FILENAME + "."
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # os.replace, not os.rename: on Windows os.rename raises
        # FileExistsError (WinError 183) when the destination exists, which
        # it does on every manifest update after the first.
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    """SHA256 hex digest of a file's content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

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
                "replace": bool(config.upload.replace.get(corpus.name, False)),
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
    saves the file atomically, records it in the corpus manifest, and
    optionally triggers indexing.

    Overwrite rule: a filename may be overwritten only if the manifest
    already records it (the server uploaded it) or the corpus grants
    replacement via ``upload.replace``. Anything else is refused without
    touching the existing file.

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

    # Reserved names are refused regardless of manifest state, before any
    # filesystem write.
    if _is_reserved_name(filename):
        return {
            "error": f"Filename '{filename}' is reserved and cannot be uploaded"
        }

    ext = Path(filename).suffix.lower()
    if ext not in config.upload.allowed_extensions:
        return {
            "error": f"Extension '{ext}' not allowed. Permitted: {config.upload.allowed_extensions}"
        }

    # Check size limit
    if len(content_bytes) > config.upload.max_file_size:
        max_mb = config.upload.max_file_size / (1024 * 1024)
        return {"error": f"File too large ({len(content_bytes)} bytes, max {max_mb:.0f}MB)"}

    # Refuse empty and whitespace-only content. This lives in the shared
    # core (not the MCP adapter) because the CLI hands save_upload raw
    # bytes and never touches base64. Checked before any manifest read or
    # filesystem write so a rejected upload leaves no file, temp file or
    # manifest entry behind. Always refused; there is no opt-in.
    if len(content_bytes) == 0:
        return {"error": f"Refusing empty upload: '{filename}' has no content"}
    if content_bytes.strip() == b"":
        return {
            "error": f"Refusing blank upload: '{filename}' contains only whitespace"
        }

    # Save to corpus directory
    corpus_path = Path(corpus.path)
    corpus_path.mkdir(parents=True, exist_ok=True)
    dest = corpus_path / filename

    # Overwrite rule: a name the manifest does not record may only be taken
    # when the corpus explicitly allows replacement. Refuse before writing
    # anything, so a refused upload leaves no file and no temp file behind.
    manifest = _read_manifest(corpus_path)
    if filename in manifest:
        # Re-uploading a file the server itself uploaded is always allowed.
        # If the on-disk content no longer matches the recorded checksum, the
        # file was edited outside the API; re-uploading is the normal fix, so
        # this is a warning, not a refusal.
        entry = manifest[filename]
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        if expected is not None and dest.is_file():
            actual = _sha256_file(dest)
            if actual != expected:
                logger.warning(
                    f"Upload of '{filename}' to corpus '{corpus_name}' overwrites a "
                    f"drifted file (expected sha256 {expected}, found {actual}); "
                    f"the file was modified outside the API"
                )
    elif dest.exists() and not config.upload.replace.get(corpus_name, False):
        return {
            "error": (
                f"Refusing to overwrite '{filename}': the server did not upload "
                f"this file (no manifest entry) and corpus '{corpus_name}' does "
                f"not allow replacement (upload.replace)"
            )
        }

    # Write atomically (temp + rename)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(corpus_path), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(content_bytes)
            f.flush()
            os.fsync(f.fileno())
        # os.replace, not os.rename: on Windows os.rename raises
        # FileExistsError (WinError 183) when the destination exists.
        # Re-uploading a file under an existing name is the normal path, so
        # this is not an edge case there.
        os.replace(tmp_path, str(dest))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return {"error": f"Failed to save file: {filename}"}

    logger.info(f"Uploaded {filename} to corpus '{corpus_name}' ({len(content_bytes)} bytes)")

    # Record the upload in the manifest so later overwrites and deletions of
    # this name are recognized as server-managed.
    manifest[filename] = {
        "sha256": hashlib.sha256(content_bytes).hexdigest(),
        "size_bytes": len(content_bytes),
    }
    try:
        _write_manifest(corpus_path, manifest)
    except OSError as e:
        logger.error(f"Failed to write upload manifest in {corpus_path}: {e}")
        return {"error": f"File saved but manifest update failed: {filename}"}

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
    deletes the file, removes its manifest entry, and optionally triggers
    re-indexing.

    Delete rule: a file may only be deleted if the manifest records it (the
    server uploaded it), its extension is allowed, and the name is not
    reserved. A manifest-tracked file whose on-disk content no longer matches
    the recorded checksum was edited outside the API and is refused as drift.

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

    # Reserved names are refused regardless of manifest state.
    if _is_reserved_name(filename):
        return {
            "error": f"Filename '{filename}' is reserved and cannot be deleted"
        }

    # Deletion applies the same extension allowlist as upload: a file the
    # API can never write is a file the API will never delete.
    ext = Path(filename).suffix.lower()
    if ext not in config.upload.allowed_extensions:
        return {
            "error": f"Extension '{ext}' not allowed. Permitted: {config.upload.allowed_extensions}"
        }

    corpus_path = Path(corpus.path)
    file_path = corpus_path / filename

    try:
        real_path = file_path.resolve()
        real_corpus = corpus_path.resolve()
        if not str(real_path).startswith(str(real_corpus)):
            return {"error": "Invalid filename (path traversal detected)"}
    except Exception:
        return {"error": "Invalid filename"}

    # Delete rule: only files the server uploaded itself may be deleted.
    manifest = _read_manifest(corpus_path)
    entry = manifest.get(filename)
    if entry is None:
        return {
            "error": (
                f"Refusing to delete '{filename}': the server did not upload "
                f"this file (no manifest entry)"
            )
        }
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
        return {
            "error": f"Refusing to delete '{filename}': manifest entry is corrupt"
        }

    if not file_path.exists():
        return {"error": f"File not found: {filename}"}

    if not file_path.is_file():
        return {"error": f"Not a file: {filename}"}

    # Drift check: the on-disk content must still match what the server
    # wrote. A mismatch means the file was edited outside the API; deleting
    # it would destroy changes the server has no record of.
    actual = _sha256_file(file_path)
    expected = entry["sha256"]
    if actual != expected:
        return {
            "error": (
                f"Refusing to delete '{filename}': content drift detected - the "
                f"file no longer matches what the server uploaded (expected "
                f"sha256 {expected}, found {actual}). Restore or re-upload the "
                f"file through the API first."
            )
        }

    try:
        file_size = file_path.stat().st_size
        os.unlink(str(file_path))
    except OSError as e:
        logger.error(f"Failed to delete {file_path}: {e}")
        return {"error": f"Failed to delete file: {filename}"}

    logger.info(f"Deleted {filename} from corpus '{corpus_name}' ({file_size} bytes)")

    # Free the name: remove the manifest entry so the filename can be
    # re-uploaded (delete-then-upload cycle) without replace permission.
    del manifest[filename]
    try:
        _write_manifest(corpus_path, manifest)
    except OSError as e:
        # The file is already gone; a stale entry is visible via
        # manifest_status and removable via prune_orphans.
        logger.error(f"Failed to update upload manifest after deleting {file_path}: {e}")

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


def manifest_status(config: GuaipecaConfig, corpus_name: str) -> dict:
    """Report the upload-manifest state of a corpus.

    Compares the manifest against the corpus directory: which tracked files
    still match, which drifted, which manifest entries lost their file, and
    which files on disk were never uploaded through the server (orphans).
    Orphans are allowed-extension files that are not in the manifest, not
    reserved, and not dotfiles; the manifest itself is never listed.

    Args:
        config: Loaded config with an ``upload`` section.
        corpus_name: Corpus to inspect (must be in ``upload.allow``).

    Returns:
        Dict with corpus, tracked, drifted, orphan_files and missing_files,
        or an ``{"error": ...}`` dict.
    """
    if corpus_name not in config.upload.allow:
        return {"error": f"Corpus '{corpus_name}' does not accept uploads"}

    corpus = config.get_corpus(corpus_name)
    if corpus is None:
        return {"error": f"Corpus not found: {corpus_name}"}

    corpus_path = Path(corpus.path)
    manifest = _read_manifest(corpus_path)

    tracked = 0
    drifted = []
    missing_files = []
    for filename, entry in manifest.items():
        file_path = corpus_path / filename
        if not file_path.is_file():
            missing_files.append(filename)
            continue
        tracked += 1
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        actual = _sha256_file(file_path)
        if expected != actual:
            drifted.append({"filename": filename, "expected": expected, "actual": actual})

    orphan_files = []
    if corpus_path.is_dir():
        for child in corpus_path.iterdir():
            name = child.name
            if not child.is_file():
                continue
            if name in manifest:
                continue
            # Reserved names (manifest, .guaipeca*, *.tmp) and dotfiles are
            # infrastructure, not orphaned documents.
            if _is_reserved_name(name) or name.startswith("."):
                continue
            if child.suffix.lower() not in config.upload.allowed_extensions:
                continue
            orphan_files.append(name)

    return {
        "corpus": corpus_name,
        "tracked": tracked,
        "drifted": drifted,
        "orphan_files": sorted(orphan_files),
        "missing_files": sorted(missing_files),
    }


def prune_orphans(config: GuaipecaConfig, corpus_name: str, apply: bool) -> dict:
    """Remove stale manifest entries (entries with no file on disk).

    Never deletes files: orphan files on disk are left alone, and only
    manifest entries whose file is gone are removed. With ``apply`` false
    this is a dry run that reports what would be removed without writing
    anything; with ``apply`` true the manifest is rewritten atomically.

    Args:
        config: Loaded config with an ``upload`` section.
        corpus_name: Corpus to prune (must be in ``upload.allow``).
        apply: When True, remove the stale entries; otherwise dry run.

    Returns:
        Dict with corpus, dry_run, would_remove and removed, or an
        ``{"error": ...}`` dict.
    """
    if corpus_name not in config.upload.allow:
        return {"error": f"Corpus '{corpus_name}' does not accept uploads"}

    corpus = config.get_corpus(corpus_name)
    if corpus is None:
        return {"error": f"Corpus not found: {corpus_name}"}

    corpus_path = Path(corpus.path)
    manifest = _read_manifest(corpus_path)

    would_remove = sorted(
        name for name in manifest if not (corpus_path / name).is_file()
    )

    removed = []
    if apply and would_remove:
        for name in would_remove:
            del manifest[name]
        _write_manifest(corpus_path, manifest)
        removed = would_remove

    return {
        "corpus": corpus_name,
        "dry_run": not apply,
        "would_remove": would_remove,
        "removed": removed,
    }
