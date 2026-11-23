"""Acceptance tests for issue #54: upload/delete must track server-uploaded files.

The manifest is ``uploaded.json`` in the corpus directory. A file may only be
overwritten or deleted through the API when the manifest says the server
uploaded it, and only when the corpus permits replacement
(``upload.replace[corpus]``).

These tests are the acceptance gate for the issue. They are written against the
*desired* behaviour, so they fail on unfixed code — that is the point.
"""

from __future__ import annotations

import hashlib
import json

from guaipeca.config import CorpusConfig, GuaipecaConfig, UploadConfig
from guaipeca.uploads import (
    delete_upload,
    list_upload_folders,
    manifest_status,
    prune_orphans,
    save_upload,
)

MANIFEST = "uploaded.json"


class FakeSearcher:
    def __init__(self):
        self.calls: list[str] = []

    def index_corpus(self, corpus_name: str = "all", force: bool = False) -> dict:
        self.calls.append(corpus_name)
        return {"corpus": corpus_name, "indexed": True}


def _config(tmp_path, replace=None, **upload_kwargs) -> GuaipecaConfig:
    return GuaipecaConfig(
        corpora=[CorpusConfig(name="uploads", path=str(tmp_path), extensions=[".md"])],
        upload=UploadConfig(
            allow=["uploads"],
            allowed_extensions=[".md", ".txt"],
            max_file_size=upload_kwargs.pop("max_file_size", 1024),
            replace=replace or {},
            **upload_kwargs,
        ),
    )


def _read_manifest(tmp_path) -> dict:
    path = tmp_path / MANIFEST
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# AC: a manifest tracks server-uploaded files per corpus
# --------------------------------------------------------------------------

def test_upload_writes_manifest_entry_with_checksum(tmp_path):
    config = _config(tmp_path)
    save_upload(config, "uploads", "note.md", b"hello", do_index=False)

    manifest = _read_manifest(tmp_path)
    assert "note.md" in manifest
    entry = manifest["note.md"]
    assert entry["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert entry["size_bytes"] == 5


def test_manifest_survives_reupload(tmp_path):
    config = _config(tmp_path, replace={"uploads": True})
    save_upload(config, "uploads", "note.md", b"one", do_index=False)
    save_upload(config, "uploads", "note.md", b"two", do_index=False)

    manifest = _read_manifest(tmp_path)
    assert manifest["note.md"]["sha256"] == hashlib.sha256(b"two").hexdigest()


# --------------------------------------------------------------------------
# AC: upload to a name not in the manifest, without replace, is refused
# --------------------------------------------------------------------------

def test_upload_over_preexisting_file_is_refused(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "contract.md").write_bytes(b"ORIGINAL CONTRACT CONTENT")

    result = save_upload(config, "uploads", "contract.md", b"ATTACKER CONTENT", do_index=False)

    assert "error" in result
    # The pre-existing content must be untouched.
    assert (tmp_path / "contract.md").read_bytes() == b"ORIGINAL CONTRACT CONTENT"
    # And nothing recorded.
    assert "contract.md" not in _read_manifest(tmp_path)


def test_upload_over_preexisting_file_honours_corpus_replace(tmp_path):
    """Replace is a per-corpus permission, not a per-call flag."""
    config = _config(tmp_path, replace={"uploads": True})
    (tmp_path / "contract.md").write_bytes(b"ORIGINAL CONTRACT CONTENT")

    result = save_upload(config, "uploads", "contract.md", b"NEW CONTENT", do_index=False)

    assert result.get("saved") is True
    assert (tmp_path / "contract.md").read_bytes() == b"NEW CONTENT"
    assert "contract.md" in _read_manifest(tmp_path)


def test_replace_permission_is_per_corpus(tmp_path):
    """A corpus without replace enabled still refuses, even when another has it."""
    other = tmp_path / "other"
    other.mkdir()
    config = _config(tmp_path, replace={"uploads": True})
    config.corpora.append(CorpusConfig(name="second", path=str(other), extensions=[".md"]))
    config.upload.allow.append("second")

    (other / "doc.md").write_bytes(b"PRE-EXISTING")
    result = save_upload(config, "second", "doc.md", b"REPLACEMENT", do_index=False)

    assert "error" in result
    assert (other / "doc.md").read_bytes() == b"PRE-EXISTING"


def test_replace_of_manifest_tracked_file_does_not_need_permission(tmp_path):
    """Re-uploading a file the server itself uploaded is always allowed."""
    config = _config(tmp_path)
    save_upload(config, "uploads", "note.md", b"first", do_index=False)
    result = save_upload(config, "uploads", "note.md", b"second", do_index=False)

    assert result.get("saved") is True
    assert (tmp_path / "note.md").read_bytes() == b"second"


def test_upload_of_new_name_writes_nothing_when_refused(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "secret.txt").write_bytes(b"keep me")
    config.upload.allowed_extensions = [".md", ".txt"]

    result = save_upload(config, "uploads", "secret.txt", b"clobbered", do_index=False)

    assert "error" in result
    assert (tmp_path / "secret.txt").read_bytes() == b"keep me"
    assert not list(tmp_path.glob("*.tmp")), "a refused upload must not leave temp files"


# --------------------------------------------------------------------------
# AC: delete of a file not in the manifest is refused
# --------------------------------------------------------------------------

def test_delete_of_preexisting_file_is_refused(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "notes.md").write_bytes(b"never uploaded")

    result = delete_upload(config, "uploads", "notes.md", do_index=False)

    assert "error" in result
    assert (tmp_path / "notes.md").exists()


def test_delete_of_uploaded_file_succeeds(tmp_path):
    config = _config(tmp_path)
    save_upload(config, "uploads", "note.md", b"hello", do_index=False)

    result = delete_upload(config, "uploads", "note.md", do_index=False)

    assert result.get("deleted") is True
    assert not (tmp_path / "note.md").exists()
    assert "note.md" not in _read_manifest(tmp_path)


# --------------------------------------------------------------------------
# AC: delete applies the same extension allowlist as upload
# --------------------------------------------------------------------------

def test_delete_refuses_disallowed_extension_even_if_in_manifest(tmp_path):
    config = _config(tmp_path)
    save_upload(config, "uploads", "note.md", b"hello", do_index=False)
    # Simulate a manifest entry that somehow names a non-allowed extension.
    manifest = _read_manifest(tmp_path)
    manifest["evil.exe"] = {"sha256": "x", "size_bytes": 1}
    (tmp_path / MANIFEST).write_text(json.dumps(manifest))
    (tmp_path / "evil.exe").write_bytes(b"x")

    result = delete_upload(config, "uploads", "evil.exe", do_index=False)

    assert "error" in result
    assert (tmp_path / "evil.exe").exists()


def test_delete_refuses_unmanaged_other_extensions(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "secret.key").write_bytes(b"key")
    (tmp_path / ".hidden").write_bytes(b"dot")

    assert "error" in delete_upload(config, "uploads", "secret.key", do_index=False)
    assert "error" in delete_upload(config, "uploads", ".hidden", do_index=False)
    assert (tmp_path / "secret.key").exists()
    assert (tmp_path / ".hidden").exists()


# --------------------------------------------------------------------------
# AC: the full delete-then-upload cycle still works end to end
# --------------------------------------------------------------------------

def test_delete_then_upload_cycle(tmp_path):
    config = _config(tmp_path)
    searcher = FakeSearcher()

    save_upload(config, "uploads", "report.md", b"v1", do_index=True, searcher=searcher)
    assert (tmp_path / "report.md").read_bytes() == b"v1"

    deleted = delete_upload(config, "uploads", "report.md", do_index=True, searcher=searcher)
    assert deleted.get("deleted") is True
    assert not (tmp_path / "report.md").exists()

    # The intended workflow: delete, then upload the replacement under the same
    # name. This must succeed even without replace permission, because the
    # manifest no longer claims the name.
    replaced = save_upload(config, "uploads", "report.md", b"v2", do_index=True, searcher=searcher)
    assert replaced.get("saved") is True
    assert (tmp_path / "report.md").read_bytes() == b"v2"
    assert _read_manifest(tmp_path)["report.md"]["sha256"] == hashlib.sha256(b"v2").hexdigest()


# --------------------------------------------------------------------------
# Reserved names: the manifest itself and dotfiles must never be API-touchable
# --------------------------------------------------------------------------

def test_upload_cannot_target_the_manifest(tmp_path):
    config = _config(tmp_path, replace={"uploads": True})
    save_upload(config, "uploads", "note.md", b"hello", do_index=False)
    before = _read_manifest(tmp_path)

    result = save_upload(config, "uploads", MANIFEST, b"{}", do_index=False)

    assert "error" in result
    assert _read_manifest(tmp_path) == before


def test_delete_cannot_target_the_manifest(tmp_path):
    config = _config(tmp_path)
    save_upload(config, "uploads", "note.md", b"hello", do_index=False)

    result = delete_upload(config, "uploads", MANIFEST, do_index=False)

    assert "error" in result
    assert (tmp_path / MANIFEST).exists()


def test_dotfile_is_not_adopted_and_not_deletable(tmp_path):
    config = _config(tmp_path)
    (tmp_path / ".gitignore").write_bytes(b"*.pyc\n")

    assert "error" in delete_upload(config, "uploads", ".gitignore", do_index=False)
    assert (tmp_path / ".gitignore").exists()


# --------------------------------------------------------------------------
# Drift detection
# --------------------------------------------------------------------------

def test_drift_detected_when_file_edited_outside_api(tmp_path):
    config = _config(tmp_path)
    save_upload(config, "uploads", "note.md", b"original", do_index=False)
    (tmp_path / "note.md").write_bytes(b"edited by hand")

    result = delete_upload(config, "uploads", "note.md", do_index=False)

    # Deleting a file whose content no longer matches what we wrote is refused.
    assert "error" in result
    assert "drift" in str(result["error"]).lower()
    assert (tmp_path / "note.md").read_bytes() == b"edited by hand"


def test_status_reports_manifest_and_drift(tmp_path):
    config = _config(tmp_path)
    save_upload(config, "uploads", "clean.md", b"same", do_index=False)
    save_upload(config, "uploads", "drifted.md", b"original", do_index=False)
    (tmp_path / "drifted.md").write_bytes(b"changed")

    status = manifest_status(config, "uploads")

    assert status["tracked"] == 2
    assert [d["filename"] for d in status["drifted"]] == ["drifted.md"]


# --------------------------------------------------------------------------
# Orphans: files with no manifest entry, and manifest entries with no file
# --------------------------------------------------------------------------

def test_status_reports_orphan_files(tmp_path):
    config = _config(tmp_path)
    save_upload(config, "uploads", "known.md", b"a", do_index=False)
    (tmp_path / "unknown.md").write_bytes(b"b")  # predates the manifest

    status = manifest_status(config, "uploads")

    assert status["orphan_files"] == ["unknown.md"]
    assert status["missing_files"] == []


def test_status_reports_manifest_entry_with_no_file(tmp_path):
    config = _config(tmp_path)
    save_upload(config, "uploads", "gone.md", b"a", do_index=False)
    (tmp_path / "gone.md").unlink()

    status = manifest_status(config, "uploads")

    assert status["missing_files"] == ["gone.md"]


def test_prune_dry_run_does_not_change_anything(tmp_path):
    config = _config(tmp_path)
    save_upload(config, "uploads", "gone.md", b"a", do_index=False)
    (tmp_path / "gone.md").unlink()
    (tmp_path / "unknown.md").write_bytes(b"b")
    before = _read_manifest(tmp_path)

    result = prune_orphans(config, "uploads", apply=False)

    assert result["dry_run"] is True
    assert "gone.md" in result["would_remove"]
    assert _read_manifest(tmp_path) == before
    assert (tmp_path / "unknown.md").exists()


def test_prune_apply_removes_orphan_manifest_entries(tmp_path):
    config = _config(tmp_path)
    save_upload(config, "uploads", "gone.md", b"a", do_index=False)
    (tmp_path / "gone.md").unlink()
    (tmp_path / "unknown.md").write_bytes(b"b")

    result = prune_orphans(config, "uploads", apply=True)

    assert result["dry_run"] is False
    assert "gone.md" not in _read_manifest(tmp_path)
    # Orphan *files* are never deleted by prune — only stale manifest entries.
    assert (tmp_path / "unknown.md").exists()


# --------------------------------------------------------------------------
# The manifest must not be listed as a corpus document
# --------------------------------------------------------------------------

def test_manifest_not_listed_as_document(tmp_path):
    config = _config(tmp_path)
    save_upload(config, "uploads", "note.md", b"hello", do_index=False)

    listing = list_upload_folders(config)
    assert listing["total"] == 1
    # The manifest is infrastructure, not a document; ensure it is excluded from
    # any per-corpus file enumeration the status command exposes.
    status = manifest_status(config, "uploads")
    assert MANIFEST not in status["orphan_files"]
