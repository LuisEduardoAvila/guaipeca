"""Regression tests from the adversarial code review of #83 + #85.

Each test here pins a defect that the review found and that was then fixed.
They fail against the pre-fix code, so they are the evidence the fixes hold.
"""

from __future__ import annotations

import base64
import json
import threading

from guaipeca.config import CorpusConfig, GuaipecaConfig, UploadConfig
from guaipeca.mcp_server import GuaipecaMCPServer
from guaipeca.uploads import manifest_status, prune_orphans, save_upload


def _config(tmp_path, **upload_kwargs) -> GuaipecaConfig:
    return GuaipecaConfig(
        corpora=[CorpusConfig(name="uploads", path=str(tmp_path), extensions=[".md"])],
        upload=UploadConfig(
            allow=["uploads"],
            allowed_extensions=[".md"],
            max_file_size=upload_kwargs.pop("max_file_size", 100_000),
            **upload_kwargs,
        ),
    )


# --------------------------------------------------------------------------
# Concurrent uploads must not lose manifest entries (review H1 / S3)
# --------------------------------------------------------------------------

def test_concurrent_uploads_keep_every_manifest_entry(tmp_path):
    """Each upload read the whole manifest and wrote it back; the last writer
    erased the others, leaving files on disk with no record."""
    config = _config(tmp_path)
    names = [f"doc{i}.md" for i in range(6)]
    barrier = threading.Barrier(len(names))
    results = []

    def upload(name):
        barrier.wait()
        results.append(save_upload(config, "uploads", name, b"content", do_index=False))

    threads = [threading.Thread(target=upload, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r.get("saved") is True for r in results), results
    manifest = json.loads((tmp_path / "uploaded.json").read_text())
    assert set(manifest) == set(names), "every uploaded file must be recorded"


def test_concurrent_upload_and_delete_keep_other_entries(tmp_path):
    """A delete that wrote back a stale manifest erased a concurrent upload."""
    config = _config(tmp_path)
    save_upload(config, "uploads", "keep.md", b"a", do_index=False)
    save_upload(config, "uploads", "drop.md", b"b", do_index=False)
    barrier = threading.Barrier(2)

    def do_delete():
        barrier.wait()
        from guaipeca.uploads import delete_upload
        delete_upload(config, "uploads", "drop.md", do_index=False)

    def do_upload():
        barrier.wait()
        save_upload(config, "uploads", "new.md", b"c", do_index=False)

    ts = [threading.Thread(target=do_delete), threading.Thread(target=do_upload)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    manifest = json.loads((tmp_path / "uploaded.json").read_text())
    assert "drop.md" not in manifest
    assert "keep.md" in manifest
    assert "new.md" in manifest, "the concurrent upload's entry must survive"


# --------------------------------------------------------------------------
# upload.replace must reject non-boolean values (review H2 / S2)
# --------------------------------------------------------------------------

def test_replace_rejects_quoted_string_false(tmp_path):
    """bool("false") is True: a quoted false silently ENABLED overwriting."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "corpora:\n"
        f"  - name: docs\n    path: {tmp_path}/corpus\n    extensions: [\".md\"]\n"
        "upload:\n  allow: [\"docs\"]\n  replace:\n    docs: \"false\"\n"
    )
    try:
        GuaipecaConfig.from_yaml(str(cfg_path))
    except ValueError as e:
        assert "must be true or false" in str(e)
    else:
        raise AssertionError("a string replace value must be rejected, not coerced")


def test_replace_accepts_real_booleans(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "corpora:\n"
        f"  - name: docs\n    path: {tmp_path}/corpus\n    extensions: [\".md\"]\n"
        "upload:\n  allow: [\"docs\"]\n  replace:\n    docs: true\n"
    )
    cfg = GuaipecaConfig.from_yaml(str(cfg_path))
    assert cfg.upload.replace == {"docs": True}


# --------------------------------------------------------------------------
# Over-long filenames must be refused, not raise (review E2 / E3)
# --------------------------------------------------------------------------

def test_overlong_filename_is_refused_without_raising(tmp_path):
    config = _config(tmp_path)
    result = save_upload(config, "uploads", "b" * 4092 + ".md", b"hi", do_index=False)
    assert "error" in result
    assert "too long" in result["error"].lower()


def test_overlong_manifest_entry_does_not_crash_status(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "uploaded.json").write_text(
        json.dumps({"b" * 4092 + ".md": {"sha256": "abc", "size_bytes": 1}})
    )
    status = manifest_status(config, "uploads")
    assert status["missing_files"], "an unstat-able entry is reported, not raised"


# --------------------------------------------------------------------------
# Wrapped base64 is accepted; junk is still refused (review S6)
# --------------------------------------------------------------------------

def test_line_wrapped_base64_is_accepted(tmp_path):
    """validate=True alone rejects MIME-wrapped payloads, which in
    delete-then-upload loses the document AFTER the delete."""
    config = _config(tmp_path)
    server = GuaipecaMCPServer(config)
    raw = b"# heading\n\n" + b"body text " * 40
    wrapped = base64.encodebytes(raw).decode()  # 76-column wrap
    assert "\n" in wrapped.strip()

    result = server._handle_upload("uploads", "doc.md", wrapped, False)

    assert result.get("saved") is True
    assert (tmp_path / "doc.md").read_bytes() == raw


def test_wrapped_decode_still_rejects_junk(tmp_path):
    config = _config(tmp_path)
    server = GuaipecaMCPServer(config)
    for bad in ("!!!", "aGVs bG8=x!!", "\n\n@@@@\n"):
        result = server._handle_upload("uploads", "doc.md", bad, False)
        assert "error" in result, bad
        assert not (tmp_path / "doc.md").exists()


# --------------------------------------------------------------------------
# prune must not raise on an unwritable manifest (review C1)
# --------------------------------------------------------------------------

def test_prune_returns_error_instead_of_raising_on_write_failure(tmp_path, monkeypatch):
    config = _config(tmp_path)
    save_upload(config, "uploads", "gone.md", b"a", do_index=False)
    (tmp_path / "gone.md").unlink()

    import guaipeca.uploads as up

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(up, "_write_manifest", boom)
    result = prune_orphans(config, "uploads", apply=True)

    assert "error" in result, "an unwritable manifest is an error result, not a traceback"
