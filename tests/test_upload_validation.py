"""Acceptance tests for issue #57: malformed or empty uploads must be rejected.

A malformed or empty payload is currently accepted and reported as saved. In
the delete-then-upload workflow that silently replaces a real document with
nothing, and the tool reports success.

Two layers are covered:

- the MCP tool, which decodes base64 and must reject anything that is not
  clean base64 (``validate=True``), and
- ``save_upload``, the shared core, which must reject empty and blank content
  regardless of how it was decoded, since the CLI hands it raw bytes and never
  touches base64 at all.

These are the acceptance gate: they fail on unfixed code.
"""

from __future__ import annotations

import base64

from guaipeca.config import CorpusConfig, GuaipecaConfig, UploadConfig
from guaipeca.mcp_server import GuaipecaMCPServer
from guaipeca.uploads import delete_upload, save_upload


class FakeSearcher:
    def __init__(self):
        self.calls: list[str] = []

    def index_corpus(self, corpus_name: str = "all", force: bool = False) -> dict:
        self.calls.append(corpus_name)
        return {"corpus": corpus_name, "indexed": True}


def _config(tmp_path) -> GuaipecaConfig:
    return GuaipecaConfig(
        corpora=[CorpusConfig(name="uploads", path=str(tmp_path), extensions=[".md"])],
        upload=UploadConfig(
            allow=["uploads"],
            allowed_extensions=[".md"],
            max_file_size=1024,
        ),
    )


def _server(tmp_path) -> GuaipecaMCPServer:
    return GuaipecaMCPServer(_config(tmp_path))


def _mcp_upload(server, filename, content_b64):
    return server._handle_upload(
        corpus_name="uploads",
        filename=filename,
        content_b64=content_b64,
        do_index=False,
    )


# --------------------------------------------------------------------------
# AC: base64 is decoded strictly
# --------------------------------------------------------------------------

def test_non_base64_junk_is_rejected(tmp_path):
    """'!!!' is silently discarded by a lenient decoder, producing b''."""
    server = _server(tmp_path)

    result = _mcp_upload(server, "doc.md", "!!!")

    assert "error" in result
    assert not (tmp_path / "doc.md").exists()


def test_trailing_junk_after_valid_payload_is_rejected(tmp_path):
    """A lenient decoder returns b'hello<' and reports success."""
    server = _server(tmp_path)

    result = _mcp_upload(server, "doc.md", "aGVsbG88!!!!")

    assert "error" in result
    assert not (tmp_path / "doc.md").exists()


def test_valid_base64_is_accepted(tmp_path):
    server = _server(tmp_path)

    result = _mcp_upload(server, "doc.md", base64.b64encode(b"# hello\n").decode())

    assert result.get("saved") is True
    assert (tmp_path / "doc.md").read_bytes() == b"# hello\n"


# --------------------------------------------------------------------------
# AC: empty content rejected
# --------------------------------------------------------------------------

def test_empty_base64_is_rejected(tmp_path):
    server = _server(tmp_path)

    result = _mcp_upload(server, "doc.md", "")

    assert "error" in result
    assert not (tmp_path / "doc.md").exists()


def test_empty_bytes_rejected_in_core(tmp_path):
    """The CLI passes raw bytes, so the core must guard empty content too."""
    config = _config(tmp_path)

    result = save_upload(config, "uploads", "doc.md", b"", do_index=False)

    assert "error" in result
    assert not (tmp_path / "doc.md").exists()


# --------------------------------------------------------------------------
# AC: whitespace-only content rejected
# --------------------------------------------------------------------------

def test_whitespace_only_bytes_rejected_in_core(tmp_path):
    config = _config(tmp_path)

    for payload in (b"\n", b"   ", b"\n\t \r\n"):
        result = save_upload(config, "uploads", "doc.md", payload, do_index=False)
        assert "error" in result, payload
        assert not (tmp_path / "doc.md").exists(), payload


def test_whitespace_only_base64_rejected_via_mcp(tmp_path):
    server = _server(tmp_path)

    result = _mcp_upload(server, "doc.md", base64.b64encode(b"   ").decode())

    assert "error" in result
    assert not (tmp_path / "doc.md").exists()


# --------------------------------------------------------------------------
# AC: nothing is written on rejection
# --------------------------------------------------------------------------

def test_rejection_changes_no_state(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "existing.md").write_bytes(b"KEEP ME")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}

    results = [
        save_upload(config, "uploads", "new.md", b"", do_index=False),
        save_upload(config, "uploads", "new.md", b"  \n ", do_index=False),
    ]

    for r in results:
        assert "error" in r
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert after == before, "a rejected upload must not change the corpus or leave temp files"
    assert not (tmp_path / "uploaded.json").exists(), "a rejected upload must not write a manifest entry"


# --------------------------------------------------------------------------
# AC: malformed replacement in the delete-then-upload cycle
# --------------------------------------------------------------------------

def test_malformed_replacement_does_not_create_empty_document(tmp_path):
    """Delete-then-upload with a bad payload must not end with an empty file."""
    config = _config(tmp_path)
    server = _server(tmp_path)

    # Seed a real document through the API.
    ok = _mcp_upload(server, "report.md", base64.b64encode(b"ORIGINAL REPORT").decode())
    assert ok.get("saved") is True

    deleted = delete_upload(config, "uploads", "report.md", do_index=False)
    assert deleted.get("deleted") is True

    bad = _mcp_upload(server, "report.md", "!!!")
    assert "error" in bad

    # Either the original survives, or it is absent -- never a 0-byte file
    # standing in its place, which is the silent-loss failure mode.
    path = tmp_path / "report.md"
    if path.exists():
        assert path.read_bytes() != b"", "a malformed upload must never leave an empty document"
    assert "report.md" not in __import__("json").loads(
        (tmp_path / "uploaded.json").read_text()
    ), "a rejected upload must not record a manifest entry"


def test_blank_replacement_does_not_blank_document(tmp_path):
    """A whitespace-only replacement is refused rather than blanking the file."""
    config = _config(tmp_path)
    server = _server(tmp_path)

    _mcp_upload(server, "report.md", base64.b64encode(b"ORIGINAL REPORT").decode())
    delete_upload(config, "uploads", "report.md", do_index=False)

    bad = _mcp_upload(server, "report.md", base64.b64encode(b"\n   \n").decode())
    assert "error" in bad

    path = tmp_path / "report.md"
    if path.exists():
        assert path.read_bytes().strip() != b""
