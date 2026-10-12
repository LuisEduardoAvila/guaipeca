"""Tests for the shared upload/delete core (src/guaipeca/uploads.py)."""

from __future__ import annotations

from guaipeca.config import CorpusConfig, GuaipecaConfig, UploadConfig
from guaipeca.uploads import delete_upload, list_upload_folders, save_upload


class FakeSearcher:
    """Minimal searcher stub recording index_corpus calls."""

    def __init__(self):
        self.calls: list[str] = []

    def index_corpus(self, corpus_name: str = "all", force: bool = False) -> dict:
        self.calls.append(corpus_name)
        return {"corpus": corpus_name, "indexed": True}


def _config(tmp_path, **upload_kwargs) -> GuaipecaConfig:
    return GuaipecaConfig(
        corpora=[CorpusConfig(name="uploads", path=str(tmp_path), extensions=[".md"])],
        upload=UploadConfig(
            allow=["uploads"],
            allowed_extensions=[".md"],
            max_file_size=upload_kwargs.pop("max_file_size", 1024),
            **upload_kwargs,
        ),
    )


def test_list_upload_folders_only_lists_allowed(tmp_path):
    config = _config(tmp_path)
    config.corpora.append(CorpusConfig(name="readonly", path=str(tmp_path)))
    result = list_upload_folders(config)
    assert result["total"] == 1
    assert result["folders"][0]["name"] == "uploads"
    assert result["allowed_extensions"] == [".md"]


def test_save_upload_writes_file(tmp_path):
    config = _config(tmp_path)
    result = save_upload(config, "uploads", "note.md", b"hello", do_index=False)
    assert result["saved"] is True
    assert result["indexed"] is False
    assert (tmp_path / "note.md").read_bytes() == b"hello"


def test_save_upload_rejects_disallowed_corpus(tmp_path):
    config = _config(tmp_path)
    result = save_upload(config, "readonly", "note.md", b"x", do_index=False)
    assert "error" in result


def test_save_upload_rejects_bad_extension(tmp_path):
    config = _config(tmp_path)
    result = save_upload(config, "uploads", "evil.exe", b"x", do_index=False)
    assert "error" in result
    assert not (tmp_path / "evil.exe").exists()


def test_save_upload_strips_path_traversal(tmp_path):
    config = _config(tmp_path)
    result = save_upload(config, "uploads", "../../escape.md", b"x", do_index=False)
    # basename is applied, so the file lands inside the corpus directory
    assert result["saved"] is True
    assert (tmp_path / "escape.md").exists()
    assert not (tmp_path.parent.parent / "escape.md").exists()


def test_save_upload_enforces_size_limit(tmp_path):
    config = _config(tmp_path, max_file_size=4)
    result = save_upload(config, "uploads", "big.md", b"too many bytes", do_index=False)
    assert "error" in result
    assert not (tmp_path / "big.md").exists()


def test_save_upload_indexes_when_requested(tmp_path):
    config = _config(tmp_path)
    searcher = FakeSearcher()
    result = save_upload(config, "uploads", "note.md", b"hello", do_index=True, searcher=searcher)
    assert result["indexed"] is True
    assert searcher.calls == ["uploads"]


def test_delete_upload_removes_file(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "note.md").write_bytes(b"hello")
    result = delete_upload(config, "uploads", "note.md", do_index=False)
    assert result["deleted"] is True
    assert not (tmp_path / "note.md").exists()


def test_delete_upload_missing_file(tmp_path):
    config = _config(tmp_path)
    result = delete_upload(config, "uploads", "nope.md", do_index=False)
    assert "error" in result


def test_delete_upload_reindexes_when_requested(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "note.md").write_bytes(b"hello")
    searcher = FakeSearcher()
    result = delete_upload(config, "uploads", "note.md", do_index=True, searcher=searcher)
    assert result["indexed"] is True
    assert searcher.calls == ["uploads"]


def test_delete_upload_rejects_disallowed_corpus(tmp_path):
    config = _config(tmp_path)
    result = delete_upload(config, "readonly", "note.md", do_index=False)
    assert "error" in result
