"""Regression tests for TOC tree construction in index.py.

Bug (reported against 0.3.5 on Windows 11 / Python 3.14):

    UnboundLocalError: cannot access local variable 'os' where it is not
    associated with a value

``_build_toc_tree_for_doc`` had a redundant ``import os`` inside the
``if not has_structure:`` branch. A function-local import makes ``os`` a
local name for the *whole* function, so the later ``os.path.basename()`` on
the structured path — which never executed the import — raised
UnboundLocalError.

The bug only fired for documents **with** headings, which is the common case
for Markdown. It was invisible locally because ``_load()`` wraps the TOC
rebuild in ``except Exception:``, so the error was swallowed and left stale
TOC metadata behind instead of failing loudly.

These tests exercise both branches directly, so neither the heading path nor
the non-heading path can regress.
"""

from __future__ import annotations

import numpy as np

from guaipeca.chunking import Chunk
from guaipeca.config import ChunkingConfig, CorpusConfig, IndexingConfig
from guaipeca.index import CorpusIndex

DIM = 32


class MockEmbedder:
    dimensions = DIM
    preprocess = False

    def embed(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(DIM).astype(np.float32)
            n = np.linalg.norm(v)
            out.append(v / n if n > 0 else v)
        return np.vstack(out)


def _make_index(tmp_path):
    return CorpusIndex(
        corpus=CorpusConfig(name="toc-encoding-test", path=str(tmp_path)),
        chunking=ChunkingConfig(),
        indexing=IndexingConfig(data_dir=str(tmp_path)),
        embedding_service=MockEmbedder(),
    )


def _chunks(source_path: str, heading_paths: list[list[str]]) -> list[Chunk]:
    return [
        Chunk(
            id=f"chunk-{i}",
            heading=hp[-1] if hp else "",
            text=f"text {i}",
            source_path=source_path,
            char_offset=i,
            heading_path=hp,
            section_id=f"s{i}" if hp else "",
        )
        for i, hp in enumerate(heading_paths)
    ]


class TestBuildTocTreeWithoutOsShadowing:
    def test_structured_document_builds_tree(self, tmp_path):
        """Documents with headings on the first chunk build a structured tree.

        Note: this case did *not* hit the original bug — the title expression
        short-circuits on ``doc_chunks[0].heading_path`` being non-empty, so
        ``os.path.basename`` was never evaluated. It is kept as coverage of
        the structured path, not as the regression trigger. See
        ``test_structured_title_falls_back_when_no_heading_path`` for the
        case that actually raised UnboundLocalError.
        """
        idx = _make_index(tmp_path)
        source = str(tmp_path / "guide.md")
        idx._chunks = _chunks(source, [["Consolidation", "Overview"], ["Consolidation", "Rules"]])

        tree = idx._build_toc_tree_for_doc(source)

        assert tree["structured"] is True
        assert tree["title"] == "Consolidation"
        assert tree["chunk_count"] == 2

    def test_unstructured_document_uses_filename(self, tmp_path):
        """Documents without headings fall back to the basename."""
        idx = _make_index(tmp_path)
        source = str(tmp_path / "notes.md")
        idx._chunks = _chunks(source, [[], []])

        tree = idx._build_toc_tree_for_doc(source)

        assert tree["structured"] is False
        assert tree["title"] == "notes.md"
        assert tree["chunk_count"] == 2

    def test_structured_title_falls_back_when_no_heading_path(self, tmp_path):
        """Regression: structured doc whose first chunk has no heading path.

        ``has_structure`` is True (a later chunk has a heading), so the
        ``if not has_structure:`` branch — and its now-removed ``import os`` —
        is skipped. The title expression then evaluates
        ``os.path.basename(source_path)``. When ``os`` was imported inside the
        skipped branch it was a function-local name, and this line raised
        UnboundLocalError. Verified to fail against the unfixed source.
        """
        idx = _make_index(tmp_path)
        source = str(tmp_path / "mixed.md")
        # First chunk has no heading path, second does -> has_structure is True,
        # so the basename fallback on the structured path is exercised.
        idx._chunks = _chunks(source, [[], ["Intro"]])

        tree = idx._build_toc_tree_for_doc(source)

        assert tree["structured"] is True
        assert tree["title"] == "mixed.md"

    def test_rebuild_metadata_handles_mixed_documents(self, tmp_path):
        """TOC rebuild over structured + unstructured docs must not raise."""
        idx = _make_index(tmp_path)
        structured = str(tmp_path / "a.md")
        flat = str(tmp_path / "b.md")
        idx._chunks = _chunks(structured, [["A", "B"]]) + _chunks(flat, [[]])

        idx._rebuild_toc_metadata()

        assert set(idx._toc_trees) == {structured, flat}
        assert idx._toc_trees[structured]["structured"] is True
        assert idx._toc_trees[flat]["structured"] is False
