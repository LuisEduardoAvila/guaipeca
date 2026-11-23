"""Acceptance tests for issue #62: incremental reindex must not keep stale chunks.

A modified file is never in the "deleted" set, because its hash is overwritten
before that set is computed. Its new chunks are then appended on top of the old
ones: the index grows, superseded content stays retrievable, and one chunk_id
ends up at more than one FAISS position.

These are the acceptance gate and fail on unfixed code.

The tests use a deterministic mock embedder (same pattern as
test_index_concurrency.py) so no model has to load.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from guaipeca.config import ChunkingConfig, CorpusConfig, IndexingConfig
from guaipeca.index import CorpusIndex

DIM = 32


class MockEmbedder:
    """Deterministic embedder - no model load, no external deps."""

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

    def embed_query(self, text):
        return self.embed([text])[0]


def _index(tmp_path, name="docs"):
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    return CorpusIndex(
        corpus=CorpusConfig(name=name, path=str(corpus)),
        chunking=ChunkingConfig(),
        indexing=IndexingConfig(data_dir=str(tmp_path / "idx")),
        embedding_service=MockEmbedder(),
    )


def _current_chunk_count(idx: CorpusIndex) -> int:
    """Chunk count the current file contents should produce, computed fresh."""
    texts = [c.text for c in idx._chunks]
    return len(texts)


# --------------------------------------------------------------------------
# AC: modify then reindex leaves no stale chunks
# --------------------------------------------------------------------------

def test_modify_and_reindex_does_not_grow_the_index(tmp_path):
    """The core defect: ntotal and len(chunks) inflate together on reindex."""
    idx = _index(tmp_path)
    doc = tmp_path / "corpus" / "doc.md"
    doc.write_text("# A\n\nalpha one\n\n# B\n\nbeta two\n\n# C\n\ngamma three\n")
    idx.index()

    before = idx._faiss_index.ntotal
    assert before > 0

    doc.write_text("# A\n\nalpha one CHANGED\n\n# B\n\nbeta two\n\n# C\n\ngamma three\n")
    idx.index()

    # Same structure, same number of chunks - but the old ones must have gone.
    assert idx._faiss_index.ntotal == before, (
        f"ntotal grew from {before} to {idx._faiss_index.ntotal}: stale chunks kept"
    )
    assert len(idx._chunks) == before


def test_old_content_is_not_retrievable_after_modification(tmp_path):
    """Superseded text must not survive in the chunk list."""
    idx = _index(tmp_path)
    doc = tmp_path / "corpus" / "doc.md"
    doc.write_text("# A\n\nORIGINAL PHRASE HERE\n")
    idx.index()
    assert any("ORIGINAL PHRASE HERE" in c.text for c in idx._chunks)

    doc.write_text("# A\n\nREPLACEMENT PHRASE HERE\n")
    idx.index()

    stale = [c for c in idx._chunks if "ORIGINAL PHRASE HERE" in c.text]
    assert not stale, f"old content still indexed: {[c.text for c in stale]}"
    assert any("REPLACEMENT PHRASE HERE" in c.text for c in idx._chunks)


def test_no_chunk_id_at_two_faiss_positions(tmp_path):
    """A duplicated chunk_id makes get_chunk ambiguous."""
    idx = _index(tmp_path)
    doc = tmp_path / "corpus" / "doc.md"
    doc.write_text("# A\n\nalpha\n\n# B\n\nbeta\n\n# C\n\ngamma\n")
    idx.index()

    doc.write_text("# A\n\nalpha edited\n\n# B\n\nbeta\n\n# C\n\ngamma\n")
    idx.index()

    counts = Counter(c.id for c in idx._chunks)
    dupes = {cid: n for cid, n in counts.items() if n > 1}
    assert not dupes, f"chunk_id appears at more than one position: {dupes}"


def test_unchanged_file_is_still_skipped(tmp_path):
    """The fix must not turn incremental indexing into a full rebuild."""
    idx = _index(tmp_path)
    corpus = tmp_path / "corpus"
    (corpus / "a.md").write_text("# A\n\nalpha\n")
    idx.index()

    stats = idx.index()
    assert stats["files_skipped"] >= 1
    assert stats["files_indexed"] == 0


def test_repeated_modifications_do_not_compound(tmp_path):
    """Three edits in a row must not leave three generations of chunks."""
    idx = _index(tmp_path)
    doc = tmp_path / "corpus" / "doc.md"
    for i in range(1, 4):
        doc.write_text(f"# A\n\nversion {i} text\n")
        idx.index()

    assert idx._faiss_index.ntotal == len(idx._chunks)
    versions = [c.text for c in idx._chunks if "version" in c.text]
    assert len(versions) == 1, f"expected only the latest version, got {versions}"
    assert "version 3" in versions[0]


# --------------------------------------------------------------------------
# AC: "has indexed content" guard must count BM25 too
# --------------------------------------------------------------------------

def test_has_indexed_content_counts_chunks_not_only_vectors(tmp_path):
    """A corpus with chunks but zero FAISS vectors must not report 'no index'."""
    from guaipeca.config import GuaipecaConfig
    from guaipeca.converter import Converter
    from guaipeca.search import Searcher

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text("# A\n\nalpha\n")

    cfg = GuaipecaConfig(corpora=[CorpusConfig(name="docs", path=str(corpus))])
    cfg.indexing.data_dir = str(tmp_path / "idx")
    emb = MockEmbedder()
    searcher = Searcher(config=cfg, embedding_service=emb, converter=Converter())

    idx = searcher.corpora["docs"]
    idx.index()
    # Simulate chunks present but no vectors (e.g. a partially written index).
    import faiss
    idx._faiss_index = faiss.IndexFlatIP(DIM)

    result = searcher.search("alpha", top_k=3)
    assert result.get("error") != "No indexed content found. Run 'guaipeca index' first.", (
        "a corpus with chunks but no vectors must not be treated as empty"
    )


def test_failed_conversion_keeps_the_document_instead_of_dropping_it(tmp_path):
    """A transient error must not remove a document from the index.

    Dropping a changed file's old chunks is only correct when the file is
    actually re-indexed. If conversion fails, the previous chunks are stale but
    present, which is strictly better than the document silently vanishing.
    """
    idx = _index(tmp_path)
    doc = tmp_path / "corpus" / "doc.md"
    doc.write_text("# A\n\nversion one\n")
    idx.index()
    assert idx._faiss_index.ntotal == 1

    real_convert = idx.converter.convert
    doc.write_text("# A\n\nversion two\n")

    def boom(path):
        raise RuntimeError("transient converter failure")

    idx.converter.convert = boom
    try:
        stats = idx.index()
    finally:
        idx.converter.convert = real_convert

    assert stats["errors"], "the failure should be reported in stats"
    assert idx._faiss_index.ntotal == 1, "the document must not vanish from the index"
    assert len(idx._chunks) == 1

    # The next successful run corrects it.
    idx.index()
    assert len([c for c in idx._chunks if "version two" in c.text]) == 1
    assert not any("version one" in c.text for c in idx._chunks)
