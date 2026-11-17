"""Relevance floor for dense search results (issue #12).

Every index is ``faiss.IndexFlatIP`` over L2-normalised vectors, so a dense hit's
``raw_score`` IS the cosine similarity in [-1, 1]. The public knob is
``max_distance``: keep only hits with ``cosine >= 1 - max_distance``. Dense hits
also carry ``"distance": 1 - raw_score``; a missing/None ``distance`` means the
hit came only from BM25.

Tests use a deterministic fake embedder and synthetic unit vectors, so no model
is loaded and no real corpus is indexed.
"""

from __future__ import annotations

import numpy as np
import pytest

from guaipeca.chunking import Chunk
from guaipeca.config import (
    ChunkingConfig,
    CorpusConfig,
    GuaipecaConfig,
    IndexingConfig,
    SearchConfig,
)
from guaipeca.index import CorpusIndex, _assert_inner_product
from guaipeca.search import Searcher

DIMS = 8

# Query vector: unit vector along axis 0.
QUERY = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Chunk whose stored vector is identical to the query (cosine 1.0 -> distance 0).
HIGH_TEXT = "high similarity chunk"
HIGH_EMBEDDING = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Chunk whose stored vector is orthogonal to the query (cosine 0.0 -> distance 1).
LOW_TEXT = "low similarity chunk"
LOW_EMBEDDING = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


class FakeEmbedder:
    """Deterministic embedder: fixed text -> unit vector map, any query -> QUERY."""

    model_name = "fake"
    dimensions = DIMS
    preprocess = False

    def __init__(self, mapping: dict[str, np.ndarray] | None = None):
        self.mapping = mapping or {HIGH_TEXT: HIGH_EMBEDDING, LOW_TEXT: LOW_EMBEDDING}

    def embed_query(self, text: str) -> np.ndarray:
        return QUERY.copy()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.mapping[t] for t in texts])


class FakeConverter:
    """Returns whatever the test registered for a path (no real conversion)."""

    def __init__(self):
        self.texts: dict[str, str] = {}

    def convert(self, path: str) -> str:
        return self.texts[path]


def _make_searcher(tmp_path, corpora: list[CorpusConfig], search: SearchConfig | None = None):
    config = GuaipecaConfig(
        corpora=corpora,
        chunking=ChunkingConfig(max_size=10000, overlap=0, split_on_headings=False),
        indexing=IndexingConfig(data_dir=str(tmp_path / "data"), incremental=True),
        search=search or SearchConfig(hybrid=False),
    )
    embedder = FakeEmbedder()
    converter = FakeConverter()
    return Searcher(config=config, embedding_service=embedder, converter=converter), converter


def _add_chunk(searcher: Searcher, converter: FakeConverter, corpus: str, tmp_path,
               filename: str, text: str, embedding: np.ndarray) -> None:
    """Write a file and inject one chunk + its vector straight into the corpus index.

    Bypasses chunking and embedding: the vector is supplied explicitly so the
    cosine similarity of every hit is exactly known.
    """
    source = tmp_path / f"{corpus}-{filename}"
    source.write_text(text, encoding="utf-8")
    converter.texts[str(source)] = text

    corpus_idx: CorpusIndex = searcher.corpora[corpus]
    with corpus_idx._rwlock.write_lock():
        corpus_idx._load()
        corpus_idx._chunks.append(
            Chunk(
                id=f"c{len(corpus_idx._chunks):04d}",
                heading=text,
                text=text,
                source_path=str(source),
            )
        )
        corpus_idx._faiss_index.add(embedding.reshape(1, -1).astype(np.float32))
        corpus_idx._rebuild_stable_id_map()
        corpus_idx._rebuild_toc_metadata()
        # The sparse side is built by index(), which this helper bypasses. Without
        # it every hybrid test silently degrades to dense-only and a "BM25-only
        # survives" assertion would pass for the wrong reason - or, as happened
        # here, return nothing at all and look like a filter bug.
        corpus_idx._build_bm25_index(corpus_idx._chunks)


def _two_chunk_corpus(tmp_path, search: SearchConfig | None = None, **corpus_kwargs):
    """One corpus with a cosine-1.0 chunk and a cosine-0.0 chunk."""
    corpus = CorpusConfig(
        name="docs", path=str(tmp_path), extensions=[".md"], **corpus_kwargs
    )
    searcher, converter = _make_searcher(tmp_path, [corpus], search)
    _add_chunk(searcher, converter, "docs", tmp_path, "high.md", HIGH_TEXT, HIGH_EMBEDDING)
    _add_chunk(searcher, converter, "docs", tmp_path, "low.md", LOW_TEXT, LOW_EMBEDDING)
    return searcher


# ---------------------------------------------------------------------------
# 1. distance field
# ---------------------------------------------------------------------------

def test_distance_field_on_dense_results(tmp_path):
    searcher = _two_chunk_corpus(tmp_path)
    result = searcher.search("anything", top_k=5)
    assert "error" not in result
    assert result["total"] == 2
    for hit in result["results"]:
        assert hit["distance"] == pytest.approx(1.0 - hit["raw_score"])

    by_text = {h["summary"]: h for h in result["results"]}
    assert by_text[HIGH_TEXT]["distance"] == pytest.approx(0.0)
    assert by_text[LOW_TEXT]["distance"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. omitted floor == current behaviour
# ---------------------------------------------------------------------------

def test_omitted_floor_matches_current_behaviour(tmp_path):
    searcher = _two_chunk_corpus(tmp_path)
    result = searcher.search("anything", top_k=5)

    # Exact pre-#12 envelope: no floor bookkeeping keys at all.
    assert set(result) == {"results", "total", "query", "hybrid", "return_mode"}
    assert result["hybrid"] is False
    assert result["return_mode"] == "chunks"
    assert result["total"] == 2
    # Both hits survive, in cosine order, with the untouched raw scores.
    assert [h["summary"] for h in result["results"]] == [HIGH_TEXT, LOW_TEXT]
    assert [h["score"] for h in result["results"]] == pytest.approx([1.0, 0.0])
    assert [h["raw_score"] for h in result["results"]] == pytest.approx([1.0, 0.0])


# ---------------------------------------------------------------------------
# 3. floor drops low similarity
# ---------------------------------------------------------------------------

def test_floor_drops_low_similarity(tmp_path):
    searcher = _two_chunk_corpus(tmp_path)
    # Floor 0.5 -> keep cosine >= 0.5 -> only the high chunk survives.
    result = searcher.search("anything", top_k=5, max_distance=0.5)
    assert "error" not in result
    assert result["total"] == 1
    assert result["results"][0]["summary"] == HIGH_TEXT
    assert result["filtered_by_distance"] == 1


# ---------------------------------------------------------------------------
# 4. all filtered -> empty success
# ---------------------------------------------------------------------------

def test_all_filtered_returns_empty_success(tmp_path):
    # A negative floor would be a config error (validation rejects it), so build
    # the case from a chunk we control instead: the best hit has cosine 0.6
    # (distance 0.4), and a floor of 0.1 (cutoff 0.9) admits nothing. A chunk in
    # hand makes the numbers explicit instead of relying on how a real embedder
    # happens to score.
    searcher, converter = _make_searcher(
        tmp_path,
        [CorpusConfig(name="docs", path=str(tmp_path), extensions=[".md"])],
    )
    mid = np.array([0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _add_chunk(searcher, converter, "docs", tmp_path, "mid.md", "mid similarity chunk", mid)

    result = searcher.search("anything", top_k=5, max_distance=0.1)
    assert "error" not in result
    assert result["results"] == []
    assert result["total"] == 0
    assert result["filtered_by_distance"] == 1
    assert result["max_distance_applied"] is True
    # An empty result here is a SUCCESS, not a validation failure.
    assert set(result) == {
        "results", "total", "query", "max_distance_applied", "filtered_by_distance",
    }


# ---------------------------------------------------------------------------
# 5. BM25-only hit survives the floor
# ---------------------------------------------------------------------------

def test_bm25_only_hit_survives_floor(tmp_path):
    # One chunk is orthogonal to the query (dense distance 1.0) but contains the
    # exact query token, so BM25 finds it. It must survive a floor that drops
    # its dense hit: hybrid exists to catch exact-token matches.
    token_text = "zebracorn"
    searcher, converter = _make_searcher(
        tmp_path,
        [CorpusConfig(name="docs", path=str(tmp_path), extensions=[".md"])],
        SearchConfig(hybrid=True),
    )
    _add_chunk(searcher, converter, "docs", tmp_path, "ortho.md", token_text, LOW_EMBEDDING)
    # Give the dense index a second, unrelated vector so the query still has a
    # dense side to filter; the token chunk stays BM25-only.
    _add_chunk(searcher, converter, "docs", tmp_path, "other.md", "other text", LOW_EMBEDDING)

    result = searcher.search(token_text, top_k=5, hybrid=True, max_distance=0.1)
    assert "error" not in result
    assert result["total"] == 1
    hit = result["results"][0]
    assert hit["summary"] == token_text
    # Fusion kept the sparse record, which has no dense score to measure.
    assert hit.get("distance") is None
    assert result["filtered_by_distance"] >= 1


# ---------------------------------------------------------------------------
# 6. per-corpus floor overrides global
# ---------------------------------------------------------------------------

def test_per_corpus_floor_overrides_global(tmp_path):
    # Global floor 0.0 means cosine >= 1.0 (only an exact match). The corpus's
    # own 2.0 (cosine >= -1.0) must win over both the global value and the
    # per-call value.
    searcher = _two_chunk_corpus(
        tmp_path, SearchConfig(hybrid=False, max_distance=0.0), max_distance=2.0
    )

    result = searcher.search("anything", top_k=5)
    assert result["total"] == 2

    result = searcher.search("anything", top_k=5, max_distance=0.0)
    assert result["total"] == 2


def test_call_floor_overrides_global(tmp_path):
    # No corpus value: the per-call floor beats the global one, both ways.
    searcher = _two_chunk_corpus(tmp_path, SearchConfig(hybrid=False, max_distance=0.0))
    assert searcher.search("anything", top_k=5)["total"] == 1  # global 0.0 -> exact only
    assert searcher.search("anything", top_k=5, max_distance=2.0)["total"] == 2
    assert searcher.search("anything", top_k=5, max_distance=0.5)["total"] == 1


# ---------------------------------------------------------------------------
# 7. floor ignored when None
# ---------------------------------------------------------------------------

def test_floor_ignored_when_none(tmp_path):
    searcher = _two_chunk_corpus(tmp_path, SearchConfig(hybrid=False, max_distance=None))
    result = searcher.search("anything", top_k=5, max_distance=None)
    assert result["total"] == 2
    assert "filtered_by_distance" not in result
    assert "max_distance_applied" not in result


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-0.1, "0.5", True])
def test_invalid_max_distance_rejected(tmp_path, bad):
    corpus = CorpusConfig(name="docs", path=str(tmp_path), max_distance=bad)
    config = GuaipecaConfig(corpora=[corpus])
    with pytest.raises(ValueError, match="max_distance"):
        config._validate_types()


def test_max_distance_parsed_from_yaml(tmp_path):
    import yaml

    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    config_path = tmp_path / "guaipeca.yaml"
    config_path.write_text(
        yaml.dump({
            "corpora": [{"name": "docs", "path": str(doc_dir), "max_distance": 0.4}],
            "search": {"max_distance": 0.75},
        }),
        encoding="utf-8",
    )
    config = GuaipecaConfig.from_yaml(str(config_path))
    assert config.search.max_distance == 0.75
    assert config.corpora[0].max_distance == 0.4


# ---------------------------------------------------------------------------
# Guard: fail closed on a non inner-product index
# ---------------------------------------------------------------------------

def test_non_inner_product_index_is_rejected():
    import faiss

    with pytest.raises(RuntimeError, match="metric is not inner product"):
        _assert_inner_product(faiss.IndexFlatL2(DIMS))
