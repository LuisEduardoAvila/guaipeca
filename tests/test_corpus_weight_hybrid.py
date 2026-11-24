"""Tests for corpus weights in hybrid search mode (issue #65).

In dense-only mode, ``CorpusIndex.search()`` multiplies the score by
``corpus.weight`` before returning. In hybrid mode the fuse step
recomputes the score from rank position and the normalized raw score,
which discarded the pre-applied weight — making ``corpus.weight`` a
no-op whenever ``hybrid=True``.
"""

from __future__ import annotations

from guaipeca.config import GuaipecaConfig, SearchConfig
from guaipeca.search import Searcher


def _make_searcher(**search_kwargs):
    """Create a minimal searcher with given search config for unit tests."""
    cfg = GuaipecaConfig(corpora=[], search=SearchConfig(**search_kwargs))
    searcher = Searcher.__new__(Searcher)
    searcher.config = cfg
    return searcher


_DENSE = [
    {"chunk_id": "a:1", "raw_score": 0.9, "summary": "dense hit", "score": 0.9},
]
_SPARSE = [
    {"chunk_id": "a:1", "raw_score": 11.2, "summary": "bm25 hit", "score": 11.2},
]


class TestCorpusWeightInHybridFuse:
    """``_rrf_fuse`` must scale its output by the corpus weight."""

    def test_weight_scales_fused_score(self):
        """A corpus weight of 2.0 must double the fused score."""
        searcher = _make_searcher()

        base = searcher._rrf_fuse(_DENSE, _SPARSE, top_k=5, corpus_weight=1.0)
        doubled = searcher._rrf_fuse(_DENSE, _SPARSE, top_k=5, corpus_weight=2.0)

        assert base[0]["chunk_id"] == doubled[0]["chunk_id"]
        assert doubled[0]["score"] == base[0]["score"] * 2.0

    def test_zero_weight_drops_score(self):
        """A disabled corpus (weight 0) must not contribute rank by score."""
        searcher = _make_searcher()

        fused = searcher._rrf_fuse(_DENSE, _SPARSE, top_k=5, corpus_weight=0.0)

        assert fused[0]["score"] == 0.0

    def test_default_weight_is_backward_compatible(self):
        """Omitting the weight keeps the previous behavior (weight 1.0)."""
        searcher = _make_searcher()

        implicit = searcher._rrf_fuse(_DENSE, _SPARSE, top_k=5)
        explicit = searcher._rrf_fuse(_DENSE, _SPARSE, top_k=5, corpus_weight=1.0)

        assert implicit[0]["score"] == explicit[0]["score"]

    def test_heavier_corpus_outranks_identical_lighter_corpus(self):
        """Identical hits from two corpora: the heavier corpus must rank first."""
        searcher = _make_searcher()

        light = searcher._rrf_fuse(
            [{"chunk_id": "light:x", "raw_score": 0.9, "summary": "s", "score": 0.9}],
            [{"chunk_id": "light:x", "raw_score": 11.2, "summary": "s", "score": 11.2}],
            top_k=5,
            corpus_weight=1.0,
        )
        heavy = searcher._rrf_fuse(
            [{"chunk_id": "heavy:x", "raw_score": 0.9, "summary": "s", "score": 0.9}],
            [{"chunk_id": "heavy:x", "raw_score": 11.2, "summary": "s", "score": 11.2}],
            top_k=5,
            corpus_weight=3.0,
        )

        merged = sorted(light + heavy, key=lambda r: r["score"], reverse=True)
        assert merged[0]["chunk_id"] == "heavy:x"
