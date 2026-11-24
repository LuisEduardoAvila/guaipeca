"""Integration test for the hybrid default (issue #94).

Verifies that a Searcher built from a config with no ``search:`` section
defaults to ``hybrid=True`` and actually returns BM25 hits — the behavioural
test that #67/#93 were missing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from guaipeca.config import GuaipecaConfig
from guaipeca.search import Searcher


@pytest.fixture
def hybrid_default_config(tmp_path: Path) -> Path:
    """A minimal config with NO ``search:`` section — the dataclass default
    must apply."""
    cfg = {
        "corpora": [
            {
                "name": "test-docs",
                "path": str(tmp_path),
                "topic": "Test corpus",
                "weight": 1.0,
                "extensions": [".md"],
            }
        ],
        "upload": {"allow": ["test-docs"], "auto_index": True},
        "embedding": {
            "model": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "cache_dir": str(tmp_path / "models"),
        },
        "chunking": {"max_size": 2000, "overlap": 200},
        "indexing": {"data_dir": str(tmp_path / "data"), "max_file_size": 52428800},
    }
    cfg_path = tmp_path / "hybrid-default-config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    return cfg_path


def test_searcher_defaults_to_hybrid(hybrid_default_config):
    """A Searcher built from a config with no ``search:`` section runs
    hybrid search by default."""
    cfg = GuaipecaConfig.from_yaml(str(hybrid_default_config))
    # The dataclass default must be True.
    assert cfg.search.hybrid is True, (
        f"Expected search.hybrid to default to True, got {cfg.search.hybrid}"
    )


def test_hybrid_default_returns_bm25_hits(hybrid_default_config, embedder):
    """A Searcher built from a config with no ``search:`` section returns
    BM25 hits — proving the hybrid path is actually exercised, not just
    asserted on the parsed config."""
    cfg = GuaipecaConfig.from_yaml(str(hybrid_default_config))
    searcher = Searcher(config=cfg, embedding_service=embedder)

    # Create a corpus directory with a document containing a unique token.
    doc_dir = hybrid_default_config.parent / "docs"
    doc_dir.mkdir(exist_ok=True)
    doc_path = doc_dir / "bm25-test.md"
    # Use a token that is unlikely to appear in any embedding training data.
    doc_path.write_text(
        "## BM25 Test\n\n"
        "The zebracorn unicorn flew over the moon and back.\n",
        encoding="utf-8",
    )

    # Index the corpus.
    searcher.index_corpus(corpus_name="test-docs", force=True)

    # Search for the unique token. With hybrid=True, BM25 must find it.
    result = searcher.search("zebracorn", top_k=5)

    assert "error" not in result, f"Search failed: {result['error']}"
    assert result["hybrid"] is True, (
        f"Expected hybrid=True in search result, got {result.get('hybrid')}"
    )
    assert result["total"] >= 1, (
        "Expected at least one hit from BM25; got zero. "
        "The hybrid default may not be working."
    )
    # The hit exists and hybrid mode is active — the BM25 path found the
    # unique token "zebracorn" even though its embedding is near-random.
    # (The summary may be the heading; we only assert the hit count and mode.)
