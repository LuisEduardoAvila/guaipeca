"""Shared fixtures for Guaipeca integration tests."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from guaipeca.config import GuaipecaConfig
from guaipeca.converter import Converter
from guaipeca.embedding import EmbeddingService
from guaipeca.search import Searcher
from guaipeca.index import CorpusIndex
from guaipeca.chunking import chunk_file


# ---- Paths ----

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
TEST_PDF = FIXTURES_DIR / "test-doc.pdf"
TEST_MD = FIXTURES_DIR / "test-doc.md"
TEST_CONFIG = FIXTURES_DIR / "test-config.yaml"


# ---- Config fixture ----

@pytest.fixture(scope="session")
def test_config_path():
    """Path to the test config YAML."""
    # Write a fresh config that uses a temp data dir
    config_data = {
        "corpora": [
            {
                "name": "test-docs",
                "path": str(FIXTURES_DIR),
                "topic": "oracle-fcc",
                "weight": 1.0,
                "extensions": [".md", ".pdf"],
            }
        ],
        "embedding": {
            "model": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "cache_dir": "~/.guaipeca/models",
        },
        "chunking": {
            "max_size": 2000,
            "overlap": 200,
            "table_aware": True,
            "split_on_headings": True,
        },
        "indexing": {
            "data_dir": "~/.guaipeca/data-test",
            "incremental": True,
            "max_file_size": 52428800,
            "exclude_dirs": ["node_modules", ".git", "venv", "__pycache__"],
        },
    }

    # Use the pre-existing test-config.yaml
    if TEST_CONFIG.exists():
        return str(TEST_CONFIG)

    # Fallback: write it
    with open(TEST_CONFIG, "w") as f:
        yaml.dump(config_data, f)
    return str(TEST_CONFIG)


@pytest.fixture(scope="session")
def config(test_config_path):
    """Loaded GuaipecaConfig."""
    return GuaipecaConfig.from_yaml(test_config_path)


# ---- Services (session-scoped, expensive to init) ----

@pytest.fixture(scope="session")
def embedder():
    """EmbeddingService instance (model loads once, ~30s on Pi 5).

    Note: cache_dir must be an expanded path — EmbeddingService does not
    call os.path.expanduser on cache_dir (bug found during testing).
    """
    return EmbeddingService(
        model_name="all-MiniLM-L6-v2",
        cache_dir=os.path.expanduser("~/.guaipeca/models"),
        dimensions=384,
    )


@pytest.fixture(scope="session")
def converter():
    """Converter instance."""
    return Converter()


@pytest.fixture(scope="session")
def searcher(config, embedder, converter):
    """Searcher instance with pre-indexed corpus."""
    return Searcher(
        config=config,
        embedding_service=embedder,
        converter=converter,
    )


# ---- Test data ----

@pytest.fixture(scope="session")
def test_pdf_path():
    """Path to the test PDF."""
    assert TEST_PDF.exists(), f"Test PDF not found: {TEST_PDF}"
    return str(TEST_PDF)


@pytest.fixture(scope="session")
def test_md_path():
    """Path to the converted markdown."""
    assert TEST_MD.exists(), f"Test markdown not found: {TEST_MD}"
    return str(TEST_MD)


@pytest.fixture(scope="session")
def test_md_content(test_md_path):
    """Content of the converted markdown."""
    with open(test_md_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="session")
def indexed_corpus(searcher):
    """Ensure the corpus is indexed (idempotent — skips if already done)."""
    searcher.index_corpus(corpus_name="all")
    return searcher


def _index_built(index_dir: Path) -> bool:
    """True when a completed forced rebuild is already on disk.

    Both the primary and its one-generation backup must exist and be non-empty:
    ``_save`` only writes the backup when a primary was already there
    (``src/guaipeca/index.py:687``), so a backup is the marker of a *second*
    successful build — which is exactly the state ``force_reindexed_corpus``
    promises. A zero-byte primary means a build was interrupted.
    """
    primary = index_dir / "faiss_index.fai"
    backup = index_dir / "faiss_index.fai.bak"
    return (
        primary.exists()
        and backup.exists()
        and primary.stat().st_size > 0
        and backup.stat().st_size > 0
    )


@pytest.fixture(scope="session")
def force_reindexed_corpus(indexed_corpus):
    """``indexed_corpus`` after a forced rebuild, shared by the session.

    A forced rebuild of the fixture corpus costs ~11 s (181 chunks through
    FAISS). Eleven call sites in ``test_fmem_port.py`` and two in
    ``test_hybrid.py`` each triggered their own, which is where most of the
    suite's runtime went (#50). The rebuild is idempotent for tests that only
    *read* its result — the ``.fai.bak`` it produces, the cache state it clears
    — so one rebuild serves all of them.

    The rebuild is **skipped when a completed one is already on disk**. Without
    this gate every session pays the rebuild up front, including a one-test run
    that never needed a forced rebuild, which made single-test iteration slower
    than before #50 (21.5 s -> 29.0 s, measured). The persistent data dir means
    the cost is paid once per index change rather than once per session.

    Tests that mutate the index on disk (corruption, deletion) must NOT use
    this fixture; they need their own throwaway corpus via
    ``isolated_indexed_corpus``, or they poison the shared one for every test
    that runs after.
    """
    corpus_idx = indexed_corpus.corpora["test-docs"]
    if not _index_built(corpus_idx.index_dir):
        stats = indexed_corpus.index_corpus(corpus_name="test-docs", force=True)
        # ``index()`` returns {"error": ...} instead of raising when the lock is
        # held by another process (src/guaipeca/index.py:916-931). Ignoring that
        # would leave every dependent test asserting against a stale index while
        # still reporting green.
        if "error" in stats:
            pytest.fail(f"force reindex of 'test-docs' failed: {stats['error']}")
    return indexed_corpus


# ---- Temp dir for isolated tests ----

@pytest.fixture
def temp_data_dir():
    """Temp directory for isolated index tests."""
    d = tempfile.mkdtemp(prefix="guaipeca-test-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def small_indexed_corpus(embedder, converter, temp_data_dir):
    """A ``Searcher`` over one small markdown file — three-ish chunks.

    For tests whose assertion is about *behaviour* (a cache cleared, a flag
    set, a call made) rather than about a realistic corpus. A rebuild here costs
    a fraction of a second instead of ~11 s (#50).
    """
    doc_dir = Path(temp_data_dir) / "docs"
    doc_dir.mkdir()
    (doc_dir / "small.md").write_text(
        "# Small Doc\n\n"
        "## First Section\n\nAlpha content about consolidation and retrieval.\n\n"
        "## Second Section\n\nBeta content about indexing and chunking.\n",
        encoding="utf-8",
    )

    cfg = GuaipecaConfig.from_yaml(str(TEST_CONFIG))
    cfg.corpora[0].path = str(doc_dir)
    cfg.indexing.data_dir = str(Path(temp_data_dir) / "index")

    s = Searcher(config=cfg, embedding_service=embedder, converter=converter)
    s.index_corpus(corpus_name="test-docs", force=True)
    return s


@pytest.fixture
def isolated_indexed_corpus(force_reindexed_corpus, temp_data_dir):
    """A ``Searcher`` whose *index directory* is a copy of the built session
    index, so a test may corrupt or delete it without poisoning the session.

    Copying the built index (~1 MB) rather than re-indexing is the whole point:
    a rebuild costs ~11 s per test, which is what #50 is about. The copy carries
    the primary, the one-generation backup and the chunk metadata, so the
    corruption and recovery tests see exactly the state a real deployment has
    after two index runs — real FAISS bytes, not a mock.

    Only the index directory is isolated. The embedder and converter are the
    session-scoped instances and are shared with every other fixture, so
    consumers must not call ``index_corpus`` on this Searcher — that would clear
    and repopulate the shared embedding cache for later tests.
    """
    src_idx = force_reindexed_corpus.corpora["test-docs"].index_dir
    dst_root = Path(temp_data_dir) / "index"
    dst_idx = dst_root / "corpora" / "test-docs"
    dst_idx.parent.mkdir(parents=True)
    shutil.copytree(src_idx, dst_idx)

    # The copy must carry a real backup, or the recovery tests below assert
    # against an empty index and pass without testing anything.
    backup = dst_idx / "faiss_index.fai.bak"
    if not backup.exists() or backup.stat().st_size == 0:
        pytest.fail(
            f"no usable .fai.bak at {backup}; cannot exercise backup recovery"
        )

    cfg = GuaipecaConfig.from_yaml(str(TEST_CONFIG))
    cfg.indexing.data_dir = str(dst_root)

    return Searcher(
        config=cfg,
        embedding_service=force_reindexed_corpus.embedder,
        converter=force_reindexed_corpus.converter,
    )
