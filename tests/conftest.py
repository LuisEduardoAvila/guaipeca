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


# ---- Temp dir for isolated tests ----

@pytest.fixture
def temp_data_dir():
    """Temp directory for isolated index tests."""
    d = tempfile.mkdtemp(prefix="guaipeca-test-")
    yield d
    shutil.rmtree(d, ignore_errors=True)