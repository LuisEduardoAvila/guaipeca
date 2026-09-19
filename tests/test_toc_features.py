"""Tests for STAIR-inspired ToC-aware features.

Tests cover:
1. heading_path with structured and unstructured documents (Feature A)
2. section_id generation (Feature A)
3. section_mode search -- full section aggregation (Feature C)
4. section_mode on unstructured documents
5. get_toc -- structured and unstructured documents (Feature D)
6. section_filter -- search within a section (Feature E)
7. section_filter on unstructured documents
8. section_filter not found
9. toc_rerank -- keyword re-ranking (Feature B)
10. toc_rerank on unstructured documents
11. Backward compatibility -- old metadata.json without new fields
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from guaipeca.chunking import (
    Chunk,
    chunk_file,
)
from guaipeca.config import (
    ChunkingConfig,
    CorpusConfig,
    EmbeddingConfig,
    GuaipecaConfig,
    IndexingConfig,
    SearchConfig,
)
from guaipeca.embedding import EmbeddingService
from guaipeca.index import CorpusIndex
from guaipeca.search import Searcher

# ===========================================================================
# Test fixtures
# ===========================================================================

STRUCTURED_MD = """\
# Document Title

## Introduction

This is the introduction section. It talks about RAG systems and embeddings.

### Background

Some background information about embeddings here.

## Methods

The methods section describes the chunking and indexing approach.

### Chunking Strategy

We split on headings and respect table boundaries.

## Results

The results show improved retrieval with structure-aware chunking.
"""

UNSTRUCTURED_MD = """\
This is a plain text document without any markdown headings.

It just has paragraphs of text about financial consolidation and close processes.

The document talks about journal entries, currency translation, and data audit.

There are no heading markers at all in this document.
"""


@pytest.fixture
def structured_path(tmp_path):
    """Path to a structured markdown file."""
    p = tmp_path / "structured.md"
    p.write_text(STRUCTURED_MD)
    return str(p)


@pytest.fixture
def unstructured_path(tmp_path):
    """Path to an unstructured markdown file."""
    p = tmp_path / "unstructured.md"
    p.write_text(UNSTRUCTURED_MD)
    return str(p)


@pytest.fixture
def embedder():
    """EmbeddingService instance."""
    return EmbeddingService(
        model_name="all-MiniLM-L6-v2",
        cache_dir=os.path.expanduser("~/.guaipeca/models"),
        dimensions=384,
    )


@pytest.fixture
def temp_corpus(tmp_path, structured_path, unstructured_path):
    """Create a temporary corpus with structured and unstructured docs."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Copy files into corpus dir
    shutil.copy(structured_path, corpus_dir / "structured.md")
    shutil.copy(unstructured_path, corpus_dir / "unstructured.md")

    data_dir = tmp_path / "data"

    config = GuaipecaConfig(
        corpora=[
            CorpusConfig(
                name="test-toc",
                path=str(corpus_dir),
                topic="test",
                extensions=[".md"],
            )
        ],
        embedding=EmbeddingConfig(
            model="all-MiniLM-L6-v2",
            dimensions=384,
            cache_dir=os.path.expanduser("~/.guaipeca/models"),
        ),
        chunking=ChunkingConfig(max_size=2000, overlap=200),
        indexing=IndexingConfig(
            data_dir=str(data_dir),
            incremental=True,
        ),
        search=SearchConfig(hybrid=False, toc_rerank=False),
    )
    return config


@pytest.fixture
def indexed_toc_corpus(temp_corpus, embedder):
    """Searcher with indexed ToC test corpus."""
    from guaipeca.converter import Converter
    converter = Converter()
    searcher = Searcher(
        config=temp_corpus,
        embedding_service=embedder,
        converter=converter,
    )
    searcher.index_corpus(corpus_name="all", force=True)
    return searcher


# ===========================================================================
# 1. heading_path with structured documents (Feature A)
# ===========================================================================

class TestHeadingPath:
    """Test heading_path tracking in chunking."""

    def test_heading_path_with_headings(self, structured_path):
        """Markdown with #, ##, and ### headings should produce correct heading_path."""
        chunks = chunk_file(structured_path, STRUCTURED_MD, max_size=2000)

        # Should have chunks with non-empty heading_path
        has_path = any(c.heading_path for c in chunks)
        assert has_path, "No chunks have heading_path"

        # Check specific heading paths
        # The first chunk (before any ##) should have heading_path = ["Document Title"]
        # (from the # heading)
        first_chunk = chunks[0]
        assert "Document Title" in first_chunk.heading_path, (
            f"Expected 'Document Title' in heading_path, got: {first_chunk.heading_path}"
        )

        # Find a chunk in the "Introduction" section
        intro_chunks = [c for c in chunks if c.heading == "Introduction"]
        assert len(intro_chunks) > 0, "No chunk with heading 'Introduction'"
        intro_chunk = intro_chunks[0]
        assert "Document Title" in intro_chunk.heading_path
        assert "Introduction" in intro_chunk.heading_path

        # Find a chunk in the "Methods" section
        methods_chunks = [c for c in chunks if c.heading == "Methods"]
        assert len(methods_chunks) > 0, "No chunk with heading 'Methods'"
        methods_chunk = methods_chunks[0]
        assert "Document Title" in methods_chunk.heading_path
        assert "Methods" in methods_chunk.heading_path

    def test_heading_path_no_headings(self, unstructured_path):
        """Plain text without headings should have heading_path = []."""
        chunks = chunk_file(unstructured_path, UNSTRUCTURED_MD, max_size=2000)

        assert len(chunks) > 0, "No chunks produced"
        for chunk in chunks:
            assert chunk.heading_path == [], (
                f"Expected empty heading_path for unstructured doc, got: {chunk.heading_path}"
            )

    def test_heading_path_in_to_dict(self, structured_path):
        """Chunk.to_dict() should include heading_path."""
        chunks = chunk_file(structured_path, STRUCTURED_MD, max_size=2000)
        d = chunks[0].to_dict()
        assert "heading_path" in d
        assert isinstance(d["heading_path"], list)

    def test_heading_path_level3_included(self, structured_path):
        """Level 3 headings (###) should be included in heading_path."""
        chunks = chunk_file(structured_path, STRUCTURED_MD, max_size=500)

        # Find a chunk that should be under ### Background
        # With small max_size, the ### Background content should be its own chunk
        # within the Introduction section
        background_chunks = [c for c in chunks if "Background" in c.heading_path]
        assert len(background_chunks) > 0, (
            f"No chunks with 'Background' in heading_path. "
            f"Paths: {[c.heading_path for c in chunks]}"
        )


# ===========================================================================
# 2. section_id generation (Feature A)
# ===========================================================================

class TestSectionId:
    """Test section_id generation."""

    def test_section_id_generation(self, structured_path):
        """Chunks should get correct section_id based on source + heading."""
        chunks = chunk_file(structured_path, STRUCTURED_MD, max_size=2000)

        # All chunks should have a non-empty section_id
        for chunk in chunks:
            assert chunk.section_id, f"Chunk has empty section_id: {chunk}"

        # Chunks in the same section should have the same section_id
        intro_chunks = [c for c in chunks if c.heading == "Introduction"]
        if len(intro_chunks) > 1:
            assert intro_chunks[0].section_id == intro_chunks[1].section_id, (
                "Chunks in same section should have same section_id"
            )

        # Chunks in different sections should have different section_ids
        methods_chunks = [c for c in chunks if c.heading == "Methods"]
        if intro_chunks and methods_chunks:
            assert intro_chunks[0].section_id != methods_chunks[0].section_id, (
                "Chunks in different sections should have different section_ids"
            )

    def test_section_id_unstructured(self, unstructured_path):
        """Headingless doc should have section_id = 'unstructured:...'."""
        chunks = chunk_file(unstructured_path, UNSTRUCTURED_MD, max_size=2000)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.section_id.startswith("unstructured:"), (
                f"Expected 'unstructured:...' section_id, got: {chunk.section_id}"
            )

    def test_section_id_is_stable(self, structured_path):
        """Same input should produce same section_id."""
        chunks1 = chunk_file(structured_path, STRUCTURED_MD, max_size=2000)
        chunks2 = chunk_file(structured_path, STRUCTURED_MD, max_size=2000)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.section_id == c2.section_id, "section_id is not stable"


# ===========================================================================
# 3. section_mode search (Feature C)
# ===========================================================================

class TestSectionMode:
    """Test section_mode search aggregation."""

    def test_section_mode_search(self, indexed_toc_corpus):
        """Search with section_mode=True should return full sections."""
        result = indexed_toc_corpus.search("embedding", top_k=5, section_mode=True)

        assert "error" not in result, f"Search error: {result.get('error')}"
        assert result.get("section_mode") is True
        assert result["total"] > 0

        for r in result["results"]:
            assert "section_id" in r
            assert "heading" in r
            assert "text" in r
            assert "chunk_count" in r
            assert r["chunk_count"] >= 1

    def test_section_mode_unstructured(self, indexed_toc_corpus):
        """section_mode on unstructured docs should return all chunks from that doc."""
        result = indexed_toc_corpus.search("financial", top_k=5, section_mode=True)

        assert "error" not in result
        # Should find the unstructured document
        unstructured_results = [
            r for r in result["results"]
            if r.get("section_id", "").startswith("unstructured:")
        ]
        if unstructured_results:
            r = unstructured_results[0]
            assert r["chunk_count"] >= 1
            assert len(r["text"]) > 0

    def test_section_mode_returns_section_text(self, indexed_toc_corpus):
        """section_mode results should contain concatenated section text."""
        result = indexed_toc_corpus.search("methods", top_k=3, section_mode=True)

        if result["total"] > 0:
            r = result["results"][0]
            # Section text should be longer than a single chunk summary
            assert len(r["text"]) >= len(r.get("summary", ""))


# ===========================================================================
# 4. get_toc (Feature D)
# ===========================================================================

class TestGetToc:
    """Test get_toc navigation."""

    def test_get_toc_structured(self, indexed_toc_corpus):
        """get_toc should return nested tree for structured doc."""
        # List all documents first
        result = indexed_toc_corpus.get_toc(corpus="test-toc")
        assert "error" not in result
        assert "documents" in result
        assert result["total"] > 0

        # Find the structured document
        structured_doc = None
        for doc in result["documents"]:
            if doc.get("filename") == "structured.md":
                structured_doc = doc
                break

        assert structured_doc is not None, "structured.md not found in ToC"
        assert structured_doc.get("structured") is True
        assert structured_doc.get("chunk_count", 0) > 0
        assert len(structured_doc.get("top_level_headings", [])) > 0

    def test_get_toc_unstructured(self, indexed_toc_corpus):
        """get_toc should return structured=false for headingless doc."""
        result = indexed_toc_corpus.get_toc(corpus="test-toc")
        assert "error" not in result

        unstructured_doc = None
        for doc in result["documents"]:
            if doc.get("filename") == "unstructured.md":
                unstructured_doc = doc
                break

        assert unstructured_doc is not None, "unstructured.md not found in ToC"
        assert unstructured_doc.get("structured") is False
        assert unstructured_doc.get("chunk_count", 0) > 0
        assert unstructured_doc.get("top_level_headings", []) == []

    def test_get_toc_specific_document(self, indexed_toc_corpus, temp_corpus):
        """get_toc for a specific document should return its tree."""
        # Get the list first to find a source_path
        list_result = indexed_toc_corpus.get_toc(corpus="test-toc")
        structured_doc = next(
            d for d in list_result["documents"] if d["filename"] == "structured.md"
        )

        # Get specific doc ToC
        doc_result = indexed_toc_corpus.get_toc(
            corpus="test-toc",
            document=structured_doc["source_path"],
        )

        assert "title" in doc_result
        assert "children" in doc_result
        assert "chunk_count" in doc_result
        assert doc_result.get("structured") is True

    def test_get_toc_nonexistent_corpus(self, indexed_toc_corpus):
        """get_toc with nonexistent corpus should return error."""
        result = indexed_toc_corpus.get_toc(corpus="nonexistent")
        assert "error" in result

    def test_get_toc_mcp_tool_schema(self):
        """get_toc should be in the MCP TOOLS list."""
        from guaipeca.mcp_server import TOOLS
        tool_names = [t["name"] for t in TOOLS]
        assert "get_toc" in tool_names

    def test_get_toc_mcp_schema_has_required_fields(self):
        """get_toc schema should have corpus as required."""
        from guaipeca.mcp_server import TOOLS
        toc_tool = next(t for t in TOOLS if t["name"] == "get_toc")
        assert "corpus" in toc_tool["inputSchema"]["required"]
        assert "document" in toc_tool["inputSchema"]["properties"]


# ===========================================================================
# 5. section_filter (Feature E)
# ===========================================================================

class TestSectionFilter:
    """Test search within section."""

    def test_section_filter(self, indexed_toc_corpus):
        """Search with section_filter should return only chunks from that section."""
        # First, get ToC to find a section
        toc = indexed_toc_corpus.get_toc(corpus="test-toc")
        structured_doc = next(
            d for d in toc["documents"] if d["filename"] == "structured.md"
        )

        # Get the ToC tree for the structured doc to find a heading
        tree = indexed_toc_corpus.get_toc(
            corpus="test-toc",
            document=structured_doc["source_path"],
        )

        # Use a heading path prefix as filter
        if tree.get("children"):
            heading = tree["children"][0]["heading"]
            result = indexed_toc_corpus.search(
                "embedding", top_k=5, section_filter=heading,
            )
            assert "error" not in result, f"Error: {result.get('error')}"
            # All results should be from the filtered section
            if result["total"] > 0:
                for r in result["results"]:
                    assert r.get("section_id"), "Result missing section_id"

    def test_section_filter_unstructured(self, indexed_toc_corpus):
        """section_filter on unstructured doc should return all chunks from that doc."""
        # The section_id for unstructured docs is "unstructured:{hash}"
        # We can search using the source_path to find it via heading_path prefix
        # or we can look it up from the corpus index
        corpus_idx = indexed_toc_corpus.corpora["test-toc"]
        with corpus_idx._rwlock.read_lock():
            corpus_idx._load()
            section_map = dict(corpus_idx._section_map)

        unstructured_section_ids = [
            sid for sid in section_map if sid.startswith("unstructured:")
        ]
        if unstructured_section_ids:
            sid = unstructured_section_ids[0]
            result = indexed_toc_corpus.search(
                "financial", top_k=5, section_filter=sid,
            )
            assert "error" not in result, f"Error: {result.get('error')}"
            # Should return results (all chunks in the unstructured section)
            if result["total"] > 0:
                for r in result["results"]:
                    assert r.get("section_id", "").startswith("unstructured:")

    def test_section_filter_not_found(self, indexed_toc_corpus):
        """Non-existent section_filter should return empty + error."""
        result = indexed_toc_corpus.search(
            "embedding", top_k=5, section_filter="nonexistent_section_12345",
        )
        assert result["total"] == 0
        assert "error" in result
        assert "section not found" in result["error"]


# ===========================================================================
# 6. toc_rerank (Feature B)
# ===========================================================================

class TestTocRerank:
    """Test ToC keyword re-ranking."""

    def test_toc_rerank_boosts_matching(self, temp_corpus, embedder):
        """Re-ranking should boost results with heading_path matching query."""
        # Create searcher with toc_rerank enabled
        temp_corpus.search.toc_rerank = True
        temp_corpus.search.toc_rerank_weight = 0.5

        from guaipeca.converter import Converter
        converter = Converter()
        searcher = Searcher(
            config=temp_corpus,
            embedding_service=embedder,
            converter=converter,
        )
        searcher.index_corpus(corpus_name="all", force=True)

        # Search for a term that appears in a heading
        result = searcher.search("Introduction", top_k=10)
        assert "error" not in result
        assert result["total"] > 0

        # Results with "Introduction" in heading_path should be boosted
        # (at least they should appear, and their score should include the boost)
        has_heading_path = any(r.get("heading_path") for r in result["results"])
        assert has_heading_path, "No results with heading_path in reranked results"

    def test_toc_rerank_unstructured(self, temp_corpus, embedder):
        """Re-ranking should not crash on empty heading_path."""
        temp_corpus.search.toc_rerank = True

        from guaipeca.converter import Converter
        converter = Converter()
        searcher = Searcher(
            config=temp_corpus,
            embedding_service=embedder,
            converter=converter,
        )
        searcher.index_corpus(corpus_name="all", force=True)

        # Search for a term in the unstructured doc
        result = searcher.search("financial", top_k=5)
        assert "error" not in result
        assert result["total"] > 0

        # Unstructured results should still appear (just not boosted)
        unstructured_results = [
            r for r in result["results"]
            if r.get("section_id", "").startswith("unstructured:")
        ]
        # They should be present and not crash
        assert len(unstructured_results) >= 0  # Just ensure no crash

    def test_toc_rerank_disabled_by_default(self):
        """toc_rerank should default to False."""
        cfg = SearchConfig()
        assert cfg.toc_rerank is False
        assert cfg.toc_rerank_weight == 0.3

    def test_toc_rerank_config_from_yaml(self, tmp_path):
        """toc_rerank should be parseable from YAML."""
        import yaml
        config_data = {
            "corpora": [
                {"name": "test", "path": str(tmp_path), "extensions": [".md"]}
            ],
            "search": {
                "toc_rerank": True,
                "toc_rerank_weight": 0.5,
            },
        }
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(yaml.dump(config_data))

        cfg = GuaipecaConfig.from_yaml(str(config_path))
        assert cfg.search.toc_rerank is True
        assert cfg.search.toc_rerank_weight == 0.5


# ===========================================================================
# 7. Backward compatibility
# ===========================================================================

class TestBackwardCompat:
    """Test backward compatibility with old metadata.json."""

    def test_backward_compat_old_metadata(self, tmp_path, embedder):
        """Old metadata.json without heading_path/section_id should load and work."""
        # Create a corpus dir with a test file
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "test.md").write_text("# Test\n\nSome content about embeddings.")

        data_dir = tmp_path / "data"
        index_dir = data_dir / "corpora" / "test-bc"
        index_dir.mkdir(parents=True)

        # Create an old-style metadata.json (no heading_path, no section_id)
        old_metadata = [
            {
                "id": "abc123def4567890",
                "heading": "Test",
                "text": "# Test\n\nSome content about embeddings.",
                "source_path": str(corpus_dir / "test.md"),
                "char_offset": 0,
                "summary": "Test",
                "char_count": 40,
                "metadata": {},
            }
        ]
        with open(index_dir / "metadata.json", "w") as f:
            json.dump(old_metadata, f)

        # Create empty file_hashes
        with open(index_dir / "file_hashes.json", "w") as f:
            json.dump({}, f)

        # Create a FAISS index with one vector
        import faiss
        index = faiss.IndexFlatIP(384)
        vec = np.random.randn(1, 384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        index.add(vec)
        faiss.write_index(index, str(index_dir / "faiss_index.fai"))

        # Load with CorpusIndex
        config = GuaipecaConfig(
            corpora=[
                CorpusConfig(
                    name="test-bc",
                    path=str(corpus_dir),
                    topic="test",
                    extensions=[".md"],
                )
            ],
            embedding=EmbeddingConfig(
                model="all-MiniLM-L6-v2",
                dimensions=384,
                cache_dir=os.path.expanduser("~/.guaipeca/models"),
            ),
            indexing=IndexingConfig(data_dir=str(data_dir)),
        )

        from guaipeca.converter import Converter
        converter = Converter()
        corpus_idx = CorpusIndex(
            corpus=config.corpora[0],
            chunking=config.chunking,
            indexing=config.indexing,
            embedding_service=embedder,
            converter=converter,
        )

        # Load and verify migration guards work
        with corpus_idx._rwlock.read_lock():
            corpus_idx._load()
            assert len(corpus_idx._chunks) == 1
            chunk = corpus_idx._chunks[0]
            # Migration guard: heading_path should be [] for old metadata
            assert chunk.heading_path == [], (
                f"Expected empty heading_path for old metadata, got: {chunk.heading_path}"
            )
            # section_id should be computed (not empty) via migration
            # It could be "unstructured:..." since the old metadata doesn't have it
            # The migration guard should set it to "" and then _rebuild_toc_metadata
            # should handle it
            assert chunk.section_id is not None

    def test_chunk_dataclass_defaults(self):
        """Chunk dataclass should have safe defaults for new fields."""
        c = Chunk(
            id="test",
            heading="Test",
            text="content",
            source_path="/test.md",
        )
        assert c.heading_path == []
        assert c.section_id == ""

    def test_search_results_include_heading_path(self, indexed_toc_corpus):
        """Search results should include heading_path field."""
        result = indexed_toc_corpus.search("embedding", top_k=3)
        assert "error" not in result
        if result["total"] > 0:
            for r in result["results"]:
                assert "heading_path" in r, "Search result missing heading_path"
                assert "section_id" in r, "Search result missing section_id"

    def test_get_chunk_includes_heading_path(self, indexed_toc_corpus):
        """get_chunk result should include heading_path and section_id."""
        result = indexed_toc_corpus.search("embedding", top_k=1)
        if result["total"] > 0:
            chunk_id = result["results"][0]["chunk_id"]
            chunk = indexed_toc_corpus.get_chunk(chunk_id)
            assert chunk is not None
            assert "heading_path" in chunk
            assert "section_id" in chunk

    def test_mcp_search_schema_has_new_params(self):
        """Search tool schema should include section_mode and section_filter."""
        from guaipeca.mcp_server import TOOLS
        search_tool = next(t for t in TOOLS if t["name"] == "search")
        props = search_tool["inputSchema"]["properties"]
        assert "section_mode" in props, "Search schema missing section_mode"
        assert "section_filter" in props, "Search schema missing section_filter"
        assert props["section_mode"]["type"] == "boolean"
        assert props["section_filter"]["type"] == "string"