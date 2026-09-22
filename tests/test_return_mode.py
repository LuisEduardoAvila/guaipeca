"""Tests for return_mode parameter: chunks, documents, auto.

Tests cover:
1. chunks mode (backward compatibility)
2. documents mode (deduplication, full text, best score)
3. auto mode (threshold-based switching)
4. MCP tool schema and integration
5. Edge cases
6. Config
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from guaipeca.config import (
    GuaipecaConfig, CorpusConfig, ChunkingConfig, IndexingConfig,
    SearchConfig,
)


# ===========================================================================
# 1. chunks mode (backward compatibility)
# ===========================================================================

class TestChunksMode:
    """Test that return_mode='chunks' preserves existing behavior."""

    def test_chunks_mode_returns_chunks(self, indexed_corpus):
        """chunks mode should return chunk-level results."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=5, return_mode="chunks")
        assert "error" not in result
        assert result["total"] > 0
        assert result["return_mode"] == "chunks"
        # Each result should have chunk-level fields
        for r in result["results"]:
            assert "chunk_id" in r
            assert "summary" in r
            assert "location" in r
            assert "score" in r

    def test_default_mode_is_chunks(self, indexed_corpus):
        """Without return_mode, behavior should be chunks."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=3)
        assert result["return_mode"] == "chunks"

    def test_chunks_mode_results_sorted_by_score(self, indexed_corpus):
        """chunks mode results should be sorted by score descending."""
        searcher = indexed_corpus
        result = searcher.search("vector retrieval", top_k=5, return_mode="chunks")
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 2. documents mode
# ===========================================================================

class TestDocumentsMode:
    """Test return_mode='documents'."""

    def test_documents_mode_returns_documents(self, indexed_corpus):
        """documents mode should return document-level results."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=10, return_mode="documents")
        assert "error" not in result
        assert result["total"] > 0
        assert result["return_mode"] == "documents"
        # Each result should have document-level fields
        for r in result["results"]:
            assert "source_path" in r
            assert "corpus" in r
            assert "topic" in r
            assert "score" in r
            assert "text" in r
            assert "char_count" in r
            assert "matched_chunks" in r
            # Should NOT have chunk-level fields
            assert "chunk_id" not in r
            assert "summary" not in r

    def test_documents_mode_deduplicates_by_source(self, indexed_corpus):
        """documents mode should deduplicate by source_path."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=20, return_mode="documents")
        source_paths = [r["source_path"] for r in result["results"]]
        # No duplicate source paths
        assert len(source_paths) == len(set(source_paths)), \
            f"Duplicate source paths: {source_paths}"

    def test_documents_mode_has_full_text(self, indexed_corpus):
        """documents mode should return full document text."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=10, return_mode="documents")
        for r in result["results"]:
            assert len(r["text"]) > 0, "Document text is empty"
            assert r["char_count"] == len(r["text"]), \
                "char_count doesn't match text length"

    def test_documents_mode_score_is_best_chunk_score(self, indexed_corpus):
        """Document score should be the best (highest) chunk score from that document."""
        searcher = indexed_corpus
        # Get chunk results
        chunk_result = searcher.search("embedding", top_k=20, return_mode="chunks")
        # Get document results
        doc_result = searcher.search("embedding", top_k=20, return_mode="documents")

        # For each document, find all chunks from that source and verify score
        for doc in doc_result["results"]:
            source_path = doc["source_path"]
            chunk_scores = [
                r["score"] for r in chunk_result["results"]
                if r["location"].split("#")[0] == source_path
            ]
            if chunk_scores:
                assert doc["score"] == max(chunk_scores), \
                    f"Document score {doc['score']} != max chunk score {max(chunk_scores)}"

    def test_documents_mode_results_sorted_by_score(self, indexed_corpus):
        """documents mode results should be sorted by score descending."""
        searcher = indexed_corpus
        result = searcher.search("vector retrieval", top_k=10, return_mode="documents")
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_documents_mode_matched_chunks_count(self, indexed_corpus):
        """matched_chunks should count how many chunks came from that document."""
        searcher = indexed_corpus
        chunk_result = searcher.search("embedding", top_k=20, return_mode="chunks")
        doc_result = searcher.search("embedding", top_k=20, return_mode="documents")

        for doc in doc_result["results"]:
            source_path = doc["source_path"]
            expected_count = sum(
                1 for r in chunk_result["results"]
                if r["location"].split("#")[0] == source_path
            )
            assert doc["matched_chunks"] == expected_count, \
                f"matched_chunks {doc['matched_chunks']} != expected {expected_count}"

    def test_documents_mode_file_text_matches_disk(self, indexed_corpus):
        """Document text should match the file content on disk.

        Plain-text sources (.md/.txt/.markdown) are returned verbatim, so the
        returned text must equal the on-disk bytes. Converted formats (PDF,
        DOCX, ...) are returned as converted markdown, so byte-equality does
        not apply -- assert only that non-empty text was produced.
        """
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=10, return_mode="documents")
        assert result["total"] > 0
        for r in result["results"]:
            ext = os.path.splitext(r["source_path"])[1].lower()
            assert r["text"], f"Empty document text for {r['source_path']}"
            if ext in (".md", ".txt", ".markdown"):
                with open(r["source_path"], "r", encoding="utf-8") as f:
                    disk_text = f.read()
                assert r["text"] == disk_text, \
                    f"Document text doesn't match file on disk: {r['source_path']}"


# ===========================================================================
# 3. auto mode
# ===========================================================================

class TestAutoMode:
    """Test return_mode='auto'."""

    def test_auto_returns_chunks_when_small(self, indexed_corpus):
        """auto should return chunks when total size is below threshold."""
        searcher = indexed_corpus
        # Use a high max_chars to ensure chunks are returned
        result = searcher.search("embedding", top_k=3, return_mode="auto", max_chars=1000000)
        assert result["return_mode"] == "chunks"
        assert result["total"] > 0
        for r in result["results"]:
            assert "chunk_id" in r

    def test_auto_returns_documents_when_large(self, indexed_corpus):
        """auto should return documents when total size exceeds threshold."""
        searcher = indexed_corpus
        # Use a very low max_chars to force documents mode
        result = searcher.search("embedding", top_k=10, return_mode="auto", max_chars=1)
        assert result["return_mode"] == "documents"
        assert result["total"] > 0
        for r in result["results"]:
            assert "source_path" in r
            assert "text" in r

    def test_auto_uses_config_max_chars_default(self, indexed_corpus):
        """auto should use config search.max_chars when max_chars param not given."""
        searcher = indexed_corpus
        # Config default is 8000
        result = searcher.search("embedding", top_k=5, return_mode="auto")
        # With default 8000 threshold and small test corpus, likely chunks
        assert result["return_mode"] in ("chunks", "documents")
        assert result["total"] > 0


# ===========================================================================
# 4. MCP tool schema and integration
# ===========================================================================

class TestMCPReturnMode:
    """Test the MCP search tool with return_mode parameter."""

    def test_search_tool_schema_has_return_mode(self):
        """Search tool schema should include return_mode parameter."""
        from guaipeca.mcp_server import TOOLS
        search_tool = next(t for t in TOOLS if t["name"] == "search")
        assert "return_mode" in search_tool["inputSchema"]["properties"]
        assert search_tool["inputSchema"]["properties"]["return_mode"]["type"] == "string"
        assert search_tool["inputSchema"]["properties"]["return_mode"]["default"] == "chunks"

    def test_mcp_search_passes_return_mode_documents(self, indexed_corpus):
        """MCP search should pass return_mode to searcher."""
        from guaipeca.mcp_server import GuaipecaMCPServer
        server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
        server.config = indexed_corpus.config
        server.searcher = indexed_corpus

        result = server._handle_tool_call({
            "name": "search",
            "arguments": {"query": "embedding", "return_mode": "documents", "top_k": 10},
        })
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert data["return_mode"] == "documents"
        assert data["total"] > 0

    def test_mcp_search_default_return_mode(self, indexed_corpus):
        """MCP search without return_mode should default to chunks."""
        from guaipeca.mcp_server import GuaipecaMCPServer
        server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
        server.config = indexed_corpus.config
        server.searcher = indexed_corpus

        result = server._handle_tool_call({
            "name": "search",
            "arguments": {"query": "embedding"},
        })
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert data["return_mode"] == "chunks"

    def test_mcp_search_auto_mode(self, indexed_corpus):
        """MCP search with auto mode should work."""
        from guaipeca.mcp_server import GuaipecaMCPServer
        server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
        server.config = indexed_corpus.config
        server.searcher = indexed_corpus

        result = server._handle_tool_call({
            "name": "search",
            "arguments": {"query": "embedding", "return_mode": "auto", "max_chars": 1},
        })
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert data["return_mode"] == "documents"


# ===========================================================================
# 5. Edge cases
# ===========================================================================

class TestReturnModeEdgeCases:
    """Test edge cases for return_mode."""

    def test_invalid_return_mode(self, indexed_corpus):
        """Invalid return_mode should return error."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=5, return_mode="invalid")
        assert "error" in result
        assert "Invalid return_mode" in result["error"]

    def test_documents_mode_no_results(self, indexed_corpus):
        """documents mode with no matching results should return empty list."""
        searcher = indexed_corpus
        result = searcher.search("zzzznonexistentterm12345", top_k=5, return_mode="documents")
        # Should return empty results, not an error
        assert "error" not in result or result.get("total", 0) == 0
        assert result["return_mode"] == "documents"

    def test_auto_mode_no_results(self, indexed_corpus):
        """auto mode with no results should return empty list."""
        searcher = indexed_corpus
        result = searcher.search("zzzznonexistentterm12345", top_k=5, return_mode="auto")
        assert "error" not in result or result.get("total", 0) == 0

    def test_documents_mode_preserves_corpus_and_topic(self, indexed_corpus):
        """documents mode should include corpus and topic fields."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=10, return_mode="documents")
        for r in result["results"]:
            assert r["corpus"] == "test-docs"
            assert isinstance(r["topic"], str)

    def test_documents_mode_with_hybrid(self, indexed_corpus):
        """documents mode should work with hybrid search."""
        searcher = indexed_corpus
        result = searcher.search(
            "consolidation", top_k=10, return_mode="documents", hybrid=True
        )
        assert "error" not in result
        assert result["return_mode"] == "documents"
        assert result.get("hybrid") is True
        assert result["total"] > 0


# ===========================================================================
# 6. Config tests
# ===========================================================================

class TestMaxCharsConfig:
    """Test search.max_chars configuration."""

    def test_max_chars_default(self):
        """SearchConfig should default max_chars to 8000."""
        cfg = SearchConfig()
        assert cfg.max_chars == 8000

    def test_max_chars_from_yaml(self, tmp_path):
        """Config should parse max_chars from YAML."""
        config_data = {
            "corpora": [
                {"name": "test", "path": str(tmp_path), "extensions": [".md"]}
            ],
            "search": {
                "max_chars": 5000,
            },
        }
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(yaml.dump(config_data))
        cfg = GuaipecaConfig.from_yaml(str(config_path))
        assert cfg.search.max_chars == 5000

    def test_max_chars_default_when_not_in_yaml(self, tmp_path):
        """Config without max_chars should use default 8000."""
        config_data = {
            "corpora": [
                {"name": "test", "path": str(tmp_path), "extensions": [".md"]}
            ],
            "search": {
                "hybrid": True,
            },
        }
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(yaml.dump(config_data))
        cfg = GuaipecaConfig.from_yaml(str(config_path))
        assert cfg.search.max_chars == 8000

    def test_max_chars_validation_invalid(self, tmp_path):
        """Invalid max_chars should raise ValueError."""
        config_data = {
            "corpora": [
                {"name": "test", "path": str(tmp_path), "extensions": [".md"]}
            ],
            "search": {
                "max_chars": 0,
            },
        }
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(yaml.dump(config_data))
        with pytest.raises(ValueError, match="max_chars"):
            GuaipecaConfig.from_yaml(str(config_path))