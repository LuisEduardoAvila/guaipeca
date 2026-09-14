"""Tests for get_document, list_documents MCP tools and /download HTTP endpoint.

Tests cover:
1. get_document — text retrieval, metadata, error cases
2. list_documents — listing, corpus filter, metadata fields
3. /download HTTP endpoint — file serving, auth, path traversal protection
4. Search result document references — source_path and filename in results
"""

from __future__ import annotations

import json
import os
import tempfile
import shutil
from pathlib import Path

import pytest

from guaipeca.config import (
    GuaipecaConfig, CorpusConfig, ChunkingConfig, IndexingConfig,
    EmbeddingConfig, SearchConfig, ServerConfig, UploadConfig,
)
from guaipeca.search import Searcher
from guaipeca.index import CorpusIndex
from guaipeca.converter import Converter
from guaipeca.embedding import EmbeddingService


# ===========================================================================
# 1. get_document
# ===========================================================================

class TestGetDocument:
    """Test the get_document MCP tool."""

    def test_get_document_returns_text(self, indexed_corpus):
        """get_document should return converted text for an indexed file."""
        searcher = indexed_corpus
        # First list documents to get a source_path
        docs = searcher.list_documents(corpus="test-docs")
        assert docs["total"] > 0
        source_path = docs["documents"][0]["source_path"]

        result = searcher.get_document("test-docs", source_path)
        assert result is not None, f"Document not found: {source_path}"
        assert "text" in result
        assert len(result["text"]) > 0, "Document text is empty"
        assert "source_path" in result
        assert "filename" in result
        assert "corpus" in result
        assert "file_size" in result
        assert "chunk_count" in result
        assert "char_count" in result
        assert result["corpus"] == "test-docs"
        assert result["source_path"] == source_path

    def test_get_document_filename_matches(self, indexed_corpus):
        """get_document filename should match the basename of source_path."""
        searcher = indexed_corpus
        docs = searcher.list_documents(corpus="test-docs")
        source_path = docs["documents"][0]["source_path"]

        result = searcher.get_document("test-docs", source_path)
        assert result is not None
        assert result["filename"] == os.path.basename(source_path)

    def test_get_document_chunk_count_matches_list(self, indexed_corpus):
        """get_document chunk_count should match list_documents chunk count."""
        searcher = indexed_corpus
        docs = searcher.list_documents(corpus="test-docs")
        doc = docs["documents"][0]
        source_path = doc["source_path"]

        result = searcher.get_document("test-docs", source_path)
        assert result is not None
        assert result["chunk_count"] == doc["chunk_count"]

    def test_get_document_nonexistent_corpus(self, indexed_corpus):
        """get_document with nonexistent corpus should return None."""
        searcher = indexed_corpus
        result = searcher.get_document("nonexistent", "/some/path.md")
        assert result is None

    def test_get_document_nonexistent_file(self, indexed_corpus):
        """get_document with untracked source_path should return None."""
        searcher = indexed_corpus
        result = searcher.get_document("test-docs", "/nonexistent/file.md")
        assert result is None

    def test_get_document_via_mcp_tool(self, indexed_corpus):
        """MCP get_document tool should return text content."""
        from guaipeca.mcp_server import GuaipecaMCPServer
        server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
        server.config = indexed_corpus.config
        server.searcher = indexed_corpus

        docs = indexed_corpus.list_documents(corpus="test-docs")
        source_path = docs["documents"][0]["source_path"]

        result = server._handle_tool_call({
            "name": "get_document",
            "arguments": {"corpus": "test-docs", "source_path": source_path},
        })
        assert "isError" not in result
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert "text" in data
        assert data["corpus"] == "test-docs"

    def test_get_document_mcp_error_on_not_found(self, indexed_corpus):
        """MCP get_document tool should return error for missing document."""
        from guaipeca.mcp_server import GuaipecaMCPServer
        server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
        server.config = indexed_corpus.config
        server.searcher = indexed_corpus

        result = server._handle_tool_call({
            "name": "get_document",
            "arguments": {"corpus": "test-docs", "source_path": "/nonexistent/file.md"},
        })
        assert result.get("isError") is True
        assert "not found" in result["content"][0]["text"].lower()


# ===========================================================================
# 2. list_documents
# ===========================================================================

class TestListDocuments:
    """Test the list_documents MCP tool."""

    def test_list_documents_all_corpora(self, indexed_corpus):
        """list_documents without corpus should list all."""
        searcher = indexed_corpus
        result = searcher.list_documents()
        assert "error" not in result
        assert result["total"] > 0
        assert len(result["documents"]) == result["total"]

    def test_list_documents_specific_corpus(self, indexed_corpus):
        """list_documents with corpus should filter."""
        searcher = indexed_corpus
        result = searcher.list_documents(corpus="test-docs")
        assert "error" not in result
        assert result["total"] > 0
        for doc in result["documents"]:
            assert doc["corpus"] == "test-docs"

    def test_list_documents_nonexistent_corpus(self, indexed_corpus):
        """list_documents with nonexistent corpus should return error."""
        searcher = indexed_corpus
        result = searcher.list_documents(corpus="nonexistent")
        assert "error" in result
        assert "corpus not found" in result["error"]

    def test_list_documents_has_required_fields(self, indexed_corpus):
        """Each document entry should have required fields."""
        searcher = indexed_corpus
        result = searcher.list_documents(corpus="test-docs")
        for doc in result["documents"]:
            assert "source_path" in doc
            assert "filename" in doc
            assert "corpus" in doc
            assert "chunk_count" in doc
            assert "file_size" in doc
            assert "last_indexed" in doc

    def test_list_documents_filename_is_basename(self, indexed_corpus):
        """filename should be the basename of source_path."""
        searcher = indexed_corpus
        result = searcher.list_documents(corpus="test-docs")
        for doc in result["documents"]:
            assert doc["filename"] == os.path.basename(doc["source_path"])

    def test_list_documents_chunk_count_positive(self, indexed_corpus):
        """Indexed documents should have positive chunk counts."""
        searcher = indexed_corpus
        result = searcher.list_documents(corpus="test-docs")
        for doc in result["documents"]:
            assert doc["chunk_count"] > 0, f"Document {doc['filename']} has 0 chunks"

    def test_list_documents_file_size_positive(self, indexed_corpus):
        """Indexed documents should have positive file size."""
        searcher = indexed_corpus
        result = searcher.list_documents(corpus="test-docs")
        for doc in result["documents"]:
            assert doc["file_size"] > 0, f"Document {doc['filename']} has 0 size"

    def test_list_documents_via_mcp_tool(self, indexed_corpus):
        """MCP list_documents tool should return documents list."""
        from guaipeca.mcp_server import GuaipecaMCPServer
        server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
        server.config = indexed_corpus.config
        server.searcher = indexed_corpus

        result = server._handle_tool_call({
            "name": "list_documents",
            "arguments": {},
        })
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert data["total"] > 0
        assert "documents" in data

    def test_list_documents_mcp_with_corpus_filter(self, indexed_corpus):
        """MCP list_documents tool with corpus argument should filter."""
        from guaipeca.mcp_server import GuaipecaMCPServer
        server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
        server.config = indexed_corpus.config
        server.searcher = indexed_corpus

        result = server._handle_tool_call({
            "name": "list_documents",
            "arguments": {"corpus": "test-docs"},
        })
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert data["total"] > 0
        for doc in data["documents"]:
            assert doc["corpus"] == "test-docs"

    def test_list_documents_mcp_error_on_nonexistent_corpus(self, indexed_corpus):
        """MCP list_documents with nonexistent corpus should return error."""
        from guaipeca.mcp_server import GuaipecaMCPServer
        server = GuaipecaMCPServer.__new__(GuaipecaMCPServer)
        server.config = indexed_corpus.config
        server.searcher = indexed_corpus

        result = server._handle_tool_call({
            "name": "list_documents",
            "arguments": {"corpus": "nonexistent"},
        })
        assert result.get("isError") is True


# ===========================================================================
# 3. /download HTTP endpoint
# ===========================================================================

class TestDownloadEndpoint:
    """Test the /download/{corpus}/{filename} HTTP endpoint."""

    @pytest.fixture
    def http_server_setup(self, indexed_corpus):
        """Start the MCP HTTP server on a test port."""
        import http.server
        import threading
        import urllib.request

        server_instance = indexed_corpus
        config = indexed_corpus.config

        # Set auth token for testing
        config.server.auth_token = "test-token-12345"

        # Build the HTTP handler using the same logic as run_http
        # but on a random available port
        host = "127.0.0.1"
        port = 0  # OS assigns a free port

        auth_token = config.server.auth_token
        cors_origin = "null" if auth_token else "*"

        DownloadTestServer = type("DownloadTestServer", (), {})

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _check_auth(self):
                if not auth_token:
                    return True
                auth_header = self.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    import hmac as _hmac
                    return _hmac.compare_digest(auth_header[7:], auth_token)
                return False

            def _send_json(self, code, data):
                body = json.dumps(data).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)

                if parsed.path.startswith("/download/"):
                    if not self._check_auth():
                        self._send_json(401, {"error": "unauthorized"})
                        return

                    parts = parsed.path.split("/", 3)
                    if len(parts) < 4:
                        self._send_json(400, {"error": "path must be /download/{corpus}/{filename}"})
                        return

                    corpus_name = parts[2]
                    filename = urllib.parse.unquote(parts[3])

                    if ".." in filename or filename.startswith("/"):
                        self._send_json(400, {"error": "invalid filename"})
                        return
                    if ".." in corpus_name or corpus_name.startswith("/"):
                        self._send_json(400, {"error": "invalid corpus"})
                        return

                    corpus_cfg = config.get_corpus(corpus_name)
                    if corpus_cfg is None:
                        self._send_json(404, {"error": f"corpus not found: {corpus_name}"})
                        return

                    corpus_path = Path(corpus_cfg.path)
                    file_path = corpus_path / filename

                    try:
                        file_path.resolve().relative_to(corpus_path.resolve())
                    except ValueError:
                        self._send_json(400, {"error": "path outside corpus directory"})
                        return

                    if not file_path.exists() or not file_path.is_file():
                        self._send_json(404, {"error": f"file not found: {filename}"})
                        return

                    import mimetypes
                    content_type, _ = mimetypes.guess_type(str(file_path))
                    if content_type is None:
                        content_type = "application/octet-stream"

                    file_size = file_path.stat().st_size
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(file_size))
                    self.send_header(
                        "Content-Disposition",
                        f'attachment; filename="{filename}"'
                    )
                    self.end_headers()
                    with open(file_path, "rb") as f:
                        self.wfile.write(f.read())

                elif parsed.path == "/health":
                    self._send_json(200, {"status": "ok"})
                else:
                    self._send_json(404, {"error": "not found"})

            def log_message(self, *args):
                pass

        httpd = http.server.HTTPServer((host, 0), Handler)
        actual_port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        yield {
            "port": actual_port,
            "config": config,
            "server": server_instance,
        }

        httpd.shutdown()
        thread.join(timeout=5)

    def test_download_serves_file(self, http_server_setup):
        """Download endpoint should serve the original file."""
        import urllib.request
        port = http_server_setup["port"]
        # The test fixture has test-doc.md
        url = f"http://127.0.0.1:{port}/download/test-docs/test-doc.md"
        req = urllib.request.Request(url)
        req.add_header("Authorization", "Bearer test-token-12345")

        with urllib.request.urlopen(req) as response:
            assert response.status == 200
            content = response.read()
            assert len(content) > 0
            # Check Content-Disposition header
            cd = response.headers.get("Content-Disposition", "")
            assert "attachment" in cd
            assert "test-doc.md" in cd
            # Content-Type should be text/markdown or text/plain
            ct = response.headers.get("Content-Type", "")
            assert "text" in ct

    def test_download_requires_auth(self, http_server_setup):
        """Download endpoint should require auth token."""
        import urllib.request
        import urllib.error
        port = http_server_setup["port"]
        url = f"http://127.0.0.1:{port}/download/test-docs/test-doc.md"

        try:
            urllib.request.urlopen(url)
            assert False, "Should have raised 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401

    def test_download_wrong_token(self, http_server_setup):
        """Download endpoint should reject wrong auth token."""
        import urllib.request
        import urllib.error
        port = http_server_setup["port"]
        url = f"http://127.0.0.1:{port}/download/test-docs/test-doc.md"
        req = urllib.request.Request(url)
        req.add_header("Authorization", "Bearer wrong-token")

        try:
            urllib.request.urlopen(req)
            assert False, "Should have raised 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401

    def test_download_nonexistent_corpus(self, http_server_setup):
        """Download with nonexistent corpus should return 404."""
        import urllib.request
        import urllib.error
        port = http_server_setup["port"]
        url = f"http://127.0.0.1:{port}/download/nonexistent/file.md"
        req = urllib.request.Request(url)
        req.add_header("Authorization", "Bearer test-token-12345")

        try:
            urllib.request.urlopen(req)
            assert False, "Should have raised 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_download_nonexistent_file(self, http_server_setup):
        """Download with nonexistent file should return 404."""
        import urllib.request
        import urllib.error
        port = http_server_setup["port"]
        url = f"http://127.0.0.1:{port}/download/test-docs/nonexistent.md"
        req = urllib.request.Request(url)
        req.add_header("Authorization", "Bearer test-token-12345")

        try:
            urllib.request.urlopen(req)
            assert False, "Should have raised 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_download_path_traversal_filename(self, http_server_setup):
        """Download with .. in filename should be rejected."""
        import urllib.request
        import urllib.error
        port = http_server_setup["port"]
        url = f"http://127.0.0.1:{port}/download/test-docs/../../../etc/passwd"
        req = urllib.request.Request(url)
        req.add_header("Authorization", "Bearer test-token-12345")

        try:
            urllib.request.urlopen(req)
            assert False, "Should have raised 400"
        except urllib.error.HTTPError as e:
            assert e.code in (400, 404)

    def test_download_url_encoded_traversal(self, http_server_setup):
        """Download with URL-encoded .. should be rejected."""
        import urllib.request
        import urllib.error
        port = http_server_setup["port"]
        # %2e%2e = ..
        url = f"http://127.0.0.1:{port}/download/test-docs/%2e%2e/%2e%2e/etc/passwd"
        req = urllib.request.Request(url)
        req.add_header("Authorization", "Bearer test-token-12345")

        try:
            urllib.request.urlopen(req)
            assert False, "Should have been rejected"
        except urllib.error.HTTPError as e:
            # Should be 400 (invalid filename) or 404 (not found)
            assert e.code in (400, 404)

    def test_download_pdf_file(self, http_server_setup):
        """Download endpoint should serve PDF files with correct content type."""
        import urllib.request
        port = http_server_setup["port"]
        url = f"http://127.0.0.1:{port}/download/test-docs/test-doc.pdf"
        req = urllib.request.Request(url)
        req.add_header("Authorization", "Bearer test-token-12345")

        with urllib.request.urlopen(req) as response:
            assert response.status == 200
            content = response.read()
            assert len(content) > 0
            ct = response.headers.get("Content-Type", "")
            assert "pdf" in ct or "octet-stream" in ct
            cd = response.headers.get("Content-Disposition", "")
            assert "test-doc.pdf" in cd


# ===========================================================================
# 4. Search result document references
# ===========================================================================

class TestSearchResultReferences:
    """Test that search results include source_path and filename fields."""

    def test_chunks_mode_has_source_path(self, indexed_corpus):
        """chunks mode results should include source_path."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=5, return_mode="chunks")
        assert result["total"] > 0
        for r in result["results"]:
            assert "source_path" in r, "Chunk result missing source_path"
            assert r["source_path"], "source_path is empty"

    def test_chunks_mode_has_filename(self, indexed_corpus):
        """chunks mode results should include filename."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=5, return_mode="chunks")
        assert result["total"] > 0
        for r in result["results"]:
            assert "filename" in r, "Chunk result missing filename"
            assert r["filename"], "filename is empty"

    def test_chunks_mode_filename_is_basename(self, indexed_corpus):
        """filename should be basename of source_path."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=5, return_mode="chunks")
        for r in result["results"]:
            assert r["filename"] == os.path.basename(r["source_path"])

    def test_documents_mode_has_source_path(self, indexed_corpus):
        """documents mode results should include source_path."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=10, return_mode="documents")
        assert result["total"] > 0
        for r in result["results"]:
            assert "source_path" in r

    def test_documents_mode_has_filename(self, indexed_corpus):
        """documents mode results should include filename."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=10, return_mode="documents")
        assert result["total"] > 0
        for r in result["results"]:
            assert "filename" in r, "Document result missing filename"
            assert r["filename"] == os.path.basename(r["source_path"])

    def test_hybrid_results_have_source_path(self, indexed_corpus):
        """Hybrid search results should also include source_path and filename."""
        searcher = indexed_corpus
        result = searcher.search("embedding", top_k=5, hybrid=True, return_mode="chunks")
        assert result["total"] > 0
        for r in result["results"]:
            assert "source_path" in r, "Hybrid result missing source_path"
            assert "filename" in r, "Hybrid result missing filename"


# ===========================================================================
# 5. MCP tool definitions
# ===========================================================================

class TestMCPToolDefinitions:
    """Test that new tools are properly defined in the TOOLS list."""

    def test_get_document_tool_exists(self):
        """get_document tool should be in TOOLS."""
        from guaipeca.mcp_server import TOOLS
        names = [t["name"] for t in TOOLS]
        assert "get_document" in names

    def test_list_documents_tool_exists(self):
        """list_documents tool should be in TOOLS."""
        from guaipeca.mcp_server import TOOLS
        names = [t["name"] for t in TOOLS]
        assert "list_documents" in names

    def test_get_document_schema(self):
        """get_document tool should have correct schema."""
        from guaipeca.mcp_server import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "get_document")
        props = tool["inputSchema"]["properties"]
        assert "corpus" in props
        assert "source_path" in props
        required = tool["inputSchema"].get("required", [])
        assert "corpus" in required
        assert "source_path" in required

    def test_list_documents_schema(self):
        """list_documents tool should have correct schema."""
        from guaipeca.mcp_server import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "list_documents")
        props = tool["inputSchema"]["properties"]
        assert "corpus" in props
        # corpus is optional for list_documents
        required = tool["inputSchema"].get("required", [])
        assert "corpus" not in required

    def test_tool_count(self):
        """Should now have 8 tools (6 original + 2 new)."""
        from guaipeca.mcp_server import TOOLS
        assert len(TOOLS) == 8, f"Expected 8 tools, got {len(TOOLS)}: {[t['name'] for t in TOOLS]}"