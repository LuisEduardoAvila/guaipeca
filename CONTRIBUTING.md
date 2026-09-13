# Contributing to Guaipeca

Thanks for your interest in contributing! Guaipeca is a small project, so the process is lightweight.

## Development Setup

```bash
# Clone
git clone https://github.com/LuisEduardoAvila/guaipeca.git
cd guaipeca

# Install with dev dependencies
pip install -e ".[all,dev]"
```

### System Dependencies (ARM64 / Pi 5)

```bash
sudo apt install libxml2-dev libxslt-dev libffi-dev libjpeg-dev build-essential python3-dev
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run integration tests only
pytest tests/test_integration.py -v --timeout=120

# Run a single test
pytest tests/test_integration.py::TestClassName::test_method -v
```

Tests use a sample PDF fixture (`tests/fixtures/test-doc.pdf`) with content about RAG concepts (embeddings, retrieval, chunking, vector stores).

## Code Style

- **Linter:** [ruff](https://github.com/astral-sh/ruff)
- **Line length:** 100 characters
- **Target:** Python 3.10+

```bash
# Check
ruff check src/ tests/

# Fix
ruff check --fix src/ tests/

# Format
ruff format src/ tests/
```

## Project Structure

```
guaipeca/
├── src/guaipeca/
│   ├── __init__.py
│   ├── cli.py            # CLI entry point
│   ├── config.py         # YAML config loader
│   ├── indexer.py        # Document indexing pipeline
│   ├── searcher.py       # FAISS search + hybrid BM25
│   ├── chunker.py        # Structure-aware chunking
│   ├── pdf_reader.py     # pymupdf PDF extraction
│   ├── embeddings.py     # fastembed wrapper + LRU cache
│   ├── mcp_server.py     # MCP tool definitions
│   └── http_server.py    # HTTP/SSE transport
├── tests/
│   ├── fixtures/
│   │   ├── test-doc.pdf      # Sample PDF for integration tests
│   │   └── test-config.yaml  # Test configuration
│   ├── test_integration.py
│   └── test_*.py
├── config/
│   └── guaipeca.yaml         # Default config template
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Reporting Issues

1. Check existing issues to avoid duplicates
2. Include:
   - Python version and OS
   - Steps to reproduce
   - Expected vs actual behavior
   - Relevant logs (remove personal paths)

## Pull Requests

1. Fork the repo and create a feature branch
2. Run `ruff check src/ tests/` and `pytest tests/ -v` before submitting
3. Keep changes focused — one feature/fix per PR
4. Write clear commit messages

## License

MIT — see [LICENSE](LICENSE)