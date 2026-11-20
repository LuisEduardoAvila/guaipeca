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
- **Target:** Python 3.12+ (3.13 recommended — this is what the Docker image and the Pi deployment run)

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

The loop, for maintainers and outside contributors alike:

1. **One issue per branch, short-lived.** Branch off `main` as
   `fix/<issue>-<slug>` or `feat/<issue>-<slug>`. Days, not weeks — if a branch
   outlives a week, the issue was too large and should be split.
2. **Test first when the change fixes a bug.** Show the test failing on the
   unfixed tree, then make it pass. A regression test that passes without the fix
   is worse than no test: it looks like coverage.
3. Run `ruff check src/ tests/` and `pytest tests/ -v` before submitting.
4. Open the PR with `Closes #N` **in the body or a commit message** — not in a
   comment. The merge then closes the issue; a comment does not.
5. CI must be green before merge. `main` requires the `Tests` check.
6. Squash-merge, and delete the branch.

Keep changes focused — one feature or fix per PR. Write commit messages that say
what a *user* experiences, not which file moved.

Direct pushes to `main` are rejected. That is deliberate: the two defects this
process exists to catch (a test that never ran its code, a CI guard on a trigger
that never fired) both passed human review and were caught by reading CI output
instead.

## Releases

Versions are cut from `main`, not from branches, and only when there is something
a user would want to know about:

| Change | Version |
|---|---|
| Bug fixes only | patch (`0.6.1`) |
| New behaviour, no breakage | minor (`0.7.0`) — the normal cadence |
| Config or wire-format break | major |

To cut one:

1. Move `## [Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] — YYYY-MM-DD`.
2. Bump `version` in `pyproject.toml` and `__version__` in `src/guaipeca/__init__.py`.
3. Merge, then tag `vX.Y.Z` at the merge commit. CI runs `tools/release_check.py`
   on `v*` tags and fails if the tag and the declared version disagree.
4. `gh release create vX.Y.Z --notes-file <the changelog section>`.

Container images are tagged `:latest`, `:<short-sha>` and `:<version>`, the last
of which is the one to pin.

An installed copy does not upgrade itself. Reporting a version is not the same as
running it: reinstall and `sudo systemctl restart guaipeca`, then check
`guaipeca --version` against `/health`.

## License

MIT — see [LICENSE](LICENSE)