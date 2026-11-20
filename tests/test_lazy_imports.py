"""The CLI/MCP entry points must not import the numerical stack (issue #32).

Reported as ``fastembed`` and ``bm25s`` being eager; both were already lazy.
The eager import was ``numpy``, reached through
``cli -> search -> index -> embedding``, which ``guaipeca.mcp_server`` also
pulls in. Measurements live in CHANGELOG.md and the PR, not here.

The leak assertion runs in a *fresh* interpreter: a warm in-process check
would see whatever an earlier test already imported.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
# One source of truth: the assertion intersects whatever is listed here.
HEAVY = ("numpy", "fastembed", "bm25s", "faiss", "onnxruntime")


def _imported_after_import(module: str) -> set[str]:
    """Heavy packages present in sys.modules after importing *module*, fresh process."""
    code = (
        f"import sys; import {module}; "
        f"print(','.join(n for n in {HEAVY!r} if n in sys.modules))"
    )
    env = dict(os.environ)
    src = str(PROJECT_ROOT / "src")
    inherited = env.get("PYTHONPATH")
    # Prepend, never replace: CI may rely on other PYTHONPATH entries.
    env["PYTHONPATH"] = f"{src}{os.pathsep}{inherited}" if inherited else src
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return {name for name in result.stdout.strip().split(",") if name}


@pytest.mark.parametrize("module", ["guaipeca.cli", "guaipeca.mcp_server"])
def test_entry_point_does_not_import_numerical_stack(module: str):
    """Neither entry point may pull the numerical stack at import time."""
    leaked = _imported_after_import(module) & set(HEAVY)
    assert not leaked, (
        f"{module} imports {sorted(leaked)} at module scope; these belong to "
        f"first use (embedding, index search, BM25)"
    )


def test_embed_path_still_works_after_deferral(monkeypatch):
    """embed() must work through the deferred import, producing L2-normalised float32.

    A stub model stands in for fastembed so this stays in the fast suite and
    exercises the real arithmetic: the in-function import, the normalisation,
    the caching, and the stacking back to an ndarray.
    """
    import numpy as np

    from guaipeca.embedding import EmbeddingService

    service = EmbeddingService(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=4,
        preprocess=False,
    )
    # Non-unit vectors: normalisation has something to correct.
    outputs = [
        np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    ]
    stub = SimpleNamespace(embed=lambda texts: iter(outputs[: len(texts)]))
    monkeypatch.setattr(EmbeddingService, "_build_model", lambda self: stub)

    result = service.embed(["alpha", "beta"])

    assert result.shape == (2, 4)
    assert result.dtype == np.float32
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), norms

    # Cached hit: same vector, and the cache really stored it.
    again = service.embed(["alpha"])
    assert np.allclose(again[0], result[0], atol=1e-6)


def test_embed_empty_batch_returns_shaped_array():
    """The empty-input early return is after the deferred import; it must still work."""
    import numpy as np

    from guaipeca.embedding import EmbeddingService

    service = EmbeddingService(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=4,
        preprocess=False,
    )
    empty = service.embed([])
    assert empty.shape == (0, 4)
    assert empty.dtype == np.float32
