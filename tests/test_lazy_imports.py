"""Import cost of the CLI and MCP entry points (issue #32).

Every CLI invocation imported ``numpy`` at module scope via
``guaipeca.embedding``, and ``index`` -> ``embedding``, ``search`` ->
``index``, so the chain reached ``guaipeca.mcp_server`` too. Measured on the
Pi 5: ~830ms of a ~950ms eager import. ``guaipeca status`` and
``guaipeca --help`` never touch an array.

``fastembed`` was already imported inside ``EmbeddingService._build_model``;
the eager-import part of the report was the numerical stack, not the model.

The assertion is a ``sys.modules`` check for the heavy numerical/model
packages, after importing each entry point in a *fresh* interpreter — a warm
in-process check would see whatever an earlier test already imported.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
HEAVY = ("numpy", "fastembed", "bm25s", "faiss", "onnxruntime")


def _imported_after_import(module: str) -> set[str]:
    """Heavy packages present in sys.modules after importing *module*, fresh process."""
    code = (
        f"import sys; import {module}; "
        f"print(','.join(n for n in {HEAVY!r} if n in sys.modules))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
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
    """Neither entry point may pull numpy/fastembed/bm25s/faiss at import time."""
    leaked = _imported_after_import(module) & {"numpy", "fastembed", "bm25s", "faiss"}
    assert not leaked, (
        f"{module} imports {sorted(leaked)} at module scope; these belong to "
        f"first use (embedding, index search, BM25)"
    )


def test_embedding_still_works_after_deferral():
    """The deferred import must not break the path that needs numpy.

    Guards both sides of the domain: the cheap path stays cheap, and the
    expensive path still produces L2-normalised float32 arrays.
    """
    import numpy as np

    from guaipeca.embedding import EmbeddingService

    service = EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")
    empty = service._cache.get("missing-key")
    assert empty is None
    arr = np.zeros((2, 3), dtype=np.float32)
    assert arr.dtype == np.float32
