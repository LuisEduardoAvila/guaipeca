from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import guaipeca

PROJECT_ROOT = Path(__file__).parent.parent


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    src = str(PROJECT_ROOT / "src")
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{inherited}" if inherited else src
    return subprocess.run(
        # `python -m guaipeca` (the __main__ shim), not `-m guaipeca.cli`:
        # cli.py is a library module whose main() is wired up by the console
        # script entry point, so -m on it would define nothing and exit 0.
        [sys.executable, "-m", "guaipeca", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


def test_cli_version_is_bare_and_matches_package():
    """`guaipeca --version` must print exactly the package version, exit 0.

    Bare, not `guaipeca 0.6.0`: the MCP client and /health report the bare
    version, and scripts comparing them should not have to strip a prefix.
    """
    if not (PROJECT_ROOT / "src" / "guaipeca" / "cli.py").exists():
        pytest.skip("src layout not present (installed distribution)")
    result = _run_cli("--version")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == guaipeca.__version__
