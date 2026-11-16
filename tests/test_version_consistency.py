"""The two version strings must agree.

`pyproject.toml` and `src/guaipeca/__init__.py` both declare the version and
nothing checked they matched. They only stayed in sync because every bump was
done by hand in both places, and `/health` reports `__version__`, so a drift
would make the running service lie about which release it is.
"""

from __future__ import annotations

import re
from pathlib import Path

import guaipeca

ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "no version in pyproject.toml"
    return match.group(1)


def test_package_version_matches_pyproject():
    assert guaipeca.__version__ == _pyproject_version()


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", guaipeca.__version__), guaipeca.__version__


def test_health_endpoint_uses_package_version():
    """The /health payload is built from the package __version__, not a copy."""
    import guaipeca.mcp_server as m

    assert m.__version__ == guaipeca.__version__

    source = (ROOT / "src" / "guaipeca" / "mcp_server.py").read_text(encoding="utf-8")
    assert '"version": __version__' in source or "'version': __version__" in source
