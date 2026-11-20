#!/usr/bin/env python3
"""Refuse a release whose tag does not match the declared version (issue #45).

`tests/test_version_consistency.py` keeps `pyproject.toml` and `__init__.py`
agreeing with each other. Nothing stopped a `v0.6.0` tag being pushed on a tree
declaring `0.6.1` — the one moment the version matters most.

    python3 tools/release_check.py v0.6.0     # or: python3 tools/release_check.py

With no argument it uses the tag pointed at by the workflow's ref, and exits 0
(not a release) when the ref is not a tag. Exit codes: 0 ok/skip, 1 mismatch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def declared_version() -> str:
    text = (ROOT / "src" / "guaipeca" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit("release_check: no __version__ in src/guaipeca/__init__.py")
    return match.group(1)


def main(argv: list[str]) -> int:
    tag = argv[1] if len(argv) > 1 else ""
    if not tag:
        print("release_check: no tag supplied; not a release, nothing to check")
        return 0
    if not tag.startswith("v"):
        print(f"release_check: {tag!r} is not a v-prefixed release tag; skipping")
        return 0
    expected = f"v{declared_version()}"
    if tag != expected:
        print(
            f"release_check: tag {tag} does not match declared version {expected}. "
            f"Bump the version in pyproject.toml and __init__.py, or retag.",
            file=sys.stderr,
        )
        return 1
    print(f"release_check: {tag} matches declared version")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
