#!/opt/venv/bin/python
"""Container health probe for Guaipeca.

``/health`` requires a Bearer token whenever ``server.auth_token`` is set in the
config, so a probe that omits it reports a healthy server as failing. That was a
real bug: enabling auth turned every container unhealthy while the app served
normally.

The token is read from the same config file the server reads, not from an
environment variable, so the probe and the server cannot disagree about it.

Exit codes: 0 when /health answers 200, 1 otherwise (with a one-line reason on
stderr so ``docker inspect`` shows something useful).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

CONFIG_PATH = os.environ.get("GUAIPECA_CONFIG", "/data/config/guaipeca.yaml")
HEALTH_URL = os.environ.get("GUAIPECA_HEALTH_URL", "http://localhost:8090/health")


def _auth_token() -> str | None:
    """Read server.auth_token from the config, tolerating a missing/broken file."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml is a hard dependency
        return None
    try:
        with open(CONFIG_PATH, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return None
    token = (data.get("server") or {}).get("auth_token")
    return token if isinstance(token, str) and token else None


def main() -> int:
    request = urllib.request.Request(HEALTH_URL, headers={"Accept": "application/json"})
    token = _auth_token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            if response.status != 200:
                print(f"health: HTTP {response.status}", file=sys.stderr)
                return 1
            body = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = "auth_token set in config but rejected" if exc.code == 401 else "unexpected status"
        print(f"health: HTTP {exc.code} ({detail})", file=sys.stderr)
        return 1
    except Exception as exc:  # any failure means "not ready"
        print(f"health: {exc}", file=sys.stderr)
        return 1

    if body.get("status") != "ok":
        print(f"health: unexpected body {body}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
