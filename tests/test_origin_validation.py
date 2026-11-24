"""Tests for Origin header validation (CORS / DNS-rebinding protection).

The only surviving tests after the Phase 4 migration are the unit tests for
``_origin_allowed`` — the e2e harness (``MockMCPServer``, ``_start_server``)
drove ``GuaipecaMCPServer.run_http()``, the hand-rolled transport removed in
Phase 4.  The e2e assertions live in ``tests/test_mcp_sdk_http.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from guaipeca.mcp_server import _origin_allowed


# ---- Unit tests for the policy function ----

class TestOriginPolicy:
    def test_absent_origin_allowed(self):
        assert _origin_allowed("", None) is True

    def test_loopback_origins_allowed(self):
        for origin in (
            "http://localhost:3000",
            "http://127.0.0.1:8090",
            "https://127.0.0.1",
            "http://[::1]:8080",
            "http://ip6-localhost:3000",
        ):
            assert _origin_allowed(origin, None) is True, origin

    def test_evil_origin_rejected(self):
        for origin in (
            "http://evil.example.com",
            "https://evil.example.com:8443",
            "http://127.0.0.1.evil.example.com",
            "null",
        ):
            assert _origin_allowed(origin, None) is False, origin

    def test_allowlist_exact_match(self):
        allowed = ["https://ui.example.com"]
        assert _origin_allowed("https://ui.example.com", allowed) is True
        assert _origin_allowed("https://ui.example.com/", allowed) is True
        assert _origin_allowed("https://other.example.com", allowed) is False
        # An allowlist replaces the loopback default rather than extending it.
        assert _origin_allowed("http://localhost:3000", allowed) is False

    def test_padded_entry_matches_both_layers(self):
        """A padded entry matches _origin_allowed (outer) and
        _allowed_origin_patterns (inner SDK) equally — never one accepts
        while the other rejects. See issue #101."""
        allowed = ["https://ui.example.com"]
        # Padded origin matches the trimmed allowlist entry.
        assert _origin_allowed("  https://ui.example.com  ", allowed) is True
        # Whitespace-only origin is rejected (both layers strip to empty).
        assert _origin_allowed("  ", allowed) is False
