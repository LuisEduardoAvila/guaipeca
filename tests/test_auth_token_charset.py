"""The configured auth token must be presentable as a Bearer token (#43).

Starlette decodes request headers as latin-1, so a presented Bearer token can
only ever contain code points U+0000-U+00FF. A configured secret containing
anything above that (CJK, emoji, most non-latin-1 scripts) could therefore never
match: the operator sees a permanent 401 with no diagnostic.

The issue offered three options (document it, remove the constraint, or reject
at config load). Rejecting is the one that turns a silent permanent 401 into an
error the operator can act on, at the moment they load the config.

These tests go through ``GuaipecaConfig.from_yaml``, which is the path an
operator actually takes and the only one that runs ``_validate_types``. A
first draft called ``validate()`` instead, which does NOT run that check - it
would have passed for the wrong reason either way.
"""

from __future__ import annotations

import textwrap

import pytest

from guaipeca.config import GuaipecaConfig


def _load(tmp_path, auth_token: str | None) -> GuaipecaConfig:
    token_line = "" if auth_token is None else f'  auth_token: "{auth_token}"\n'
    path = tmp_path / "guaipeca.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            corpora:
              - name: docs
                path: /tmp/docs
            """
        )
        + "server:\n"
        + (token_line or "  port: 8090\n"),
        encoding="utf-8",
    )
    return GuaipecaConfig.from_yaml(str(path))


def test_ascii_token_is_accepted(tmp_path):
    cfg = _load(tmp_path, "a-plain-ascii-token")
    assert cfg.server.auth_token == "a-plain-ascii-token"


def test_latin1_token_is_accepted(tmp_path):
    """U+00FF and below survive starlette's latin-1 decode, so they can match."""
    cfg = _load(tmp_path, "pässwörd-ÿ")
    assert cfg.server.auth_token == "pässwörd-ÿ"


def test_emoji_token_is_rejected_at_load(tmp_path):
    """An emoji secret can never be presented; fail at startup, not as a 401."""
    with pytest.raises(ValueError, match="U\\+00FF"):
        _load(tmp_path, "pässwörd-\U0001F600")


def test_cjk_token_is_rejected_at_load(tmp_path):
    with pytest.raises(ValueError, match="U\\+00FF"):
        _load(tmp_path, "秘密のトークン")


def test_no_token_still_loads(tmp_path):
    cfg = _load(tmp_path, None)
    assert cfg.server.auth_token is None


def test_bearer_helpers_are_listed_in_dunder_all():
    """#43: the tests import these private helpers, so a future __all__ cleanup
    must not silently drop them."""
    from guaipeca.mcp_sdk import http

    assert "_check_bearer" in http.__all__
    assert "_HmacTokenVerifier" in http.__all__


def test_documented_helpers_still_importable_by_name():
    """The other side: adding them to __all__ did not rename or remove anything."""
    from guaipeca.mcp_sdk import http

    assert callable(http._check_bearer)
    assert http._HmacTokenVerifier is not None
