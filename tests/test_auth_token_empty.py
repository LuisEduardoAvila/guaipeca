"""An explicitly configured empty auth_token must not silently disable auth (#52 review).

Found by the local review arm on PR #52: `auth_token: ""` is falsy, so the
charset check is skipped and `_check_bearer` hits `if not auth_token: return
True` before reading the header at all. Measured: no Authorization header,
`Bearer `, `Bearer anything` and `Basic user:pass` all authenticate.

That is worse than an open server, because a configured token makes the server
*look* protected. The operator intent behind `auth_token: ""` is "auth is on";
honouring it as "auth is off" is the silent failure this rejects.

Absent/null remains the documented way to disable auth.
"""

from __future__ import annotations

import textwrap

import pytest

from guaipeca.config import GuaipecaConfig


def _load(tmp_path, server_block: str) -> GuaipecaConfig:
    path = tmp_path / "guaipeca.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            corpora:
              - name: docs
                path: /tmp/docs
            """
        )
        + server_block,
        encoding="utf-8",
    )
    return GuaipecaConfig.from_yaml(str(path))


def test_empty_auth_token_is_rejected(tmp_path):
    """`auth_token: ""` must fail at load, not silently run unauthenticated."""
    with pytest.raises(ValueError, match="empty"):
        _load(tmp_path, 'server:\n  auth_token: ""\n')


def test_whitespace_only_auth_token_is_rejected(tmp_path):
    """A token of only spaces is either a typo or a weak secret; reject it."""
    with pytest.raises(ValueError, match="empty|whitespace"):
        _load(tmp_path, 'server:\n  auth_token: "   "\n')


def test_absent_auth_token_still_means_auth_off(tmp_path):
    """The other side of the domain: omitting the key must stay valid."""
    cfg = _load(tmp_path, "server:\n  port: 8090\n")
    assert cfg.server.auth_token is None


def test_real_token_still_accepted(tmp_path):
    cfg = _load(tmp_path, 'server:\n  auth_token: "a-real-token"\n')
    assert cfg.server.auth_token == "a-real-token"


def test_control_character_token_is_rejected(tmp_path):
    """A token with a newline cannot appear in a header, so it could never work (#43)."""
    with pytest.raises(ValueError, match="control characters"):
        _load(tmp_path, 'server:\n  auth_token: "a\\nb"\n')


def test_printable_ascii_still_accepted(tmp_path):
    """The rejection must not catch ordinary tokens."""
    cfg = _load(tmp_path, 'server:\n  auth_token: "a-B_c.123"\n')
    assert cfg.server.auth_token == "a-B_c.123"
