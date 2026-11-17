"""Tests for ``guaipeca.service``, the systemd unit shipped with the repo.

The unit is a real artefact with real failure modes that no runtime test
covers: a directive in the wrong section is silently ignored by systemd, and a
wrong transport makes the daemon spawn a listener it cannot serve. Both
happened in this repo — ``StartLimitIntervalSec`` sat in ``[Service]`` (systemd
logged "Unknown key ... ignoring") and ``--transport both`` started a stdio
listener with no stdin, which exits immediately.

``systemd-analyze verify`` is the authoritative check, but it is not present
everywhere and it validates the *installed* copy, not the repo's. These tests
parse the repo file directly so they run in CI on any machine, and they assert
the specific mistakes that actually occurred rather than restating the file.
"""

from __future__ import annotations

import configparser
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
# The installable unit (``guaipeca.service``) is gitignored because it embeds
# absolute personal paths. The tracked artefact is the template, so that is
# what these tests validate - otherwise they would pass locally and fail on a
# fresh clone.
UNIT_PATH = PROJECT_ROOT / "guaipeca.service.template"

# Directives systemd only honours in [Unit]. In [Service] they are ignored and
# logged as "Unknown key ... in section [Service], ignoring".
UNIT_ONLY_DIRECTIVES = (
    "StartLimitIntervalSec",
    "StartLimitBurst",
    "After",
    "Before",
    "Wants",
    "Requires",
    "Conflicts",
    "Documentation",
    "Description",
    "PartOf",
    "BindsTo",
)

# Directives systemd only honours in [Service].
SERVICE_ONLY_DIRECTIVES = (
    "ExecStart",
    "Restart",
    "RestartSec",
    "User",
    "Group",
    "WorkingDirectory",
    "MemoryMax",
    "CPUQuota",
    "Type",
)


@pytest.fixture(scope="module")
def unit() -> configparser.ConfigParser:
    """The unit file parsed as INI.

    ``strict=False`` because systemd allows repeated keys (Environment= twice)
    and configparser otherwise rejects them. ``optionxform`` is the identity so
    keys keep their CamelCase spelling, which is how systemd names them.
    """
    # The template uses systemd specifiers (``%h``) and ``CHANGEME`` user
    # placeholders; neither affects directive placement or values under test.
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # type: ignore[assignment]
    assert UNIT_PATH.exists(), f"missing unit file: {UNIT_PATH}"
    parser.read_string(UNIT_PATH.read_text(encoding="utf-8"))
    return parser


def test_unit_has_the_expected_sections(unit):
    assert "Unit" in unit
    assert "Service" in unit
    assert "Install" in unit


@pytest.mark.parametrize("directive", UNIT_ONLY_DIRECTIVES)
def test_unit_only_directives_are_not_in_service(unit, directive):
    """A [Unit] directive in [Service] is silently ignored.

    This is not hypothetical: StartLimitIntervalSec was in [Service] here, so
    the configured start-rate limit never applied.
    """
    assert directive not in unit["Service"], (
        f"{directive} is a [Unit] directive but sits in [Service]; systemd will "
        f"ignore it and log 'Unknown key'. Move it to [Unit]."
    )


@pytest.mark.parametrize("directive", SERVICE_ONLY_DIRECTIVES)
def test_service_only_directives_are_not_in_unit(unit, directive):
    """The mirror case: a [Service] directive in [Unit] is also ignored."""
    assert directive not in unit["Unit"], (
        f"{directive} belongs in [Service], not [Unit]; systemd would ignore it."
    )


def test_start_rate_limit_is_configured(unit):
    """Restart=on-failure without a start limit can spin forever on a bad config."""
    assert unit["Unit"].get("StartLimitIntervalSec")
    assert unit["Unit"].get("StartLimitBurst")


def test_exec_start_serves_http_only(unit):
    """The daemon must not spawn a stdio transport.

    systemd gives the process no stdin, so a stdio listener sees EOF at once and
    its thread exits immediately. 'both' therefore means "HTTP, plus a thread
    that dies". It also made the two transports import the SDK concurrently,
    which could crash startup with a circular-import error in ``mcp``.
    """
    exec_start = unit["Service"]["ExecStart"]
    assert "--transport" in exec_start, f"ExecStart does not pin a transport: {exec_start}"

    match = re.search(r"--transport[= ]+(\S+)", exec_start)
    assert match, f"cannot parse the transport from ExecStart: {exec_start}"
    transport = match.group(1).strip("\"'")

    assert transport == "http", (
        f"ExecStart uses --transport {transport!r}; a systemd daemon should use "
        f"'http'. 'both' starts a stdio listener that exits immediately (no "
        f"stdin) and can race the HTTP transport's SDK import at startup."
    )


def test_exec_start_points_at_the_configured_port(unit):
    """The port in ExecStart must match the one the health check and docs use."""
    exec_start = unit["Service"]["ExecStart"]
    match = re.search(r"--port[= ]+(\d+)", exec_start)
    assert match, f"cannot parse the port from ExecStart: {exec_start}"
    assert match.group(1) == "8090"


def test_service_does_not_run_as_root(unit):
    """A network-facing daemon should not be root."""
    user = unit["Service"].get("User", "")
    assert user, "service should declare a User"
    assert user != "root", f"service must not run as root, got {user!r}"


def test_service_has_a_memory_cap(unit):
    """This host is a 7.9 GB Pi shared with other services (see AGENTS.md)."""
    assert unit["Service"].get("MemoryMax"), "no MemoryMax; a runaway index could exhaust the host"


def test_wanted_by_multi_user_target(unit):
    assert "multi-user.target" in unit["Install"].get("WantedBy", "")


@pytest.mark.skipif(
    shutil.which("systemd-analyze") is None,
    reason="systemd-analyze not available on this machine",
)
def test_systemd_analyze_verify_accepts_the_unit(tmp_path):
    """Authoritative cross-check with systemd's own parser.

    Run against a copy in a temp dir so a machine's installed unit cannot make
    this pass or fail: the repo file is what ships.
    """
    target = tmp_path / "guaipeca.service"
    target.write_text(UNIT_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    result = subprocess.run(
        ["systemd-analyze", "verify", str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    output = (result.stdout + result.stderr).strip()
    assert "Unknown key" not in output, f"systemd rejects a directive in our unit:\n{output}"
    assert "Unknown section" not in output, f"systemd rejects a section in our unit:\n{output}"
