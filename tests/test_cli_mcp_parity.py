"""CLI/MCP parity checks (issue #10).

The CLI and the MCP server expose the same read/write capabilities. These
tests fail if an MCP tool or ``search`` parameter is added/renamed without a
matching CLI surface, so the two adapters cannot drift apart silently.
"""

from __future__ import annotations

import argparse

import pytest

from guaipeca.cli import (
    MCP_TOOL_TO_CLI,
    SEARCH_PROP_TO_CLI_DEST,
    build_parser,
)
from guaipeca.mcp_server import TOOLS


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError("no subparsers registered")


def _cli_commands() -> set[str]:
    return set(_subparsers(build_parser()).choices)


def _subparser_options(command: str) -> set[str]:
    """Return the set of argparse dests exposed by ``command``."""
    sub = _subparsers(build_parser()).choices[command]
    dests = set()
    for action in sub._actions:
        if action.dest and action.dest != "help":
            dests.add(action.dest)
    return dests


def _mcp_tool_names() -> set[str]:
    return {tool["name"] for tool in TOOLS}


def test_every_mcp_tool_has_a_cli_mapping():
    assert _mcp_tool_names() <= set(MCP_TOOL_TO_CLI), (
        "MCP tools without a CLI counterpart: "
        f"{_mcp_tool_names() - set(MCP_TOOL_TO_CLI)}"
    )


def test_mapping_has_no_stale_entries():
    assert set(MCP_TOOL_TO_CLI) <= _mcp_tool_names(), (
        "CLI mappings referencing unknown MCP tools: "
        f"{set(MCP_TOOL_TO_CLI) - _mcp_tool_names()}"
    )


def test_every_mapped_cli_command_is_registered():
    registered = _cli_commands()
    missing = {cmd for cmd in MCP_TOOL_TO_CLI.values() if cmd not in registered}
    assert not missing, f"Mapped CLI commands not registered as subparsers: {missing}"


@pytest.mark.parametrize("cmd", sorted(set(MCP_TOOL_TO_CLI.values())))
def test_mapped_command_parses(cmd):
    """Sanity: each mapped command is a real subparser with a handler."""
    sub = _subparsers(build_parser()).choices[cmd]
    assert sub.get_default("func") is not None


def test_every_mcp_search_param_has_a_cli_dest():
    search_tool = next(t for t in TOOLS if t["name"] == "search")
    props = set(search_tool["inputSchema"].get("properties", {}))
    missing = props - set(SEARCH_PROP_TO_CLI_DEST)
    assert not missing, f"MCP search params without CLI mapping: {missing}"


def test_search_cli_exposes_all_mapped_dests():
    dests = _subparser_options("search")
    missing = {d for d in SEARCH_PROP_TO_CLI_DEST.values() if d not in dests}
    assert not missing, f"search subparser missing options for dests: {missing}"


def test_search_cli_accepts_extra_flags():
    args = build_parser().parse_args([
        "search", "query",
        "--return-mode", "auto",
        "--max-chars", "1234",
        "--section-mode",
        "--section-filter", "Chapter 3",
    ])
    assert args.return_mode == "auto"
    assert args.max_chars == 1234
    assert args.section_mode is True
    assert args.section_filter == "Chapter 3"


@pytest.mark.parametrize(
    "argv,handler",
    [
        (["get-chunk", "corpus:abc123"], "cmd_get_chunk"),
        (["get-document", "--corpus", "c", "--path", "/x/y.md"], "cmd_get_document"),
        (["get-document", "--corpus", "c", "--source-path", "/x/y.md"], "cmd_get_document"),
        (["list-documents"], "cmd_list_documents"),
        (["list-documents", "c"], "cmd_list_documents"),
        (["list-folders"], "cmd_list_folders"),
        (["get-toc", "--corpus", "c"], "cmd_get_toc"),
        (["get-toc", "--corpus", "c", "--document", "/x/y.md"], "cmd_get_toc"),
        (["upload", "report.pdf", "--corpus", "c"], "cmd_upload"),
        (["upload", "report.pdf", "--corpus", "c", "--no-index"], "cmd_upload"),
        (["delete", "report.pdf", "--corpus", "c"], "cmd_delete"),
        (["delete", "report.pdf", "--corpus", "c", "--no-index"], "cmd_delete"),
    ],
)
def test_new_subcommands_parse(argv, handler):
    args = build_parser().parse_args(argv)
    assert args.func.__name__ == handler


def test_upload_and_delete_default_to_indexing():
    assert build_parser().parse_args(["upload", "f.md", "--corpus", "c"]).no_index is False
    assert build_parser().parse_args(["delete", "f.md", "--corpus", "c"]).no_index is False
