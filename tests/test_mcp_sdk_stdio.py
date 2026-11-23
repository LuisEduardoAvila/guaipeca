"""End-to-end test of the SDK stdio transport (Phase 3 of the SDK migration).

Drives the real entry point in a child process:

    python -m guaipeca --config <tmp yaml> serve --transport stdio

and speaks newline-delimited JSON-RPC to its stdin/stdout, exactly as a stdio
MCP client (Claude Code, the SDK client) does. Nothing here loads an embedding
model on purpose: the child's corpus is an empty temp directory, and a session
that only initializes and lists tools never reaches the embedder. If the child
cannot get that far the test fails with its stderr, not a hang.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from guaipeca.mcp_server import TOOLS

PROJECT_ROOT = Path(__file__).parent.parent

# Generous enough for a Pi 5 importing fastembed/faiss/markitdown, small enough
# that a hung child fails the job instead of stalling CI.
RESPONSE_TIMEOUT = 30.0
# How long a child gets to exit after stdin EOF.
EXIT_TIMEOUT = 30.0

EXPECTED_TOOLS = {tool["name"] for tool in TOOLS}

INITIALIZE_PARAMS = {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": {"name": "guaipeca-e2e", "version": "0.0.0"},
}


class _StdioChild:
    """A ``guaipeca serve --transport stdio`` child, read line by line."""

    def __init__(self, config_path: Path):
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_chunks: list[str] = []
        # ``-m guaipeca`` (not the ``guaipeca`` script): the same code path,
        # but independent of how the venv lays out console scripts.
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "guaipeca",
                "--config",
                str(config_path),
                "serve",
                "--transport",
                "stdio",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Own process group: killing the child cannot take the test runner
            # with it, and a stray grandchild does not survive the fixture.
            start_new_session=True,
        )
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()

    def _pump_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr_chunks.append(line)

    def stderr_text(self) -> str:
        return "".join(self._stderr_chunks)

    def send(self, message: dict) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def read_message(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"No JSON-RPC response within {timeout:.0f}s. "
                    f"Child alive={self.process.poll() is None}.\n"
                    f"stderr:\n{self.stderr_text()}"
                )
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(
                    "Child closed stdout before replying. "
                    f"exit={self.process.poll()}\nstderr:\n{self.stderr_text()}"
                )
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Not a JSON-RPC line (a stray log line on stdout): keep reading.
                continue

    def request(self, msg_id: int, method: str, params: dict | None = None) -> dict:
        message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        response = self.read_message(RESPONSE_TIMEOUT)
        assert response.get("id") == msg_id, f"response for id {msg_id} expected, got {response}"
        return response

    def close(self) -> tuple[int | None, str]:
        """Close stdin (EOF), wait, then kill if the child did not exit."""
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            exit_code = self.process.wait(timeout=EXIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.process.kill()
            exit_code = self.process.wait(timeout=10)
        self._stderr_thread.join(timeout=5)
        return exit_code, self.stderr_text()


def _startup_or_skip_or_fail(child: _StdioChild, step: str, exc: BaseException) -> None:
    """Classify a child that never answered.

    An environment that cannot import the package or load the native model
    stack is a skip, not a Phase 3 failure; a child that died for any other
    reason is a real failure and must show its stderr, not hang the job.
    """
    stderr = child.stderr_text()
    environmental = (
        "ModuleNotFoundError",
        "ImportError",
        "No module named",
        "Failed to load embedding model",
        "cannot open shared object",
    )
    if any(marker in stderr for marker in environmental):
        pytest.skip(f"environment cannot run `guaipeca serve`: {step} failed\n{stderr[-2000:]}")
    raise AssertionError(
        f"`guaipeca serve --transport stdio` never answered {step} "
        f"({type(exc).__name__}: {exc})\nstderr:\n{stderr[-4000:]}"
    ) from exc


def _answer_or_classify(child: _StdioChild, step: str, request: Callable[[], dict]) -> dict:
    """Run one request, turning a dead/timed-out child into skip-or-fail."""
    try:
        return request()
    except (TimeoutError, RuntimeError) as exc:
        _startup_or_skip_or_fail(child, step, exc)
        raise  # unreachable: the helper skips or raises


def _initialize(child: _StdioChild) -> dict:
    init = _answer_or_classify(
        child, "initialize", lambda: child.request(1, "initialize", INITIALIZE_PARAMS)
    )
    assert "error" not in init, f"initialize failed: {init}\nstderr:\n{child.stderr_text()}"
    return init["result"]


@pytest.fixture
def stdio_config(tmp_path: Path) -> Path:
    """Minimal config: one empty corpus, temp data dir, no real model needed."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    config_path = tmp_path / "guaipeca.yaml"
    config_path.write_text(
        "\n".join(
            [
                "corpora:",
                "  - name: e2e-empty",
                f"    path: {json.dumps(str(corpus_dir))}",
                '    topic: ""',
                "    weight: 1.0",
                '    extensions: [".md", ".txt"]',
                "embedding:",
                "  model: all-MiniLM-L6-v2",
                "  dimensions: 384",
                "  cache_dir: " + json.dumps(str(tmp_path / "models")),
                "indexing:",
                "  data_dir: " + json.dumps(str(tmp_path / "data")),
                "server:",
                "  transport: stdio",
                "  host: 127.0.0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def stdio_child(stdio_config: Path):
    child = _StdioChild(stdio_config)
    try:
        yield child
    finally:
        if child.process.poll() is None:
            child.process.kill()
            child.process.wait(timeout=10)


def test_stdio_initialize_then_tools_list(stdio_child: _StdioChild):
    """A real stdio client can initialize and list all twelve tools."""
    result = _initialize(stdio_child)
    assert result["serverInfo"]["name"] == "guaipeca"
    assert result.get("protocolVersion"), f"no protocolVersion in initialize result: {result}"

    # Required before any request that depends on initialized state.
    stdio_child.send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    listed = _answer_or_classify(
        stdio_child, "tools/list", lambda: stdio_child.request(2, "tools/list")
    )

    assert "error" not in listed, f"tools/list failed: {listed}"
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert names == EXPECTED_TOOLS, (
        f"tool set differs from the registry: "
        f"missing={sorted(EXPECTED_TOOLS - names)}, unexpected={sorted(names - EXPECTED_TOOLS)}"
    )
    assert len(names) == 12


def test_stdio_exits_cleanly_on_eof(stdio_child: _StdioChild):
    """Closing stdin ends the session instead of leaving the child hung."""
    _initialize(stdio_child)
    exit_code, stderr = stdio_child.close()
    assert exit_code is not None, "child did not exit after stdin EOF"
    # EOF is a normal end of a stdio session, not an error.
    assert exit_code == 0, f"child exited {exit_code} on EOF\nstderr:\n{stderr}"
