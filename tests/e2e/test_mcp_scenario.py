"""No-backend unit coverage for the MCP scenario and its local fixture."""
from __future__ import annotations

import subprocess
import signal
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest


E2E_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(E2E_DIR))

from fixtures import mcp_process  # noqa: E402
from scenarios import mcp  # noqa: E402


def test_fixture_prints_sentinel_and_stays_alive():
    process = subprocess.Popen(
        [sys.executable, str(Path(mcp_process.__file__).resolve())],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        line = process.stdout.readline().strip()
        assert line.startswith(mcp_process.READY_SENTINEL)
        assert mcp_process.parse_ready_pid(line) == process.pid
        assert process.poll() is None
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=2)
    assert process.returncode == 0


def test_fixture_exit_code_mode_is_short_and_nonzero():
    result = subprocess.run(
        [sys.executable, str(Path(mcp_process.__file__).resolve()), "--exit-code", "7"],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    assert result.returncode == 7
    line = result.stdout.strip()
    assert line.startswith(mcp_process.READY_SENTINEL)
    assert mcp_process.parse_ready_pid(line) > 0


def test_fixture_command_is_local_and_supports_exit_mode(tmp_path):
    command = mcp.fixture_command(tmp_path)
    exit_command = mcp.fixture_command(tmp_path, exit_code=7)
    assert str(mcp.fixture_path(tmp_path)) in command
    assert "--exit-code 7" in exit_command
    assert command.startswith("exec ")
    assert exit_command.startswith("exec ")
    assert command == mcp.fixture_command(tmp_path)
    assert "http://" not in command
    assert "https://" not in command


def test_mcp_validator_allows_only_e2e_local_mutations():
    root = E2E_DIR.parents[1]
    payload = {
        "id": mcp.MCP_SERVER_ID,
        "name": mcp.MCP_SERVER_NAME,
        "cwd": str(root.resolve()),
        "command": mcp.fixture_command(root),
    }
    mcp.validate_mcp_request("POST", "http://127.0.0.1:8420/api/mcp", payload, root=root)
    mcp.validate_mcp_request("POST", f"http://127.0.0.1:8420/api/mcp/{mcp.MCP_SERVER_ID}/start")
    mcp.validate_mcp_request("GET", "http://127.0.0.1:8420/api/mcp")
    with pytest.raises(Exception, match="loopback"):
        mcp.validate_mcp_request("POST", "http://192.0.2.1:8420/api/mcp", payload, root=root)
    with pytest.raises(Exception, match="allowlisted"):
        mcp.validate_mcp_request("DELETE", "http://127.0.0.1:8420/api/mcp/production")
    with pytest.raises(Exception, match="remote MCP"):
        mcp.validate_mcp_request(
            "POST", "http://127.0.0.1:8420/api/mcp", {**payload, "command": "https://example.invalid"}, root=root,
        )


def test_backend_environment_never_allows_remote_mcp():
    assert mcp.MCP_ENV == {
        "LLM_LAUNCHER_ENABLE_MCP": "1",
        "LLM_LAUNCHER_ALLOW_REMOTE_MCP": "0",
    }
    assert mcp.MCP_DISABLED_ENV == {
        "LLM_LAUNCHER_ENABLE_MCP": "0",
        "LLM_LAUNCHER_ALLOW_REMOTE_MCP": "0",
    }


def test_wait_for_mcp_tab_polls_until_visible_and_records_duration():
    class Locator:
        def __init__(self, state):
            self.state = state

        def count(self):
            return self.state[0]

        def is_visible(self):
            return self.state[1]

    class Page:
        def __init__(self):
            self.calls = 0

        def get_by_test_id(self, _name):
            self.calls += 1
            return Locator((0, False) if self.calls < 3 else (1, True))

    result = mcp.wait_for_mcp_tab(Page(), timeout=0.2, poll=0.001)
    assert result["count"] == 1
    assert result["visible"] is True
    assert result["wait_seconds"] >= 0


def test_cleanup_attempts_stop_delete_and_orphan_check_without_backend(monkeypatch):
    class Response:
        status = 200

    class API:
        def __init__(self):
            self.calls = []
            self.running = True

        def get(self, path):
            self.calls.append(("GET", path))
            return Response()

        def post(self, path):
            self.calls.append(("POST", path))
            self.running = False
            return Response()

        def delete(self, path):
            self.calls.append(("DELETE", path))
            return Response()

        def json(self, _response):
            return [{"id": mcp.MCP_SERVER_ID, "status": {"running": self.running}}]

    api = API()
    ctx = SimpleNamespace(root=E2E_DIR.parents[1], api=api)
    orphan_checks = []
    monkeypatch.setattr(mcp, "fixture_pids", lambda root: orphan_checks.append(root) or [])
    mcp._cleanup_server(ctx)  # type: ignore[arg-type]
    assert ("POST", f"/api/mcp/{mcp.MCP_SERVER_ID}/stop") in api.calls
    assert ("DELETE", f"/api/mcp/{mcp.MCP_SERVER_ID}") in api.calls
    assert orphan_checks == [ctx.root]


def test_lifecycle_cleanup_runs_when_enabled_subscenario_raises(monkeypatch):
    events = []
    context = SimpleNamespace(guard=object())

    class Lifecycle:
        def close_browser_context(self):
            events.append("close")

        def stop_backend(self):
            events.append("stop")

        def start_backend(self, env):
            events.append(("start", dict(env)))

        def open_browser_context(self) -> Any:
            events.append("open")
            return context

        def install_mcp_guard_extensions(self, guard):
            events.append("guard")

    monkeypatch.setattr(mcp, "_run_enabled", lambda _ctx: (_ for _ in ()).throw(RuntimeError("item failed")))
    monkeypatch.setattr(mcp, "_cleanup_server", lambda _ctx: events.append("cleanup"))
    with pytest.raises(RuntimeError, match="item failed"):
        mcp.run(Lifecycle())  # type: ignore[arg-type]
    cleanup_index = events.index("cleanup")
    close_index = events.index("close", cleanup_index + 1)
    stop_index = events.index("stop", close_index + 1)
    assert cleanup_index < close_index < stop_index
