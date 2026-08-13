from __future__ import annotations

import json
import sys
import threading
import time
import types

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from api import server
from api.core import backends, builder, launch_events, mcp_config, settings_store

if "questionary" not in sys.modules:
    _questionary_stub = types.ModuleType("questionary")
    _questionary_stub.Choice = lambda *args, **kwargs: args
    sys.modules["questionary"] = _questionary_stub

import models


def test_custom_without_settings_has_null_identity_and_no_capabilities(monkeypatch):
    monkeypatch.setattr(settings_store, "get_backend_dir", lambda name: None)
    status = {item["name"]: item for item in backends.backend_status()}["custom"]
    assert status["server_path"] == status["cli_path"] == ""
    assert status["default_dir"] is None
    assert status["server_available"] is False
    assert status["cli_available"] is False
    assert status["kv_types_server"] == status["kv_types_cli"] == []
    with pytest.raises(builder.BackendBinaryMissing):
        builder.build_auto_server_command(__import__("pathlib").Path("/tmp/model.gguf"), "custom")


@pytest.mark.parametrize("mode", ["server", "cli"])
@pytest.mark.parametrize("auto", [False, True])
def test_cli_custom_builders_use_required_binary_boundary(monkeypatch, mode, auto):
    monkeypatch.setattr(models, "get_backend_dir", lambda name: None)
    model = Path("/tmp/model.gguf")
    if auto:
        build = models.build_auto_command if mode == "server" else models.build_auto_cli_command
        with pytest.raises(models.BackendBinaryMissing):
            build(model, "custom")
        return
    if mode == "server":
        with pytest.raises(models.BackendBinaryMissing):
            models.build_command(
                model, "custom", 4096, "f16", True, 99, 1, 512, 128,
                2, 2, None, False, 8, 2048, 0,
            )
    else:
        with pytest.raises(models.BackendBinaryMissing):
            models.build_cli_command(
                model, "custom", 4096, "f16", True, 99, 512, 128,
                2, 2, None, False, 8,
            )


def test_cli_resilient_reconstruction_uses_required_binary(monkeypatch):
    monkeypatch.setattr(models, "get_backend_dir", lambda name: None)
    base = {"model": "/tmp/model.gguf", "backend": "custom", "llama_auto": True}
    with pytest.raises(models.BackendBinaryMissing):
        models._build_cmd_from_cfg(base)
    normal = {**base, "llama_auto": False, "context_window": 4096, "kv_cache": "f16"}
    with pytest.raises(models.BackendBinaryMissing):
        models._build_cmd_from_cfg(normal)


def test_settings_store_serializes_custom_default_as_null(tmp_path):
    shape = settings_store.read_settings(tmp_path / "settings.json")
    assert set(shape["backend_paths_defaults"]) == set(settings_store.CONFIGURABLE_BACKENDS)
    assert shape["backend_paths_defaults"]["custom"] is None


def test_custom_launch_rejects_before_save_or_session(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [tmp_path])
    monkeypatch.setattr(server.settings_store, "get_backend_dir", lambda name: None)
    monkeypatch.setattr(server.config_store, "save_config", lambda *_: pytest.fail("saved unavailable custom"))
    monkeypatch.setattr(server, "LaunchSession", lambda *_args, **_kwargs: pytest.fail("created unavailable custom"))
    with pytest.raises(server.HTTPException) as exc:
        server.launch(server.LaunchConfig(model=str(model), backend="custom"))
    assert exc.value.status_code == 400


def test_replay_is_late_subscriber_safe_and_cursors_are_independent():
    replay = launch_events.LaunchEventReplay()
    replay.publish({"type": "start"})
    first = replay.subscribe(0)
    second = replay.subscribe(0)
    assert [r.seq for r in first.wait_after().events] == [1]
    assert [r.seq for r in second.wait_after().events] == [1]
    replay.publish({"type": "done"})
    assert [r.seq for r in first.wait_after().events] == [2]
    assert [r.seq for r in second.wait_after().events] == [2]


def test_replay_wait_race_and_close_drain():
    replay = launch_events.LaunchEventReplay()
    result = []

    def reader():
        result.append(replay.subscribe().wait_after(timeout=1).events[0].event["type"])

    thread = threading.Thread(target=reader)
    thread.start()
    time.sleep(0.01)
    replay.publish({"type": "giveup", "reason": "cancelled"})
    replay.close()
    thread.join(1)
    assert result == ["giveup"]
    assert replay.close() is False


def test_stdout_eviction_keeps_control_and_reports_gap():
    replay = launch_events.LaunchEventReplay()
    replay.publish({"type": "start"})
    for _ in range(launch_events.MAX_STDOUT_EVENTS + 10):
        replay.publish({"type": "stdout", "line": "x"})
    replay.publish({"type": "done"})
    batch = replay.wait_after(1, timeout=0)
    assert batch.history_gap
    assert batch.events[-1].event["type"] == "done"
    assert batch.events[0].event["type"] == "stdout"


def test_mcp_schema_rejects_url_and_secrets_do_not_enter_error(tmp_path, monkeypatch):
    path = tmp_path / "config" / "mcp" / "servers.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"mcpServers": {"x": {"command": "https://secret.example"}}}))
    monkeypatch.setattr(mcp_config, "MCP_CONFIG_FILE", path)
    with pytest.raises(mcp_config.McpConfigError) as exc:
        mcp_config.validate(path)
    assert "secret.example" not in str(exc.value)


def test_mcp_schema_rejects_alternate_transport_and_unknown_fields(tmp_path, monkeypatch):
    path = tmp_path / "config" / "mcp" / "servers.json"
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(mcp_config, "MCP_CONFIG_FILE", path)
    for entry in (
        {"command": "/bin/true", "url": "https://alternate.example"},
        {"command": "/bin/true", "unexpected": "value"},
    ):
        path.write_text(json.dumps({"mcpServers": {"x": entry}}))
        with pytest.raises(mcp_config.McpConfigError):
            mcp_config.validate(path)


def test_options_exposes_only_mcp_metadata(monkeypatch):
    monkeypatch.setattr(server.constants, "MCP_CONFIG_FILE", server.constants.MCP_CONFIG_DIR / "servers.json")
    response = TestClient(server.app).get("/api/options")
    assert response.status_code == 200
    payload = response.json()
    assert "mcp_config" not in payload
    assert set(payload["mcp_runtime_config"]) == {"path", "exists", "valid"}
    assert "command" not in json.dumps(payload["mcp_runtime_config"])
