"""Contratos da configuração MCP da lane backend da Fase 3."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from api import server
from api.core import builder, config_store, path_policy, mcp_config


def _model(root: Path) -> Path:
    model = root / "owner" / "repo" / "model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"GGUF")
    return model


def _cfg(model: Path, mcp: Path | None = None, *, auto: bool = False) -> dict:
    return {
        "model": str(model),
        "backend": "vanilla",
        "context_window": 4096,
        "kv_cache": "q5_0",
        "gpu_layers": 99,
        "parallel_slots": 1,
        "flash_attn": True,
        "batch_size": 512,
        "ubatch_size": 128,
        "threads_gen": 2,
        "threads_batch": 2,
        "reasoning_budget": None,
        "mlock": False,
        "max_tokens": 8,
        "cache_ram": 2048,
        "ctx_checkpoints": 0,
        "llama_auto": auto,
        "mcp_servers_config": str(mcp) if mcp else None,
    }


def _canonical(tmp_path: Path, payload: dict, monkeypatch) -> Path:
    path = tmp_path / "config" / "mcp" / "servers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(mcp_config, "MCP_CONFIG_FILE", path)
    return path


@pytest.fixture
def fake_backend(monkeypatch, tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("binary")
    monkeypatch.setattr(builder, "backend_binary", lambda *_args: binary)


def test_mcp_config_is_optional_and_is_emitted_for_normal_and_auto_server(
    tmp_path, monkeypatch, fake_backend,
):
    root = tmp_path / "models"
    root.mkdir()
    model = _model(root)
    mcp = _canonical(tmp_path, {"mcpServers": {}}, monkeypatch)
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [root])

    absent = server._validated_config_paths(_cfg(model))
    assert absent["mcp_servers_config"] is None
    assert "--mcp-servers-config" not in builder.build_command_from_cfg(absent)

    normal = server._validated_config_paths(_cfg(model, mcp))
    assert normal["mcp_servers_config"] == str(mcp.resolve())
    normal_command = builder.build_command_from_cfg(normal)
    assert f'--mcp-servers-config "{mcp.resolve()}"' in normal_command

    auto_command = builder.build_command_from_cfg(
        server._validated_config_paths(_cfg(model, mcp, auto=True)),
    )
    assert f'--mcp-servers-config "{mcp.resolve()}"' in auto_command


def test_mcp_config_is_never_emitted_for_cli_or_as_ui_proxy(
    tmp_path, monkeypatch, fake_backend,
):
    root = tmp_path / "models"
    root.mkdir()
    model = _model(root)
    mcp = _canonical(tmp_path, {"mcpServers": {}}, monkeypatch)
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [root])

    command = builder.build_command_from_cfg(
        server._validated_config_paths(_cfg(model, mcp)), mode="cli",
    )
    assert "--mcp-servers-config" not in command
    assert "--ui-mcp-proxy" not in command


def test_router_rejects_mcp_config_without_creating_a_session(
    tmp_path, monkeypatch,
):
    root = tmp_path / "models"
    root.mkdir()
    model = _model(root)
    mcp = _canonical(tmp_path, {"mcpServers": {}}, monkeypatch)
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [root])
    cfgs = [
        {"id": "one", **_cfg(model, mcp)},
        {"id": "two", **_cfg(model)},
    ]
    monkeypatch.setattr(server.config_store, "read_all_configs", lambda: cfgs)
    created = []
    started = []

    class DummySession:
        id = "router-test"

        def __init__(self, cfg, *, member_cfgs):
            created.append(True)
            self.cfg = cfg
            self.member_cfgs = member_cfgs

        def start(self):
            started.append(True)

    monkeypatch.setattr(server, "LaunchSession", DummySession)
    with pytest.raises(HTTPException) as exc:
        server.launch_router(server.RouterLaunchRequest(ids=["one", "two"]))
    assert exc.value.status_code == 400
    assert "router" in exc.value.detail.lower()
    assert "mcp_servers_config" in exc.value.detail
    assert created == []
    assert started == []


@pytest.mark.parametrize(
    ("value", "status", "absolute"),
    [
        ("mcp.json", 400, False),
        ("../mcp.json", 400, False),
        ("missing.json", 400, True),
        ("mcp.txt", 400, True),
    ],
)
def test_mcp_config_path_policy_maps_invalid_paths(
    tmp_path, monkeypatch, value, status, absolute,
):
    root = tmp_path / "models"
    root.mkdir()
    model = _model(root)
    (root / "mcp.txt").write_text("{}")
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [root])

    with pytest.raises(HTTPException) as exc:
        candidate = root / value if absolute else Path(value)
        server._validated_config_paths(_cfg(model, candidate))
    assert exc.value.status_code == status


def test_mcp_config_rejects_external_symlink(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    model = _model(root)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"mcpServers": {}}))
    link = tmp_path / "config" / "mcp" / "servers.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    monkeypatch.setattr(mcp_config, "MCP_CONFIG_FILE", link)
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [root])

    with pytest.raises(HTTPException) as exc:
        server._validated_config_paths(_cfg(model, link))
    assert exc.value.status_code == 400


def test_mcp_config_rejects_internal_symlink_to_non_json(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    model = _model(root)
    target = root / "config.txt"
    target.write_text("{}")
    link = tmp_path / "config" / "mcp" / "servers.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    monkeypatch.setattr(mcp_config, "MCP_CONFIG_FILE", link)
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [root])

    with pytest.raises(HTTPException) as exc:
        server._validated_config_paths(_cfg(model, link))
    assert exc.value.status_code == 400


def test_mcp_config_persistence_and_legacy_config_compatibility(tmp_path):
    config_file = tmp_path / "last_config.json"
    model = tmp_path / "model.gguf"
    mcp = tmp_path / "mcp.json"
    mcp.write_text("{}")
    saved = config_store.save_config(
        {"model": str(model), "backend": "vanilla", "mcp_servers_config": str(mcp)},
        config_file,
    )
    loaded = config_store.read_all_configs(config_file)[0]
    assert loaded["mcp_servers_config"] == str(mcp)
    assert saved["mcp_servers_config"] == str(mcp)

    config_file.write_text(json.dumps([{"model": str(model)}]))
    legacy = config_store.read_all_configs(config_file)[0]
    assert "mcp_servers_config" not in legacy
    assert server.LaunchConfig(**legacy).mcp_servers_config is None


def test_mcp_path_policy_returns_canonical_path_and_rejects_traversal(tmp_path, monkeypatch):
    root = tmp_path / "settings"
    root.mkdir()
    mcp = _canonical(tmp_path, {"mcpServers": {}}, monkeypatch)
    assert path_policy.validate_existing_json_config(mcp, [root]) == mcp.resolve()
    with pytest.raises(path_policy.MalformedPath):
        path_policy.validate_existing_json_config(
            root / ".." / "settings" / "mcp.json", [root],
        )
