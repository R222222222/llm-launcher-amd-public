"""Fase9 backend boundaries; all tests are filesystem/unit or ASGI-only."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api import server
from api.core import builder, constants, hf, models_repo, path_policy, runner, sampling, settings_store, updates


def _model(root: Path, rel: str = "owner/repo/model.gguf") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF fixture")
    return path


def _fake_binaries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Injeta binários falsos reais (is_file()==True) em builder.backend_binary.

    Os testes de contrato do builder não devem depender de vendor/llama.cpp
    (ausente em clone limpo): só a resolução do binário é substituída; a lógica
    do builder (porta, host, rejeição de aspas, revalidação do router) continua
    sendo o código real.
    """
    server_bin = tmp_path / "llama-server"
    server_bin.write_bytes(b"")
    cli_bin = tmp_path / "llama-cli"
    cli_bin.write_bytes(b"")
    monkeypatch.setattr(
        builder, "backend_binary",
        lambda backend, mode: server_bin if mode == "server" else cli_bin,
    )
    return server_bin


def test_host_default_and_trusted_override(monkeypatch):
    monkeypatch.delenv("LLM_LAUNCHER_HOST", raising=False)
    assert server.get_bind_host() == "127.0.0.1"
    assert server.get_bind_host({"LLM_LAUNCHER_HOST": "100.64.0.42"}) == "100.64.0.42"


def test_health_probe_host_normalization_is_generic():
    assert runner._health_host("") == "127.0.0.1"
    assert runner._health_host("0.0.0.0") == "127.0.0.1"
    assert runner._health_host("::") == "127.0.0.1"
    assert runner._health_host("127.0.0.1") == "127.0.0.1"
    assert runner._health_host("::1") == "127.0.0.1"
    assert runner._health_host("100.64.0.42") == "100.64.0.42"


def test_path_policy_accepts_in_root_and_rejects_missing_outside_and_symlink(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    model = _model(root)
    assert path_policy.validate_existing_gguf(model, [root]) == model.resolve()

    outside = _model(tmp_path / "outside", "other/model.gguf")
    with pytest.raises(path_policy.OutsideRoot):
        path_policy.validate_existing_gguf(outside, [root])
    with pytest.raises(path_policy.MissingPath):
        path_policy.validate_existing_gguf(root / "missing.gguf", [root])

    escaped = root / "escaped.gguf"
    escaped.symlink_to(outside)
    with pytest.raises(path_policy.OutsideRoot):
        path_policy.validate_existing_gguf(escaped, [root])


def test_scanner_skips_symlink_escaping_scan_root(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    good = _model(root)
    outside = _model(tmp_path / "outside", "outside.gguf")
    (root / "escaped.gguf").symlink_to(outside)
    assert models_repo.collect_models([root]) == [good]


def test_scanner_ignores_missing_roots_quotes_and_escaped_mmproj(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    good = _model(root)
    bad_quote = _model(root, 'owner/repo/bad"model.gguf')
    outside = _model(tmp_path / "outside", "mmproj.gguf")
    link = root / "owner" / "repo" / "mmproj-model.gguf"
    link.symlink_to(outside)
    missing = tmp_path / "not-installed"
    not_dir = tmp_path / "file-root"
    not_dir.write_text("not a root")
    assert bad_quote.exists()
    assert models_repo.collect_models([missing, not_dir, root]) == [good]
    assert models_repo.describe_model(good, [root])["mmproj"] is None
    link.unlink()
    (root / "owner" / "repo" / "mmproj-model.gguf").write_bytes(b"GGUF")
    assert models_repo.describe_model(good, [root])["mmproj"].endswith("mmproj-model.gguf")


def test_scanner_and_models_route_tolerate_missing_roots(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    good = _model(root)
    missing = tmp_path / "missing"
    not_dir = tmp_path / "not-dir"
    not_dir.write_text("x")
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [missing, root, not_dir])
    response = TestClient(server.app).get("/api/models")
    assert response.status_code == 200
    assert response.json()[0]["path"] == str(good)
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [missing, not_dir])
    assert TestClient(server.app).get("/api/models").json() == []


def test_delete_outside_preserved_and_valid_split_mmproj_keeps_root(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    outside = _model(tmp_path / "outside", "outside.gguf")
    with pytest.raises(path_policy.OutsideRoot):
        models_repo.plan_delete_model(outside, roots=[root])
    assert outside.exists()

    folder = root / "owner" / "repo"
    folder.mkdir(parents=True)
    first = folder / "model-00001-of-00002.gguf"
    second = folder / "model-00002-of-00002.gguf"
    mmproj = folder / "mmproj-model.gguf"
    for path in (first, second, mmproj):
        path.write_bytes(b"GGUF")
    result = models_repo.delete_model(first, roots=[root])
    assert not result["errors"]
    assert not first.exists() and not second.exists() and not mmproj.exists()
    assert root.exists()


@pytest.mark.parametrize("bad", ["", ".", "..", "a/../b", "/abs", r"a\\b", "C:/x", "//server/x", "a//b", "a\x00b"])
def test_download_subdir_rejects_traversal_and_windows_forms(bad):
    with pytest.raises(path_policy.MalformedPath):
        path_policy.validate_subdir(bad)


def test_download_destination_rejects_symlink_dest_and_part(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link_dir = root / "owner"
    link_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(path_policy.SymlinkEscape):
        path_policy.validate_write_destination(root, link_dir / "repo" / "model.gguf")

    safe = root / "owner" / "repo"
    safe.mkdir(parents=True)
    part = safe / "model.gguf.part"
    part.symlink_to(outside / "part")
    with pytest.raises(path_policy.SymlinkEscape):
        path_policy.validate_write_destination(root, safe / "model.gguf")
    with pytest.raises(path_policy.MalformedPath):
        path_policy.validate_write_destination(root, root / "owner" / ".." / "escape.gguf")


def test_sidecar_writes_reject_symlink_targets_and_preserve_outside_targets(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    folder = root / "owner" / "repo"
    folder.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("keep")

    sampling_dest = folder / "sampling.json"
    sampling_dest.symlink_to(outside)
    assert sampling.write_sidecar(folder, {"temp": 0.3}, root=root) is None
    assert outside.read_text() == "keep"
    sampling_dest.unlink()

    origin_dest = folder / "origin.json"
    origin_dest.symlink_to(outside)
    assert updates.write_origin(folder, "owner/repo", "main", [], root=root) is None
    assert outside.read_text() == "keep"
    origin_dest.unlink()

    cache = root / "oid_cache.json"
    cache.symlink_to(outside)
    monkeypatch.setattr(updates, "_OID_CACHE_FILE", cache)
    updates._write_oid_cache({"x": {"size": 1}}, root=root)
    assert outside.read_text() == "keep"


def test_builder_rejects_quote_injected_model_and_uses_safe_llama_host(tmp_path, monkeypatch):
    _fake_binaries(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        builder.build_auto_server_command(Path('/tmp/model";touch-pwned.gguf'), "vanilla")
    monkeypatch.setenv("LLM_LAUNCHER_LLAMA_HOST", "100.64.0.42")
    assert "--host 100.64.0.42" in builder.build_auto_server_command(Path("/tmp/model.gguf"), "vanilla")
    assert "--host 100.64.0.42" in builder.build_router_command(
        [{"model": "/tmp/model.gguf", "backend": "vanilla"}],
        tmp_path / "router.ini",
    )
    monkeypatch.delenv("LLM_LAUNCHER_LLAMA_HOST", raising=False)
    assert "--host 127.0.0.1" in builder.build_auto_server_command(Path("/tmp/model.gguf"), "vanilla")


def test_constants_bin_override_and_lm_studio_port_unchanged(tmp_path, monkeypatch):
    override = tmp_path / "llama-bin"
    monkeypatch.setenv("LLM_LAUNCHER_LLAMA_CPP_BIN", str(override))
    importlib.reload(constants)
    assert constants.LLAMA_CPP_BIN == override
    assert constants.LLAMA_SERVER_PORT == 8421
    assert "1234" in " ".join(["--port", "1234"])
    monkeypatch.delenv("LLM_LAUNCHER_LLAMA_CPP_BIN", raising=False)
    importlib.reload(constants)


def test_model_routes_map_malformed_outside_and_missing(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [root])
    client = TestClient(server.app)
    assert client.post("/api/models/meta", json={"model": "relative.gguf"}).status_code == 400
    assert client.post("/api/models/meta", json={"model": str(root / "missing.gguf")}).status_code == 404
    outside = _model(tmp_path / "outside", "outside.gguf")
    assert client.post("/api/models/meta", json={"model": str(outside)}).status_code == 403


def test_launch_router_restart_and_mmproj_are_validated_without_spawn(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    model = _model(root)
    outside_mmproj = _model(tmp_path / "outside", "mmproj.gguf")
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [root])
    with pytest.raises(HTTPException) as exc:
        server._validated_config_paths({"model": str(model), "mmproj": str(outside_mmproj)})
    assert exc.value.status_code == 403

    cfgs = [{"id": "one", "model": str(outside_mmproj), "backend": "vanilla"},
            {"id": "two", "model": str(model), "backend": "vanilla"}]
    monkeypatch.setattr(server.config_store, "read_all_configs", lambda: cfgs)
    with pytest.raises(HTTPException) as exc:
        server.launch_router(server.RouterLaunchRequest(ids=["one", "two"]))
    assert exc.value.status_code == 403

    session = server.LaunchSession({"model": str(outside_mmproj), "backend": "vanilla"})
    session.done = False
    with server._SESSIONS_LOCK:
        server._SESSIONS[session.id] = session
    try:
        with pytest.raises(HTTPException) as exc:
            server.launch_restart(session.id)
        assert exc.value.status_code == 403
    finally:
        with server._SESSIONS_LOCK:
            server._SESSIONS.pop(session.id, None)


def test_router_restart_revalidates_members_and_updates_synthetic_models(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    first = _model(root, "owner/repo/first.gguf")
    second = _model(root, "owner/repo/second.gguf")
    _fake_binaries(monkeypatch, tmp_path)
    monkeypatch.setattr(server.settings_store, "get_scan_paths", lambda: [root])
    member_cfgs = [
        {"model": str(first), "backend": "vanilla"},
        {"model": str(second), "backend": "vanilla"},
    ]
    session = server.LaunchSession({"router": True, "models": ["stale"]}, member_cfgs=member_cfgs)
    session.done = False
    called = []
    monkeypatch.setattr(session.handle, "request_restart", lambda: called.append(True) or True)
    with server._SESSIONS_LOCK:
        server._SESSIONS[session.id] = session
    try:
        assert server.launch_restart(session.id) == {"ok": True}
        assert called == [True]
        assert session.cfg["models"] == [str(first.resolve()), str(second.resolve())]
    finally:
        with server._SESSIONS_LOCK:
            server._SESSIONS.pop(session.id, None)

    outside = _model(tmp_path / "outside", "bad.gguf")
    blocked = server.LaunchSession(
        {"router": True, "models": [str(first), str(outside)]},
        member_cfgs=[member_cfgs[0], {"model": str(outside), "backend": "vanilla"}],
    )
    blocked.done = False
    called = []
    monkeypatch.setattr(blocked.handle, "request_restart", lambda: called.append(True) or True)
    with server._SESSIONS_LOCK:
        server._SESSIONS[blocked.id] = blocked
    try:
        with pytest.raises(HTTPException) as exc:
            server.launch_restart(blocked.id)
        assert exc.value.status_code == 403
        assert called == []
    finally:
        with server._SESSIONS_LOCK:
            server._SESSIONS.pop(blocked.id, None)


def test_hf_plan_validates_repo_subdir_and_remote_paths(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr("api.core.hf.hf_list_files", lambda *args: [])
    plan = hf.plan_download(
        "owner/repo", "main", ["nested/model.gguf"], "quant/a", root,
    )
    assert plan["root"] == str(root.resolve())
    assert plan["items"][0]["dest"].endswith("owner/repo/quant/a/model.gguf")
    for repo, subdir, rel in [
        ("owner/repo/extra", None, ["model.gguf"]),
        ("owner/repo", "../escape", ["model.gguf"]),
        ("owner/repo", None, ["../escape.gguf"]),
        ("owner/repo", None, [r"C:\\escape.gguf"]),
    ]:
        with pytest.raises(path_policy.MalformedPath):
            hf.plan_download(repo, "main", rel, subdir, root)


def test_settings_post_is_loopback_only():
    payload = {"model_paths": [], "backend_paths": {}}
    remote = TestClient(server.app, client=("203.0.113.9", 50000))
    assert remote.post("/api/settings", json=payload).status_code == 403


def test_settings_post_invalid_model_path_is_400(tmp_path):
    missing = tmp_path / "missing-model-root"
    client = TestClient(server.app, client=("127.0.0.1", 50000))
    assert client.post(
        "/api/settings",
        json={"model_paths": [str(missing)], "backend_paths": {}},
    ).status_code == 400


def test_settings_save_rejects_invalid_dirs_without_corrupting_file(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model_paths": ["/keep"]}))
    with pytest.raises(path_policy.PathPolicyError):
        settings_store.save_settings({"model_paths": [str(tmp_path / "missing")], "backend_paths": {}}, settings)
    assert json.loads(settings.read_text()) == {"model_paths": ["/keep"]}
    with pytest.raises(path_policy.PathPolicyError):
        settings_store.save_settings({"model_paths": ["relative-models"], "backend_paths": {}}, settings)
    assert json.loads(settings.read_text()) == {"model_paths": ["/keep"]}


def test_mcp_default_is_absent_from_routes_and_openapi(monkeypatch):
    monkeypatch.delenv("LLM_LAUNCHER_ENABLE_MCP", raising=False)
    reloaded = importlib.reload(server)
    assert reloaded.MCP_ENABLED is False
    assert not any(r.path.startswith("/api/mcp") for r in reloaded.app.routes)
    assert "/api/mcp" not in reloaded.app.openapi()["paths"]
    assert TestClient(reloaded.app).get("/api/options").json()["features"]["mcp"] is False


def test_mcp_enabled_loopback_and_remote_guard_without_spawn(monkeypatch):
    monkeypatch.setenv("LLM_LAUNCHER_ENABLE_MCP", "1")
    monkeypatch.delenv("LLM_LAUNCHER_ALLOW_REMOTE_MCP", raising=False)
    reloaded = importlib.reload(server)
    assert reloaded.MCP_ENABLED is True
    assert "/api/mcp" in reloaded.app.openapi()["paths"]
    assert TestClient(reloaded.app, client=("127.0.0.1", 50000)).get("/api/mcp").status_code == 200
    assert TestClient(reloaded.app, client=("203.0.113.9", 50000)).get("/api/mcp").status_code == 403
    assert TestClient(reloaded.app, client=("127.0.0.1", 50000)).get("/api/options").json()["features"]["mcp"] is True
    assert TestClient(reloaded.app, client=("203.0.113.9", 50000)).get("/api/options").json()["features"]["mcp"] is False

    monkeypatch.setenv("LLM_LAUNCHER_ALLOW_REMOTE_MCP", "1")
    reloaded = importlib.reload(server)
    assert TestClient(reloaded.app, client=("203.0.113.9", 50000)).get("/api/mcp").status_code == 200
    assert TestClient(reloaded.app, client=("203.0.113.9", 50000)).get("/api/options").json()["features"]["mcp"] is True
