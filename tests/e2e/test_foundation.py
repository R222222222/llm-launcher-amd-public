"""No-side-effect unit tests for the Phase 6 foundation."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


E2E_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(E2E_DIR))

import checklist  # noqa: E402
import critical_path  # noqa: E402
import harness  # noqa: E402
import run as e2e_run  # noqa: E402


def _guard(monkeypatch, tmp_path: Path) -> harness.MutationGuard:
    model = tmp_path / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(harness, "allowed_model_path", lambda _root: model)
    return harness.MutationGuard(tmp_path, "e2e-unit")


def test_mutation_guard_requires_allowlisted_model_and_expected_config_id(monkeypatch, tmp_path):
    guard = _guard(monkeypatch, tmp_path)
    model = guard.expected_model
    config = {"id": "e2e-critical-e2e-unit", "model": model, "context_window": 4096}
    guard.expect_config(config["id"], {"context_window": 4096})

    guard.validate("POST", "http://127.0.0.1:8420/api/configs", config)
    guard.register_config(config["id"], model)
    guard.validate("POST", "http://127.0.0.1:8420/api/launch", config)
    assert guard.ui_launch_requests == [config]
    guard.ui_launch_requests.clear()
    guard.validate("POST", "http://127.0.0.1:8420/api/launch", json.dumps(config))
    assert guard.ui_launch_requests == [config]

    with pytest.raises(harness.GuardViolation, match="owned"):
        guard.validate("POST", "http://127.0.0.1:8420/api/launch", {**config, "id": "production"})
    with pytest.raises(harness.GuardViolation, match="owned"):
        guard.validate("POST", "http://127.0.0.1:8420/api/launch", {**config, "id": "e2e-other"})
    with pytest.raises(harness.GuardViolation, match="allowlisted"):
        guard.validate("POST", "http://127.0.0.1:8420/api/launch", {**config, "model": "/tmp/production.gguf"})


def test_mutation_guard_blocks_production_delete_and_router_namespace(monkeypatch, tmp_path):
    guard = _guard(monkeypatch, tmp_path)
    owned = "e2e-owned-e2e-unit"
    guard.register_config(owned)
    guard.register_launch("opaque123", owned)
    guard.validate("POST", "http://127.0.0.1:8420/api/launch", {"id": owned, "model": guard.expected_model})
    guard.validate("POST", "http://127.0.0.1:8420/api/launch/opaque123/cancel")
    with pytest.raises(harness.GuardViolation, match="não registrado"):
        guard.validate("POST", "http://127.0.0.1:8420/api/launch/unknown/cancel")
    with pytest.raises(harness.GuardViolation):
        guard.validate(
            "DELETE", "http://127.0.0.1:8420/api/models",
            {"model": "/path/to/llm-launcher-amd/runtime/production-models/model.gguf"},
        )
    with pytest.raises(harness.GuardViolation, match="router"):
        guard.validate("POST", "http://127.0.0.1:8420/api/launch-router", {"ids": ["production-id", "e2e-two"]})
    with pytest.raises(harness.GuardViolation, match="owned"):
        guard.validate("DELETE", "http://127.0.0.1:8420/api/configs", {"id": "e2e-other-e2e-unit"})


def test_mutation_guard_allows_estimate_many_but_blocks_unknown_post(monkeypatch, tmp_path):
    guard = _guard(monkeypatch, tmp_path)
    guard.validate("POST", "http://127.0.0.1:8420/api/estimate-many", {"items": []})
    with pytest.raises(harness.GuardViolation, match="mutação não autorizada"):
        guard.validate("POST", "http://127.0.0.1:8420/api/unknown-computation", {})


def test_mutation_guard_scopes_settings_to_e2e_root_and_cancel_to_registered_id(monkeypatch, tmp_path):
    guard = _guard(monkeypatch, tmp_path)
    root = guard.prepare_download_root()
    good_settings = {"model_paths": [str(root)], "backend_paths": {"custom": str(root)}}
    guard.validate("POST", "http://127.0.0.1:8420/api/settings", good_settings)

    with pytest.raises(harness.GuardViolation, match="raiz de download"):
        guard.validate("POST", "http://127.0.0.1:8420/api/settings", {"model_paths": [str(tmp_path)], "backend_paths": {}})
    with pytest.raises(harness.GuardViolation, match="não registrado"):
        guard.validate("POST", "http://127.0.0.1:8420/api/hf/download/unknown/cancel")

    guard.register_download("download-e2e-1")
    guard.validate("POST", "http://127.0.0.1:8420/api/hf/download/download-e2e-1/cancel")


def test_guarded_api_registers_config_and_opaque_launch_response(monkeypatch, tmp_path):
    guard = _guard(monkeypatch, tmp_path)
    config_id = "e2e-api-e2e-unit"
    model = guard.expected_model
    guard.expect_config(config_id, {})

    class Response:
        ok = True
        status = 200

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

        def text(self):
            return ""

    class RequestContext:
        def post(self, url, data=None, **_kwargs):
            if url.endswith("/api/configs"):
                return Response({"config": {"id": config_id, "model": model}})
            return Response({"launch_id": "opaque-api-123"})

    api = harness.GuardedAPI(RequestContext(), guard)
    api.post("/api/configs", data={"id": config_id, "model": model})
    assert guard.owned_configs[config_id] == model
    api.post("/api/launch", data={"id": config_id, "model": model})
    assert guard.owned_launches["opaque-api-123"] == (config_id,)
    guard.validate("POST", "http://127.0.0.1:8420/api/launch/opaque-api-123/restart")
    with pytest.raises(harness.GuardViolation):
        guard.register_launch("bad/id", config_id)


def test_settings_guard_preserves_baseline_model_and_non_custom_backend_paths(monkeypatch, tmp_path):
    (tmp_path / "app_settings.json").write_text(json.dumps({
        "model_paths": ["/models/a", "/models/b"],
        "backend_paths": {"vanilla": "/backends/vanilla", "custom": "/old/custom"},
    }), encoding="utf-8")
    guard = _guard(monkeypatch, tmp_path)
    root = guard.prepare_download_root()
    good = {
        "model_paths": ["/models/a", "/models/b", str(root)],
        "backend_paths": {"vanilla": "/backends/vanilla", "custom": str(root)},
    }
    guard.validate("POST", "http://127.0.0.1:8420/api/settings", good)
    with pytest.raises(harness.GuardViolation, match="baseline"):
        guard.validate("POST", "http://127.0.0.1:8420/api/settings", {
            **good, "model_paths": ["/models/b", str(root)],
        })
    with pytest.raises(harness.GuardViolation, match="backend baseline"):
        guard.validate("POST", "http://127.0.0.1:8420/api/settings", {
            **good, "backend_paths": {"vanilla": "/backends/other", "custom": str(root)},
        })


def test_backend_process_accepts_only_mcp_env_and_forces_remote_zero(monkeypatch, tmp_path):
    python = tmp_path / "app" / ".venv" / "bin" / "python"
    script = tmp_path / "app" / "api" / "server.py"
    python.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(harness, "assert_ports_free", lambda: None)
    captured = {}

    class FakeProcess:
        pid = 12345

    def fake_popen(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(harness.subprocess, "Popen", fake_popen)
    backend = harness.BackendProcess(tmp_path, tmp_path / "backend.log")
    backend.start({"LLM_LAUNCHER_ENABLE_MCP": "1", "LLM_LAUNCHER_ALLOW_REMOTE_MCP": "0"})
    assert captured["kwargs"]["env"]["LLM_LAUNCHER_ENABLE_MCP"] == "1"
    assert captured["kwargs"]["env"]["LLM_LAUNCHER_ALLOW_REMOTE_MCP"] == "0"
    assert captured["kwargs"]["env"]["LLM_LAUNCHER_HOST"] == "127.0.0.1"
    assert captured["kwargs"]["env"]["LLM_LAUNCHER_LLAMA_HOST"] == "127.0.0.1"
    assert backend._stream is not None
    backend._stream.close()
    with pytest.raises(harness.HarnessError, match="não allowlisted"):
        backend.start({"NOT_MCP": "1"})
    with pytest.raises(harness.HarnessError, match="remote"):
        backend.start({"LLM_LAUNCHER_ALLOW_REMOTE_MCP": "1"})


def test_runner_report_failure_keeps_emergency_adjudication(tmp_path, monkeypatch):
    runner = e2e_run.E2ERunner("e2e-report-failure")
    runner.evidence_dir = tmp_path

    def fail_write(*args, **kwargs):
        raise OSError("simulated report write failure")

    monkeypatch.setattr(runner.checklist, "write_json", fail_write)
    assert runner._write_report_safely() is False
    assert (tmp_path / "report-error.txt").read_text(encoding="utf-8").strip() == "simulated report write failure"
    emergency = json.loads((tmp_path / "checklist-emergency.json").read_text(encoding="utf-8"))
    assert emergency["validated"] is False
    assert "CP-01" in emergency["items"]


def test_checklist_rejects_missing_evidence_or_reason_and_gate_is_incomplete():
    report = checklist.Checklist("e2e-check")
    with pytest.raises(ValueError, match="PASS sem evidência"):
        report.record("CP-01", "PASS", observed="ok")
    with pytest.raises(ValueError, match="FAIL sem evidência"):
        report.record("CP-01", "FAIL", observed="bad", reason="erro")
    report.record("CP-01", "NÃO VERIFICADO", observed="")
    with pytest.raises(RuntimeError, match="sem motivo"):
        report.as_dict()

    report = checklist.Checklist("e2e-incomplete")
    report.mark_unimplemented()
    for item in checklist.CHECKLIST_ITEMS:
        if item.critical:
            report.record(item.id, "PASS", observed="synthetic evidence", evidence=[f"{item.id}.json"])
    payload = report.as_dict()
    assert payload["suite_complete"] is False
    assert payload["gate_6"] == "FAIL"
    assert all(
        report.results[item.id].reason == "cenário ainda não implementado nesta fundação"
        for item in checklist.CHECKLIST_ITEMS
        if not item.critical
    )


def test_checklist_file_gate_rejects_label_only_pass(tmp_path):
    report = checklist.Checklist("e2e-evidence")
    report.mark_dependents_unverified("not exercised")
    report.record("CP-01", "PASS", observed="ok", evidence=["API /api/gpu"])
    with pytest.raises(RuntimeError, match="arquivo"):
        report.as_dict(tmp_path)
    artifact = tmp_path / "gpu.json"
    artifact.write_text("{}\n", encoding="utf-8")
    report.record("CP-01", "PASS", observed="ok", evidence=["API /api/gpu", "gpu.json"])
    assert report.as_dict(tmp_path)["items"]["CP-01"]["status"] == "PASS"


def test_checklist_markdown_escapes_newlines_and_pipes(tmp_path):
    report = checklist.Checklist("e2e-markdown")
    report.mark_dependents_unverified("not exercised")
    artifact = tmp_path / "failure.txt"
    artifact.write_text("x\n", encoding="utf-8")
    report.record("CP-01", "PASS", observed="line1\nline2 | safe", evidence=["failure.txt"])
    report.write_markdown(tmp_path / "CHECKLIST.md", evidence_dir=tmp_path)
    markdown = (tmp_path / "CHECKLIST.md").read_text(encoding="utf-8")
    assert "line1<br>line2 \\| safe" in markdown


def test_gpu_bracket_accepts_mutable_used_and_preserves_card_sums():
    before = {"gpu_count": 1, "vram_total_mib": 100, "vram_used_mib": 10, "cards": [{"total_mib": 100, "used_mib": 10}]}
    after = {"gpu_count": 1, "vram_total_mib": 100, "vram_used_mib": 14, "cards": [{"total_mib": 100, "used_mib": 14}]}
    good = critical_path.gpu_coherence_from_snapshots(
        before, {"gpu_count": 1, "vram_total_mib": 100, "vram_used_mib": 12}, after,
    )
    assert good["ok"] is True

    bad = critical_path.gpu_coherence_from_snapshots(
        before, {"gpu_count": 1, "vram_total_mib": 101, "vram_used_mib": 20}, after,
    )
    assert bad["checks"]["vram_total_mib"] is False
    assert bad["checks"]["vram_used_mib"] is False


def test_runtime_snapshot_restores_bytes_mode_and_absence(tmp_path):
    existing = tmp_path / "last_config.json"
    existing.write_bytes(b"before\n")
    existing.chmod(0o640)
    state = harness.RuntimeState(tmp_path)
    state.snapshot()
    existing.write_bytes(b"changed\n")
    created = tmp_path / "app_settings.json"
    created.write_bytes(b"created\n")

    with pytest.raises(harness.HarnessError, match="backend"):
        state.restore(lambda: False)
    state.restore(lambda: True)
    assert existing.read_bytes() == b"before\n"
    assert (existing.stat().st_mode & 0o7777) == 0o640
    assert not created.exists()


def test_runtime_restore_refuses_current_symlink(tmp_path):
    existing = tmp_path / "last_config.json"
    target = tmp_path / "target"
    existing.write_bytes(b"before\n")
    target.write_bytes(b"target\n")
    state = harness.RuntimeState(tmp_path)
    state.snapshot()
    existing.unlink()
    existing.symlink_to(target)
    with pytest.raises(harness.HarnessError, match="symlink"):
        state.restore(lambda: True)
    existing.unlink()
    state.restore(lambda: True)
    assert existing.read_bytes() == b"before\n"


def test_cleanup_failure_is_fail_closed_and_persists_inventory(tmp_path):
    runner = e2e_run.E2ERunner("e2e-cleanup-failure")
    runner.evidence_dir = tmp_path
    runner.root = tmp_path
    runner.ggufs = harness.GGUFInventory(tmp_path)
    runner.ggufs.capture()
    runner.runtime = harness.RuntimeState(tmp_path)

    class FailingBackend:
        process = None

        def stop(self):
            raise RuntimeError("stop failed")

        def stopped(self):
            return True

    runner.backend = FailingBackend()  # type: ignore[assignment]
    assert runner._cleanup() is False
    assert runner.cleaned is False
    assert runner.checklist.suite_complete is False
    assert runner.checklist.gate_pass() is False
    cleanup = json.loads((tmp_path / "cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["ok"] is False
    assert any("backend stop" in error for error in cleanup["errors"])
    assert "gguf_inventory_before" in cleanup
    assert "gguf_inventory_after" in cleanup
