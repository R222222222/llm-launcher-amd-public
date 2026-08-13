"""Side-effect-free unit coverage for the Models/Download safety helpers."""
from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


E2E_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(E2E_DIR))

from scenarios import models_download  # noqa: E402
import checklist  # noqa: E402


def _small_model(root: Path, data: bytes = b"GGUF") -> Path:
    model = root / models_download.ALLOWED_MODEL_RELATIVE
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(data)
    return model


def test_delete_plan_allows_only_exact_model_and_rejects_part(tmp_path: Path):
    model = _small_model(tmp_path)
    production = tmp_path / "runtime" / "production-models"
    production.mkdir(parents=True)

    allowed = models_download.validate_delete_plan(
        {"files": [str(model)]}, model, production,
    )
    assert allowed == (model.resolve(),)

    with pytest.raises(models_download.DeletePlanViolation, match=r"\.part"):
        models_download.validate_delete_plan(
            {"files": [str(model), f"{model}.part"]}, model, production,
        )

    with pytest.raises(models_download.DeletePlanViolation):
        models_download.validate_delete_plan(
            {"files": [str(model), str(tmp_path / "mmproj.gguf")]}, model, production,
        )
    with pytest.raises(models_download.DeletePlanViolation, match="production"):
        models_download.validate_delete_plan(
            {"files": [str(model), str(production / "other.gguf")]}, model, production,
        )


def test_hardlink_manifest_is_inode_hash_size_and_nlink_checked_and_restored(tmp_path: Path):
    model = _small_model(tmp_path, b"stable GGUF bytes\n")
    manifest = models_download.create_hardlink_manifest(model, root=tmp_path, run_id="e2e-unit")
    backup = models_download.recovery_backup_path(model)
    manifest_path = models_download.recovery_manifest_path(model)
    assert backup.name == "qwen2.5-1.5b-instruct-q4_k_m.gguf.e2e-backup"
    assert model.stat().st_ino == backup.stat().st_ino == manifest["inode"]
    assert model.stat().st_size == backup.stat().st_size == manifest["size"]
    assert model.stat().st_nlink >= 2
    assert manifest["root"] == str(tmp_path.resolve())
    assert manifest["run_id"] == "e2e-unit"
    assert manifest_path.is_file()
    models_download.validate_hardlink_manifest(manifest, model)

    model.unlink()
    models_download.restore_hardlink_manifest(manifest)
    assert model.read_bytes() == b"stable GGUF bytes\n"
    assert model.stat().st_nlink == 1
    assert not backup.exists()
    assert not manifest_path.exists()


def test_pending_recovery_model_missing_stale_and_inconsistent(tmp_path: Path):
    model = _small_model(tmp_path, b"stable")
    manifest = models_download.create_hardlink_manifest(model, root=tmp_path, run_id="e2e-recover")
    backup = models_download.recovery_backup_path(model)
    sidecar = models_download.recovery_manifest_path(model)

    model.unlink()
    recovered = models_download.recover_pending_model(tmp_path)
    assert recovered["status"] == "recovered"
    assert model.read_bytes() == b"stable"
    assert not backup.exists() and not sidecar.exists()

    manifest = models_download.create_hardlink_manifest(model, root=tmp_path, run_id="e2e-stale")
    cleaned = models_download.recover_pending_model(tmp_path)
    assert cleaned["status"] == "cleaned"
    assert model.exists() and not backup.exists() and not sidecar.exists()

    manifest = models_download.create_hardlink_manifest(model, root=tmp_path, run_id="e2e-bad")
    model.unlink()
    model.write_bytes(b"different")
    with pytest.raises(models_download.HarnessError, match="diverge|inconsistente"):
        models_download.recover_pending_model(tmp_path)
    assert model.read_bytes() == b"different"
    assert backup.exists() and sidecar.exists()


def test_recovery_rejects_other_gguf_manifest_and_preserves_bytes(tmp_path: Path):
    model = _small_model(tmp_path, b"small-original")
    other = tmp_path / "runtime" / "production-models" / "other.gguf"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"production-original")

    with pytest.raises(models_download.HarnessError, match="modelo literal"):
        models_download.create_hardlink_manifest(other, root=tmp_path, run_id="e2e-other")
    assert other.read_bytes() == b"production-original"
    assert not models_download.recovery_backup_path(other).exists()

    manifest = models_download.create_hardlink_manifest(model, root=tmp_path, run_id="e2e-manifest")
    manifest["model"] = str(other)
    models_download.recovery_manifest_path(model).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(models_download.HarnessError, match="modelo literal"):
        models_download.recover_pending_model(tmp_path)
    assert model.read_bytes() == b"small-original"
    assert other.read_bytes() == b"production-original"
    assert models_download.recovery_backup_path(model).exists()
    assert models_download.recovery_manifest_path(model).exists()


def test_external_off_marks_downloads_nv_without_page_api_or_filesystem_effects():
    calls: list[str] = []

    class Checklist:
        def record(self, item_id, status, **kwargs):
            calls.append(f"record:{item_id}:{status}:{kwargs.get('reason', '')}")

    ctx = SimpleNamespace(checklist=Checklist(), current_item="MODELS-01")
    models_download._mark_downloads_external_off(ctx)  # type: ignore[arg-type]

    assert len(calls) == 9
    assert all(":NÃO VERIFICADO:" in call for call in calls)
    assert all("external_hf=False" in call and "sem efeitos" in call for call in calls)


def test_cancel_quiet_fails_when_new_artifact_appears(tmp_path: Path, monkeypatch):
    destination = tmp_path / "qwen.gguf"
    plan = {"items": [{"dest": str(destination)}]}

    original = models_download._download_snapshots
    calls = 0

    def snapshots(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.with_name(destination.name + ".part").write_bytes(b"late")
        return original(*args, **kwargs)

    monkeypatch.setattr(models_download, "_download_snapshots", snapshots)
    ctx = SimpleNamespace()
    with pytest.raises(models_download.HarnessError, match="artefato novo"):
        models_download._wait_cancel_quiet(ctx, plan, tmp_path, timeout=1.1)  # type: ignore[arg-type]


def test_cancel_quiet_window_accepts_stable_local_session(tmp_path: Path):
    plan = {"items": [{"dest": str(tmp_path / "cancel.gguf")}]}
    evidence = models_download._wait_cancel_quiet(SimpleNamespace(), plan, tmp_path, timeout=0.01)  # type: ignore[arg-type]
    assert evidence["quiet"] is True
    assert evidence["quiet_seconds"] == 0.01


def test_models03_is_nv_when_vision_fixture_is_absent():
    models = [{"path": "/models/qwen.gguf", "is_thinking": False, "is_mtp": False, "mmproj": None}]
    status, reason, evidence = models_download.evaluate_models03_flags(
        models, {"/models/qwen.gguf": {"thinking": False, "mtp": False, "vision": False}},
    )
    assert status == "NÃO VERIFICADO"
    assert "vision" in reason
    assert evidence["fixtures"]["vision"] is False


def test_record_requires_explicit_status_for_models_items(tmp_path: Path):
    def evidence(name: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    ctx = SimpleNamespace(
        evidence_dir=tmp_path,
        current_item="MODELS-01",
        evidence=evidence,
        checklist=checklist.Checklist("e2e-model-record"),
    )
    models_download._record(ctx, "MODELS-01", status="PASS", observed="listed", payload={"models": []})  # type: ignore[arg-type]
    models_download._record(ctx, "MODELS-02", status="NÃO VERIFICADO", reason="NV: no real shards")  # type: ignore[arg-type]
    models_download._record(ctx, "MODELS-04", status="PASS", observed="safe delete", payload={"delete": "allowlisted"})  # type: ignore[arg-type]
    assert ctx.checklist.results["MODELS-01"].status == "PASS"
    assert ctx.checklist.results["MODELS-02"].status == "NÃO VERIFICADO"
    assert ctx.checklist.results["MODELS-04"].status == "PASS"


def _pinned_listing() -> dict:
    return {
        "repo_id": models_download.HF_REPO,
        "requested_revision": models_download.HF_REQUESTED_REVISION,
        "revision": models_download.HF_RESOLVED_REVISION,
        "files": [{
            "path": models_download.HF_FILE,
            "size": models_download.HF_SIZE,
            "oid": models_download.HF_OID,
        }],
    }


def test_pinned_listing_and_request_validation_reject_mutable_or_wrong_metadata():
    assert models_download.validate_fixture_listing(_pinned_listing())["oid"] == models_download.HF_OID
    with pytest.raises(models_download.HarnessError, match="requested_revision"):
        models_download.validate_fixture_listing({**_pinned_listing(), "requested_revision": "main"})
    with pytest.raises(models_download.HarnessError, match="metadados"):
        models_download.validate_fixture_listing({**_pinned_listing(), "files": [{"path": models_download.HF_FILE, "size": 1, "oid": models_download.HF_OID}]})

    request = {
        "repo_id": models_download.HF_REPO,
        "revision": models_download.HF_RESOLVED_REVISION,
        "rel_paths": [models_download.HF_FILE],
        "subdir": "cancel",
        "expected_files": [{"rel": models_download.HF_FILE, "expected_size": models_download.HF_SIZE, "expected_oid": models_download.HF_OID}],
    }
    models_download.validate_download_request(request, subdir="cancel")
    with pytest.raises(models_download.HarnessError, match="pinado"):
        models_download.validate_download_request({**request, "revision": "main"}, subdir="cancel")


def test_terminal_error_is_not_completion_and_cancel_requires_join_confirmation():
    with pytest.raises(models_download.HarnessError, match="terminal esperado"):
        models_download._terminal([{"type": "error", "message": "fake"}], "done")
    models_download.validate_cancel_response({"ok": True, "joined": True, "state": "CANCELLED"})
    with pytest.raises(models_download.HarnessError, match="join"):
        models_download.validate_cancel_response({"ok": True})


def test_independent_destination_sets_are_required():
    models_download.assert_independent_destinations([
        {"items": [{"dest": "/tmp/phase2/a.gguf"}]},
        {"items": [{"dest": "/tmp/phase2/b.gguf"}]},
    ])
    with pytest.raises(models_download.HarnessError, match="compartilham"):
        models_download.assert_independent_destinations([
            {"items": [{"dest": "/tmp/phase2/a.gguf"}]},
            {"items": [{"dest": "/tmp/phase2/a.gguf"}]},
        ])
