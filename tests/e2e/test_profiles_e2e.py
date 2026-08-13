"""Unit tests for the parametrized model allowlist and seeded-profile validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

E2E_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(E2E_DIR))

import harness  # noqa: E402
from harness import GuardViolation, HarnessError, MutationGuard  # noqa: E402


def _model(root: Path, relative: str = "runtime/fase4-models/qwen2.5-1.5b-instruct-q4_k_m.gguf") -> Path:
    model = root / relative
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"GGUF")
    return model


def _guard(root: Path, monkeypatch) -> MutationGuard:
    _model(root)
    monkeypatch.setenv("E2E_MODEL_PATH", "")
    return MutationGuard(root, "e2e-unit-test")


def _seed_manifest(root: Path) -> dict:
    small = _model(root, "runtime/fase4-models/small-test.gguf")
    vis = _model(root, "runtime/production-models/vis-test.gguf")
    manifest = {
        "version": 1,
        "profiles": [
            {
                "id": "chat-ferramentas-ornith",
                "model": "runtime/production-models/vis-test.gguf",
                "backend": "vanilla",
                "context_window": 65536,
                "kv_cache": "q8_0",
                "flash_attn": True,
                "gpu_layers": 99,
                "parallel_slots": 4,
                "mmproj": None,
            },
            {
                "id": "ocr-glm",
                "model": "runtime/fase4-models/small-test.gguf",
                "mmproj": "runtime/production-models/vis-test.gguf",
                "backend": "vanilla",
                "context_window": 131072,
                "kv_cache": "q8_0",
                "flash_attn": True,
                "gpu_layers": 99,
                "parallel_slots": 1,
                "sampler_source": "manual",
                "temp": 0.1,
                "top_k": 1,
            },
        ],
    }
    manifest_path = root / "docs" / "profiles" / "seed-profiles.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {"small": str(small), "vis": str(vis)}


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


# ─── parametrized allowlist ────────────────────────────────────────────────────

def test_allowed_model_path_default(tmp_path, monkeypatch):
    model = _model(tmp_path)
    monkeypatch.setenv("E2E_MODEL_PATH", "")
    assert harness.allowed_model_path(tmp_path) == model.resolve()


def test_allowed_model_path_missing_default_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("E2E_MODEL_PATH", "")
    with pytest.raises(HarnessError):
        harness.allowed_model_path(tmp_path)


def test_allowed_model_path_env_override_absolute(tmp_path, monkeypatch):
    model = _model(tmp_path, "elsewhere/custom.gguf")
    monkeypatch.setenv("E2E_MODEL_PATH", str(model))
    assert harness.allowed_model_path(tmp_path) == model.resolve()


def test_allowed_model_path_env_override_relative(tmp_path, monkeypatch):
    model = _model(tmp_path, "runtime/fase4-models/custom.gguf")
    monkeypatch.setenv("E2E_MODEL_PATH", "runtime/fase4-models/custom.gguf")
    assert harness.allowed_model_path(tmp_path) == model.resolve()


def test_allowed_model_path_env_override_rejects_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("E2E_MODEL_PATH", str(tmp_path / "missing.gguf"))
    with pytest.raises(HarnessError):
        harness.allowed_model_path(tmp_path)


def test_allowed_model_path_env_override_rejects_non_gguf(tmp_path, monkeypatch):
    bad = tmp_path / "model.txt"
    bad.write_text("not a gguf", encoding="utf-8")
    monkeypatch.setenv("E2E_MODEL_PATH", str(bad))
    with pytest.raises(HarnessError):
        harness.allowed_model_path(tmp_path)


def test_allowed_model_path_env_override_rejects_symlink(tmp_path, monkeypatch):
    real = _model(tmp_path, "runtime/fase4-models/real.gguf")
    link = tmp_path / "link.gguf"
    link.symlink_to(real)
    monkeypatch.setenv("E2E_MODEL_PATH", str(link))
    with pytest.raises(HarnessError):
        harness.allowed_model_path(tmp_path)


# ─── profile loading ───────────────────────────────────────────────────────────

def test_guard_load_profile_configs_resolves_paths(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    paths = _seed_manifest(tmp_path)
    profiles = guard.load_profile_configs()
    assert set(profiles) == {"chat-ferramentas-ornith", "ocr-glm"}
    assert profiles["ocr-glm"]["model"] == paths["small"]
    assert profiles["ocr-glm"]["mmproj"] == paths["vis"]
    assert profiles["ocr-glm"]["temp"] == 0.1
    assert profiles["ocr-glm"]["top_k"] == 1
    assert profiles["ocr-glm"]["sampler_source"] == "manual"
    assert profiles["chat-ferramentas-ornith"]["mmproj"] is None


def test_guard_load_profile_configs_rejects_traversal(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    manifest = tmp_path / "docs" / "profiles" / "seed-profiles.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "version": 1,
        "profiles": [{"id": "evil", "model": "../outside.gguf", "mmproj": None}],
    }), encoding="utf-8")
    with pytest.raises(GuardViolation):
        guard.load_profile_configs()


def test_guard_load_profile_configs_rejects_missing_gguf(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    manifest = tmp_path / "docs" / "profiles" / "seed-profiles.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "version": 1,
        "profiles": [{"id": "ghost", "model": "runtime/fase4-models/absent.gguf", "mmproj": None}],
    }), encoding="utf-8")
    with pytest.raises(GuardViolation):
        guard.load_profile_configs()


def test_guard_load_profile_configs_rejects_duplicate_ids(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    small = _model(tmp_path, "runtime/fase4-models/a.gguf")
    manifest = tmp_path / "docs" / "profiles" / "seed-profiles.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "version": 1,
        "profiles": [
            {"id": "dup", "model": str(small), "mmproj": None},
            {"id": "dup", "model": str(small), "mmproj": None},
        ],
    }), encoding="utf-8")
    with pytest.raises(GuardViolation):
        guard.load_profile_configs()


# ─── profile launch validation ─────────────────────────────────────────────────

def _persisted_body(profile_id: str, paths: dict) -> dict:
    return {
        "id": profile_id,
        "model": paths["small"],
        "mmproj": paths["vis"],
        "backend": "vanilla",
        "context_window": 131072,
        "kv_cache": "q8_0",
        "flash_attn": True,
        "gpu_layers": 99,
        "parallel_slots": 1,
        "sampler_source": "manual",
        "temp": 0.1,
        "top_k": 1,
        "batch_size": 2048,
        "max_tokens": -1,
    }


def test_guard_accepts_profile_launch(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    paths = _seed_manifest(tmp_path)
    guard.load_profile_configs()
    body = _persisted_body("ocr-glm", paths)
    guard.validate("POST", "http://127.0.0.1:8420/api/launch", body)
    assert guard.ui_launch_requests == [body]


def test_guard_rejects_profile_launch_with_wrong_model(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    paths = _seed_manifest(tmp_path)
    guard.load_profile_configs()
    body = _persisted_body("ocr-glm", paths)
    body["model"] = paths["vis"]
    with pytest.raises(GuardViolation):
        guard.validate("POST", "http://127.0.0.1:8420/api/launch", body)


def test_guard_rejects_profile_launch_with_template_sampling(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    paths = _seed_manifest(tmp_path)
    guard.load_profile_configs()
    body = _persisted_body("ocr-glm", paths)
    body["temp"] = 0.6
    body["top_k"] = 20
    body["sampler_source"] = "template"
    with pytest.raises(GuardViolation):
        guard.validate("POST", "http://127.0.0.1:8420/api/launch", body)


def test_guard_rejects_unknown_profile_id_launch(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    _seed_manifest(tmp_path)
    guard.load_profile_configs()
    body = {"id": "not-a-profile", "model": "whatever", "mmproj": None}
    with pytest.raises(GuardViolation):
        guard.validate("POST", "http://127.0.0.1:8420/api/launch", body)


def test_guard_profile_delete_prohibited(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    paths = _seed_manifest(tmp_path)
    guard.load_profile_configs()
    # 27B/35B/profile models are never deletable: only the allowlisted model is.
    with pytest.raises(GuardViolation):
        guard.validate("DELETE", "http://127.0.0.1:8420/api/models", {"model": paths["small"]})
    expected = harness.allowed_model_path(guard.root)
    guard.validate("DELETE", "http://127.0.0.1:8420/api/models", {"model": str(expected)})


def test_guard_profile_cancel_requires_registration(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    paths = _seed_manifest(tmp_path)
    guard.load_profile_configs()
    launch_id = guard.register_profile_launch_response(
        _StubResponse({"launch_id": "abc123def456"}), "ocr-glm",
    )
    assert guard.owned_profile_launches == {launch_id: "ocr-glm"}
    guard.validate("POST", f"http://127.0.0.1:8420/api/launch/{launch_id}/cancel", {})
    guard.validate("POST", f"http://127.0.0.1:8420/api/launch/{launch_id}/restart", {})
    with pytest.raises(GuardViolation):
        guard.validate("POST", "http://127.0.0.1:8420/api/launch/otherid123/cancel", {})


def test_guard_profile_launch_response_rejects_unknown_profile(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    _seed_manifest(tmp_path)
    guard.load_profile_configs()
    with pytest.raises(GuardViolation):
        guard.register_profile_launch_response(_StubResponse({"launch_id": "abc123def456"}), "ghost")


def test_guard_profile_launch_response_rejects_missing_launch_id(tmp_path, monkeypatch):
    guard = _guard(tmp_path, monkeypatch)
    _seed_manifest(tmp_path)
    guard.load_profile_configs()
    with pytest.raises(GuardViolation):
        guard.register_profile_launch_response(_StubResponse({}), "ocr-glm")
