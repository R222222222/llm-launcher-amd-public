"""Unit tests for the repository profile seed manifest validation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "seed-profiles.py"
SPEC = importlib.util.spec_from_file_location("seed_profiles", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
seed_profiles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seed_profiles)


def _load_with_manifest(tmp_path: Path, profiles: list[dict]):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    manifest = repo / "seed-profiles.json"
    manifest.write_text(
        json.dumps({"version": 1, "profiles": profiles}), encoding="utf-8",
    )
    setattr(seed_profiles, "REPO_ROOT", repo)
    setattr(seed_profiles, "REPO_ROOT_RESOLVED", repo.resolve())
    setattr(seed_profiles, "MANIFEST", manifest)
    return repo, seed_profiles._load_profiles()


def _profile(profile_id: str, model: str = "models/model.gguf", **extra) -> dict:
    return {"id": profile_id, "model": model, **extra}


def test_loads_four_profiles_and_resolves_valid_mmproj(tmp_path):
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "model.gguf").write_bytes(b"GGUF")
    (repo / "models" / "mmproj.gguf").write_bytes(b"GGUF")

    profiles = [
        _profile("one"),
        _profile("two"),
        _profile("three"),
        _profile("four", mmproj="models/mmproj.gguf"),
    ]
    loaded_repo, loaded = _load_with_manifest(tmp_path, profiles)

    assert loaded_repo == repo
    assert [profile["id"] for profile in loaded] == ["one", "two", "three", "four"]
    assert loaded[-1]["mmproj"] == str((repo / "models/mmproj.gguf").resolve())


def test_empty_profile_list_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="version=1"):
        _load_with_manifest(tmp_path, [])


def test_duplicate_profile_id_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "model.gguf").write_bytes(b"GGUF")
    profiles = [_profile("same"), _profile("same")]
    with pytest.raises(RuntimeError, match="id duplicado"):
        _load_with_manifest(tmp_path, profiles)


@pytest.mark.parametrize(
    "bad_model",
    [
        "../outside/model.gguf",
        "/absolute/model.gguf",
        "models/missing.gguf",
        "models/model.txt",
    ],
)
def test_invalid_model_paths_are_rejected(tmp_path, bad_model):
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "model.txt").write_bytes(b"not GGUF")
    with pytest.raises(RuntimeError):
        _load_with_manifest(tmp_path, [_profile("bad", model=bad_model)])


def test_invalid_mmproj_paths_are_rejected(tmp_path):
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "model.gguf").write_bytes(b"GGUF")
    (repo / "models" / "mmproj.txt").write_bytes(b"not GGUF")
    with pytest.raises(RuntimeError):
        _load_with_manifest(
            tmp_path,
            [_profile("bad", mmproj="models/mmproj.txt")],
        )
