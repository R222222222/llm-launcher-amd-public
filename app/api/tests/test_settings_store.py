"""Testes focados da persistência de Settings e do scan de modelos."""
import sys
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_ROOT))

from api.core import constants, models_repo, settings_store  # noqa: E402


def test_settings_file_is_absolute_at_fork_root():
    assert settings_store.SETTINGS_FILE.is_absolute()
    assert settings_store.SETTINGS_FILE == constants.APP_SETTINGS_FILE
    assert settings_store.SETTINGS_FILE == constants._REPO_ROOT / "app_settings.json"


def test_settings_save_read_and_scan_survive_cwd_changes(tmp_path, monkeypatch):
    settings_path = tmp_path / "runtime-settings.json"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "tiny-checkpoint.gguf"
    model.write_bytes(b"GGUF test fixture")

    app_dir = Path(__file__).resolve().parents[2]
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()

    monkeypatch.chdir(app_dir)
    settings_store.save_settings({"model_paths": [str(model_dir)], "backend_paths": {}}, settings_path)

    monkeypatch.chdir(other_cwd)
    saved = settings_store.read_settings(settings_path)
    scan_paths = settings_store.get_scan_paths(settings_path)
    collected = models_repo.collect_models(scan_paths)

    assert saved["model_paths"] == [str(model_dir)]
    assert scan_paths == [model_dir]
    assert collected == [model]
