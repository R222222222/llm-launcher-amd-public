"""Testes focados da persistência de configs e do histórico de falhas."""
import json
import sys
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_ROOT))

from api.core import config_store, constants  # noqa: E402


def test_config_and_fail_paths_are_absolute_at_fork_root():
    assert constants.CONFIG_FILE.is_absolute()
    assert constants.CONFIG_FILE == constants._REPO_ROOT / "last_config.json"
    assert constants.FAIL_HISTORY_FILE.is_absolute()
    assert constants.FAIL_HISTORY_FILE == constants._REPO_ROOT / "fail_history.jsonl"
    assert config_store.CONFIG_FILE == constants.CONFIG_FILE
    assert config_store.FAIL_HISTORY_FILE == constants.FAIL_HISTORY_FILE


def test_config_save_load_and_fail_history_survive_cwd_changes(tmp_path, monkeypatch):
    config_path = tmp_path / "last_config.json"
    fail_path = tmp_path / "fail_history.jsonl"
    app_dir = Path(__file__).resolve().parents[2]
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    cfg = {
        "model": "/tmp/test-model.gguf",
        "backend": "vanilla",
        "context_window": 4096,
    }

    monkeypatch.chdir(app_dir)
    saved = config_store.save_config(dict(cfg), config_path)
    config_store.append_fail_history(
        saved, "test_failure", "test excerpt", None, 1, fail_path,
    )

    monkeypatch.chdir(other_cwd)
    loaded = config_store.load_config(cfg["model"], cfg["backend"], config_path)
    history = [json.loads(line) for line in fail_path.read_text(encoding="utf-8").splitlines()]

    assert loaded is not None
    assert loaded["model"] == cfg["model"]
    assert loaded["backend"] == cfg["backend"]
    assert history[0]["failure"] == "test_failure"
    assert config_path.is_absolute()
    assert fail_path.is_absolute()
