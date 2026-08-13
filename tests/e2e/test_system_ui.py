"""Pure unit tests for the system-ui scenario helpers.

No Patchright, browser, backend, network request or filesystem mutation is
used here.  The real browser path belongs to ``scenarios.system_ui.run``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


E2E_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(E2E_DIR))

from scenarios import system_ui  # noqa: E402


class _FakePage:
    def __init__(self) -> None:
        self.handlers = {}

    def on(self, name, handler):
        self.handlers[name] = handler


def test_api_traffic_counts_gpu_requests_and_responses_without_browser():
    traffic = system_ui.ApiTraffic()
    page = _FakePage()
    traffic.attach(page)

    request = SimpleNamespace(url="http://127.0.0.1:8420/api/gpu", post_data=None)
    response = SimpleNamespace(url=request.url, status=200)
    page.handlers["request"](request)
    page.handlers["response"](response)

    assert traffic.snapshot("/api/gpu") == {"/api/gpu": {"requests": 1, "responses": 1, "http_2xx": 1}}


def test_gpu_payload_contract_covers_aggregate_and_per_gpu_fields():
    card = {field: "N/A" for field in system_ui.GPU_API_FIELDS}
    payload = {
        "available": True,
        "gpu_count": 1,
        "gpus": [card],
        "vram_total_mib": 100,
        "vram_used_mib": None,
        "vram_free_mib": None,
    }
    checks = system_ui.gpu_payload_assertions(payload)
    assert checks["ok"] is True
    assert checks["checks"]["gpu_count_matches_cards"] is True


def test_settings_validation_allows_only_the_guard_download_root(tmp_path):
    root = tmp_path / "run-root"
    root.mkdir()
    (root / ".llm-launcher-amd-e2e").write_text("run_id=e2e-unit\n", encoding="utf-8")
    good = {"model_paths": [str(root)], "backend_paths": {"custom": str(root)}}
    reset = {"model_paths": [str(root)], "backend_paths": {}}
    assert system_ui.settings_payload_is_root_only(good, root)
    assert system_ui.settings_payload_is_root_only(reset, root)
    assert not system_ui.settings_payload_is_root_only(
        {"model_paths": [str(root)], "backend_paths": {"vanilla": str(root)}}, root,
    )
    assert not system_ui.settings_payload_is_root_only(
        {"model_paths": [str(tmp_path)], "backend_paths": {}}, root,
    )


def test_backend_badges_require_exact_api_label_set():
    api = [{"name": "vanilla", "label": "vanilla"}, {"name": "turbo", "label": "turbo"}]
    assert system_ui.backend_labels_match(api, ["turbo", "vanilla"])
    assert not system_ui.backend_labels_match(api, ["vanilla"])
    assert not system_ui.backend_labels_match(api, ["vanilla", "turbo", "fake"])


def test_remote_403_is_explicitly_not_verified_and_no_request_rule():
    reason = system_ui.remote_403_reason()
    assert "peer remoto" in reason
    assert "falsificado" in reason
    assert system_ui.MUTATING_ENDPOINTS_REQUIRED_BY_GUARD[0]["path"] == "/api/settings"


class _EvidenceChecklist:
    def __init__(self):
        self.recorded = None

    def record(self, item_id, status, **kwargs):
        self.recorded = (item_id, status, kwargs)


class _ScreenshotPage:
    def screenshot(self, path, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"png")

    def get_by_role(self, *args, **kwargs):
        class NoModal:
            def count(self):
                return 0

        return NoModal()


def test_visual_screenshot_paths_are_added_to_checklist_evidence(tmp_path):
    def evidence(name):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    checklist = _EvidenceChecklist()
    ctx = SimpleNamespace(page=_ScreenshotPage(), evidence=evidence, evidence_dir=tmp_path, checklist=checklist)

    system_ui._item(ctx, "HEADER-03", lambda: (
        "refresh visual",
        {"_evidence_paths": [system_ui._screenshot(ctx, "header-refresh-fake")]},
    ))
    assert checklist.recorded is not None
    assert "screenshots/header-refresh-fake.png" in checklist.recorded[2]["evidence"]
