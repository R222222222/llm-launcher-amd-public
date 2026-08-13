"""Pure/fake-locator tests for the Configs/Editor scenario."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


E2E_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(E2E_DIR))
sys.path.insert(0, str(E2E_DIR / "scenarios"))

import configs_editor  # noqa: E402


def _cells(kv: str = "q8_0") -> list[str]:
    return ["", "", "", "qwen2.5-1.5b-instruct-q4_k_m qwen2.5-1.5b-instruct-q4_k_m.gguf", "vanilla", "4,096", kv, "99", "—", "1"]


def test_row_matcher_requires_alias_backend_context_kv_and_slots_without_full_path():
    assert configs_editor.row_cells_match(
        _cells(),
        alias="qwen2.5-1.5b-instruct-q4_k_m",
        backend="vanilla",
        context_window=4096,
        kv_cache="q8_0",
        parallel_slots=1,
    )
    assert not configs_editor.row_cells_match(
        _cells("q5_0"),
        alias="qwen2.5-1.5b-instruct-q4_k_m",
        backend="vanilla",
        context_window=4096,
        kv_cache="q8_0",
        parallel_slots=1,
    )


class _FakeCellRow:
    def __init__(self, cells: list[str]):
        self.cells = cells

    def locator(self, selector: str):
        assert selector == "td"
        return self

    def all_inner_texts(self):
        return self.cells


class _FakeRows:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class _FakePage:
    def __init__(self, rows):
        self.rows = _FakeRows(rows)

    def get_by_role(self, role: str):
        assert role == "row"
        return self.rows


def test_find_row_refuses_multiple_candidates_without_last(monkeypatch, tmp_path):
    model = Path("qwen2.5-1.5b-instruct-q4_k_m.gguf")
    def evidence(name):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    ctx = SimpleNamespace(
        model=model,
        model_alias=model.stem,
        page=_FakePage([_FakeCellRow(_cells()), _FakeCellRow(_cells())]),
        evidence_dir=tmp_path,
        evidence=evidence,
    )
    clock = iter((0.0, 16.0))
    monkeypatch.setattr(configs_editor.time, "monotonic", lambda: next(clock))
    with pytest.raises(Exception, match="não é única"):
        configs_editor._find_row(ctx, configs_editor.config_payload(model, "e2e-row"))


def test_guard_endpoint_manifest_and_config_payload_are_e2e_scoped():
    endpoints = configs_editor.required_guard_endpoints()
    assert "POST /api/configs" in endpoints
    assert "POST /api/launch-router" in endpoints
    payload = configs_editor.config_payload(Path("/tmp/model.gguf"), "e2e-test")
    assert payload["id"].startswith("e2e-")
    assert payload["mmproj"] is None
    assert payload["mcp_servers_config"] is None


def test_prepare_grid_refreshes_react_before_find(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(configs_editor, "_save_api", lambda ctx, payload: calls.append("api-save"))
    monkeypatch.setattr(configs_editor, "_refresh_configs_grid", lambda ctx: calls.append("refresh"))
    monkeypatch.setattr(configs_editor, "_configs_grid", lambda ctx, payload: calls.append("find-grid"))
    configs_editor._prepare_grid(object(), {"id": "e2e-refresh"})  # type: ignore[arg-type]
    assert calls == ["api-save", "refresh"]


class _FakeChecklist:
    def __init__(self):
        self.records = {}

    def record(self, item_id, status, **kwargs):
        self.records[item_id] = (status, kwargs)


class _FakeCancel:
    def __init__(self, page):
        self.page = page

    def count(self):
        return 1

    def is_visible(self):
        return self.page.modal

    def click(self, **kwargs):
        self.page.events.append("config-cancel")
        self.page.modal = False

    def nth(self, index):
        assert index == 0
        return self


class _FakeModal:
    def __init__(self, page):
        self.page = page

    def count(self):
        return 1

    def is_visible(self):
        return self.page.modal

    def wait_for(self, *, state, timeout):
        assert state == "hidden"
        self.page.events.append("wait-hidden")
        assert not self.page.modal


class _GridFakePage:
    def __init__(self):
        self.modal = False
        self.events = []
        self.cancel = _FakeCancel(self)
        self.modal_locator = _FakeModal(self)

    def get_by_test_id(self, name):
        if name == "config-editor-modal":
            return self.modal_locator
        if name == "config-editor-cancel":
            return self.cancel
        raise AssertionError(name)

    def get_by_role(self, role, **kwargs):
        raise AssertionError(f"unexpected role lookup: {role}")

    def screenshot(self, path, **kwargs):
        self.events.append("screenshot")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"png")

    def wait_for_timeout(self, timeout):
        pass


def test_grid_failure_is_granular_and_does_not_replace_config01(tmp_path):
    def evidence(name):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    page = _GridFakePage()
    ctx = SimpleNamespace(page=page, evidence_dir=tmp_path, evidence=evidence, checklist=_FakeChecklist())
    configs_editor._grid_step(ctx, "CONFIG-01", lambda: ("filter ok", ["filter.json"]))

    def duplicate_timeout():
        page.modal = True
        raise TimeoutError("duplicate timeout")

    configs_editor._grid_step(
        ctx, "CONFIG-05", duplicate_timeout, failure_screenshot="configs-duplicate-failure"
    )
    assert ctx.checklist.records["CONFIG-01"][0] == "PASS"
    assert ctx.checklist.records["CONFIG-05"][0] == "FAIL"
    assert page.events.index("screenshot") < page.events.index("config-cancel")


def test_step_closes_modal_after_success(tmp_path):
    def evidence(name):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    page = _GridFakePage()
    ctx = SimpleNamespace(page=page, evidence_dir=tmp_path, evidence=evidence, checklist=_FakeChecklist())

    def opens_modal_successfully():
        page.modal = True

    assert configs_editor._step(ctx, "CONFIG-01", (), opens_modal_successfully) is True
    assert page.modal is False
    assert page.events == ["config-cancel", "wait-hidden"]


def test_register_launch_uses_response_id_and_config_ids_contract():
    class Guard:
        def __init__(self):
            self.calls = []

        def register_launch(self, launch_id, config_ids):
            self.calls.append((launch_id, config_ids))

    guard = Guard()
    ctx = SimpleNamespace(guard=guard)
    configs_editor._register_owned_launch(ctx, "e2e-launch-1", ["e2e-a", "e2e-b"])
    assert guard.calls == [("e2e-launch-1", ["e2e-a", "e2e-b"])]
    assert ctx.owned_launch_id == "e2e-launch-1"


class _FakeSaveLocator:
    def __init__(self, page, label):
        self.page = page
        self.label = label

    def click(self, **kwargs):
        self.page.calls.append((self.label, kwargs))

    def wait_for(self, **kwargs):
        self.page.calls.append(("wait", kwargs))


class _FakeSavePage:
    def __init__(self):
        self.calls = []
        self.lookups = []

    def get_by_role(self, role, *, name, exact):
        assert role == "button"
        assert exact is True
        self.lookups.append((name, exact))
        return _FakeSaveLocator(self, name)

    def get_by_text(self, text, *, exact):
        assert text == "Nova configuração"
        assert exact is True
        return _FakeSaveLocator(self, "editor-hidden")


@pytest.mark.parametrize("launch, expected", [(False, "Salvar"), (True, "Salvar e launch")])
def test_save_editor_uses_exact_button_name_for_both_save_paths(launch, expected):
    page = _FakeSavePage()
    guard = SimpleNamespace(expect_config=lambda *_args: None)
    ctx = SimpleNamespace(page=page, guard=guard)
    payload = {"id": "e2e-save-test", "backend": "vanilla"}

    configs_editor._save_editor(ctx, payload, launch=launch)

    assert page.lookups == [(expected, True)]
    assert page.calls[0] == (expected, {})
    if not launch:
        assert page.calls[1] == ("wait", {"state": "hidden", "timeout": 10_000})


class _FakeOwnedModalPage:
    def __init__(self):
        self.visible = True
        self.events = []

    class _Modal:
        def __init__(self, page):
            self.page = page

        def count(self):
            return 1

        def is_visible(self):
            return self.page.visible

        def wait_for(self, *, state, timeout):
            assert state == "hidden"
            self.page.events.append(("wait-hidden", timeout))
            assert not self.page.visible

    class _Dismiss:
        def __init__(self, page):
            self.page = page

        def count(self):
            return 1

        def click(self, **kwargs):
            self.page.events.append(("dismiss", kwargs))
            self.page.visible = False

    def get_by_test_id(self, test_id):
        if test_id == "launch-modal":
            return self._Modal(self)
        if test_id == "launch-modal-dismiss":
            return self._Dismiss(self)
        raise AssertionError(f"unexpected testid: {test_id}")


@pytest.mark.parametrize("fails", [False, True])
def test_owned_launch_teardown_dismisses_modal_on_success_and_failure(monkeypatch, fails):
    page = _FakeOwnedModalPage()
    ctx = SimpleNamespace(page=page)
    cancelled = []

    def cancel(_ctx, launch_id):
        cancelled.append(launch_id)
        if fails:
            raise RuntimeError("cancel failed")

    monkeypatch.setattr(configs_editor, "_cancel_owned_launch", cancel)
    if fails:
        with pytest.raises(RuntimeError, match="cancel failed"):
            configs_editor._teardown_owned_launch(ctx, "launch-1")
    else:
        configs_editor._teardown_owned_launch(ctx, "launch-1")

    assert cancelled == ["launch-1"]
    assert page.events[0][0] == "dismiss"
    assert page.events[1][0] == "wait-hidden"
