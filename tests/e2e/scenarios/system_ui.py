"""System, header and Settings checks for the Phase 6 E2E run.

This scenario is deliberately independent from the runner.  The runner (or an
orchestrator) calls :func:`run` after ``critical_path.run`` with the same
``RunContext``.  It never writes launcher state directly: Settings mutations
are made by the page and are consequently covered by the browser guard and by
the runner's central snapshot/restore.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from critical_path import gpu_coherence_from_snapshots
from harness import HarnessError, RunContext, read_sysfs_gpu


# The central guard permits this state-changing route only with the narrow
# rules below; the scenario itself cannot weaken the guard.  No remote variant
# is allowed.
MUTATING_ENDPOINTS_REQUIRED_BY_GUARD = (
    {
        "method": "POST",
        "path": "/api/settings",
        "validation": (
            "loopback only; model_paths must equal the captured baseline plus "
            "guard.download_root; baseline backend paths stay unchanged except "
            "custom, which may be empty or guard.download_root"
        ),
    },
)

GPU_METRIC_LABELS = (
    "VRAM",
    "Edge",
    "Memória",
    "Hotspot",
    "Limite GPU",
    "Folga térmica",
    "GPU",
    "Fan",
    "Power draw",
    "Limite",
    "Clock GPU",
    "Clock memória",
)
GPU_API_FIELDS = (
    "memory.total",
    "memory.used",
    "memory.free",
    "temperature.gpu",
    "temperature.memory",
    "temperature.hotspot",
    "temperature.gpu.limit",
    "temperature.gpu.tlimit",
    "utilization.gpu",
    "utilization.memory",
    "fan.speed",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
)


def _endpoint(value: Any) -> str:
    return urlparse(str(getattr(value, "url", value))).path


class ApiTraffic:
    """Small, browser-event-only counter used as evidence for refresh checks."""

    def __init__(self) -> None:
        self.requests: Counter[str] = Counter()
        self.responses: Counter[str] = Counter()
        self.response_statuses: defaultdict[str, Counter[int]] = defaultdict(Counter)
        self.request_bodies: defaultdict[str, list[str]] = defaultdict(list)

    def attach(self, page: Any) -> None:
        def on_request(request: Any) -> None:
            path = _endpoint(request)
            if path.startswith("/api/"):
                self.requests[path] += 1
                body = getattr(request, "post_data", None)
                if callable(body):
                    body = body()
                if body:
                    self.request_bodies[path].append(str(body))

        def on_response(response: Any) -> None:
            path = _endpoint(response)
            if path.startswith("/api/"):
                self.responses[path] += 1
                status = getattr(response, "status", None)
                if callable(status):
                    status = status()
                if isinstance(status, int):
                    self.response_statuses[path][status] += 1

        page.on("request", on_request)
        page.on("response", on_response)

    def snapshot(self, *paths: str) -> dict[str, dict[str, int]]:
        wanted = paths or tuple(sorted(set(self.requests) | set(self.responses)))
        return {
            path: {
                "requests": self.requests[path],
                "responses": self.responses[path],
                "http_2xx": sum(
                    count for status, count in self.response_statuses[path].items()
                    if 200 <= status < 300
                ),
            }
            for path in wanted
        }


def gpu_payload_assertions(payload: Any) -> dict[str, Any]:
    """Return pure, testable assertions about the public ``/api/gpu`` shape."""
    checks: dict[str, bool] = {
        "object": isinstance(payload, dict),
        "available": isinstance(payload, dict) and payload.get("available") is True,
        "gpu_count": isinstance(payload, dict) and isinstance(payload.get("gpu_count"), int),
        "gpus": isinstance(payload, dict) and isinstance(payload.get("gpus"), list),
    }
    if not all(checks.values()):
        return {"checks": checks, "ok": False}

    gpus = payload["gpus"]
    checks["gpu_count_matches_cards"] = payload["gpu_count"] == len(gpus) and len(gpus) > 0
    for aggregate in ("vram_total_mib", "vram_used_mib", "vram_free_mib"):
        # None is a legitimate unavailable sensor value, but the key must be
        # present so the UI can render N/A rather than silently omit it.
        checks[f"aggregate_{aggregate}"] = aggregate in payload
    checks["gpu_metric_keys"] = all(
        isinstance(card, dict) and all(field in card for field in GPU_API_FIELDS)
        for card in gpus
    )
    return {"checks": checks, "gpu_count": payload.get("gpu_count"), "ok": all(checks.values())}


def settings_payload_is_root_only(
    payload: Any,
    root: Path,
    baseline_model_paths: Iterable[str] = (),
    baseline_backend_paths: dict[str, str] | None = None,
) -> bool:
    """Pure validation predicate for the guarded Settings mutation."""
    if not isinstance(payload, dict) or not isinstance(payload.get("model_paths"), list):
        return False
    root = root.resolve()
    expected_paths = list(baseline_model_paths)
    if str(root) not in expected_paths:
        expected_paths.append(str(root))
    if payload["model_paths"] != expected_paths:
        return False
    backend_paths = payload.get("backend_paths")
    baseline_backend_paths = baseline_backend_paths or {}
    if not isinstance(backend_paths, dict) or set(backend_paths) - (set(baseline_backend_paths) | {"custom"}):
        return False
    for name, value in baseline_backend_paths.items():
        if name != "custom" and backend_paths.get(name) != value:
            return False
    custom = backend_paths.get("custom")
    baseline_custom = baseline_backend_paths.get("custom")
    if custom in {None, "", baseline_custom}:
        return True
    return isinstance(custom, str) and Path(custom).is_absolute() and Path(custom).resolve(strict=False) == root


def backend_labels_match(api_backends: Any, ui_labels: Iterable[str]) -> bool:
    """Pure equality check used by HEADER-02 and unit tests."""
    if not isinstance(api_backends, list):
        return False
    expected: list[str] = []
    for item in api_backends:
        if isinstance(item, dict) and isinstance(item.get("label"), str):
            expected.append(item["label"])
    return len(expected) == len(api_backends) and sorted(expected) == sorted(ui_labels)


def remote_403_reason() -> str:
    """Explain why SETTINGS-06 is NV without manufacturing a remote request."""
    return "não verificado: não há peer remoto controlado nesta execução; nenhum header remoto foi falsificado"


def _json(ctx: RunContext, response: Any) -> Any:
    return ctx.api.json(response)


def _write(ctx: RunContext, name: str, value: Any) -> Path:
    path = ctx.evidence(name)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _relative(path: Path, ctx: RunContext) -> str:
    return str(path.relative_to(ctx.evidence_dir))


def _screenshot(ctx: RunContext, name: str) -> str:
    path = ctx.evidence(f"screenshots/{name}.png")
    ctx.page.screenshot(path=str(path), full_page=True)
    return _relative(path, ctx)


def _wait(page: Any, predicate: Callable[[], bool], timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        page.wait_for_timeout(100)
    return predicate()


def _record_pass(ctx: RunContext, item_id: str, observed: str, data: Any) -> None:
    path = _write(ctx, f"system-ui/{item_id.lower()}.json", data)
    ctx.current_item = item_id
    evidence = [_relative(path, ctx)]
    if isinstance(data, dict):
        for visual_path in data.get("_evidence_paths", []):
            if str(visual_path) not in evidence:
                evidence.append(str(visual_path))
    ctx.checklist.record(item_id, "PASS", observed=observed, evidence=evidence)


def _record_failure(ctx: RunContext, item_id: str, exc: Exception) -> None:
    message = str(exc) or repr(exc)
    path = _write(ctx, f"system-ui/{item_id.lower()}-failure.json", {
        "error": message, "error_fallback": repr(exc), "type": type(exc).__name__,
    })
    ctx.current_item = item_id
    ctx.checklist.record(item_id, "FAIL", observed="assertion failed", reason=message, evidence=[_relative(path, ctx)])


def _assert_no_visible_dialog(ctx: RunContext) -> None:
    """Refuse to start system_ui with a dialog leaked by an earlier owner."""
    visible: list[str] = []
    for test_id in ("config-editor-modal", "launch-modal"):
        modal = ctx.page.get_by_test_id(test_id)
        if modal.count() and modal.is_visible():
            visible.append(test_id)
    if visible:
        raise HarnessError(
            "system_ui precondition failed: prior Configs/Editor left visible modal(s): "
            + ", ".join(visible)
        )


def _item(ctx: RunContext, item_id: str, action: Callable[[], tuple[str, Any]]) -> None:
    try:
        observed, evidence = action()
        _record_pass(ctx, item_id, observed, evidence)
    except Exception as exc:
        _record_failure(ctx, item_id, exc)


def _click_tab(ctx: RunContext, tab_id: str, text: str) -> None:
    ctx.page.get_by_test_id(tab_id).click()
    ctx.page.get_by_role("heading", name=text, exact=True).wait_for(timeout=10_000)


def _amd_section(page: Any) -> Any:
    heading = page.get_by_role("heading", name="AMD GPU", exact=True)
    return page.locator("section").filter(has=heading)


def _gpu(ctx: RunContext) -> dict[str, Any]:
    payload = _json(ctx, ctx.api.get("/api/gpu"))
    checks = gpu_payload_assertions(payload)
    if not checks["ok"]:
        raise HarnessError(f"payload /api/gpu inválido: {checks}")
    return payload


def _gpu_refresh(ctx: RunContext, traffic: ApiTraffic) -> tuple[str, Any]:
    page = ctx.page
    _click_tab(ctx, "tab-amd", "AMD GPU")
    checkbox = page.get_by_role("checkbox", name="atualizar a cada 2 s")
    checkbox.wait_for(timeout=10_000)
    if not checkbox.is_checked():
        checkbox.check()

    # AmdPage polls immediately and then every 2,000 ms.  Allow a small
    # scheduling margin, while retaining the measured interval in evidence.
    started = time.monotonic()
    assert _wait(page, lambda: traffic.requests["/api/gpu"] >= 1, 5.0), "primeira request /api/gpu não ocorreu"
    initial = traffic.requests["/api/gpu"]
    assert _wait(page, lambda: traffic.requests["/api/gpu"] >= initial + 1, 2.8), "auto-refresh não gerou segunda request em ~2s"
    automatic = traffic.snapshot("/api/gpu")
    automatic_screenshot = _screenshot(ctx, "gpu-auto-refresh-enabled")

    checkbox.uncheck()
    page.wait_for_timeout(350)  # settles the generation change and its one-shot read
    settled = traffic.requests["/api/gpu"]
    page.wait_for_timeout(2_250)
    paused = traffic.requests["/api/gpu"]
    assert paused == settled, f"auto-refresh não pausou: {settled}->{paused} requests"
    paused_screenshot = _screenshot(ctx, "gpu-auto-refresh-paused")

    before_manual = traffic.requests["/api/gpu"]
    page.get_by_role("button", name="atualizar", exact=True).click()
    assert _wait(page, lambda: traffic.requests["/api/gpu"] > before_manual, 3.0), "botão atualizar não causou request"
    after_manual = traffic.snapshot("/api/gpu")
    elapsed = time.monotonic() - started
    assert after_manual["/api/gpu"]["responses"] >= after_manual["/api/gpu"]["requests"] - 1
    assert after_manual["/api/gpu"]["http_2xx"] >= 2, "menos de duas respostas 2xx de /api/gpu"
    manual_screenshot = _screenshot(ctx, "gpu-manual-refresh")
    return (
        f"requests={after_manual['/api/gpu']['requests']}, responses={after_manual['/api/gpu']['responses']}, pause={settled}->{paused}",
        {
            "automatic": automatic,
            "after_manual": after_manual,
            "settled_before_pause": settled,
            "paused_after_2_25s": paused,
            "elapsed_seconds": round(elapsed, 3),
            "assertions": {"auto_refresh_2s": True, "pause": True, "manual_update": True},
            "_evidence_paths": [automatic_screenshot, paused_screenshot, manual_screenshot],
        },
    )


def _gpu_vram(ctx: RunContext) -> tuple[str, Any]:
    payload = _gpu(ctx)
    section = _amd_section(ctx.page)
    labels = {"VRAM total": re.compile(r"VRAM total", re.I), "VRAM agregada": re.compile(r"VRAM agregada", re.I), "livres": re.compile(r"\blivres\b", re.I)}
    counts = {label: section.get_by_text(pattern).count() for label, pattern in labels.items()}
    assert all(counts.values()), f"campos VRAM ausentes: {counts}"
    assert payload["vram_total_mib"] is None or isinstance(payload["vram_total_mib"], int)
    assert payload["vram_used_mib"] is None or isinstance(payload["vram_used_mib"], int)
    assert payload["vram_free_mib"] is None or isinstance(payload["vram_free_mib"], int)
    return "VRAM total/used/free presentes na API e na UI", {"payload": payload, "ui_label_counts": counts}


def _gpu_sensors(ctx: RunContext) -> tuple[str, Any]:
    payload = _gpu(ctx)
    labels = ["Edge", "Memória", "Hotspot", "Limite GPU", "Folga térmica", "GPU", "Fan"]
    counts = {label: ctx.page.get_by_text(label, exact=True).count() for label in labels}
    assert all(counts.values()), f"campos de sensores ausentes: {counts}"
    return "temperatura, utilization e fan visíveis por GPU", {"labels": counts, "gpu_count": payload["gpu_count"]}


def _gpu_power_clocks(ctx: RunContext) -> tuple[str, Any]:
    payload = _gpu(ctx)
    labels = ["Power draw", "Limite", "Clock GPU", "Clock memória"]
    counts = {label: ctx.page.get_by_text(label, exact=True).count() for label in labels}
    assert all(counts.values()), f"campos de energia/clocks ausentes: {counts}"
    return "power e clocks visíveis por GPU", {"labels": counts, "gpu_count": payload["gpu_count"]}


def _gpu_charts(ctx: RunContext) -> tuple[str, Any]:
    charts = ctx.page.get_by_role("img", name=re.compile(r"Histórico de (VRAM|RAM)"))
    count = charts.count()
    assert count == 2, f"esperados dois gráficos VRAM/RAM, encontrados {count}"
    return "gráficos de histórico de VRAM e RAM presentes", {"charts": charts.all_inner_texts(), "count": count}


def _gpu_cards(ctx: RunContext) -> tuple[str, Any]:
    payload = _gpu(ctx)
    count = ctx.page.locator("article").count()
    assert count == payload["gpu_count"], f"cards={count}, gpu_count={payload['gpu_count']}"
    card_labels = [f"GPU {index}" for index in range(1, payload["gpu_count"] + 1)]
    visible = {label: ctx.page.get_by_text(label, exact=True).count() for label in card_labels}
    assert all(visible.values()), f"cards sem identificador: {visible}"
    return f"{count} card(s), um por GPU", {"gpu_count": payload["gpu_count"], "cards": visible}


def _gpu_sysfs(ctx: RunContext) -> tuple[str, Any]:
    before = read_sysfs_gpu()
    payload = _gpu(ctx)
    after = read_sysfs_gpu()
    result = gpu_coherence_from_snapshots(before, payload, after)
    path_data = {**result, "assertions": result["checks"]}
    assert result["ok"], f"GPU/API/sysfs incoerentes: {result}"
    return "API coerente com sysfs em bracket before/after", path_data


def _header_bars(ctx: RunContext) -> tuple[str, Any]:
    system = _json(ctx, ctx.api.get("/api/system"))
    header = ctx.page.locator("header")
    assert header.get_by_text("VRAM", exact=True).count() == 1
    assert header.get_by_text("RAM", exact=True).count() == 1
    assert isinstance(system, dict) and "vram_total_mib" in system and "ram_total_mib" in system
    return "barras VRAM e RAM visíveis; /api/system contém os totais", {"system": system, "bars": ["VRAM", "RAM"]}


def _header_backends(ctx: RunContext) -> tuple[str, Any]:
    backends = _json(ctx, ctx.api.get("/api/backends"))
    header = ctx.page.locator("header")
    labels = [value.strip() for value in header.locator("span").all_inner_texts() if value.strip()]
    expected: list[str] = []
    for item in backends:
        if isinstance(item, dict) and isinstance(item.get("label"), str):
            expected.append(item["label"])
    # The header contains other spans (version and memory values); only exact
    # backend labels are selected for the equality assertion.
    visible: list[str] = [label for label in expected if header.get_by_text(label, exact=True).count()]
    assert backend_labels_match(backends, visible), f"badges UI/API divergentes: {expected} != {visible}"
    return "badges exibidos exatamente para /api/backends", {"api": backends, "ui_backend_labels": visible, "header_spans": labels}


def _header_refresh(ctx: RunContext, traffic: ApiTraffic) -> tuple[str, Any]:
    paths = ("/api/system", "/api/backends", "/api/configs")
    before = traffic.snapshot(*paths)
    ctx.page.get_by_title("Recarregar sistema/backends/configs").click()
    assert _wait(ctx.page, lambda: all(
        traffic.requests[path] > before[path]["requests"]
        and traffic.responses[path] > before[path]["responses"]
        for path in paths
    ), 5.0), f"refresh não completou requests/responses: {before} -> {traffic.snapshot(*paths)}"
    after = traffic.snapshot(*paths)
    assert all(after[path]["responses"] > before[path]["responses"] for path in paths)
    return "refresh do header disparou os três endpoints", {
        "before": before,
        "after": after,
        "assertions": {path: True for path in paths},
        "_evidence_paths": [_screenshot(ctx, "header-refresh")],
    }


def _model_inputs(page: Any) -> Any:
    return page.get_by_placeholder(r"C:\caminho\para\modelos")


def _settings_model_paths(ctx: RunContext) -> tuple[str, Any]:
    page = ctx.page
    root = str(ctx.guard.download_root)
    inputs = _model_inputs(page)
    original_count = inputs.count()
    baseline_paths = list(ctx.guard.baseline_model_paths)
    expected_paths = list(baseline_paths)
    if root not in expected_paths:
        expected_paths.append(root)

    # Demonstrate removal using a transient row, while preserving every
    # production/model root captured during preflight.
    page.get_by_role("button", name="adicionar caminho", exact=True).click()
    inputs = _model_inputs(page)
    transient = inputs.nth(inputs.count() - 1)
    transient.fill(str(ctx.guard.download_root / "transient-not-saved"))
    transient_count = inputs.count()
    page.get_by_title("remover este caminho").nth(inputs.count() - 1).click()
    assert inputs.count() == original_count

    # Add the guarded root as the only persistent change.
    page.get_by_role("button", name="adicionar caminho", exact=True).click()
    inputs = _model_inputs(page)
    inputs.nth(inputs.count() - 1).fill(root)
    assert [inputs.nth(index).input_value() for index in range(inputs.count())] == expected_paths
    page.get_by_role("button", name="salvar", exact=True).click()
    page.get_by_text("salvo", exact=True).wait_for(timeout=5_000)
    saved = _json(ctx, ctx.api.get("/api/settings"))
    assert saved.get("model_paths") == expected_paths, f"model_paths baseline não preservado: {saved}"
    return "roots baseline preservadas e raiz E2E adicionada após remoção transitória", {
        "original_input_count": original_count,
        "transient_input_count": transient_count,
        "final_input_count": inputs.count(),
        "baseline_paths": baseline_paths,
        "expected_paths": expected_paths,
        "saved": saved,
        "root": root,
        "_evidence_paths": [_screenshot(ctx, "settings-model-roots")],
    }


def _custom_backend(ctx: RunContext, traffic: ApiTraffic) -> tuple[str, Any]:
    page = ctx.page
    root = str(ctx.guard.download_root)
    before = _json(ctx, ctx.api.get("/api/settings"))
    custom_row = page.locator("li").filter(has_text="custom (build de um modelo)")
    custom_input = custom_row.locator("input").first
    # Do not fill or click turbo/vanilla controls.  Compare their persisted
    # values before/after to catch an accidental broad Settings mutation.
    backend_before = dict(before.get("backend_paths", {}))
    custom_input.fill(root)
    page.get_by_role("button", name="salvar", exact=True).click()
    page.get_by_text("salvo", exact=True).wait_for(timeout=5_000)
    after = _json(ctx, ctx.api.get("/api/settings"))
    assert after.get("backend_paths", {}).get("custom") == root
    for name, value in backend_before.items():
        if name != "custom":
            assert after.get("backend_paths", {}).get(name) == value, f"backend {name} foi alterado"
    posts: list[dict[str, Any]] = []
    # Bodies are captured from browser traffic, not reconstructed or sent by
    # the test, so this also proves the mutation originated in the UI.
    for raw in traffic.request_bodies["/api/settings"]:
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(body, dict):
            posts.append(body)
    assert posts and all(set(post.get("backend_paths", {})) <= {"custom"} for post in posts[-1:])
    return "somente custom apontado para a raiz E2E pela UI", {
        "before": before,
        "after": after,
        "ui_posts": posts,
        "_evidence_paths": [_screenshot(ctx, "settings-custom")],
    }


def _settings_default_reset(ctx: RunContext) -> tuple[str, Any]:
    page = ctx.page
    row = page.locator("li").filter(has_text="custom (build de um modelo)")
    default_text = row.inner_text()
    # custom não tem default implícito: a linha mostra "sem default — configure
    # um build" quando vazio; aceita também o texto antigo "default: ...".
    assert "default:" in default_text or "sem default" in default_text
    reset_button = row.get_by_title("usar default")
    assert reset_button.is_enabled(), "botão reset do custom não ficou disponível"
    reset_button.click()
    assert row.locator("input").first.input_value() == ""
    page.get_by_role("button", name="salvar", exact=True).click()
    page.get_by_text("salvo", exact=True).wait_for(timeout=5_000)
    saved = _json(ctx, ctx.api.get("/api/settings"))
    expected_backends = {name: value for name, value in ctx.guard.baseline_backend_paths.items() if name != "custom"}
    assert saved.get("backend_paths", {}) == expected_backends
    row_text = row.inner_text()
    assert "usando default:" in row_text or "sem default" in row_text
    return "default exibido e reset persistido pela UI", {
        "default_text": default_text,
        "saved": saved,
        "reset": True,
        "_evidence_paths": [_screenshot(ctx, "settings-default-reset")],
    }


def _settings_reset_assertion(ctx: RunContext) -> tuple[str, Any]:
    page = ctx.page
    row = page.locator("li").filter(has_text="custom (build de um modelo)")
    # Self-sufficient: SETTINGS-03 pode ter morrido antes do reset; refaz o
    # reset aqui e persiste, sem depender do efeito colateral do item anterior.
    if row.locator("input").first.input_value() != "":
        reset_button = row.get_by_title("usar default")
        assert reset_button.is_enabled(), "botão reset do custom não ficou disponível"
        reset_button.click()
        _wait(page, lambda: row.locator("input").first.input_value() == "")
        page.get_by_role("button", name="salvar", exact=True).click()
        page.get_by_text("salvo", exact=True).wait_for(timeout=5_000)
    saved = _json(ctx, ctx.api.get("/api/settings"))
    assert row.locator("input").first.input_value() == ""
    expected_backends = {name: value for name, value in ctx.guard.baseline_backend_paths.items() if name != "custom"}
    assert saved.get("backend_paths", {}) == expected_backends
    row_text = row.inner_text()
    assert "usando default:" in row_text or "sem default" in row_text
    return "botão reset deixou custom vazio e usando default", {
        "saved": saved,
        "input_empty": True,
        "_evidence_paths": [_screenshot(ctx, "settings-reset")],
    }


def _settings_custom_unlaunchable(ctx: RunContext) -> tuple[str, Any]:
    page = ctx.page
    backends = _json(ctx, ctx.api.get("/api/backends"))
    if not isinstance(backends, list):
        raise HarnessError(f"/api/backends não retornou lista: {backends!r}")
    custom = next((item for item in backends if isinstance(item, dict) and item.get("name") == "custom"), None)
    if not isinstance(custom, dict):
        raise HarnessError(f"backend custom ausente por name em /api/backends: {backends!r}")
    if custom.get("server_available") is not False:
        raise HarnessError(
            "SETTINGS-05 FAIL: custom usa o default sem path próprio, "
            f"mas permanece available (server_available={custom.get('server_available')!r}): {custom!r}"
        )
    _click_tab(ctx, "tab-configs", "Configurações salvas")
    page.get_by_role("button", name="nova", exact=True).click()
    page.get_by_text("Nova configuração", exact=True).wait_for(timeout=10_000)
    try:
        if page.get_by_role("combobox").count() == 0:
            raise HarnessError(f"editor não hidratou após abrir Nova; backend={custom!r}")
        # O card do ConfigEditor renderiza b.label ("custom") + b.description
        # ("build/fork específico — caminho configurado em Settings"); o label
        # "custom (build de um modelo)" é da SettingsPage, não do editor. O
        # wrapper Field também é um <label> e contém o grid inteiro — escopar
        # ao grid dentro do modal evita o strict mode (4 radios).
        custom_label = page.get_by_test_id("config-editor-modal").locator(
            "div.grid.grid-cols-2.gap-2 label"
        ).filter(has_text="build/fork específico")
        custom_radio = custom_label.locator('input[type="radio"]')
        if not _wait(page, lambda: custom_radio.count() == 1 and custom_radio.is_disabled(), 5.0):
            raise HarnessError(
                f"custom continuou lançável/DOM não hidratou: radio_count={custom_radio.count()} "
                f"disabled={custom_radio.is_disabled() if custom_radio.count() else None}; backend={custom!r}"
            )
        return "custom sem path/binário não lançável: radio disabled", {
            "backend": custom,
            "radio_disabled": True,
            "_evidence_paths": [_screenshot(ctx, "settings-custom-availability")],
        }
    finally:
        modal = page.get_by_test_id("config-editor-modal")
        if modal.count() and modal.is_visible():
            cancel = page.get_by_test_id("config-editor-cancel")
            try:
                cancel.click(timeout=5_000)
                modal.wait_for(state="hidden", timeout=10_000)
            except Exception:
                pass


def run(ctx: RunContext) -> None:
    """Run GPU, Header and Settings checks using the supplied RunContext."""
    traffic = ApiTraffic()
    traffic.attach(ctx.page)
    _assert_no_visible_dialog(ctx)
    # Kept on the context solely to share browser-observed Settings POST bodies
    # between item helpers; no harness or launcher state is monkey-patched.

    _item(ctx, "GPU-01", lambda: _gpu_refresh(ctx, traffic))
    _item(ctx, "GPU-02", lambda: _gpu_vram(ctx))
    _item(ctx, "GPU-03", lambda: _gpu_sensors(ctx))
    _item(ctx, "GPU-04", lambda: _gpu_power_clocks(ctx))
    _item(ctx, "GPU-05", lambda: _gpu_charts(ctx))
    _item(ctx, "GPU-06", lambda: _gpu_cards(ctx))
    _item(ctx, "GPU-07", lambda: _gpu_sysfs(ctx))

    _item(ctx, "HEADER-01", lambda: _header_bars(ctx))
    _item(ctx, "HEADER-02", lambda: _header_backends(ctx))
    _item(ctx, "HEADER-03", lambda: _header_refresh(ctx, traffic))

    _click_tab(ctx, "tab-settings", "Configurações")
    _item(ctx, "SETTINGS-01", lambda: _settings_model_paths(ctx))
    _item(ctx, "SETTINGS-02", lambda: _custom_backend(ctx, traffic))
    _item(ctx, "SETTINGS-03", lambda: _settings_default_reset(ctx))
    _item(ctx, "SETTINGS-04", lambda: _settings_reset_assertion(ctx))
    _item(ctx, "SETTINGS-05", lambda: _settings_custom_unlaunchable(ctx))

    ctx.current_item = "SETTINGS-06"
    ctx.checklist.record("SETTINGS-06", "NÃO VERIFICADO", observed="", reason=remote_403_reason())
    _write(ctx, "system-ui/settings-06-not-verified.json", {"status": "NÃO VERIFICADO", "reason": remote_403_reason(), "request_made": False})
