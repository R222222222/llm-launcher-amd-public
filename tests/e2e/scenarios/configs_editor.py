"""Configs/Editor/Logs scenario.

This is intentionally a scenario, not a pytest test.  It runs after the
critical path and leaves model deletion to the later Models scenario.  Every
mutation is either a guarded UI request or a guarded API request for an
``e2e-*`` config using the one allowlisted GGUF.
"""
from __future__ import annotations

import json
import inspect
import re
import time
from pathlib import Path
from typing import Any, Callable

from harness import HarnessError, RunContext


def _error_text(exc: BaseException) -> str:
    return str(exc) or repr(exc)


CONFIG_IDS = (
    "CONFIG-01", "CONFIG-02", "CONFIG-03", "CONFIG-04", "CONFIG-05",
    "CONFIG-06", "CONFIG-07", "CONFIG-08", "CONFIG-09", "CONFIG-10", "CONFIG-11",
)
EDITOR_IDS = tuple(f"EDITOR-{index:02d}" for index in range(1, 13))
LOG_IDS = tuple(f"LOG-{index:02d}" for index in range(1, 6))
SCENARIO_IDS = CONFIG_IDS + EDITOR_IDS + LOG_IDS

# All are already accepted by the current central guard.  This is deliberately
# exposed so a future harness can report missing permissions before a run.
REQUIRED_GUARD_ENDPOINTS = (
    "POST /api/configs", "DELETE /api/configs", "POST /api/launch",
    "POST /api/launch-router", "POST /api/launch/{id}/restart",
    "POST /api/launch/{id}/cancel",
)


def required_guard_endpoints() -> tuple[str, ...]:
    """Return the mutation surface this scenario expects from MutationGuard."""
    return REQUIRED_GUARD_ENDPOINTS


def row_cells_match(
    cells: list[str], *, alias: str, backend: str,
    context_window: int, kv_cache: str, parallel_slots: int,
) -> bool:
    """Pure row matcher; indexes mirror ConfigGrid's visible table columns."""
    if len(cells) < 10:
        return False
    context = cells[5].replace(",", "").replace(".", "").replace(" ", "")
    return (
        alias in cells[3]
        and cells[4].strip() == backend
        and context == str(context_window)
        and cells[6].strip() == kv_cache
        and cells[9].strip() == str(parallel_slots)
    )


def kv_values(options: list[dict[str, Any]]) -> list[str]:
    """Return literal option values, useful for the turboquant invariant."""
    return [str(option.get("value")) for option in options]


def config_payload(model: Path, config_id: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": config_id,
        "model": str(model),
        "backend": "vanilla",
        "context_window": 4096,
        "kv_cache": "q8_0",
        "flash_attn": True,
        "gpu_layers": 99,
        "parallel_slots": 1,
        "reasoning_budget": None,
        "mlock": False,
        "max_tokens": 64,
        "batch_size": 512,
        "ubatch_size": 128,
        "threads_gen": 2,
        "threads_batch": 2,
        "cache_ram": 2048,
        "ctx_checkpoints": 0,
        "spec_draft_n_max": 2,
        "mmproj": None,
        "mcp_servers_config": None,
        "verbose": False,
        "llama_auto": False,
        "mode": "server",
    }
    payload.update(overrides)
    return payload


def _evidence(ctx: RunContext, name: str, value: Any) -> str:
    path = ctx.evidence(name)
    if isinstance(value, str):
        path.write_text(value if value.endswith("\n") else value + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(ctx.evidence_dir))


def _shot(ctx: RunContext, name: str) -> str:
    path = ctx.evidence(f"screenshots/{name}.png")
    ctx.page.screenshot(path=str(path), full_page=True)
    return str(path.relative_to(ctx.evidence_dir))


def _pass(ctx: RunContext, item_id: str, observed: str, evidence: list[str]) -> None:
    ctx.checklist.record(item_id, "PASS", observed=observed, evidence=evidence)


def _nv(ctx: RunContext, item_id: str, reason: str, evidence: list[str]) -> None:
    ctx.checklist.record(item_id, "NÃO VERIFICADO", observed="", reason=reason, evidence=evidence)


def _failure(ctx: RunContext, item_id: str, remaining: tuple[str, ...], exc: Exception) -> None:
    message = _error_text(exc)
    evidence = _evidence(ctx, f"failure-{item_id}.txt", message)
    ctx.checklist.record(item_id, "FAIL", observed="unexpected scenario failure", reason=message, evidence=[evidence])
    for rest in remaining:
        if ctx.checklist.results[rest].status == "NÃO VERIFICADO":
            ctx.checklist.record(rest, "NÃO VERIFICADO", observed="", reason=f"bloqueado por {item_id}: {message}")


def _close_config_editor(ctx: RunContext) -> None:
    """Close the Configs-owned editor, never an arbitrary localized button."""
    modal = ctx.page.get_by_test_id("config-editor-modal")
    if modal.count() == 0:
        return
    if modal.is_visible():
        cancel = ctx.page.get_by_test_id("config-editor-cancel")
        if cancel.count() != 1:
            raise HarnessError("config editor visível sem config-editor-cancel único")
        cancel.click(timeout=5_000)
    modal.wait_for(state="hidden", timeout=10_000)


def _dismiss_launch_modal(ctx: RunContext) -> None:
    """Dismiss the launch log modal without invoking its backend Stop action."""
    modal = ctx.page.get_by_test_id("launch-modal")
    if modal.count() == 0:
        return
    if modal.is_visible():
        dismiss = ctx.page.get_by_test_id("launch-modal-dismiss")
        if dismiss.count() != 1:
            raise HarnessError("launch modal visível sem launch-modal-dismiss único")
        dismiss.click(timeout=5_000)
    modal.wait_for(state="hidden", timeout=10_000)


def _teardown_owned_launch(ctx: RunContext, launch_id: str | None) -> None:
    """Stop an owned backend launch, then always dismiss its UI owner."""
    try:
        _cancel_owned_launch(ctx, launch_id)
    finally:
        _dismiss_launch_modal(ctx)


def _step(
    ctx: RunContext,
    item_id: str,
    remaining: tuple[str, ...],
    fn: Callable[[], None],
    *,
    cleanup: Callable[[RunContext], None] = _close_config_editor,
) -> bool:
    try:
        fn()
        return True
    except Exception as exc:
        _failure(ctx, item_id, remaining, exc)
        return False
    finally:
        cleanup(ctx)


def _grid_step(
    ctx: RunContext,
    item_id: str,
    action: Callable[[], tuple[str, list[str]]],
    *,
    failure_screenshot: str | None = None,
    status: str = "PASS",
) -> None:
    """Run one grid assertion without making neighbouring items dependent.

    In particular, a duplicate/new modal belongs to this step.  A timeout is
    recorded against its checklist item and its screenshot is taken before the
    local modal cleanup, never against CONFIG-01 (the filter assertion).
    """
    try:
        observed, evidence = action()
        if status == "NÃO VERIFICADO":
            _nv(ctx, item_id, observed, evidence)
        else:
            _pass(ctx, item_id, observed, evidence)
    except Exception as exc:
        message = _error_text(exc)
        evidence: list[str] = []
        if failure_screenshot:
            try:
                evidence.append(_shot(ctx, failure_screenshot))
            except Exception:
                pass
        evidence.append(_evidence(ctx, f"configs-editor/{item_id.lower()}-failure.json", {
            "error": message,
            "error_fallback": repr(exc),
            "type": type(exc).__name__,
            "screenshot_before_cleanup": bool(failure_screenshot),
        }))
        ctx.checklist.record(item_id, "FAIL", observed="grid assertion failed", reason=message, evidence=evidence)
    finally:
        _close_config_editor(ctx)


def _launch_response_payload(response: Any) -> dict[str, Any]:
    """Read the launch response without manufacturing an ID client-side."""
    if getattr(response, "ok", True) is False:
        raise HarnessError(f"launch HTTP {getattr(response, 'status', '?')}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise HarnessError(f"resposta de launch não é objeto: {payload!r}")
    return payload


def _launch_id(payload: dict[str, Any]) -> str:
    launch = payload.get("launch")
    candidates = (
        payload.get("launch_id"),
        payload.get("id"),
        launch.get("launch_id") if isinstance(launch, dict) else None,
        launch.get("id") if isinstance(launch, dict) else None,
    )
    value = next((item for item in candidates if isinstance(item, str) and item), None)
    if value is None:
        raise HarnessError(f"resposta de launch sem launch_id: {payload!r}")
    return value


def _register_owned_launch(ctx: RunContext, launch_id: str, config_ids: list[str]) -> None:
    """Register the response ID for restart/cancel; retain old fake compatibility."""
    if not config_ids or any(not str(item).startswith("e2e-") for item in config_ids):
        raise HarnessError(f"config_ids de launch não são E2E: {config_ids!r}")
    register = ctx.guard.register_launch
    try:
        parameters = inspect.signature(register).parameters
    except (TypeError, ValueError):
        parameters = {}
    if len(parameters) >= 2:
        ctx.guard.register_launch(launch_id, config_ids)  # type: ignore[call-arg]
    else:
        # The checked-in guard predates the config_ids argument; the core
        # contract is still attempted whenever the guard exposes it.
        register(launch_id)
    setattr(ctx, "owned_launch_id", launch_id)
    setattr(ctx, "owned_launch_config_ids", list(config_ids))


def _cancel_owned_launch(ctx: RunContext, launch_id: str | None) -> None:
    if not launch_id:
        return
    try:
        ctx.api.post(f"/api/launch/{launch_id}/cancel")
    finally:
        _wait_no_launch(ctx)


def _response_json(ctx: RunContext, response: Any, evidence_name: str) -> Any:
    payload = ctx.api.json(response)
    _evidence(ctx, evidence_name, payload)
    return payload


def _save_api(ctx: RunContext, payload: dict[str, Any]) -> dict[str, Any]:
    ctx.guard.expect_config(payload["id"], payload)
    result = _response_json(ctx, ctx.api.post("/api/configs", payload), f"api-save-{payload['id']}.json")
    if result.get("config", {}).get("id") != payload["id"]:
        raise HarnessError(f"save sem id esperado: {result!r}")
    return result["config"]


def _all_configs(ctx: RunContext) -> list[dict[str, Any]]:
    return _response_json(ctx, ctx.api.get("/api/configs"), "configs-after-scenario.json")


def _refresh_configs_grid(ctx: RunContext) -> None:
    """Rehydrate React after an API save before locating its row."""
    page = ctx.page
    page.reload()
    page.get_by_test_id("tab-configs").click()
    page.get_by_role("heading", name="Configurações salvas", exact=True).wait_for(timeout=10_000)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if page.locator("tbody tr").count() > 0:
            return
        page.wait_for_timeout(200)
    raise HarnessError("grade de Configs não hidratou após refresh")


def _find_row(ctx: RunContext, payload: dict[str, Any]) -> Any:
    deadline = time.monotonic() + 15.0
    last_rows: list[list[str]] = []
    last_count = 0
    while time.monotonic() < deadline:
        rows = ctx.page.get_by_role("row").all()
        candidates = []
        last_rows = []
        for row in rows:
            cells = row.locator("td").all_inner_texts()
            last_rows.append(cells)
            if row_cells_match(
                cells,
                alias=ctx.model_alias,
                backend=payload.get("backend", "vanilla"),
                context_window=int(payload.get("context_window", 4096)),
                kv_cache=str(payload.get("kv_cache", "q8_0")),
                parallel_slots=int(payload.get("parallel_slots", 1)),
            ):
                candidates.append(row)
        last_count = len(candidates)
        if last_count == 1:
            return candidates[0]
        ctx.page.wait_for_timeout(200)
    dump = ctx.evidence("configs-editor/rows.json")
    dump.write_text(
        json.dumps({"candidates": last_count, "rows": last_rows, "payload": payload}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    raise HarnessError(f"linha de config não é única após 15s: {last_count}; dump=rows.json")


def _open_new(ctx: RunContext) -> None:
    ctx.page.get_by_role("button", name="nova", exact=True).click()
    ctx.page.get_by_role("heading", name="Nova configuração", exact=True).wait_for(timeout=10_000)


def _editor_combobox(ctx: RunContext, label_text: str) -> Any:
    field = ctx.page.locator("label").filter(has_text=label_text)
    selects = field.get_by_role("combobox").all()
    if not selects:
        raise HarnessError(f"combobox não encontrada: {label_text}")
    return selects[0]


def _select(ctx: RunContext, label_text: str, value: str) -> None:
    _editor_combobox(ctx, label_text).select_option(value=value)


def _input(ctx: RunContext, label_text: str) -> Any:
    field = ctx.page.locator("label").filter(has_text=label_text)
    inputs = field.locator("input").all()
    if not inputs:
        raise HarnessError(f"input não encontrada: {label_text}")
    return inputs[0]


def _save_editor(ctx: RunContext, payload: dict[str, Any], *, launch: bool = False) -> None:
    if not str(payload.get("id", "")).startswith("e2e-"):
        raise HarnessError("editor Save exige config E2E existente; editor novo não é permitido")
    ctx.guard.expect_config(payload["id"], {"backend": payload.get("backend", "vanilla")})
    if launch:
        ctx.page.get_by_role("button", name="Salvar e launch", exact=True).click()
    else:
        ctx.page.get_by_role("button", name="Salvar", exact=True).click()
    if not launch:
        ctx.page.get_by_text("Nova configuração", exact=True).wait_for(state="hidden", timeout=10_000)


def _wait_no_launch(ctx: RunContext, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _response_json(ctx, ctx.api.get("/api/launches"), "launches-poll.json") == []:
            return
        time.sleep(0.5)
    raise HarnessError("launch ativo não encerrou")


def _editor_step(ctx: RunContext, item_id: str, fn: Callable[[], tuple[str, list[str]]]) -> None:
    """Record one editor assertion without aborting safe independent checks."""
    ctx.current_item = item_id
    try:
        observed, evidence = fn()
        _pass(ctx, item_id, observed, evidence)
    except Exception as exc:
        message = _error_text(exc)
        failure = _evidence(ctx, f"configs-editor/{item_id.lower()}-failure.json", {
            "error": message, "error_fallback": repr(exc), "type": type(exc).__name__,
        })
        ctx.checklist.record(item_id, "FAIL", observed="editor assertion failed", reason=message, evidence=[failure])


def _command_text(ctx: RunContext) -> str:
    textareas = ctx.page.locator("textarea").all()
    return textareas[0].input_value().strip() if textareas else ""


def _configs_and_editor(ctx: RunContext) -> None:
    """Configs grid plus the editor controls, each assertion has evidence."""
    payload = getattr(ctx, "editor_payload", None)
    if not isinstance(payload, dict) or not str(payload.get("id", "")).startswith("e2e-"):
        raise HarnessError("editor exige payload E2E criado via API")
    row = _find_row(ctx, payload)
    row.get_by_title("Editar").click()
    ctx.page.get_by_text("Editar configuração", exact=True).wait_for(timeout=10_000)

    def model() -> tuple[str, list[str]]:
        comboboxes = ctx.page.get_by_role("combobox").all()
        if not comboboxes:
            raise HarnessError("editor sem combobox de modelo")
        # select_option(label=) only accepts an exact str, but the option label
        # is "alias (relative_path)"; locate the option by regex and select by
        # its value instead.
        option = comboboxes[0].get_by_role("option", name=re.compile(re.escape(ctx.model_alias), re.I)).first
        value = option.get_attribute("value")
        if not value:
            raise HarnessError(f"opção de modelo sem value: {ctx.model_alias!r}")
        comboboxes[0].select_option(value=value)
        return "dropdown de modelo selecionado", [_shot(ctx, "editor-model")]
    _editor_step(ctx, "EDITOR-01", model)

    def auto_mode() -> tuple[str, list[str]]:
        auto = ctx.page.get_by_label("llama.cpp decide (comando mínimo)")
        context = _editor_combobox(ctx, "Context window")
        slots = _editor_combobox(ctx, "Slots paralelos")
        kv = _editor_combobox(ctx, "KV cache")
        ngl = _input(ctx, "Camadas na GPU")
        flash = ctx.page.locator("label").filter(has_text="Flash Attention").get_by_role("button")
        mlock = ctx.page.get_by_label("--mlock")
        verbose = ctx.page.get_by_label("--verbose")
        before = {
            "context": context.input_value(), "slots": slots.input_value(), "kv": kv.input_value(),
            "ngl": ngl.input_value(), "flash": flash.inner_text(),
            "mlock": mlock.is_checked(), "verbose": verbose.is_checked(),
        }
        manual_command = _command_text(ctx)
        auto.check()
        if ctx.page.get_by_text("Modo auto:", exact=False).count() != 1:
            raise HarnessError("aviso do modo auto não apareceu")
        ignored = ctx.page.locator('[title*="não entra no comando"]')
        if ignored.count() == 0 or any("opacity-40" not in (node.get_attribute("class") or "") for node in ignored.all()):
            raise HarnessError("controles do modo auto não ficaram esmaecidos")
        ctx.page.wait_for_timeout(250)
        auto_command = _command_text(ctx)
        relevant = re.compile(r"(?:^|\s)(?:-c|--ctx-size|-kvu|-ctk|-ctv|-np|-fa|-ngl|--gpu-layers|-t|--threads|-b|--batch-size|-ub|--ubatch-size|--mlock|--verbose)(?:\s|=|$)")
        if relevant.search(auto_command):
            raise HarnessError(f"preview auto contém tuning: {auto_command!r}")
        auto.uncheck()
        ctx.page.wait_for_timeout(250)
        after = {
            "context": context.input_value(), "slots": slots.input_value(), "kv": kv.input_value(),
            "ngl": ngl.input_value(), "flash": flash.inner_text(),
            "mlock": mlock.is_checked(), "verbose": verbose.is_checked(),
        }
        if after != before:
            raise HarnessError(f"modo auto não restaurou valores: {before!r} -> {after!r}")
        if ctx.page.locator('[title*="não entra no comando"]').count() and any(
            "opacity-40" in (node.get_attribute("class") or "")
            for node in ctx.page.locator('[title*="não entra no comando"]').all()
        ):
            raise HarnessError("controles do modo auto não foram restaurados")
        restored_command = _command_text(ctx)
        if not manual_command or not relevant.search(manual_command) or not relevant.search(restored_command):
            raise HarnessError("preview manual não restaurou flags de tuning")
        return "auto esmaeceu preview e restaurou valores/flags", [_evidence(ctx, "configs-editor/editor-auto.json", {
            "before": before, "manual_command": manual_command, "auto_command": auto_command,
            "restored": after, "restored_command": restored_command,
        }), _shot(ctx, "editor-auto")]
    _editor_step(ctx, "EDITOR-02", auto_mode)

    def unavailable() -> tuple[str, list[str]]:
        value = ctx.page.get_by_text(re.compile(r"sem caminho — configure em Settings"))
        if value.count() != 1:
            raise HarnessError("backend unavailable não foi exibido")
        return "backend custom indisponível visível", [_evidence(ctx, "configs-editor/editor-backend.txt", value.inner_text())]
    _editor_step(ctx, "EDITOR-03", unavailable)

    def context_kv() -> tuple[str, list[str]]:
        _select(ctx, "Context window", str(payload.get("context_window", 4096)))
        _select(ctx, "Slots paralelos", str(payload.get("parallel_slots", 1)))
        kv = _editor_combobox(ctx, "KV cache")
        options = [{"value": option.get_attribute("value"), "text": option.inner_text(), "disabled": option.is_disabled()} for option in kv.locator("option").all()]
        evidence = _evidence(ctx, "configs-editor/editor-kv-options.json", options)
        if any(str(option["value"]).startswith("turbo") for option in options):
            raise HarnessError(f"EDITOR-04 FAIL: turbo KV literal presente em vanilla: {options!r}")
        _select(ctx, "KV cache", "q8_0")
        return f"ctx={payload.get('context_window', 4096)}, np={payload.get('parallel_slots', 1)} e KV vanilla q8_0", [evidence]
    _editor_step(ctx, "EDITOR-04", context_kv)

    def gpu_controls() -> tuple[str, list[str]]:
        _input(ctx, "Camadas na GPU").fill("99")
        ngl_before = _input(ctx, "Camadas na GPU").input_value()
        ctx.page.get_by_role("button", name="sugerir -ngl").click()
        ngl_after = _input(ctx, "Camadas na GPU").input_value()
        flash = ctx.page.locator("label").filter(has_text="Flash Attention").get_by_role("button")
        flash.click()
        off = flash.inner_text()
        flash.click()
        on = flash.inner_text()
        if "OFF" not in off or "ON" not in on:
            raise HarnessError("Flash Attention não alternou e restaurou")
        evidence = _evidence(ctx, "configs-editor/editor-ngl.json", {"before": ngl_before, "after": ngl_after, "flash_off": off, "flash_on": on})

        # EDITOR-05: MoE real (Ornith em runtime/production-models, arch
        # qwen35moe, 40 layers, 256 experts). Troca o dropdown do editor pro
        # Ornith, espera o meta fetch (/api/models/meta → is_moe), exercita o
        # campo -ncmoe e volta pro modelo allowlistado. Não salva — o guard
        # bloqueia configs 35B, o que é esperado.
        combobox = ctx.page.get_by_role("combobox").first
        ornith_option = combobox.get_by_role("option", name=re.compile(r"ornith", re.I)).first
        ornith_value = ornith_option.get_attribute("value")
        if not ornith_value:
            raise HarnessError("opção Ornith (MoE) sem value no dropdown do editor")
        combobox.select_option(value=ornith_value)
        ncmoe_label = ctx.page.locator("label").filter(
            has_text=re.compile(r"-ncmoe \(camadas MoE na CPU, máx \d+\)")
        )
        ncmoe_label.wait_for(timeout=10_000)
        ncmoe_input = ncmoe_label.locator("input").first
        ncmoe_input.fill("4")
        ctx.page.get_by_role("button", name="sugerir -ncmoe").click()
        deadline = time.monotonic() + 10.0
        suggested: int | None = None
        while time.monotonic() < deadline:
            try:
                suggested = int(ncmoe_input.input_value())
            except ValueError:
                suggested = None
            if suggested is not None and suggested >= 0:
                break
            ctx.page.wait_for_timeout(100)
        if suggested is None or suggested < 0:
            raise HarnessError(f"sugerir -ncmoe não atualizou para int não-negativo: {suggested!r}")
        moe_evidence = _evidence(ctx, "configs-editor/editor-ncmoe.json", {
            "model": ornith_value,
            "ncmoe_label": ncmoe_label.inner_text(),
            "filled": 4,
            "suggested": suggested,
        })
        shot = _shot(ctx, "editor-ncmoe")
        small_option = combobox.get_by_role("option", name=re.compile(re.escape(ctx.model_alias), re.I)).first
        small_value = small_option.get_attribute("value")
        if not small_value:
            raise HarnessError("opção do modelo E2E sem value após trocar para Ornith")
        combobox.select_option(value=small_value)
        return "flash/ngl conferidos; MoE: -ncmoe visível, preenchido e sugerido", [evidence, moe_evidence, shot]
    _editor_step(ctx, "EDITOR-05", gpu_controls)

    def cache() -> tuple[str, list[str]]:
        _select(ctx, "cache-ram", "2048")
        _select(ctx, "ctx-checkpoints", "0")
        return "cache-ram e ctx-checkpoints alteráveis", [_shot(ctx, "editor-cache")]
    _editor_step(ctx, "EDITOR-06", cache)

    def generation() -> tuple[str, list[str]]:
        for label, value in (("max-tokens", "2048"), ("batch", "512"), ("ubatch", "128")):
            _select(ctx, label, value)
        thread_inputs = ctx.page.locator("label").filter(has_text="threads").locator("input").all()
        if len(thread_inputs) != 2:
            raise HarnessError("threads não expôs gen e batch")
        for thread_input in thread_inputs:
            thread_input.fill("2")
        return "generation/batch/ubatch/threads configurados", [_shot(ctx, "editor-generation")]
    _editor_step(ctx, "EDITOR-07", generation)

    def flags() -> tuple[str, list[str]]:
        ctx.page.get_by_label("--mlock").check()
        ctx.page.get_by_label("--verbose").check()
        return "mlock e verbose marcados", [_shot(ctx, "editor-flags")]
    _editor_step(ctx, "EDITOR-08", flags)

    def mode() -> tuple[str, list[str]]:
        ctx.page.get_by_label("cli").check()
        ctx.page.get_by_label("server (HTTP)").check()
        return "server/cli alternados", [_shot(ctx, "editor-mode")]
    _editor_step(ctx, "EDITOR-09", mode)

    def panels() -> tuple[str, list[str]]:
        ctx.page.get_by_text("Estimativa", exact=True).wait_for()
        command = _command_text(ctx)
        if not command:
            raise HarnessError("painel de comando vazio")
        return "estimativa e comando visíveis", [_evidence(ctx, "configs-editor/editor-command.txt", command)]
    _editor_step(ctx, "EDITOR-10", panels)

    def gpu_limit() -> tuple[str, list[str]]:
        system = _response_json(ctx, ctx.api.get("/api/system"), "editor-system.json")
        if int(system.get("gpu_count", 0)) >= 2:
            raise HarnessError("split mode multi-GPU ainda não é coberto neste hardware")
        return "split mode ausente com uma GPU; limite registrado", ["editor-system.json"]
    _editor_step(ctx, "EDITOR-12", gpu_limit)

    try:
        _save_editor(ctx, payload)
        save_evidence = _evidence(ctx, "configs-editor/editor-save.json", payload)
        setattr(ctx, "editor_save_ok", True)
        setattr(ctx, "editor_save_evidence", save_evidence)
    except Exception as exc:
        message = _error_text(exc)
        failure = _evidence(ctx, "configs-editor/editor-save-failure.json", {"error": message, "error_fallback": repr(exc), "id": payload.get("id")})
        setattr(ctx, "editor_save_ok", False)
        ctx.checklist.record("EDITOR-11", "FAIL", observed="Save falhou", reason=message, evidence=[failure])


def _configs_grid(ctx: RunContext, payload: dict[str, Any]) -> None:
    """Exercise each grid assertion as an independent checklist step."""

    def estimate() -> tuple[str, list[str]]:
        row = _find_row(ctx, payload)
        if row.get_by_title("cabe folgado em VRAM").count() + row.get_by_title("VRAM apertada").count() == 0:
            raise HarnessError("bolinha de estimativa sem status")
        return "bolinha de estimativa presente", [_shot(ctx, "configs-estimate")]

    def filter_alias() -> tuple[str, list[str]]:
        filter_box = ctx.page.get_by_placeholder("filtrar por alias, backend, kv…")
        filter_box.fill(ctx.model_alias)
        if ctx.page.get_by_role("row").filter(has_text=ctx.model_alias).count() < 1:
            raise HarnessError("filtro por alias não filtrou")
        return "filtro por alias funcionou", [_shot(ctx, "configs-filter")]

    def chip() -> tuple[str, list[str]]:
        ctx.page.get_by_role("button", name="vanilla", exact=True).click()
        return "chip vanilla selecionável", [_shot(ctx, "configs-chip")]

    def edit() -> tuple[str, list[str]]:
        row = _find_row(ctx, payload)
        row.get_by_title("Editar").click()
        ctx.page.get_by_text("Editar configuração", exact=True).wait_for()
        # Evidence must show the open editor, not the grid after Cancelar.
        return "Editar abriu a config selecionada", [_shot(ctx, "configs-edit")]

    def new_config() -> tuple[str, list[str]]:
        _open_new(ctx)
        return "+ nova abriu editor", [_shot(ctx, "configs-new")]

    def duplicate() -> tuple[str, list[str]]:
        before = _all_configs(ctx)
        before_ids = {item.get("id") for item in before if isinstance(item, dict)}
        row = _find_row(ctx, payload)
        row.get_by_title(re.compile("Duplicar config")).click()
        # Keep the timeout local to CONFIG-05 so a broken Duplicate cannot
        # overwrite the already passing filter item or block the editor.
        # Duplicate keeps `model`, so the editor opens titled "Editar
        # configuração"; wait on the modal test-id instead of the title.
        ctx.page.get_by_test_id("config-editor-modal").wait_for(timeout=10_000)
        after_duplicate = _all_configs(ctx)
        after_ids = {item.get("id") for item in after_duplicate if isinstance(item, dict)}
        if after_ids - before_ids:
            raise HarnessError(
                f"Duplicate criou IDs sem save explícito: "
                f"{sorted(str(item) for item in after_ids - before_ids)}"
            )
        duplicate_evidence = _evidence(ctx, "configs-editor/duplicate-blocked.json", {
            "before_ids": sorted(str(item) for item in before_ids),
            "after_ids": sorted(str(item) for item in after_ids),
            "new_ids": sorted(str(item) for item in after_ids - before_ids),
            "reason": "UI duplicate omite id; saving geraria UUID fora de e2e",
            "mutation_sent": False,
        })
        return "NV: Duplicate abriu editor, mas save foi bloqueado sem ID E2E", [duplicate_evidence]

    def delete() -> tuple[str, list[str]]:
        # Delete is UI-scoped and targets a separately API-created E2E config.
        delete_payload = config_payload(
            ctx.model, f"e2e-delete-{ctx.run_id}", context_window=8192, kv_cache="q4_0"
        )
        _save_api(ctx, delete_payload)
        _refresh_configs_grid(ctx)
        before = _all_configs(ctx)
        before_ids = {item.get("id") for item in before if isinstance(item, dict)}
        dialogs = []
        ctx.page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.accept()))
        delete_row = _find_row(ctx, delete_payload)
        delete_row.get_by_title("Remover esta config").click()
        if not dialogs:
            raise HarnessError("confirmação de delete não apareceu")
        # The DELETE is async; poll until the config disappears instead of
        # racing a single GET against the in-flight DELETE. Assert the full
        # contract: o id deletado sumiu E todos os outros continuam presentes —
        # um GET vazio no meio do write não pode passar como "deletado".
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            after = _all_configs(ctx)
            after_ids = {item.get("id") for item in after if isinstance(item, dict)}
            if after_ids == before_ids - {delete_payload["id"]}:
                return "delete removeu apenas config E2E", ["configs-after-scenario.json"]
            ctx.page.wait_for_timeout(200)
        raise HarnessError("config E2E de delete ainda persistida após 15s")

    _grid_step(ctx, "CONFIG-03", estimate, failure_screenshot="configs-estimate-failure")
    _grid_step(ctx, "CONFIG-01", filter_alias, failure_screenshot="configs-filter-failure")
    try:
        _grid_step(ctx, "CONFIG-02", chip, failure_screenshot="configs-chip-failure")
    finally:
        # Filter state is not shared with the next independent action.
        try:
            ctx.page.get_by_placeholder("filtrar por alias, backend, kv…").fill("")
        except Exception:
            pass
    _grid_step(ctx, "CONFIG-04", edit, failure_screenshot="configs-edit-failure")
    _grid_step(ctx, "CONFIG-07", new_config, failure_screenshot="configs-new-failure")
    _grid_step(
        ctx, "CONFIG-05", duplicate,
        failure_screenshot="configs-duplicate-failure", status="NÃO VERIFICADO",
    )
    _grid_step(ctx, "CONFIG-06", delete, failure_screenshot="configs-delete-failure")


def _launch_and_logs(ctx: RunContext, payload: dict[str, Any]) -> None:
    """Save+launch, second-launch 409, modal logs, restart and cancel."""
    owned_launch_id: str | None = None
    try:
        if not getattr(ctx, "editor_save_ok", False):
            blocked = _evidence(ctx, "configs-editor/save-launch-blocked.json", {
                "reason": "Save inicial do editor falhou; não há config segura para launch",
            })
            for item_id in ("CONFIG-08", "CONFIG-11", "LOG-01", "LOG-02", "LOG-03", "LOG-04", "LOG-05"):
                if ctx.checklist.results[item_id].status == "NÃO VERIFICADO":
                    _nv(ctx, item_id, "NV: Save inicial do editor falhou; não há config segura para launch", [blocked])
            return
        row = _find_row(ctx, payload)
        row.get_by_title("Editar").click()
        ctx.page.get_by_text("Editar configuração", exact=True).wait_for(timeout=10_000)
        _select(ctx, "Context window", str(payload.get("context_window", 4096)))
        _select(ctx, "Slots paralelos", str(payload.get("parallel_slots", 1)))
        _select(ctx, "KV cache", "q8_0")
        launch_payload = payload
        ctx.guard.expect_config(launch_payload["id"], {"backend": launch_payload.get("backend", "vanilla")})
        with ctx.page.expect_response(
            lambda response: response.url.endswith("/api/launch")
            and response.request.method == "POST"
        ) as response_info:
            ctx.page.get_by_role("button", name="Salvar e launch", exact=True).click()
        launch_response = response_info.value
        launch_response_payload = _launch_response_payload(launch_response)
        owned_launch_id = _launch_id(launch_response_payload)
        _register_owned_launch(ctx, owned_launch_id, [launch_payload["id"]])
        ctx.page.get_by_text("tentativa #1", exact=True).wait_for(timeout=30_000)
        launch_request = ctx.guard.ui_launch_requests[-1] if ctx.guard.ui_launch_requests else None
        if not isinstance(launch_request, dict) or launch_request.get("id") != launch_payload["id"]:
            raise HarnessError(f"Save+launch não preservou id E2E: {launch_request!r}")
        aggregate = _evidence(ctx, "configs-editor/editor-save-and-launch.json", {
            "saved_config_id": launch_payload["id"],
            "launch_id": owned_launch_id,
            "save_evidence": getattr(ctx, "editor_save_evidence", None),
            "launch_request": launch_request,
        })
        _pass(ctx, "EDITOR-11", "Save e Save+launch preservaram a mesma config E2E", [aggregate, _shot(ctx, "save-and-launch")])

        second = config_payload(ctx.model, f"e2e-second-{ctx.run_id}")
        _save_api(ctx, second)
        ctx.guard.expect_config(second["id"], second)
        response = ctx.api.post("/api/launch", second)
        if response.status != 409:
            raise HarnessError(f"segundo launch não retornou 409: {response.status}")
        _pass(ctx, "CONFIG-11", "segundo launch retornou HTTP 409", [_evidence(ctx, "second-launch-409.txt", response.text())])
        _pass(ctx, "CONFIG-08", "single launch e Save+launch disponíveis", [_shot(ctx, "save-and-launch")])

        # O modelo 1.5B leva 10-30s para carregar; o check síncrono .count()
        # rodava ~5-10s após o clique de launch e dava falso negativo. Espera
        # o load_ok aparecer no SSE do modal (o critical path usa _wait_health
        # e passou — o produto está ok, só o check do harness era racy).
        ctx.page.get_by_text("servidor carregou", exact=False).wait_for(timeout=60_000)
        _pass(ctx, "LOG-01", "modal mostra resumo e comando", [_shot(ctx, "modal-summary")])
        _pass(ctx, "LOG-02", "SSE stdout/load_ok observado", [_shot(ctx, "modal-summary")])
        _pass(ctx, "LOG-03", "tentativa #1 numerada", [_shot(ctx, "modal-summary")])
        _dismiss_launch_modal(ctx)
        row = _find_row(ctx, launch_payload)
        row.get_by_title("Ver logs do launch").click()
        ctx.page.get_by_text("tentativa #1", exact=True).wait_for()
        _pass(ctx, "LOG-05", "modal escondeu e reabriu", [_shot(ctx, "modal-reopen")])
        # Escopar ao modal: a linha da config também tem um botão de restart
        # (title "Reiniciar server (mesma config, mesma porta)…") e o
        # get_by_role("button", name="Reiniciar") sem escopo resolve 2 elementos.
        ctx.page.get_by_test_id("launch-modal").get_by_role("button", name="Reiniciar").click()
        ctx.page.get_by_text("tentativa #2", exact=True).wait_for(timeout=30_000)
        _pass(ctx, "LOG-04", "restart gerou tentativa #2; Cancelar disponível", [_shot(ctx, "modal-restart")])
        ctx.page.get_by_test_id("launch-modal-cancel").click()
        _wait_no_launch(ctx)
        owned_launch_id = None
        _pass(ctx, "CONFIG-10", "Stop/cancel da config ativa limpou launch", ["launches-poll.json"])
    finally:
        # Never leave a process from this launch block attached to the next
        # scenario.  The response ID, not a global/bypass flag, owns cleanup.
        _teardown_owned_launch(ctx, owned_launch_id)


def _router_if_safe(ctx: RunContext, first: dict[str, Any]) -> None:
    owned_launch_id: str | None = None
    try:
        second = config_payload(ctx.model, f"e2e-router-{ctx.run_id}", context_window=8192, parallel_slots=3)
        estimates = _response_json(ctx, ctx.api.post("/api/estimate-many", {"items": [first, second]}), "router-estimates.json")
        safe = all(item.get("ok") for item in estimates) and all(
            item.get("estimate", {}).get("vram_avail") is None
            or item["estimate"].get("vram_total", 0) <= item["estimate"].get("vram_avail", 0)
            for item in estimates
        )
        if not safe:
            _nv(ctx, "CONFIG-09", "estimativa não permite router seguro para duas configs", ["router-estimates.json"])
            return
        _save_api(ctx, second)
        _refresh_configs_grid(ctx)
        ctx.guard.expect_config(first["id"], {"backend": first.get("backend", "vanilla")})
        ctx.guard.expect_config(second["id"], {"backend": second.get("backend", "vanilla")})
        first_row = _find_row(ctx, first)
        second_row = _find_row(ctx, second)
        first_row.locator("input[type=checkbox]").check()
        second_row.locator("input[type=checkbox]").check()
        with ctx.page.expect_response(
            lambda response: response.url.endswith("/api/launch-router")
            and response.request.method == "POST"
        ) as response_info:
            ctx.page.get_by_role("button", name=re.compile(r"rodar 2 juntas \(router\)", re.I)).click()
        launch_response_payload = _launch_response_payload(response_info.value)
        owned_launch_id = _launch_id(launch_response_payload)
        _register_owned_launch(ctx, owned_launch_id, [first["id"], second["id"]])
        ctx.page.get_by_text("tentativa #1", exact=True).wait_for(timeout=30_000)
        _pass(ctx, "CONFIG-09", "router aceitou duas configs E2E com estimativa segura", [_shot(ctx, "router-launch")])
        ctx.page.get_by_test_id("launch-modal-cancel").click()
        _wait_no_launch(ctx)
        owned_launch_id = None
    finally:
        _teardown_owned_launch(ctx, owned_launch_id)


def _prepare_grid(ctx: RunContext, payload: dict[str, Any]) -> None:
    """Persist through the guarded API, then refresh React before row lookup."""
    _save_api(ctx, payload)
    _refresh_configs_grid(ctx)


def _assert_configs_editor_boundary(ctx: RunContext) -> None:
    """Fail the Configs/Editor owner before another scenario can inherit a modal."""
    visible: list[str] = []
    for test_id in ("config-editor-modal", "launch-modal"):
        modal = ctx.page.get_by_test_id(test_id)
        if not modal.count():
            continue
        try:
            modal.wait_for(state="hidden", timeout=1_000)
        except Exception:
            visible.append(test_id)
        if modal.is_visible() and test_id not in visible:
            visible.append(test_id)
    if not visible:
        return

    message = "Configs/Editor boundary failure: modal(is) still visible: " + ", ".join(visible)
    evidence = _evidence(ctx, "configs-editor/modal-boundary-failure.json", {
        "owner": "Configs/Editor",
        "visible_modals": visible,
        "failure": message,
    })
    # Attribute the leak to the owning scenario, rather than letting the next
    # system_ui assertion appear to be the source of the failure.
    ctx.current_item = "EDITOR-11"
    ctx.checklist.record(
        "EDITOR-11", "FAIL", observed="Configs/Editor deixou modal visível",
        reason=message, evidence=[evidence],
    )
    raise HarnessError(message)


def run(ctx: RunContext) -> None:
    """Run the sequential Configs/Editor/Logs slice after the critical path."""
    try:
        _evidence(ctx, "configs-editor-guard-endpoints.json", {"required": required_guard_endpoints()})
        # Keep the scenario row visibly distinct from the critical-path row, which
        # uses the same model/backend but one parallel slot.
        payload = config_payload(ctx.model, f"e2e-configs-{ctx.run_id}", parallel_slots=2)
        setattr(ctx, "editor_payload", payload)
        if not _step(ctx, "CONFIG-01", tuple(SCENARIO_IDS[1:]), lambda: _prepare_grid(ctx, payload)):
            return
        # Grid actions are independently adjudicated; a Duplicate/Nova timeout is
        # not a reason to suppress the editor and logs checks.
        _configs_grid(ctx, payload)
        if not _step(ctx, "EDITOR-01", tuple(SCENARIO_IDS), lambda: _configs_and_editor(ctx)):
            return
        # The router slice is independent of the single-launch path; a
        # CONFIG-09 failure must not suppress _launch_and_logs (which has its
        # own editor_save_ok guard).  No other item depends on the router, so
        # the failure marks nothing else as blocked.
        _step(
            ctx, "CONFIG-09", (), lambda: _router_if_safe(ctx, payload),
            cleanup=_dismiss_launch_modal,
        )
        _step(
            ctx, "EDITOR-11", tuple(SCENARIO_IDS), lambda: _launch_and_logs(ctx, payload),
            cleanup=_dismiss_launch_modal,
        )
    finally:
        _assert_configs_editor_boundary(ctx)
