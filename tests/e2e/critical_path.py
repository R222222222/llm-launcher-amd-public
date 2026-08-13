"""Fail-fast critical path for Phase 6.

All control mutations go through ``RunContext.api`` and all launch/delete
actions are performed by UI locators scoped to the target config row.
"""
from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any

from harness import HarnessError, RunContext, median_vram, port_occupied, read_sysfs_gpu


def _json(api: Any, response: Any) -> Any:
    return api.json(response)


def _dismiss_launch_modal(ctx: RunContext) -> None:
    """Dismiss the launch modal through its owned UI control."""
    modal = ctx.page.get_by_test_id("launch-modal")
    if modal.count() == 0:
        return
    if modal.is_visible():
        dismiss = ctx.page.get_by_test_id("launch-modal-dismiss")
        if dismiss.count() != 1:
            raise HarnessError("launch modal visível sem launch-modal-dismiss único")
        dismiss.click(timeout=5_000)
    modal.wait_for(state="hidden", timeout=10_000)


def _record(ctx: RunContext, item_id: str, observed: str, evidence: list[str], *, reason: str = "") -> None:
    ctx.current_item = item_id
    ctx.checklist.record(item_id, "PASS", observed=observed, evidence=evidence, reason=reason)


def _baseline(ctx: RunContext) -> int:
    readings: list[int] = []
    evidence: list[str] = []
    for _ in range(3):
        payload = _json(ctx.api, ctx.api.get("/api/gpu"))
        value = payload.get("vram_used_mib") if isinstance(payload, dict) else None
        if not isinstance(value, int):
            raise HarnessError(f"/api/gpu sem vram_used_mib: {payload!r}")
        readings.append(value)
        time.sleep(0.25)
    baseline = median_vram(readings)
    path = ctx.evidence("baseline-vram.json")
    path.write_text(f'{{"readings":{readings},"median":{baseline}}}\n', encoding="utf-8")
    return baseline


def _wait_health(ctx: RunContext, timeout: float = 60.0) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            response = ctx.api.get("/health", base="http://127.0.0.1:8421")
            if response.status == 200:
                try:
                    payload = response.json()
                except Exception as exc:
                    last = f"JSON inválido: {exc}"
                else:
                    last = payload
                    if isinstance(payload, dict) and payload.get("status") == "ok":
                        ctx.evidence("health.json").write_text(
                            json.dumps({"http_status": response.status, "body": payload}, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        return response
        except Exception as exc:  # startup race only
            last = str(exc)
        time.sleep(0.5)
    raise HarnessError(f"/health não ficou pronto: {last}")


def _wait_launch_empty(ctx: RunContext, timeout: float = 30.0) -> list[Any]:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = _json(ctx.api, ctx.api.get("/api/launches"))
        if last == []:
            return last
        time.sleep(0.5)
    raise HarnessError(f"/api/launches não esvaziou: {last!r}")


def _wait_stable_vram(ctx: RunContext, baseline: int, timeout: float = 60.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        readings: list[int] = []
        for _ in range(3):
            payload = _json(ctx.api, ctx.api.get("/api/gpu"))
            value = payload.get("vram_used_mib") if isinstance(payload, dict) else None
            if not isinstance(value, int):
                break
            readings.append(value)
            time.sleep(0.5)
        if len(readings) == 3 and all(value <= baseline + 64 for value in readings):
            return readings
        time.sleep(1)
    raise HarnessError("VRAM pós-stop não estabilizou em baseline +64 MiB em 60s")


def gpu_coherence_from_snapshots(
    before: dict[str, Any], api_payload: dict[str, Any], after: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    counts = [before.get("gpu_count"), after.get("gpu_count")]
    totals = [before.get("vram_total_mib"), after.get("vram_total_mib")]
    used = [before.get("vram_used_mib"), after.get("vram_used_mib")]
    checks["gpu_count"] = isinstance(api_payload.get("gpu_count"), int) and counts[0] == counts[1] == api_payload["gpu_count"]
    checks["vram_total_mib"] = isinstance(api_payload.get("vram_total_mib"), int) and totals[0] == totals[1] == api_payload["vram_total_mib"]
    used_values = [value for value in used if isinstance(value, int)]
    if isinstance(api_payload.get("vram_used_mib"), int) and len(used_values) == 2:
        checks["vram_used_mib"] = min(used_values) - 1 <= api_payload["vram_used_mib"] <= max(used_values) + 1
    else:
        checks["vram_used_mib"] = False
    for name, snapshot in (("before", before), ("after", after)):
        cards = snapshot.get("cards", [])
        checks[f"{name}_card_sum"] = (
            isinstance(cards, list)
            and sum(card.get("total_mib", 0) for card in cards) == snapshot.get("vram_total_mib")
            and sum(card.get("used_mib", 0) for card in cards) == snapshot.get("vram_used_mib")
        )
    return {"before": before, "api": api_payload, "after": after, "checks": checks, "ok": all(checks.values())}


def _gpu_coherence(ctx: RunContext) -> dict[str, Any]:
    before = read_sysfs_gpu()
    api_payload = _json(ctx.api, ctx.api.get("/api/gpu"))
    after = read_sysfs_gpu()
    return gpu_coherence_from_snapshots(before, api_payload, after)


def _find_config_row(ctx: RunContext) -> Any:
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
            if len(cells) < 10:
                continue
            ctx_text = cells[5].replace(",", "").replace(".", "").replace(" ", "")
            if (
                ctx.model_alias in cells[3]
                and cells[4].strip() == "vanilla"
                and ctx_text == "4096"
                and cells[6].strip() == "q8_0"
                and cells[9].strip() == "1"
            ):
                candidates.append(row)
        last_count = len(candidates)
        if last_count == 1:
            return candidates[0]
        ctx.page.wait_for_timeout(200)
    dump = ctx.evidence("configs-editor/rows.json")
    dump.write_text(
        json.dumps({"candidates": last_count, "rows": last_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    raise HarnessError(f"candidatas determinísticas da linha E2E após 15s: {last_count}; dump=rows.json")


def _wait_port_closed(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_occupied(port):
            return
        time.sleep(0.25)
    raise HarnessError(f"porta {port} não fechou após Stop")


def run(ctx: RunContext) -> None:
    """Execute CP-* in order; exceptions are handled by the outer runner."""
    baseline = _baseline(ctx)
    baseline_path = ctx.evidence("baseline-vram.json")
    _record(ctx, "CP-01", "baseline capturado", [str(baseline_path.relative_to(ctx.evidence_dir))])

    config = {
        "id": f"e2e-critical-{ctx.run_id}",
        "model": str(ctx.model),
        "backend": "vanilla",
        "context_window": 4096,
        "kv_cache": "q8_0",
        "flash_attn": True,
        "gpu_layers": 99,
        "parallel_slots": 1,
        "max_tokens": 64,
        "batch_size": 512,
        "ubatch_size": 128,
        "threads_gen": 2,
        "threads_batch": 2,
        "cache_ram": 2048,
        "ctx_checkpoints": 0,
        "mlock": False,
        "mmproj": None,
        "mcp_servers_config": None,
        "verbose": False,
    }
    ctx.current_item = "CP-02"
    ctx.guard.expect_config(config["id"], config)
    saved = _json(ctx.api, ctx.api.post("/api/configs", config))
    if saved.get("config", {}).get("id") != config["id"]:
        raise HarnessError(f"config E2E não persistida: {saved!r}")
    configs = _json(ctx.api, ctx.api.get("/api/configs"))
    matches = [item for item in configs if item.get("id") == config["id"]]
    if len(matches) != 1:
        raise HarnessError(f"config E2E tem {len(matches)} ocorrências após POST")
    position = configs.index(matches[0])
    fields_equal = all(matches[0].get(field) == value for field, value in config.items())
    config_evidence = ctx.evidence("config-api.json")
    config_evidence.write_text(
        json.dumps({"id": config["id"], "count": len(matches), "position": position, "fields_equal": fields_equal}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not fields_equal:
        raise HarnessError(f"campos da config E2E divergentes: {matches[0]!r}")
    _record(ctx, "CP-02", "config única, posição e campos conferidos", ["config-api.json"])

    ctx.page.goto("http://127.0.0.1:8420")
    ctx.page.get_by_test_id("tab-configs").click()
    ctx.current_item = "CP-03"
    row = _find_config_row(ctx)
    with ctx.page.expect_response(
        lambda response: response.url.endswith("/api/launch") and response.request.method == "POST",
    ) as launch_response_info:
        row.get_by_title("Subir o llama-server com esta config").click()
    launch_response = launch_response_info.value
    if launch_response.status < 200 or launch_response.status >= 300:
        raise HarnessError(f"POST /api/launch falhou: HTTP {launch_response.status}")
    launch_payload = launch_response.json()
    launch_id = launch_payload.get("launch_id") if isinstance(launch_payload, dict) else None
    if not isinstance(launch_id, str):
        raise HarnessError(f"resposta de /api/launch sem launch_id: {launch_payload!r}")
    ctx.guard.register_launch_response(launch_response, [config["id"]])
    ctx.page.get_by_text("tentativa #1").wait_for(timeout=30_000)
    ctx.page.screenshot(path=str(ctx.evidence("screenshots/critical-launch.png")), full_page=True)
    launch_evidence = ctx.evidence("ui-launch-request.json")
    ctx.guard.write_ui_launch_evidence(launch_evidence)
    if len(ctx.guard.ui_launch_requests) != 1:
        raise HarnessError("UI não produziu exatamente um POST /api/launch")
    launch_body = ctx.guard.ui_launch_requests[0]
    if launch_body.get("id") != config["id"] or launch_body.get("model") != str(ctx.model):
        raise HarnessError(f"POST /api/launch divergente: {launch_body!r}")
    launch_response_path = ctx.evidence("critical-launch-response.json")
    launch_response_path.write_text(
        json.dumps({"launch_id": launch_id, "status": launch_response.status, "response": launch_payload}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _record(ctx, "CP-03", "launch acionado na linha determinística, payload UI e launch_id conferidos", ["ui-launch-request.json", "critical-launch-response.json"])
    _record(ctx, "CP-05", "modal mostra tentativa e comando do launch", ["screenshots/critical-launch.png"])

    ctx.current_item = "CP-04"
    _wait_health(ctx)
    _record(ctx, "CP-04", "8421 /health HTTP 200 com status=ok", ["health.json"])
    ctx.current_item = "CP-06"
    completion = _json(ctx.api, ctx.api.post(
        "/v1/chat/completions",
        {
            "model": ctx.model_alias,
            "messages": [{"role": "user", "content": "Responda apenas: E2E OK"}],
            "max_tokens": 16,
        },
        base="http://127.0.0.1:8421",
    ))
    choice = completion.get("choices", [{}])[0] if isinstance(completion, dict) else {}
    content = choice.get("message", {}).get("content", "") if isinstance(choice, dict) else ""
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if not completion or not isinstance(content, str) or not content.strip() or not finish_reason:
        raise HarnessError(f"completion inválida: {completion!r}")
    completion_path = ctx.evidence("completion.json")
    completion_path.write_text(__import__("json").dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _record(ctx, "CP-06", f"HTTP 200, finish_reason={finish_reason}, conteúdo não vazio", ["completion.json"])

    _dismiss_launch_modal(ctx)
    row = _find_config_row(ctx)
    ctx.current_item = "CP-07"
    row.get_by_title("Interromper launch").click()
    ctx.current_item = "CP-08"
    launches = _wait_launch_empty(ctx)
    _wait_port_closed(8421)
    ctx.page.screenshot(path=str(ctx.evidence("screenshots/critical-stop.png")), full_page=True)
    _record(ctx, "CP-07", "Stop acionado por botão da linha na UI", ["screenshots/critical-stop.png"])
    registry = ""
    if Path(ctx.root / "app/api/api_running.json").exists():
        registry = (ctx.root / "app/api/api_running.json").read_text(encoding="utf-8").strip()
        if registry not in {"", "{}", "[]"}:
            raise HarnessError(f"registry não vazio após Stop: {registry}")
    port_state = {"8420": port_occupied(8420), "8421": port_occupied(8421)}
    if port_state["8421"]:
        raise HarnessError("8421 continua ocupada após Stop")
    cleanup_path = ctx.evidence("critical-cleanup.json")
    cleanup_path.write_text(json.dumps({
        "launches": launches,
        "registry": registry,
        "ports": port_state,
        "registry_empty": registry in {"", "{}", "[]"},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _record(ctx, "CP-08", "launches=[], registry vazio e 8421 fechada", ["critical-cleanup.json", "API /api/launches"])

    ctx.current_item = "CP-09"
    stable = _wait_stable_vram(ctx, baseline)
    stable_path = ctx.evidence("post-stop-vram.json")
    stable_path.write_text(__import__("json").dumps({"baseline": baseline, "readings": stable}) + "\n", encoding="utf-8")
    _record(ctx, "CP-09", f"leituras={stable}, baseline={baseline}", ["post-stop-vram.json"])

    ctx.current_item = "CP-10"
    coherence = _gpu_coherence(ctx)
    coherence_path = ctx.evidence("gpu-coherence.json")
    coherence_path.write_text(__import__("json").dumps(coherence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not coherence["ok"]:
        raise HarnessError(f"GPU/API/sysfs incoerentes: {coherence}")
    _record(ctx, "CP-10", "GPU API e sysfs coerentes em bracket before/after", ["gpu-coherence.json"])
