"""Opt-in scenario: launch every seeded profile once, complete, stop (Fase 3).

Sequential by design: port 8421 is unique and the launcher refuses a second
launch with 409. For each profile: estimate with positive folga -> launch via
the guarded API -> /health 200 -> one real completion (image + "OCR" for the
vision profiles) -> stop -> VRAM back to baseline -> no orphan llama-server.
Evidence per profile goes to the run evidence dir and to ``logs/diario/``.
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from harness import (
    INFERENCE_BASE,
    HarnessError,
    median_vram,
    port_occupied,
)

# Profiles that must exist in the seed manifest and their completion protocol.
EXPECTED_PROFILE_IDS = (
    "agente-codigo-27b-mtp",
    "contexto-longo-27b",
    "chat-ferramentas-ornith",
    "visao-27b-mtp",
    "ocr-glm",
)
VISION_PROFILES = frozenset({"visao-27b-mtp", "ocr-glm"})

LAUNCH_TIMEOUT_S = 900.0      # big 27B models take a while to load
COMPLETION_TIMEOUT_S = 300.0
VRAM_SETTLE_TIMEOUT_S = 240.0
VRAM_TOLERANCE_MIB = 64


def _json(api: Any, response: Any) -> Any:
    return api.json(response)


def _gpu_used_mib(ctx: Any) -> int:
    payload = _json(ctx.api, ctx.api.get("/api/gpu"))
    value = payload.get("vram_used_mib") if isinstance(payload, dict) else None
    if not isinstance(value, int):
        raise HarnessError(f"/api/gpu sem vram_used_mib: {payload!r}")
    return value


def _median_vram_readings(ctx: Any) -> int:
    readings = [_gpu_used_mib(ctx) for _ in range(3)]
    return median_vram(readings)


def _wait_health(ctx: Any, timeout: float = LAUNCH_TIMEOUT_S) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            response = ctx.api.get("/health", base=INFERENCE_BASE)
            if response.status == 200:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("status") == "ok":
                    return payload
                last = payload
        except Exception as exc:  # startup race only
            last = str(exc)
        time.sleep(0.5)
    raise HarnessError(f"/health não ficou pronto: {last}")


def _wait_launches_empty(ctx: Any, timeout: float = 120.0) -> list[Any]:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = _json(ctx.api, ctx.api.get("/api/launches"))
        if last == []:
            return last
        time.sleep(0.5)
    raise HarnessError(f"/api/launches não esvaziou: {last!r}")


def _wait_port_free(port: int, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_occupied(port):
            return
        time.sleep(0.5)
    raise HarnessError(f"porta {port} não ficou livre após Stop")


def _wait_vram_baseline(ctx: Any, baseline: int, timeout: float = VRAM_SETTLE_TIMEOUT_S) -> list[int]:
    deadline = time.monotonic() + timeout
    readings: list[int] = []
    while time.monotonic() < deadline:
        readings = []
        for _ in range(3):
            try:
                readings.append(_gpu_used_mib(ctx))
            except HarnessError:
                break
            time.sleep(0.5)
        if len(readings) == 3 and all(value <= baseline + VRAM_TOLERANCE_MIB for value in readings):
            return readings
        time.sleep(1)
    raise HarnessError(
        f"VRAM pós-Stop não estabilizou em baseline +{VRAM_TOLERANCE_MIB} MiB "
        f"(baseline={baseline}, última leitura={readings[-1] if readings else 'n/a'})"
    )


def _orphan_llama_servers() -> list[str]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", "llama-server"], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"pgrep indisponível: {exc}"]
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    # pgrep -f casa a própria linha de comando de wrappers (ex.: bash -c com o
    # padrão no argumento); exclui essas linhas — um llama-server real nunca
    # contém "pgrep" na linha de comando, e o pgrep nunca lista a si mesmo.
    return [line for line in lines if "pgrep" not in line]


def _drain_launch_events(ctx: Any, launch_id: str) -> list[dict[str, Any]]:
    """Replay endpoint: drains after the session has closed (post-Stop)."""
    response = ctx.api.get(f"/api/launch/{launch_id}/events", timeout=60_000)
    if not response.ok:
        raise HarnessError(f"eventos do launch HTTP {response.status}: {response.text()[:300]}")
    events: list[dict[str, Any]] = []
    for line in response.text().splitlines():
        if not line.startswith("data:"):
            continue
        try:
            value = json.loads(line[5:].strip())
        except json.JSONDecodeError as exc:
            raise HarnessError(f"evento SSE inválido: {line!r}") from exc
        if isinstance(value, dict):
            events.append(value)
    return events


def _ocr_image_png() -> bytes:
    """Deterministic local image with known text (no personal documents)."""
    from PIL import Image, ImageDraw, ImageFont  # lazy: only vision profiles

    image = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
    except Exception:
        font = ImageFont.load_default()
    draw.text((40, 40), "Hello OCR 12345", fill="black", font=font)
    draw.text((40, 130), "GLM-OCR test", fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _complete(ctx: Any, profile_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    alias = Path(str(cfg["model"])).stem
    if profile_id in VISION_PROFILES:
        image_b64 = base64.b64encode(_ocr_image_png()).decode("ascii")
        content: Any = [
            {"type": "text", "text": "OCR"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
        expected = "Hello OCR 12345"
    else:
        content = "Hello"
        expected = None
    payload = {
        "model": alias,
        "messages": [{"role": "user", "content": content}],
        # 27B são modelos de raciocínio: com max_tokens baixo a resposta inteira
        # pode ir para reasoning_content e content fica vazio (flake observado).
        "max_tokens": 256,
    }
    started = time.monotonic()
    response = ctx.api.post(
        "/v1/chat/completions", data=payload, base=INFERENCE_BASE,
        timeout=int(COMPLETION_TIMEOUT_S * 1000),
    )
    elapsed = time.monotonic() - started
    data = _json(ctx.api, response)
    try:
        message = data["choices"][0]["message"]
        usage = data.get("usage", {})
    except (KeyError, IndexError, TypeError) as exc:
        raise HarnessError(f"{profile_id}: completion sem schema esperado: {data!r}") from exc
    content_text = message.get("content")
    reasoning_text = message.get("reasoning_content")
    text = content_text if isinstance(content_text, str) and content_text.strip() else reasoning_text
    if not isinstance(text, str) or not text.strip():
        raise HarnessError(f"{profile_id}: completion vazia: {data!r}")
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        raise HarnessError(f"{profile_id}: usage sem completion_tokens: {usage!r}")
    tokens_per_s = completion_tokens / elapsed if elapsed > 0 else 0.0
    expected_found = expected is None or expected in text
    if expected is not None and not expected_found:
        raise HarnessError(f"{profile_id}: OCR sem o texto esperado: {text!r}")
    return {
        "model_alias": alias,
        "prompt_kind": "image+OCR" if profile_id in VISION_PROFILES else "text",
        "elapsed_s": round(elapsed, 3),
        "usage": usage,
        "tokens_per_s": round(tokens_per_s, 3),
        "text": text,
        "expected_found": expected_found,
    }


def _run_profile(ctx: Any, profile_id: str, cfg: dict[str, Any], baseline_vram: int) -> dict[str, Any]:
    result: dict[str, Any] = {"profile_id": profile_id, "config": cfg}

    estimate_body = {
        "model": cfg["model"],
        "mmproj": cfg.get("mmproj"),
        "backend": cfg["backend"],
        "context_window": cfg["context_window"],
        "kv_cache": cfg["kv_cache"],
        "parallel_slots": cfg.get("parallel_slots", 1),
        "gpu_layers": cfg.get("gpu_layers", 99),
        "cache_ram": cfg.get("cache_ram", 8192),
        "mode": "server",
    }
    estimate = _json(ctx.api, ctx.api.post("/api/estimate", data=estimate_body))
    folga = int(estimate["vram_avail"]) - int(estimate["vram_total"])
    result["estimate"] = {
        "vram_weights": estimate["vram_weights"],
        "vram_mmproj_weights": estimate["vram_mmproj_weights"],
        "vram_kv": estimate["vram_kv"],
        "vram_total": estimate["vram_total"],
        "vram_avail": estimate["vram_avail"],
        "folga_mib": folga,
    }
    if folga <= 0:
        raise HarnessError(f"{profile_id}: estimativa sem folga positiva (folga={folga} MiB)")

    result["baseline_vram_mib"] = _median_vram_readings(ctx)

    response = ctx.api.post("/api/launch", data=cfg, timeout=60_000)
    launch_payload = _json(ctx.api, response)
    launch_id = launch_payload.get("launch_id") if isinstance(launch_payload, dict) else None
    if not isinstance(launch_id, str):
        raise HarnessError(f"{profile_id}: /api/launch sem launch_id: {launch_payload!r}")
    result["launch_id"] = launch_id

    try:
        health = _wait_health(ctx)
        result["health"] = health
        completion = _complete(ctx, profile_id, cfg)
        result["completion"] = completion
        if profile_id in VISION_PROFILES:
            launch_log = Path(ctx.root) / "logs" / "launches" / f"{launch_id}.log"
            lines = launch_log.read_text(encoding="utf-8", errors="replace") if launch_log.is_file() else ""
            result["multimodal_log_line"] = "loaded multimodal model" in lines
            if "loaded multimodal model" not in lines:
                raise HarnessError(f"{profile_id}: log sem 'loaded multimodal model': {launch_log}")
    finally:
        cancel = _json(ctx.api, ctx.api.post(f"/api/launch/{launch_id}/cancel", data={}, timeout=60_000))
        result["cancel"] = cancel
        _wait_launches_empty(ctx)
        _wait_port_free(8421)
        settle = _wait_vram_baseline(ctx, result["baseline_vram_mib"])
        result["vram_after_stop_mib"] = median_vram(settle)
        orphans = _orphan_llama_servers()
        result["orphan_llama_servers"] = orphans
        if orphans:
            raise HarnessError(f"{profile_id}: processos llama-server órfãos após Stop: {orphans}")
        events = _drain_launch_events(ctx, launch_id)
        result["events_types"] = [event.get("type") for event in events]
        result["degrade_events"] = [
            event for event in events if event.get("type") == "degrade"
        ]
        if not any(event.get("type") == "load_ok" for event in events):
            raise HarnessError(f"{profile_id}: eventos sem load_ok: {result['events_types']}")
        result["stop_ok"] = True
    return result


def run(runner: Any) -> list[dict[str, Any]]:
    """Run the profile scenario; ``runner`` is the E2ERunner with an open RunContext."""
    ctx = runner.run_context
    if ctx is None:
        raise HarnessError("profile-e2e exige RunContext aberto")
    profile_configs = ctx.guard.load_profile_configs()
    missing = [profile_id for profile_id in EXPECTED_PROFILE_IDS if profile_id not in profile_configs]
    if missing:
        raise HarnessError(f"perfis esperados ausentes do seed: {missing}")
    persisted = _json(ctx.api, ctx.api.get("/api/configs"))
    by_id = {item["id"]: item for item in persisted if isinstance(item, dict) and item.get("id") in EXPECTED_PROFILE_IDS}
    if set(by_id) != set(EXPECTED_PROFILE_IDS):
        raise HarnessError(f"configs persistidas divergem dos perfis esperados: {sorted(by_id)}")

    baseline_vram = _median_vram_readings(ctx)
    diario_dir = Path(ctx.root) / "logs" / "diario"
    diario_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for profile_id in EXPECTED_PROFILE_IDS:
        print(f"PROFILE-E2E: {profile_id}")
        result = _run_profile(ctx, profile_id, by_id[profile_id], baseline_vram)
        results.append(result)
        evidence = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        ctx.evidence(f"profiles-e2e/{profile_id}.json").write_text(evidence, encoding="utf-8")
        (diario_dir / f"{profile_id}.json").write_text(evidence, encoding="utf-8")
        print(f"PROFILE-E2E: {profile_id} -> {result.get('completion', {}).get('tokens_per_s')} tok/s")

    summary = {
        "scenario": "profiles-e2e",
        "baseline_vram_mib": baseline_vram,
        "profiles": results,
        "ok": True,
    }
    ctx.evidence("profiles-e2e/summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (diario_dir / "profiles-e2e-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return results
