#!/usr/bin/env python3
"""Reproducible loopback/API-to-sysfs check for the AMD telemetry contract.

The API and sysfs are sampled consecutively, so only values which can change
between those two samples have tolerances.  Static values (VRAM total, sensor
limits, labels, units, driver and the derived ``tlimit`` formula) are checked
exactly.  Dynamic tolerances, in API units, are:

* VRAM used/free: 2 MiB;
* temperatures and host temperature: 2 C;
* fan: 100 RPM; utilisation: 5 percentage points;
* power draw: 2 W; clocks: 100 MHz.

This file deliberately uses only the Python standard library and psutil.  The
endpoint is fixed to loopback; it is not configurable from argv or env.
"""
from __future__ import annotations

import json
import http.client
import math
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil


ENDPOINT = "http://127.0.0.1:8420/api/gpu"
CARD_GLOB = "/sys/class/drm/card*/device"
AMD_VENDOR = "0x1002"
MIB = 1024 * 1024
TIMEOUT_S = 5
TEMP_LABEL_RE = re.compile(r"(temp\d+)_label$", re.IGNORECASE)
CLOCK_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mhz\b", re.IGNORECASE)

DYNAMIC_TOLERANCES: dict[str, float] = {
    "memory.used": 2,
    "memory.free": 2,
    "temperature.gpu": 2,
    "temperature.memory": 2,
    "temperature.hotspot": 2,
    "temperature.gpu.tlimit": 2,
    "fan.speed": 100,
    "utilization.gpu": 5,
    "utilization.memory": 5,
    "power.draw": 2,
    "clocks.sm": 100,
    "clocks.mem": 100,
    "host_temp_c": 2,
}


@dataclass(frozen=True)
class Card:
    path: Path
    total_bytes: int
    used_bytes: int | None


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def read_number(path: Path) -> float | None:
    raw = read_text(path)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def format_number(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def discover_cards() -> list[Card]:
    cards: list[Card] = []
    for path in sorted(Path("/sys/class/drm").glob("card*/device")):
        vendor = read_text(path / "vendor")
        if vendor is None or vendor.lower() != AMD_VENDOR:
            continue
        raw_total = read_text(path / "mem_info_vram_total")
        try:
            total = int(raw_total or "", 10)
        except ValueError:
            continue
        if total <= 0:
            continue
        raw_used = read_text(path / "mem_info_vram_used")
        try:
            used = int(raw_used or "", 10)
            used = used if used >= 0 else None
        except ValueError:
            used = None
        cards.append(Card(path, total, used))
    return cards


def hwmon_dirs(card: Card) -> list[Path]:
    try:
        return sorted((card.path / "hwmon").glob("hwmon*"))
    except (OSError, RuntimeError):
        return []


def sensor(card: Card, wanted: str) -> tuple[float | None, float | None, float | None]:
    for hwmon in hwmon_dirs(card):
        for label_path in sorted(hwmon.glob("temp*_label")):
            match = TEMP_LABEL_RE.fullmatch(label_path.name)
            label = read_text(label_path)
            if match is None or label is None or label.lower() != wanted:
                continue
            name = match.group(1)
            return (
                read_number(hwmon / f"{name}_input"),
                read_number(hwmon / f"{name}_max"),
                read_number(hwmon / f"{name}_crit"),
            )
    return (None, None, None)


def card_value(card: Card, filename: str, scale: float = 1) -> str:
    value = read_number(card.path / filename)
    return format_number(None if value is None else value / scale)


def hwmon_value(card: Card, filename: str, scale: float = 1) -> str:
    for hwmon in hwmon_dirs(card):
        value = read_number(hwmon / filename)
        if value is not None:
            return format_number(value / scale)
    return "N/A"


def clock_value(card: Card, filename: str) -> str:
    raw = read_text(card.path / filename)
    if raw is None:
        return "N/A"
    for line in raw.splitlines():
        if "*" in line:
            match = CLOCK_RE.search(line)
            if match:
                return f"{format_number(float(match.group(1)))}Mhz"
    return "N/A"


def driver_version() -> str:
    version = read_text(Path("/sys/module/amdgpu/version"))
    if version:
        return version
    return platform.release().strip() or "N/A"


def host_temp() -> float | None:
    priority = ("coretemp", "k10temp", "acpitz", "cpu_thermal")
    try:
        readings = psutil.sensors_temperatures()
        if not isinstance(readings, dict):
            return None
        ordered = []
        for preferred in priority:
            ordered.extend((name, values) for name, values in readings.items() if str(name).lower() == preferred)
        ordered.extend((name, values) for name, values in readings.items() if (name, values) not in ordered)
        for _, values in ordered:
            for item in values:
                current = getattr(item, "current", None)
                if isinstance(item, dict):
                    current = item.get("current", current)
                if current is None:
                    continue
                try:
                    value = float(current)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    return value
    except Exception:
        return None
    return None


def sysfs_gpu(card: Card) -> dict[str, str]:
    edge, edge_max, edge_crit = sensor(card, "edge")
    memory, _, _ = sensor(card, "mem")
    hotspot, _, _ = sensor(card, "junction")
    limit = edge_max if edge_max is not None else edge_crit
    tlimit = None if edge is None or limit is None else limit - edge
    total_mib = card.total_bytes // MIB
    used_mib = None if card.used_bytes is None else card.used_bytes // MIB
    free_mib = (
        None if card.used_bytes is None
        else max(card.total_bytes - (card.used_bytes or 0), 0) // MIB
    )
    return {
        "name": f"AMD {card.path.parent.name}",
        "vendor": AMD_VENDOR,
        "memory.total": str(total_mib),
        "memory.used": "N/A" if used_mib is None else str(used_mib),
        "memory.free": "N/A" if free_mib is None else str(free_mib),
        "temperature.gpu": format_number(None if edge is None else edge / 1000),
        "temperature.memory": format_number(None if memory is None else memory / 1000),
        "temperature.hotspot": format_number(None if hotspot is None else hotspot / 1000),
        "temperature.gpu.limit": format_number(None if limit is None else limit / 1000),
        "temperature.gpu.tlimit": format_number(None if tlimit is None else tlimit / 1000),
        "fan.speed": hwmon_value(card, "fan1_input"),
        "utilization.gpu": card_value(card, "gpu_busy_percent"),
        "utilization.memory": card_value(card, "mem_busy_percent"),
        "power.draw": hwmon_value(card, "power1_average", 1_000_000),
        "power.limit": hwmon_value(card, "power1_cap", 1_000_000),
        "clocks.sm": clock_value(card, "pp_dpm_sclk"),
        "clocks.mem": clock_value(card, "pp_dpm_mclk"),
        "driver_version": driver_version(),
    }


def aggregate(cards: list[Card]) -> tuple[int | None, int | None, int | None]:
    if not cards:
        return (None, None, None)
    total = sum(card.total_bytes for card in cards)
    if any(card.used_bytes is None for card in cards):
        return (total // MIB, None, None)
    used = sum(card.used_bytes or 0 for card in cards)
    free = sum(max(card.total_bytes - (card.used_bytes or 0), 0) for card in cards)
    return (total // MIB, used // MIB, free // MIB)


def api_json() -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", 8420, timeout=TIMEOUT_S)
    try:
        connection.request("GET", "/api/gpu", headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"GET {ENDPOINT}: HTTP {response.status}")
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()
    if not isinstance(payload, dict):
        raise RuntimeError("/api/gpu não retornou um objeto JSON")
    return payload


def numeric(value: object, clock: bool = False) -> float | None:
    if value is None or (isinstance(value, str) and value.strip().upper() in {"", "N/A"}):
        return None
    text = str(value).strip()
    if clock:
        match = CLOCK_RE.search(text)
        if not match:
            return None
        text = match.group(1)
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def close_enough(api_value: object, sys_value: object, tolerance: float | None, clock: bool = False) -> bool:
    a = numeric(api_value, clock)
    b = numeric(sys_value, clock)
    if a is None or b is None:
        return api_value == sys_value or (a is None and b is None and str(api_value).upper() == str(sys_value).upper())
    return tolerance is not None and abs(a - b) <= tolerance if tolerance is not None else a == b


def check(label: str, ok: bool, details: str, checks: list[dict[str, object]]) -> None:
    checks.append({"label": label, "ok": ok, "details": details})


def verify() -> tuple[dict[str, object], list[dict[str, object]]]:
    api = api_json()
    cards = discover_cards()
    checks: list[dict[str, object]] = []
    if not cards:
        raise RuntimeError("nenhum card AMD com mem_info_vram_total foi encontrado no sysfs")
    if api.get("available") is not True:
        raise RuntimeError(f"API indisponível: {api.get('error', 'sem motivo')}")
    api_gpus = api.get("gpus")
    if not isinstance(api_gpus, list) or len(api_gpus) != len(cards):
        raise RuntimeError(f"sysfs encontrou {len(cards)} card(s), API retornou {len(api_gpus) if isinstance(api_gpus, list) else 'não-lista'}")

    sys_gpus = [sysfs_gpu(card) for card in cards]
    api_gpus = sorted(api_gpus, key=lambda gpu: str(gpu.get("name", "")))
    sys_gpus = sorted(sys_gpus, key=lambda gpu: gpu["name"])
    total, used, free = aggregate(cards)
    for field, expected in (("vram_total_mib", total), ("vram_used_mib", used), ("vram_free_mib", free)):
        tolerance = 2 if field != "vram_total_mib" else None
        ok = close_enough(api.get(field), expected, tolerance)
        check(f"aggregate {field}", ok, f"api={api.get(field)!r} sysfs={expected!r}", checks)
    check("gpu_count", api.get("gpu_count") == len(cards), f"api={api.get('gpu_count')!r} sysfs={len(cards)}", checks)

    fields = (
        "name", "vendor", "memory.total", "memory.used", "memory.free",
        "temperature.gpu", "temperature.memory", "temperature.hotspot",
        "temperature.gpu.limit", "temperature.gpu.tlimit", "fan.speed",
        "utilization.gpu", "utilization.memory", "power.draw", "power.limit",
        "clocks.sm", "clocks.mem", "driver_version",
    )
    for index, (api_gpu, sys_gpu) in enumerate(zip(api_gpus, sys_gpus), 1):
        for field in fields:
            tolerance = DYNAMIC_TOLERANCES.get(field)
            ok = close_enough(api_gpu.get(field), sys_gpu[field], tolerance, field.startswith("clocks."))
            check(f"GPU {index} {field}", ok, f"api={api_gpu.get(field)!r} sysfs={sys_gpu[field]!r}", checks)
        edge = numeric(api_gpu.get("temperature.gpu"))
        limit = numeric(api_gpu.get("temperature.gpu.limit"))
        tlimit = numeric(api_gpu.get("temperature.gpu.tlimit"))
        derived_ok = edge is None or limit is None or (tlimit is not None and abs(tlimit - (limit - edge)) <= DYNAMIC_TOLERANCES["temperature.gpu.tlimit"])
        check("GPU %d tlimit=limit-current" % index, derived_ok, f"limit={limit!r} current={edge!r} tlimit={tlimit!r}", checks)

    api_host = api.get("host_temp_c")
    local_host = host_temp()
    check("host temperature", close_enough(api_host, local_host, DYNAMIC_TOLERANCES["host_temp_c"]), f"api={api_host!r} psutil={local_host!r}", checks)
    result = {
        "status": "PASS" if all(item["ok"] for item in checks) else "FAIL",
        "endpoint": ENDPOINT,
        "cards": len(cards),
        "tolerances_dynamic_only": DYNAMIC_TOLERANCES,
        "checks": checks,
    }
    return result, checks


def main() -> int:
    try:
        result, checks = verify()
    except Exception as error:
        result = {"status": "FAIL", "endpoint": ENDPOINT, "error": str(error), "checks": []}
        checks = []
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    for item in checks:
        print(f"{'PASS' if item['ok'] else 'FAIL'} {item['label']}: {item['details']}")
    if result["status"] == "PASS":
        print("PASS AMD telemetry API/sysfs coherence")
        return 0
    print(f"FAIL AMD telemetry: {result.get('error', 'mismatch')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
