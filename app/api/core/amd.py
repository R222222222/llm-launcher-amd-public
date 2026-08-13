"""Telemetria AMD via sysfs e ambiente HIP.

Cada chamada enumera novamente os devices DRM para suportar múltiplas GPUs e
hot-plug sem estado fixo. Um card só entra na contagem quando é AMD (vendor
0x1002) e expõe ``mem_info_vram_total``.
"""
from __future__ import annotations

import glob
import math
import os
import platform
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import psutil

from .memory import _ram_mib_pair

_MIB = 1024 * 1024
_CARD_DEVICE_GLOB = "/sys/class/drm/card*/device"
_AMD_VENDOR = "0x1002"
_AMD_MODULE_VERSION = Path("/sys/module/amdgpu/version")
_TEMP_LABEL_RE = re.compile(r"(temp\d+)_label$", re.IGNORECASE)
_CLOCK_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mhz\b", re.IGNORECASE)
_HOST_TEMP_SENSOR_PRIORITY = ("coretemp", "k10temp", "acpitz", "cpu_thermal")


@dataclass(frozen=True)
class _AmdCard:
    path: Path
    total_bytes: int
    used_bytes: int | None


def _read_int(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip(), 10)
        return value if value >= 0 else None
    except (OSError, ValueError, TypeError):
        return None


def _read_number(path: Path) -> float | None:
    """Lê um número sysfs sem permitir valores inválidos na resposta."""
    try:
        value = float(path.read_text(encoding="utf-8").strip())
        return value if math.isfinite(value) and value >= 0 else None
    except (OSError, ValueError, TypeError):
        return None


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def _hwmon_dirs(card: _AmdCard) -> list[Path]:
    try:
        return sorted((card.path / "hwmon").glob("hwmon*"))
    except (OSError, RuntimeError, TypeError):
        return []


def _hwmon_value(card: _AmdCard, filename: str, scale: float = 1) -> str:
    """Lê a primeira instância válida de um arquivo hwmon.

    Os diretórios hwmon são independentes: um sensor quebrado ou ausente não
    impede a leitura dos demais sensores nem dos demais cards.
    """
    for hwmon in _hwmon_dirs(card):
        value = _read_number(hwmon / filename)
        if value is not None:
            return _format_number(value / scale)
    return "N/A"


def _card_value(card: _AmdCard, filename: str, scale: float = 1) -> str:
    value = _read_number(card.path / filename)
    return _format_number(None if value is None else value / scale)


def _temperature_sensor(
    card: _AmdCard, label: str, with_limits: bool = False,
) -> tuple[float | None, float | None, float | None]:
    """Lê uma vez o sensor identificado pelo label.

    Devolve (current, max, crit), em milicelsius. Os limites só são lidos para
    o sensor edge, evitando leituras extras para memória/junction.
    """
    for hwmon in _hwmon_dirs(card):
        try:
            labels = sorted(hwmon.glob("temp*_label"))
        except (OSError, RuntimeError, TypeError):
            continue
        for label_path in labels:
            match = _TEMP_LABEL_RE.fullmatch(label_path.name)
            if match is None:
                continue
            try:
                sensor_label = label_path.read_text(encoding="utf-8").strip().lower()
            except (OSError, ValueError, TypeError):
                continue
            if sensor_label != label:
                continue
            sensor = match.group(1)
            current = _read_number(hwmon / f"{sensor}_input")
            maximum = _read_number(hwmon / f"{sensor}_max") if with_limits else None
            critical = _read_number(hwmon / f"{sensor}_crit") if with_limits else None
            return (current, maximum, critical)
    return (None, None, None)


def _clock_value(card: _AmdCard, filename: str) -> str:
    """Obtém a frequência da entrada atualmente marcada com ``*``."""
    for line_path in (card.path / filename,):
        try:
            lines = line_path.read_text(encoding="utf-8").splitlines()
        except (OSError, ValueError, TypeError):
            continue
        for line in lines:
            if "*" not in line:
                continue
            match = _CLOCK_RE.search(line)
            if match is not None:
                return f"{_format_number(float(match.group(1)))}Mhz"
    return "N/A"


def _driver_version() -> str:
    try:
        version = _AMD_MODULE_VERSION.read_text(encoding="utf-8").strip()
        if version:
            return version
    except (OSError, ValueError, TypeError):
        pass
    try:
        version = platform.release().strip()
        return version or "N/A"
    except (OSError, ValueError, TypeError):
        return "N/A"


def _first_sensor_temperature(readings: object) -> float | None:
    if isinstance(readings, (str, bytes)) or not isinstance(readings, Iterable):
        return None
    try:
        for reading in readings:
            current = getattr(reading, "current", None)
            if current is None and isinstance(reading, Mapping):
                current = reading.get("current")
            if current is None:
                continue
            try:
                value = float(current)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    except (TypeError, ValueError, OSError):
        return None
    return None


def _host_temp_c() -> float | None:
    """Retorna a temperatura do host sem deixar sensores quebrarem a API."""
    try:
        sensors = psutil.sensors_temperatures()
        if not isinstance(sensors, Mapping):
            return None
        items = list(sensors.items())
        for preferred in _HOST_TEMP_SENSOR_PRIORITY:
            for name, readings in items:
                if str(name).lower() == preferred:
                    value = _first_sensor_temperature(readings)
                    if value is not None:
                        return value
        for _, readings in items:
            value = _first_sensor_temperature(readings)
            if value is not None:
                return value
    except Exception:
        return None
    return None


def _discover_cards() -> list[_AmdCard]:
    """Descobre cards AMD válidos, isolando erro de cada entrada sysfs."""
    cards: list[_AmdCard] = []
    for raw_path in sorted(glob.glob(_CARD_DEVICE_GLOB)):
        path = Path(raw_path)
        try:
            vendor = (path / "vendor").read_text(encoding="utf-8").strip().lower()
            if vendor != _AMD_VENDOR:
                continue
            total_bytes = _read_int(path / "mem_info_vram_total")
            if total_bytes is None or total_bytes <= 0:
                continue
            cards.append(_AmdCard(
                path=path,
                total_bytes=total_bytes,
                used_bytes=_read_int(path / "mem_info_vram_used"),
            ))
        except (OSError, ValueError, TypeError):
            # Um card com sysfs incompleto não pode derrubar a telemetria dos
            # demais cards AMD.
            continue
    return cards


def _aggregate(cards: list[_AmdCard]) -> tuple[int | None, int | None, int | None]:
    if not cards:
        return (None, None, None)
    total_bytes = sum(card.total_bytes for card in cards)
    if any(card.used_bytes is None for card in cards):
        return (total_bytes // _MIB, None, None)
    used_bytes = sum(card.used_bytes or 0 for card in cards)
    free_bytes = sum(max(card.total_bytes - (card.used_bytes or 0), 0) for card in cards)
    return (total_bytes // _MIB, used_bytes // _MIB, free_bytes // _MIB)


def gpu_total_mib() -> int | None:
    """VRAM total em MiB, somada sobre todas as GPUs AMD válidas."""
    return _aggregate(_discover_cards())[0]


def gpu_free_mib() -> int | None:
    """VRAM livre em MiB, somada sobre todas as GPUs AMD válidas."""
    return _aggregate(_discover_cards())[2]


def gpu_count() -> int:
    """Número de GPUs AMD válidas visíveis pelo DRM/sysfs."""
    return len(_discover_cards())


def hip_env() -> dict[str, str]:
    """Cópia do ambiente, honrando HIP_VISIBLE_DEVICES sem reordenar GPUs.

    A seleção de GPU continua sob controle do ambiente/configuração do
    llama.cpp; não inventamos uma ordem AMD nem sobrescrevemos a seleção do
    operador com um índice potencialmente incorreto.
    """
    return dict(os.environ)


def _card_status(card: _AmdCard) -> dict[str, str]:
    total_mib = card.total_bytes // _MIB
    used_mib = None if card.used_bytes is None else card.used_bytes // _MIB
    free_mib = (
        None if used_mib is None
        else max(card.total_bytes - (card.used_bytes or 0), 0) // _MIB
    )
    name = card.path.parent.name or card.path.name
    edge, edge_max, edge_crit = _temperature_sensor(card, "edge", with_limits=True)
    memory_temp, _, _ = _temperature_sensor(card, "mem")
    hotspot, _, _ = _temperature_sensor(card, "junction")
    edge_limit = edge_max if edge_max is not None else edge_crit
    edge_tlimit = (
        None
        if edge is None or edge_limit is None
        else edge_limit - edge
    )
    return {
        "name": f"AMD {name}",
        "vendor": _AMD_VENDOR,
        "memory.total": str(total_mib),
        "memory.used": "N/A" if used_mib is None else str(used_mib),
        "memory.free": "N/A" if free_mib is None else str(free_mib),
        "temperature.gpu": _format_number(None if edge is None else edge / 1000),
        "temperature.memory": _format_number(
            None if memory_temp is None else memory_temp / 1000,
        ),
        "temperature.hotspot": _format_number(None if hotspot is None else hotspot / 1000),
        "temperature.gpu.limit": _format_number(
            None if edge_limit is None else edge_limit / 1000,
        ),
        "temperature.gpu.tlimit": _format_number(
            None if edge_tlimit is None else edge_tlimit / 1000,
        ),
        "fan.speed": _hwmon_value(card, "fan1_input"),
        "utilization.gpu": _card_value(card, "gpu_busy_percent"),
        "utilization.memory": _card_value(card, "mem_busy_percent"),
        "power.draw": _hwmon_value(card, "power1_average", scale=1_000_000),
        "power.limit": _hwmon_value(card, "power1_cap", scale=1_000_000),
        "clocks.sm": _clock_value(card, "pp_dpm_sclk"),
        "clocks.mem": _clock_value(card, "pp_dpm_mclk"),
        "driver_version": _driver_version(),
    }


def amd_status() -> dict:
    """Status detalhado compatível com o contrato de status da GPU existente."""
    cards = _discover_cards()
    total_mib, used_mib, free_mib = _aggregate(cards)
    if not cards:
        return {
            "available": False,
            "error": "Nenhuma GPU AMD com VRAM sysfs disponível",
            "gpus": [],
            "gpu_count": 0,
            "vram_total_mib": None,
            "vram_used_mib": None,
            "vram_free_mib": None,
        }
    return {
        "available": True,
        "gpus": [_card_status(card) for card in cards],
        "gpu_count": len(cards),
        "vram_total_mib": total_mib,
        "vram_used_mib": used_mib,
        "vram_free_mib": free_mib,
        "host_temp_c": _host_temp_c(),
    }
