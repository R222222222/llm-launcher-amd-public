"""Testes da telemetria AMD sem depender do hardware real."""
import os
import sys
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_ROOT))

from api.core import amd  # noqa: E402


def _card(tmp_path: Path, name: str, vendor: str = "0x1002") -> Path:
    device = tmp_path / name / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(vendor, encoding="utf-8")
    return device


def test_multi_gpu_totals_are_summed_dynamically(tmp_path, monkeypatch):
    card0 = _card(tmp_path, "card0")
    card1 = _card(tmp_path, "card1")
    (card0 / "mem_info_vram_total").write_text(str(4 * amd._MIB), encoding="utf-8")
    (card0 / "mem_info_vram_used").write_text(str(1 * amd._MIB), encoding="utf-8")
    (card1 / "mem_info_vram_total").write_text(str(8 * amd._MIB), encoding="utf-8")
    (card1 / "mem_info_vram_used").write_text(str(3 * amd._MIB), encoding="utf-8")
    monkeypatch.setattr(amd.glob, "glob", lambda _pattern: [str(card1), str(card0)])

    assert amd.gpu_count() == 2
    assert amd.gpu_total_mib() == 12
    assert amd.gpu_free_mib() == 8
    status = amd.amd_status()
    assert status["available"] is True
    assert status["gpu_count"] == 2
    assert status["vram_used_mib"] == 4
    assert status["vram_free_mib"] == 8
    assert [gpu["name"] for gpu in status["gpus"]] == ["AMD card0", "AMD card1"]


def test_fractional_vram_free_uses_bytes_for_card_and_aggregate(tmp_path, monkeypatch):
    card = _card(tmp_path, "card0")
    total_bytes = 4 * amd._MIB + 100
    used_bytes = 1 * amd._MIB + 200
    (card / "mem_info_vram_total").write_text(str(total_bytes), encoding="utf-8")
    (card / "mem_info_vram_used").write_text(str(used_bytes), encoding="utf-8")
    monkeypatch.setattr(amd.glob, "glob", lambda _pattern: [str(card)])

    status = amd.amd_status()

    # Floor(total) - floor(used) would incorrectly report 3 MiB; the byte
    # difference floors to 2 MiB and must match the per-card value.
    assert status["vram_total_mib"] == 4
    assert status["vram_used_mib"] == 1
    assert status["vram_free_mib"] == 2
    assert status["gpus"][0]["memory.total"] == "4"
    assert status["gpus"][0]["memory.used"] == "1"
    assert status["gpus"][0]["memory.free"] == "2"


def test_invalid_cards_and_missing_used_are_tolerated(tmp_path, monkeypatch):
    valid = _card(tmp_path, "card0")
    invalid_vendor = _card(tmp_path, "card1", vendor="0x10de")
    missing_total = _card(tmp_path, "card2")
    (valid / "mem_info_vram_total").write_text(str(2 * amd._MIB), encoding="utf-8")
    (valid / "mem_info_vram_used").write_text("not-a-number", encoding="utf-8")
    (invalid_vendor / "mem_info_vram_total").write_text(str(9 * amd._MIB), encoding="utf-8")
    monkeypatch.setattr(
        amd.glob, "glob", lambda _pattern: [str(missing_total), str(invalid_vendor), str(valid)],
    )

    assert amd.gpu_count() == 1
    assert amd.gpu_total_mib() == 2
    assert amd.gpu_free_mib() is None
    status = amd.amd_status()
    assert status["available"] is True
    assert status["gpus"][0]["memory.used"] == "N/A"


def test_sysfs_telemetry_values_units_labels_and_clock_selection(tmp_path, monkeypatch):
    card = _card(tmp_path, "card0")
    (card / "mem_info_vram_total").write_text(str(24 * amd._MIB), encoding="utf-8")
    (card / "mem_info_vram_used").write_text(str(6 * amd._MIB), encoding="utf-8")
    (card / "gpu_busy_percent").write_text("37", encoding="utf-8")
    (card / "mem_busy_percent").write_text("12", encoding="utf-8")
    (card / "pp_dpm_sclk").write_text("0: 500Mhz\n1: 2371Mhz *\n", encoding="utf-8")
    (card / "pp_dpm_mclk").write_text("0: 96Mhz *\n1: 1200Mhz\n", encoding="utf-8")
    hwmon = card / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "temp1_label").write_text("edge", encoding="utf-8")
    (hwmon / "temp1_input").write_text("42000", encoding="utf-8")
    (hwmon / "temp1_max").write_text("105000", encoding="utf-8")
    (hwmon / "temp1_crit").write_text("110000", encoding="utf-8")
    (hwmon / "temp2_label").write_text("junction", encoding="utf-8")
    (hwmon / "temp2_input").write_text("47000", encoding="utf-8")
    (hwmon / "temp3_label").write_text("mem", encoding="utf-8")
    (hwmon / "temp3_input").write_text("55000", encoding="utf-8")
    (hwmon / "fan1_input").write_text("1200", encoding="utf-8")
    (hwmon / "power1_average").write_text("17000000", encoding="utf-8")
    (hwmon / "power1_cap").write_text("300000000", encoding="utf-8")
    driver = tmp_path / "amdgpu-version"
    driver.write_text("6.3.0", encoding="utf-8")
    monkeypatch.setattr(amd, "_AMD_MODULE_VERSION", driver)
    monkeypatch.setattr(amd.glob, "glob", lambda _pattern: [str(card)])

    gpu = amd.amd_status()["gpus"][0]

    assert gpu["temperature.gpu"] == "42"
    assert gpu["temperature.memory"] == "55"
    assert gpu["temperature.hotspot"] == "47"
    assert gpu["temperature.gpu.limit"] == "105"
    assert gpu["temperature.gpu.tlimit"] == "63"
    assert gpu["fan.speed"] == "1200"
    assert gpu["utilization.gpu"] == "37"
    assert gpu["utilization.memory"] == "12"
    assert gpu["power.draw"] == "17"
    assert gpu["power.limit"] == "300"
    assert gpu["clocks.sm"] == "2371Mhz"
    assert gpu["clocks.mem"] == "96Mhz"
    assert gpu["driver_version"] == "6.3.0"
    assert gpu["memory.total"] == "24"
    assert gpu["memory.used"] == "6"
    assert gpu["memory.free"] == "18"


def test_partial_card_fields_are_na_and_do_not_affect_other_cards(tmp_path, monkeypatch):
    partial = _card(tmp_path, "card0")
    complete = _card(tmp_path, "card1")
    for card in (partial, complete):
        (card / "mem_info_vram_total").write_text(str(4 * amd._MIB), encoding="utf-8")
        (card / "mem_info_vram_used").write_text(str(1 * amd._MIB), encoding="utf-8")

    (partial / "gpu_busy_percent").write_text("invalid", encoding="utf-8")
    (partial / "pp_dpm_sclk").write_text("0: not-a-clock *", encoding="utf-8")
    partial_hwmon = partial / "hwmon" / "hwmon0"
    partial_hwmon.mkdir(parents=True)
    (partial_hwmon / "temp1_label").write_text("edge", encoding="utf-8")
    (partial_hwmon / "temp1_input").write_text("broken", encoding="utf-8")
    (partial_hwmon / "power1_average").write_text("-1", encoding="utf-8")

    (complete / "gpu_busy_percent").write_text("80", encoding="utf-8")
    complete_hwmon = complete / "hwmon" / "hwmon0"
    complete_hwmon.mkdir(parents=True)
    (complete_hwmon / "temp1_label").write_text("edge", encoding="utf-8")
    (complete_hwmon / "temp1_input").write_text("60000", encoding="utf-8")
    (complete_hwmon / "fan1_input").write_text("900", encoding="utf-8")
    (complete / "pp_dpm_sclk").write_text("0: 1800Mhz *", encoding="utf-8")
    monkeypatch.setattr(amd, "_AMD_MODULE_VERSION", tmp_path / "missing-driver")
    monkeypatch.setattr(amd.platform, "release", lambda: "kernel-test")
    monkeypatch.setattr(
        amd.glob, "glob", lambda _pattern: [str(complete), str(partial)],
    )

    status = amd.amd_status()
    first, second = status["gpus"]

    assert status["gpu_count"] == 2
    assert first["temperature.gpu"] == "N/A"
    assert first["temperature.gpu.limit"] == "N/A"
    assert first["temperature.gpu.tlimit"] == "N/A"
    assert first["temperature.memory"] == "N/A"
    assert first["power.draw"] == "N/A"
    assert first["clocks.sm"] == "N/A"
    assert first["utilization.gpu"] == "N/A"
    assert second["temperature.gpu"] == "60"
    assert second["fan.speed"] == "900"
    assert second["clocks.sm"] == "1800Mhz"
    assert second["utilization.gpu"] == "80"
    assert first["driver_version"] == second["driver_version"] == "kernel-test"


def test_edge_temperature_limit_falls_back_to_crit(tmp_path, monkeypatch):
    card = _card(tmp_path, "card0")
    (card / "mem_info_vram_total").write_text(str(2 * amd._MIB), encoding="utf-8")
    hwmon = card / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "temp1_label").write_text("edge", encoding="utf-8")
    (hwmon / "temp1_input").write_text("42000", encoding="utf-8")
    (hwmon / "temp1_crit").write_text("100000", encoding="utf-8")
    monkeypatch.setattr(amd.glob, "glob", lambda _pattern: [str(card)])

    gpu = amd.amd_status()["gpus"][0]

    assert gpu["temperature.gpu.limit"] == "100"
    assert gpu["temperature.gpu.tlimit"] == "58"


def test_edge_temperature_above_limit_preserves_negative_tlimit(tmp_path, monkeypatch):
    card = _card(tmp_path, "card0")
    (card / "mem_info_vram_total").write_text(str(2 * amd._MIB), encoding="utf-8")
    hwmon = card / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "temp1_label").write_text("edge", encoding="utf-8")
    (hwmon / "temp1_input").write_text("120000", encoding="utf-8")
    (hwmon / "temp1_max").write_text("105000", encoding="utf-8")
    monkeypatch.setattr(amd.glob, "glob", lambda _pattern: [str(card)])

    gpu = amd.amd_status()["gpus"][0]

    assert gpu["temperature.gpu.limit"] == "105"
    assert gpu["temperature.gpu.tlimit"] == "-15"


class _Temperature:
    def __init__(self, current):
        self.current = current


def test_host_temperature_prioritizes_known_sensor_names(monkeypatch):
    monkeypatch.setattr(
        amd.psutil,
        "sensors_temperatures",
        lambda: {
            "acpitz": [_Temperature(31.0)],
            "coretemp": [_Temperature(44.5)],
        },
    )

    assert amd._host_temp_c() == 44.5


def test_host_temperature_falls_back_to_first_available_sensor(monkeypatch):
    monkeypatch.setattr(
        amd.psutil,
        "sensors_temperatures",
        lambda: {"other_sensor": [_Temperature(37.25)]},
    )

    assert amd._host_temp_c() == 37.25


def test_host_temperature_empty_and_exception_return_none(monkeypatch):
    monkeypatch.setattr(amd.psutil, "sensors_temperatures", lambda: {})
    assert amd._host_temp_c() is None

    def fail():
        raise RuntimeError("sensor unavailable")

    monkeypatch.setattr(amd.psutil, "sensors_temperatures", fail)
    assert amd._host_temp_c() is None


def test_amd_status_available_uses_host_temperature(tmp_path, monkeypatch):
    card = _card(tmp_path, "card0")
    (card / "mem_info_vram_total").write_text(str(2 * amd._MIB), encoding="utf-8")
    monkeypatch.setattr(amd.glob, "glob", lambda _pattern: [str(card)])
    monkeypatch.setattr(
        amd.psutil,
        "sensors_temperatures",
        lambda: {"cpu_thermal": [_Temperature(39.75)]},
    )

    assert amd.amd_status()["host_temp_c"] == 39.75


def test_hip_env_preserves_environment_and_selection(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "do-not-rewrite")
    env = amd.hip_env()

    assert env["HIP_VISIBLE_DEVICES"] == "1,0"
    assert env["CUDA_VISIBLE_DEVICES"] == "do-not-rewrite"
    env["HIP_VISIBLE_DEVICES"] = "changed-only-in-copy"
    assert os.environ["HIP_VISIBLE_DEVICES"] == "1,0"
