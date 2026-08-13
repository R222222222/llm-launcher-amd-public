#!/usr/bin/env python3
"""Patchright acceptance probe for the AMD GPU page.

The URL is intentionally a constant loopback URL.  This probe has no URL
argument and never navigates to an external origin.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from patchright.sync_api import sync_playwright


URL = "http://127.0.0.1:8420/"
SCREENSHOT = Path(__file__).resolve().parents[1] / "logs/final/fase2-amd.png"
POLL_WAIT_MS = 8_500


def main() -> int:
    console_errors: list[str] = []
    page_errors: list[str] = []
    checks: list[dict[str, object]] = []
    screenshot_saved = False

    def record(label: str, ok: bool, details: str = "") -> None:
        checks.append({"label": label, "ok": ok, "details": details})

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1440, "height": 1200}, device_scale_factor=1)
            page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            response = page.goto(URL, wait_until="domcontentloaded", timeout=10_000)
            record("loopback root", response is not None and response.status == 200, f"status={response.status if response else None}")
            record("fixed URL", page.url == URL, f"url={page.url}")

            page.get_by_test_id("tab-amd").wait_for(state="visible", timeout=10_000)
            page.get_by_test_id("tab-amd").click()
            page.get_by_text("Dispositivos AMD", exact=True).wait_for(state="visible", timeout=10_000)
            page.wait_for_timeout(POLL_WAIT_MS)

            body = page.locator("body").inner_text()
            body_folded = body.casefold()
            required_labels = (
                "AMD GPU", "GPUs detectadas", "VRAM total", "Temperatura do host",
                "VRAM agregada", "Histórico de VRAM", "Histórico de RAM", "Dispositivos AMD",
                "Temperatura", "Edge", "Memória", "Hotspot", "Limite GPU", "Folga térmica",
                "Utilização e ventilação", "GPU", "Fan", "Energia e clocks", "Power draw",
                "Limite", "Clock GPU", "Clock memória", "driver",
            )
            missing = [label for label in required_labels if label.casefold() not in body_folded]
            record("AMD labels", not missing, f"missing={missing}")
            record("fan unit RPM", "RPM" in body, "RPM present" if "RPM" in body else "RPM absent")
            record("no object object", "[object Object]" not in body, "clean text")

            sparklines = page.locator('svg[role="img"]')
            sparkline_count = sparklines.count()
            path_details: list[list[str]] = []
            sparkline_paths_ok = sparkline_count == 2
            for index in range(sparkline_count):
                paths = sparklines.nth(index).locator("path")
                d_values = [paths.nth(path_index).get_attribute("d") or "" for path_index in range(paths.count())]
                path_details.append(d_values)
                sparkline_paths_ok = sparkline_paths_ok and any(
                    value.strip() and re.search(r"(?:^|\s)L(?:\s|$)", value) is not None
                    for value in d_values
                )
            record("two SVG charts", sparkline_paths_ok, f"charts={sparkline_count} paths={path_details}")

            point_counts = [
                int(match.group(1))
                for match in re.finditer(r"aria-label=\"[^\"]+: (\d+) leituras\"", page.content())
            ]
            point_counts = point_counts[:2]
            record("history points <= 60", len(point_counts) == 2 and all(0 < count <= 60 for count in point_counts), f"points={point_counts}")
            record("history window label", "até 60 pontos" in body, "até 60 pontos present")

            progressbars = page.locator('[role="progressbar"]')
            progress_count = progressbars.count()
            aria_ok = progress_count >= 2
            for index in range(progress_count):
                item = progressbars.nth(index)
                aria_ok = aria_ok and all(item.get_attribute(name) is not None for name in ("aria-label", "aria-valuemin", "aria-valuemax", "aria-valuetext"))
            record("ARIA progressbars", aria_ok, f"count={progress_count}")
            record("console errors", not console_errors and not page_errors, f"console={console_errors} page={page_errors}")

            SCREENSHOT.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(SCREENSHOT), full_page=True)
            screenshot_saved = SCREENSHOT.is_file() and SCREENSHOT.stat().st_size > 0
            record("screenshot", screenshot_saved, str(SCREENSHOT))
            browser.close()
    except Exception as error:
        record("probe exception", False, f"{type(error).__name__}: {error}")

    result = {
        "status": "PASS" if all(item["ok"] for item in checks) and screenshot_saved else "FAIL",
        "url": URL,
        "screenshot": str(SCREENSHOT),
        "checks": checks,
        "console_errors": console_errors,
        "page_errors": page_errors,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    for item in checks:
        print(f"{'PASS' if item['ok'] else 'FAIL'} {item['label']}: {item['details']}")
    if result["status"] == "PASS":
        print("PASS Patchright AMD telemetry UI")
        return 0
    print("FAIL Patchright AMD telemetry UI")
    return 1


if __name__ == "__main__":
    sys.exit(main())
