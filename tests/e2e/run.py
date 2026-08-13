#!/usr/bin/env python3
"""Sequential, runner-owned Patchright suite for Phase 6."""
from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from checklist import CHECKLIST_ITEMS, Checklist
from critical_path import run as run_critical_path
from harness import (
    CONTROL_BASE,
    LOCK_PATH,
    BackendProcess,
    GGUFInventory,
    GuardedAPI,
    HarnessError,
    MutationGuard,
    REPO_ROOT,
    RuntimeState,
    RunContext,
    assert_ports_free,
    allowed_model_path,
    port_occupied,
)
from scenarios.configs_editor import run as run_configs_editor
from scenarios.mcp import run as run_mcp
from scenarios.models_download import run as run_models_download
from scenarios.profiles_e2e import run as run_profiles_e2e
from scenarios.system_ui import run as run_system_ui


class ExclusiveLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle: Any = None

    def acquire(self) -> None:
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise HarnessError(f"lock E2E ocupado: {self.path}") from exc
        self.handle.write(f"pid={os.getpid()}\n")
        self.handle.flush()

    def release(self) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class E2ERunner:
    def __init__(self, run_id: str):
        self.root = REPO_ROOT
        self.run_id = run_id
        self.evidence_dir = self.root / "logs" / "fase6-e2e" / run_id
        self.lock = ExclusiveLock(LOCK_PATH)
        self.checklist = Checklist(run_id)
        self.guard: MutationGuard | None = None
        self.runtime = RuntimeState(self.root)
        self.ggufs = GGUFInventory(self.root)
        self.backend = BackendProcess(self.root, self.evidence_dir / "backend.log")
        self.cleaned = False
        self.previous_handlers: dict[int, Any] = {}
        self.run_context: RunContext | None = None
        self.patchright_version: str | None = None
        self.external_hf = False
        self.profile_e2e = False
        self.playwright: Any = None
        self.browser: Any = None
        self.browser_context: Any = None
        self.last_item = "CP-01"

    def _browser_env(self) -> dict[str, str]:
        env = dict(os.environ)
        local_libs = Path.home() / ".cache/llm-launcher-amd/e2e-libs/root/usr/lib/x86_64-linux-gnu"
        if local_libs.is_dir():
            previous = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{local_libs}:{previous}" if previous else str(local_libs)
        return env

    def _write_environment(self) -> None:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "repo_root": str(self.root),
            "control_base": CONTROL_BASE,
            "inference_base": "http://127.0.0.1:8421",
            "allowlisted_model": str(self.root / "runtime/fase4-models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
            "python": sys.executable,
            "patchright_version": self.patchright_version,
            "profile_e2e": self.profile_e2e,
            "e2e_model_path": os.environ.get("E2E_MODEL_PATH", ""),
            "pid": os.getpid(),
            "started_at": time.time(),
        }
        (self.evidence_dir / "environment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _preflight(self) -> None:
        Checklist.validate_manifest()
        self.recover_pending_model()
        model = allowed_model_path(self.root)
        if not (self.root / "app" / "dist").is_dir():
            raise HarnessError("app/dist ausente; build da UI é pré-requisito")
        if not (self.root / "app" / ".venv" / "bin" / "python").is_file():
            raise HarnessError("app/.venv/bin/python ausente")
        from importlib.metadata import PackageNotFoundError, version
        try:
            self.patchright_version = version("patchright")
        except PackageNotFoundError as exc:
            raise HarnessError("Patchright não instalado no ambiente do runner") from exc
        if self.patchright_version != "1.61.2":
            raise HarnessError(f"Patchright incompatível: {self.patchright_version}; esperado 1.61.2")
        self._write_environment()
        assert_ports_free()
        if (self.root / "app/api/api_running.json").exists():
            raw = (self.root / "app/api/api_running.json").read_text(encoding="utf-8").strip()
            if raw not in {"", "{}", "[]"}:
                raise HarnessError(f"registry já possui launch: {raw}")
        self.guard = MutationGuard(self.root, self.run_id)
        self.guard.prepare_download_root()
        self.ggufs.capture()
        print(f"preflight ok: model={model}")

    def recover_pending_model(self) -> None:
        """Delegate pending model recovery without coupling to the Models lane."""
        try:
            from scenarios import models_download
        except ImportError:
            models_download = None
        helper = getattr(models_download, "recover_pending_model", None) if models_download is not None else None
        if callable(helper):
            helper(self.root)
            return
        model = self.root / "runtime/fase4-models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        backup = model.with_name(f"{model.stem}.gguf.e2e-backup")
        if backup.is_symlink() or backup.exists():
            raise HarnessError(f"recovery helper ausente com backup pendente: {backup}")

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt(f"sinal recebido: {signum}")
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)

    def _restore_signal_handlers(self) -> None:
        for signum, previous in self.previous_handlers.items():
            signal.signal(signum, previous)
        self.previous_handlers.clear()

    def close_browser_context(self) -> None:
        """Close only the browser owned by this runner."""
        context, browser = self.browser_context, self.browser
        if self.run_context is not None:
            self.last_item = self.run_context.current_item
        self.browser_context = None
        self.browser = None
        self.run_context = None
        if context is not None:
            context.close()
        if browser is not None:
            browser.close()

    def stop_backend(self) -> None:
        self.backend.stop()
        assert_ports_free()

    def start_backend(self, env: Mapping[str, str]) -> None:
        self.backend.start(env)
        self.backend.wait_options()

    def open_browser_context(self) -> RunContext:
        if self.playwright is None:
            raise HarnessError("Patchright não inicializado")
        if self.guard is None:
            raise HarnessError("guard E2E não inicializado")
        self.close_browser_context()
        self.browser = self.playwright.chromium.launch(headless=True, env=self._browser_env())
        self.browser_context = self.browser.new_context()
        self.guard.install_browser(self.browser_context)
        page = self.browser_context.new_page()
        api = GuardedAPI(self.browser_context.request, self.guard)
        self.run_context = RunContext(
            self.root, self.run_id, self.evidence_dir, page, api, self.guard, self.checklist,
        )
        return self.run_context

    def install_mcp_guard_extensions(self, guard: Any) -> None:
        from scenarios.mcp import install_mcp_guard_extension
        install_mcp_guard_extension(guard, root=self.root)

    def _control_request(self, path: str, method: str = "GET", payload: object | None = None) -> tuple[int, Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{CONTROL_BASE}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urlopen(request, timeout=2) as response:
                raw = response.read()
                return response.status, json.loads(raw.decode("utf-8")) if raw else None
        except HTTPError as exc:
            raw = exc.read()
            try:
                body = json.loads(raw.decode("utf-8")) if raw else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = raw.decode("utf-8", errors="replace")
            return exc.code, body
        except (OSError, URLError) as exc:
            raise HarnessError(f"controle indisponível {method} {path}: {exc}") from exc

    def _wait_cleanup_launches(self, timeout: float = 30.0) -> list[Any]:
        deadline = time.monotonic() + timeout
        last: list[Any] = []
        while time.monotonic() < deadline:
            status, payload = self._control_request("/api/launches")
            if status != 200 or not isinstance(payload, list):
                raise HarnessError(f"/api/launches inválido durante cleanup: HTTP {status}: {payload!r}")
            last = payload
            if not last:
                return last
            time.sleep(0.25)
        raise HarnessError(f"/api/launches não esvaziou no cleanup: {last!r}")

    def _wait_cleanup_port_free(self, port: int, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not port_occupied(port):
                return
            time.sleep(0.25)
        raise HarnessError(f"porta {port} não ficou livre no cleanup")

    def _cleanup(self) -> bool:
        if self.cleaned:
            return True
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        steps: list[dict[str, Any]] = []
        result: dict[str, Any] = {"run_id": self.run_id, "steps": steps, "errors": errors}
        try:
            result["ports_before"] = {str(port): port_occupied(port) for port in (8420, 8421)}
        except Exception as exc:
            errors.append(f"ports before: {str(exc) or repr(exc)}")
        try:
            result["state_hashes_expected"] = self.runtime._expected_inventory()
            result["state_hashes_before"] = self.runtime.inventory()
        except Exception as exc:
            errors.append(f"state hashes before: {str(exc) or repr(exc)}")
            result["state_hashes_before"] = None
        result["gguf_inventory_before"] = dict(self.ggufs.items)

        backend_running = self.backend.process is not None and self.backend.process.poll() is None
        if backend_running:
            owned_launches = sorted(self.guard.owned_launches) if self.guard is not None else []
            profile_launches = sorted(self.guard.owned_profile_launches) if self.guard is not None else []
            cancel_step: dict[str, Any] = {
                "name": "cancel_owned_launches", "owned": owned_launches,
                "profiles": profile_launches, "cancelled": [],
            }
            try:
                if self.guard is None:
                    raise HarnessError("guard ausente para cancelamento fail-closed")
                for launch_id in [*owned_launches, *profile_launches]:
                    url = f"{CONTROL_BASE}/api/launch/{launch_id}/cancel"
                    self.guard.validate("POST", url)
                    status, _ = self._control_request(f"/api/launch/{launch_id}/cancel", "POST")
                    if status not in {200, 404}:
                        raise HarnessError(f"cancel launch {launch_id}: HTTP {status}")
                    cancel_step["cancelled"].append({
                        "launch_id": launch_id,
                        "status": status,
                        "profile": self.guard.owned_profile_launches.get(launch_id),
                    })
                steps.append({**cancel_step, "ok": True})
            except Exception as exc:
                message = str(exc) or repr(exc)
                errors.append(f"cancel owned launches: {message}")
                steps.append({**cancel_step, "ok": False, "error": message})
            try:
                launches = self._wait_cleanup_launches()
                steps.append({"name": "wait_launches_empty", "ok": True, "launches": launches})
            except Exception as exc:
                message = str(exc) or repr(exc)
                errors.append(f"wait launches empty: {message}")
                steps.append({"name": "wait_launches_empty", "ok": False, "error": message})
            try:
                self._wait_cleanup_port_free(8421)
                steps.append({"name": "wait_inference_port_free", "ok": True, "port": 8421})
            except Exception as exc:
                message = str(exc) or repr(exc)
                errors.append(f"wait 8421 free: {message}")
                steps.append({"name": "wait_inference_port_free", "ok": False, "error": message, "port": 8421})
        else:
            steps.append({"name": "cancel_owned_launches", "ok": True, "skipped": "backend not started"})

        try:
            self.stop_backend()
            steps.append({"name": "stop_backend", "ok": True})
        except Exception as exc:
            message = str(exc) or repr(exc)
            errors.append(f"backend stop: {message}")
            steps.append({"name": "stop_backend", "ok": False, "error": message})

        try:
            self.recover_pending_model()
            steps.append({"name": "recover_pending_model", "ok": True})
        except Exception as exc:
            message = str(exc) or repr(exc)
            errors.append(f"pending model recovery: {message}")
            steps.append({"name": "recover_pending_model", "ok": False, "error": message})

        try:
            after_gguf = GGUFInventory(self.root).capture()
            result["gguf_inventory_after"] = after_gguf
            gguf_match = not self.ggufs.captured or after_gguf == self.ggufs.items
            result["gguf_inventory_match"] = gguf_match
            if not gguf_match:
                raise HarnessError("GGUF de runtime/production-models mudou durante o E2E")
            steps.append({"name": "verify_gguf_inventory", "ok": True})
        except Exception as exc:
            message = str(exc) or repr(exc)
            errors.append(f"GGUF verify: {message}")
            result["gguf_inventory_after"] = None
            steps.append({"name": "verify_gguf_inventory", "ok": False, "error": message})

        try:
            self.runtime.restore(self.backend.stopped)
            steps.append({"name": "restore_runtime_state", "ok": True})
        except Exception as exc:
            message = str(exc) or repr(exc)
            errors.append(f"state restore: {message}")
            steps.append({"name": "restore_runtime_state", "ok": False, "error": message})

        try:
            if self.backend.stopped() and self.guard is not None and self.guard.download_root.exists():
                sentinel = self.guard.download_root / ".llm-launcher-amd-e2e"
                if sentinel.is_file() and sentinel.read_text(encoding="utf-8") == f"run_id={self.run_id}\n":
                    import shutil
                    shutil.rmtree(self.guard.download_root)
            steps.append({"name": "cleanup_download_root", "ok": True})
        except Exception as exc:
            message = str(exc) or repr(exc)
            errors.append(f"download cleanup: {message}")
            steps.append({"name": "cleanup_download_root", "ok": False, "error": message})

        try:
            result["state_hashes_after"] = self.runtime.inventory()
            result["state_inventory_match"] = result["state_hashes_after"] == result.get("state_hashes_expected")
            if not result["state_inventory_match"]:
                raise HarnessError("inventário de estado pós-restore diverge do snapshot")
        except Exception as exc:
            message = str(exc) or repr(exc)
            errors.append(f"state hashes after: {message}")
            result["state_hashes_after"] = None
            result["state_inventory_match"] = False
        result["ports_after"] = {str(port): port_occupied(port) for port in (8420, 8421)}
        result["ok"] = not errors and not result["ports_after"].get("8421", True)
        try:
            (self.evidence_dir / "cleanup.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
            )
        except Exception as exc:
            errors.append(f"cleanup artifact: {str(exc) or repr(exc)}")
            result["ok"] = False
        if errors:
            self.checklist.suite_complete = False
            for error in errors:
                print(f"CLEANUP ERROR: {error}", file=sys.stderr)
            return False
        self.cleaned = True
        self._restore_signal_handlers()
        self.lock.release()
        return True

    def _write_report_safely(self) -> bool:
        """Write the validated report, retaining an emergency adjudication on error."""
        try:
            self.checklist.mark_unimplemented()
            self.checklist.write_json(self.evidence_dir / "checklist.json", evidence_dir=self.evidence_dir)
            self.checklist.write_markdown(self.evidence_dir / "CHECKLIST.md", evidence_dir=self.evidence_dir)
            return True
        except Exception as exc:
            message = str(exc) or repr(exc)
            try:
                (self.evidence_dir / "report-error.txt").write_text(message + "\n", encoding="utf-8")
            except Exception:
                pass
            emergency = {
                "validated": False,
                "report_error": message,
                "run_id": self.checklist.run_id,
                "suite_complete": self.checklist.suite_complete,
                "items": {
                    item.id: {
                        "section": item.section,
                        "spec_line": item.spec_line,
                        **asdict(self.checklist.results[item.id]),
                    }
                    for item in CHECKLIST_ITEMS
                },
            }
            try:
                (self.evidence_dir / "checklist-emergency.json").write_text(
                    json.dumps(emergency, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                )
            except Exception:
                pass
            return False

    def execute(self) -> int:
        self.lock.acquire()
        self._write_environment()
        self._install_signal_handlers()
        atexit.register(self._cleanup)
        exit_code = 1
        try:
            self.runtime.snapshot()
            self._preflight()
            self.backend.start()
            self.backend.wait_options()
            from patchright.sync_api import sync_playwright  # type: ignore[import-not-found]
            assert self.guard is not None
            with sync_playwright() as playwright:
                self.playwright = playwright
                self.open_browser_context()
                try:
                    assert self.run_context is not None
                    run_critical_path(self.run_context)
                    run_configs_editor(self.run_context)
                    run_system_ui(self.run_context)
                    run_models_download(self.run_context, external_hf=self.external_hf)
                    run_mcp(self)
                    if self.profile_e2e:
                        self.start_backend({})
                        if self.run_context is None:
                            self.open_browser_context()
                        run_profiles_e2e(self)
                    self.checklist.suite_complete = True
                finally:
                    self.close_browser_context()
                exit_code = 0 if self.checklist.gate_pass() else 1
        except KeyboardInterrupt as exc:
            item_id = self.run_context.current_item if self.run_context else self.last_item
            failure_path = self.evidence_dir / "failure.txt"
            message = str(exc) or repr(exc)
            failure_path.write_text(message + "\n", encoding="utf-8")
            self.checklist.fail_critical(item_id, message, ["failure.txt"])
        except Exception as exc:
            item_id = self.run_context.current_item if self.run_context else self.last_item
            failure_path = self.evidence_dir / "failure.txt"
            message = str(exc) or repr(exc)
            failure_path.write_text(message + "\n", encoding="utf-8")
            self.checklist.fail_critical(item_id, message, ["failure.txt"])
            print(f"E2E FAIL: {message}", file=sys.stderr)
        finally:
            try:
                if not self._cleanup():
                    exit_code = 1
            finally:
                if not self._write_report_safely():
                    exit_code = 1
        return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the sequential Patchright Phase 6 E2E suite")
    parser.add_argument("--run-id", default=None, help="safe evidence directory ID")
    parser.add_argument("--dry-run", action="store_true", help="validate static checklist and paths without touching runtime")
    parser.add_argument("--external-hf", action="store_true", help="enable the real HuggingFace download scenario")
    parser.add_argument("--profile-e2e", action="store_true", help="launch/complete/stop every seeded profile once (Fase 3)")
    parser.add_argument("--model-path", default=None, metavar="GGUF",
                        help="override do modelo allowlist para launch/delete (default: qwen2.5-1.5b); também via E2E_MODEL_PATH")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.model_path:
        os.environ["E2E_MODEL_PATH"] = args.model_path
    Checklist.validate_manifest()
    if args.dry_run:
        model = allowed_model_path(REPO_ROOT)
        print(json.dumps({"dry_run": True, "root": str(REPO_ROOT), "model": str(model), "checklist_items": len(CHECKLIST_ITEMS)}, ensure_ascii=False))
        return 0
    run_id = args.run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    runner = E2ERunner(run_id)
    runner.external_hf = args.external_hf
    runner.profile_e2e = args.profile_e2e
    return runner.execute()


if __name__ == "__main__":
    raise SystemExit(main())
