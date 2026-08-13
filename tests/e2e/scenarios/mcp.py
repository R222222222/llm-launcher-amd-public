"""MCP-01..06 scenario and its runner integration contract.

The runner supplies a :class:`McpLifecycle` implementation and calls
``run(lifecycle)``.  The lifecycle owns all real resources:

* ``close_browser_context()`` closes the current page, context and browser;
* ``stop_backend()`` stops the normal backend and proves its ports are free;
* ``start_backend(env)`` starts a *new* backend with exactly the supplied
  environment; and
* ``open_browser_context()`` creates a new browser/context, installs the
  browser route guard, creates a new ``GuardedAPI`` and returns a
  :class:`harness.RunContext`.

The implementation must also provide ``install_mcp_guard_extensions(guard)``.
It should call :func:`install_mcp_guard_extension` (or an equivalent audited
extension) before the returned context is used.  This is required because the
base Phase 6 guard deliberately has no MCP mutation routes.

The enabled backend must be started with ``MCP_ENV`` and the cleanup backend
with ``MCP_DISABLED_ENV``.  In particular, remote MCP is never enabled.  No
backend or browser is started on import, and the unit tests below this module
only exercise pure helpers and the local fixture.

Additional API surface that the guard integration must explicitly audit:

* ``POST /api/mcp`` (only the fixed ``e2e-mcp-stdio`` entry and fixture
  command/cwd);
* ``POST /api/mcp/{id}/start`` and ``POST /api/mcp/{id}/stop`` (same ID);
* ``DELETE /api/mcp/{id}`` (same ID, and only after it is stopped).

The read-only endpoints used for evidence are ``GET /api/options``,
``GET /api/mcp``, ``GET /api/mcp/{id}/logs`` and
``GET /api/mcp/{id}/events``.  All requests must remain loopback requests.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlparse

try:  # Works both as tests.e2e.scenarios.mcp and from the E2E script cwd.
    from ..harness import GuardedAPI, HarnessError, RunContext
except ImportError:  # pragma: no cover - exercised by the existing runner style
    from harness import GuardedAPI, HarnessError, RunContext


MCP_SERVER_ID = "e2e-mcp-stdio"
MCP_SERVER_NAME = "e2e-mcp-stdio"
READY_SENTINEL = "LLM_LAUNCHER_E2E_MCP_READY"
READY_SENTINEL_RE = re.compile(r"^LLM_LAUNCHER_E2E_MCP_READY pid=(\d+)$")
CONTROL_BASE = "http://127.0.0.1:8420"
MCP_ENV: dict[str, str] = {
    "LLM_LAUNCHER_ENABLE_MCP": "1",
    "LLM_LAUNCHER_ALLOW_REMOTE_MCP": "0",
}
MCP_DISABLED_ENV: dict[str, str] = {
    "LLM_LAUNCHER_ENABLE_MCP": "0",
    "LLM_LAUNCHER_ALLOW_REMOTE_MCP": "0",
}
MCP_MUTATING_ENDPOINTS = (
    "POST /api/mcp",
    "POST /api/mcp/{e2e-server-id}/start",
    "POST /api/mcp/{e2e-server-id}/stop",
    "DELETE /api/mcp/{e2e-server-id}",
)
MCP_READ_ENDPOINTS = (
    "GET /api/options",
    "GET /api/mcp",
    "GET /api/mcp/{e2e-server-id}/logs",
    "GET /api/mcp/{e2e-server-id}/events",
)


class McpLifecycle(Protocol):
    """Callbacks required by :func:`run`; no concrete runner is assumed."""

    def close_browser_context(self) -> None: ...

    def stop_backend(self) -> None: ...

    def start_backend(self, env: Mapping[str, str]) -> None: ...

    def open_browser_context(self) -> RunContext: ...

    def install_mcp_guard_extensions(self, guard: Any) -> None: ...


def fixture_path(root: Path | None = None) -> Path:
    """Return the versioned fixture without touching or creating files."""
    here = Path(__file__).resolve()
    candidate = here.parents[1] / "fixtures" / "mcp_process.py"
    if root is not None:
        candidate = root.resolve() / "tests" / "e2e" / "fixtures" / "mcp_process.py"
    return candidate


def fixture_command(root: Path | None = None, *, exit_code: int | None = None) -> str:
    """Build the shell command accepted by the local MCP fixture runner."""
    command = ["exec", sys.executable, str(fixture_path(root))]
    if exit_code is not None:
        command.extend(("--exit-code", str(exit_code)))
    return shlex.join(command)


def parse_ready_pid(line: str) -> int:
    """Parse the fixture's PID-bearing READY line."""
    match = READY_SENTINEL_RE.fullmatch(line.strip())
    if match is None:
        raise HarnessError(f"sentinela MCP inválida: {line!r}")
    return int(match.group(1))


def _body(data: object | None) -> object | None:
    if isinstance(data, (dict, list)):
        return data
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    return None


def _mcp_path(path: str, server_id: str) -> bool:
    return path in {"/api/mcp", f"/api/mcp/{server_id}",
                    f"/api/mcp/{server_id}/start", f"/api/mcp/{server_id}/stop",
                    f"/api/mcp/{server_id}/logs", f"/api/mcp/{server_id}/events"}


def validate_mcp_request(
    method: str,
    url: str,
    data: object | None = None,
    *,
    root: Path | None = None,
    server_id: str = MCP_SERVER_ID,
) -> None:
    """Validate the narrow MCP allowlist used by a guard extension.

    This validator is deliberately independent from ``MutationGuard`` so it
    can be unit-tested without a backend.  It rejects remote hosts, unknown
    IDs, remote-looking command values and every mutating endpoint outside the
    four routes listed in ``MCP_MUTATING_ENDPOINTS``.
    """
    parsed = urlparse(url if url.startswith("http") else f"http://127.0.0.1:8420{url}")
    if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port not in {None, 8420}:
        raise HarnessError(f"MCP fora do loopback/porta de controle: {url}")
    path = parsed.path
    if not _mcp_path(path, server_id):
        raise HarnessError(f"endpoint MCP não allowlisted: {method.upper()} {path}")
    method = method.upper()
    if method in {"GET", "HEAD", "OPTIONS"}:
        return
    if method not in {"POST", "DELETE"}:
        raise HarnessError(f"método MCP não allowlisted: {method}")
    if not path.startswith(f"/api/mcp/{server_id}") and path != "/api/mcp":
        raise HarnessError("ID MCP fora do namespace E2E")
    if method == "DELETE":
        return
    if path != "/api/mcp":
        if path not in {f"/api/mcp/{server_id}/start", f"/api/mcp/{server_id}/stop"}:
            raise HarnessError(f"mutação MCP não allowlisted: {method} {path}")
        return
    payload = _body(data)
    if not isinstance(payload, dict):
        raise HarnessError("POST /api/mcp exige JSON objeto")
    if payload.get("id") != server_id:
        raise HarnessError("servidor MCP não usa o ID E2E fixo")
    effective_root = (root or Path(__file__).resolve().parents[2]).resolve()
    expected_cwd = str(effective_root)
    if payload.get("cwd") != expected_cwd:
        raise HarnessError("cwd MCP não é a raiz E2E")
    expected_command = fixture_command(effective_root)
    if any(isinstance(value, str) and ("http://" in value or "https://" in value) for value in payload.values()):
        raise HarnessError("remote MCP nunca é permitido")
    if payload.get("command") not in {expected_command, fixture_command(effective_root, exit_code=7)}:
        raise HarnessError("comando MCP não é a fixture local versionada")


def install_mcp_guard_extension(guard: Any, *, root: Path | None = None, server_id: str = MCP_SERVER_ID) -> None:
    """Extend an existing MutationGuard without weakening non-MCP rules.

    The runner may call this directly from its lifecycle callback.  All other
    routes continue through the original validator; installing twice is safe.
    """
    if getattr(guard, "_e2e_mcp_extension", False):
        return
    original = guard.validate

    def validate(method: str, url: str, data: object | None = None) -> None:
        parsed = urlparse(url if url.startswith("http") else f"http://127.0.0.1:8420{url}")
        if _mcp_path(parsed.path, server_id):
            validate_mcp_request(method, url, data, root=root, server_id=server_id)
            return
        original(method, url, data)

    guard.validate = validate
    guard._e2e_mcp_extension = True


def _json(api: GuardedAPI, response: Any) -> Any:
    return api.json(response)


def assert_loopback_url(url: str) -> None:
    """Reject browser/API URLs which could escape the local E2E process."""
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise HarnessError(f"URL E2E não é loopback HTTP: {url}")
    if parsed.port not in {None, 8420, 8421}:
        raise HarnessError(f"porta E2E não allowlisted: {url}")


def _wait(api: GuardedAPI, predicate: Callable[[Any], bool], timeout: float = 12.0) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = _json(api, api.get("/api/mcp"))
        try:
            if predicate(last):
                return last
        except HarnessError as exc:
            # mcp_servers.json pode ser lido no meio de um write não-atômico
            # (truncate-then-write) e vir vazio; "ausente" é transitório —
            # continua sondando até o deadline em vez de falhar na 1ª leitura.
            if "servidor MCP E2E ausente" not in str(exc):
                raise
        time.sleep(0.15)
    raise HarnessError(f"timeout esperando estado MCP: {last!r}")


def wait_for_mcp_tab(page: Any, *, expected: bool = True, timeout: float = 30.0, poll: float = 0.05) -> dict[str, Any]:
    """Poll tab presence/visibility and return timing evidence.

    A fresh page can have ``/api/options`` available before React has hydrated
    its tabs.  Counting once is therefore not a valid assertion.  This helper
    is intentionally locator-shaped and is covered with fakes in unit tests.
    The default timeout is generous because the MCP tab only renders after the
    frontend's async options load completes.
    """
    started = time.monotonic()
    deadline = started + timeout
    last_count = 0
    last_visible = False
    while time.monotonic() < deadline:
        tab = page.get_by_test_id("tab-mcp")
        last_count = tab.count()
        last_visible = bool(last_count and tab.is_visible())
        ready = last_visible if expected else last_count == 0
        if ready:
            return {
                "expected": expected,
                "count": last_count,
                "visible": last_visible,
                "wait_seconds": time.monotonic() - started,
            }
        time.sleep(poll)
    raise HarnessError(
        f"timeout aguardando tab-mcp expected={expected} count={last_count} visible={last_visible}"
    )


def _server(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, list):
        raise HarnessError(f"/api/mcp não retornou lista: {payload!r}")
    for entry in payload:
        if isinstance(entry, dict) and entry.get("id") == MCP_SERVER_ID:
            return entry
    raise HarnessError("servidor MCP E2E ausente")


def fixture_pids(root: Path) -> list[int]:
    """List live processes whose command line names this exact fixture."""
    target = str(fixture_path(root).resolve()).encode()
    found: list[int] = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            command_line = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if target in command_line and _pid_alive(pid):
            found.append(pid)
    return sorted(found)


def ready_pid_from_logs(logs_payload: Any) -> int:
    """Extract exactly one real fixture PID from the runner log payload."""
    lines = logs_payload.get("logs", []) if isinstance(logs_payload, dict) else []
    pids = [parse_ready_pid(event["text"]) for event in lines
            if isinstance(event, dict) and isinstance(event.get("text"), str)
            and event["text"].startswith(READY_SENTINEL)]
    if not pids:
        raise HarnessError("sentinela READY com PID ausente nos logs MCP")
    if len(set(pids)) != 1:
        raise HarnessError(f"mais de um PID no sentinela MCP: {pids!r}")
    return pids[0]


def _evidence(ctx: RunContext, name: str, payload: Any) -> str:
    path = ctx.evidence(name)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return str(path.relative_to(ctx.evidence_dir))


def _pass(ctx: RunContext, item: str, observed: str, name: str, payload: Any) -> None:
    evidence = _evidence(ctx, name, payload)
    ctx.checklist.record(item, "PASS", observed=observed, evidence=[evidence])


def _fail(ctx: RunContext, item: str, exc: Exception, name: str, details: Any | None = None) -> None:
    message = str(exc) or repr(exc)
    payload = {
        "error": message, "error_fallback": repr(exc), "type": type(exc).__name__,
    }
    if details is not None:
        payload["details"] = details
    evidence = _evidence(ctx, name, payload)
    ctx.checklist.record(item, "FAIL", observed="cenário MCP falhou", reason=message, evidence=[evidence])


def _nv(ctx: RunContext, item: str, reason: str) -> None:
    ctx.checklist.record(item, "NÃO VERIFICADO", observed="", reason=reason)


def _row(page: Any) -> Any:
    return page.locator("li").filter(has_text=MCP_SERVER_NAME).first


def _open_app(ctx: RunContext) -> dict[str, Any]:
    """Hydrate a newly-created browser page before inspecting feature tabs."""
    assert_loopback_url(CONTROL_BASE)
    ctx.page.goto(CONTROL_BASE)
    assert_loopback_url(ctx.page.url)
    ctx.page.get_by_test_id("tab-configs").wait_for(timeout=10_000)
    options = _json(ctx.api, ctx.api.get("/api/options"))
    return options


def _run_enabled(ctx: RunContext) -> None:
    page, api = ctx.page, ctx.api
    ctx.current_item = "MCP-01"
    options: Any = None
    tab_wait: dict[str, Any] | None = None
    try:
        options = _open_app(ctx)
        assert options.get("features", {}).get("mcp") is True, options
        tab_wait = wait_for_mcp_tab(page, expected=True)
        tab = page.get_by_test_id("tab-mcp")
        remote_blocked = False
        try:
            validate_mcp_request("GET", "http://198.51.100.10:8420/api/mcp")
        except HarnessError:
            remote_blocked = True
        assert remote_blocked, "política MCP aceitaria host remoto"
        tab.click()
        page.get_by_role("heading", name="Supervisão MCP").wait_for(timeout=10_000)
        _pass(ctx, "MCP-01", "feature mcp=true, tab MCP hidratada e host remoto bloqueado", "mcp-01.json", {
            "options": options, "tab_wait": tab_wait, "remote_blocked": remote_blocked,
        })
    except Exception as exc:
        _fail(ctx, "MCP-01", exc, "mcp-01-failure.json", {"options": options, "tab_wait": tab_wait})
        for item in ("MCP-02", "MCP-03", "MCP-04", "MCP-05"):
            _nv(ctx, item, "MCP-01 não passou")
        return

    created: dict[str, Any] = {}
    ready_pid: int | None = None
    try:
        root = ctx.root.resolve()
        assert fixture_path(root).is_file(), f"fixture MCP ausente: {fixture_path(root)}"
        payload = {
            "id": MCP_SERVER_ID,
            "name": MCP_SERVER_NAME,
            "cwd": str(root),
            "command": fixture_command(root),
            "enabled": False,
        }
        created = _json(api, api.post("/api/mcp", data=payload))
        assert _server([created["server"]]), created
        # Registration uses the guarded API because the current editor has no
        # ID field; all subsequent start/stop/log/edit/delete actions are UI.
        page.reload()
        page.get_by_test_id("tab-configs").wait_for(timeout=10_000)
        page.get_by_test_id("tab-mcp").click()
        row = _row(page)
        row.get_by_role("button", name="ligar").click()
        state = _wait(api, lambda value: _server(value).get("status", {}).get("running") is True)
        server = _server(state)
        status = server["status"]
        assert isinstance(status.get("pid"), int) and status["pid"] > 0, status
        # A fixture imprime a sentinela READY logo após ficar "running"; o
        # endpoint de logs pode não ter capturado a linha ainda. Espera curta
        # e limitada antes de desistir (flake de timing observado em runs reais).
        ready_pid: int | None = None
        logs: Any = None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            logs = _json(api, api.get(f"/api/mcp/{MCP_SERVER_ID}/logs"))
            try:
                ready_pid = ready_pid_from_logs(logs)
                break
            except HarnessError:
                time.sleep(1.0)
        assert ready_pid is not None, "sentinela READY não apareceu nos logs MCP em 15s"
        assert ready_pid == status["pid"], {"ready_pid": ready_pid, "status": status}
        _pass(ctx, "MCP-02", "servidor E2E ligado; PID do status igual ao sentinel real", "mcp-02.json", {
            "created": created, "status": status, "ready_pid": ready_pid, "logs": logs,
        })
    except Exception as exc:
        _fail(ctx, "MCP-02", exc, "mcp-02-failure.json")
        for item in ("MCP-03", "MCP-04", "MCP-05"):
            _nv(ctx, item, "MCP-02 não passou")
        return

    try:
        row = _row(page)
        row.locator('[title="ver logs"]').click()
        page.get_by_role("heading", name=f"Logs — {MCP_SERVER_NAME}").wait_for(timeout=5_000)
        page.get_by_text(READY_SENTINEL).wait_for(timeout=5_000)
        log_evidence = _json(api, api.get(f"/api/mcp/{MCP_SERVER_ID}/logs"))
        page.get_by_role("button", name="fechar").click()
        row.get_by_role("button", name="desligar").click()
        stopped = _wait(api, lambda value: _server(value).get("status", {}).get("running") is False)
        pid = _server(stopped).get("status", {}).get("pid")
        # mcp_runner retains the numeric PID in its status after exit; inspect
        # the OS as well, which is the meaningful "PID morto" assertion.
        assert isinstance(pid, int) and pid == ready_pid, {"pid": pid, "ready_pid": ready_pid}
        pid_dead = not _pid_alive(pid)
        orphan_pids = fixture_pids(ctx.root)
        assert pid_dead, f"PID ainda vivo: {pid}"
        assert not orphan_pids, f"fixtures MCP órfãs: {orphan_pids}"
        _pass(ctx, "MCP-03", "desligar pela UI deixou running=false e PID morto", "mcp-03.json", {
            "stopped": stopped, "pid": pid, "ready_pid": ready_pid,
            "pid_dead": pid_dead, "orphan_pids": orphan_pids,
        })
    except Exception as exc:
        _fail(ctx, "MCP-03", exc, "mcp-03-failure.json")
        _nv(ctx, "MCP-04", "MCP-03 não passou")
        _nv(ctx, "MCP-05", "MCP-03 não passou")
        return

    try:
        row = _row(page)
        row.locator('[title="editar"]').click()
        page.get_by_role("heading", name=f"Editar MCP — {MCP_SERVER_NAME}").wait_for(timeout=5_000)
        page.get_by_label("Comando").fill(fixture_command(ctx.root, exit_code=7))
        page.get_by_role("button", name="salvar").click()
        _wait(api, lambda value: _server(value).get("command") == fixture_command(ctx.root, exit_code=7))
        edit_evidence = {"logs": log_evidence, "command": fixture_command(ctx.root, exit_code=7)}
    except Exception as exc:
        _fail(ctx, "MCP-04", exc, "mcp-04-failure.json")
        _nv(ctx, "MCP-05", "MCP-04 não passou")
        return

    try:
        row = _row(page)
        row.get_by_role("button", name="ligar").click()
        auto = _wait(api, lambda value: (
            _server(value).get("status", {}).get("running") is False
            and _server(value).get("status", {}).get("auto_stopped") is True
        ))
        status = _server(auto)["status"]
        assert status.get("last_exit_code") == 7, status
        auto_logs = _json(api, api.get(f"/api/mcp/{MCP_SERVER_ID}/logs"))
        auto_ready_pid = ready_pid_from_logs(auto_logs)
        assert auto_ready_pid == status.get("pid"), {"ready_pid": auto_ready_pid, "status": status}
        auto_pid_dead = not _pid_alive(auto_ready_pid)
        auto_orphans = fixture_pids(ctx.root)
        assert auto_pid_dead, f"PID auto-stop ainda vivo: {auto_ready_pid}"
        assert not auto_orphans, f"fixtures MCP órfãs após auto-stop: {auto_orphans}"
        page.reload()
        page.get_by_test_id("tab-configs").wait_for(timeout=10_000)
        page.get_by_test_id("tab-mcp").click()
        page.on("dialog", lambda dialog: dialog.accept())
        row = _row(page)
        row.locator('[title="remover"]').click()
        _wait(api, lambda value: not any(
            isinstance(entry, dict) and entry.get("id") == MCP_SERVER_ID for entry in value
        ))
        after_delete = _json(api, api.get("/api/mcp"))
        evidence = {
            "edit_and_logs": edit_evidence, "status": status,
            "auto_ready_pid": auto_ready_pid, "auto_pid_dead": auto_pid_dead,
            "auto_orphans": auto_orphans, "auto_logs": auto_logs,
            "servers_after_delete": after_delete,
        }
        # MCP-04 includes delete in the manifest; record it only after the
        # deletion assertion has passed rather than claiming a partial PASS.
        _pass(ctx, "MCP-04", "logs, edição rc7 e exclusão foram observados", "mcp-04.json", evidence)
        _pass(ctx, "MCP-05", "rc7 causou auto-stop e exclusão deixou lista sem o servidor", "mcp-05.json", evidence)
    except Exception as exc:
        _fail(ctx, "MCP-04", exc, "mcp-04-delete-failure.json")
        _fail(ctx, "MCP-05", exc, "mcp-05-failure.json")


def _pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _run_disabled(ctx: RunContext) -> None:
    ctx.current_item = "MCP-06"
    options: Any = None
    try:
        options = _open_app(ctx)
        tab_wait = wait_for_mcp_tab(ctx.page, expected=False)
        tab = ctx.page.get_by_test_id("tab-mcp")
        assert options.get("features", {}).get("mcp") is False, options
        response = ctx.api.get("/api/mcp")
        assert response.status == 404, f"HTTP {response.status}"
        _pass(ctx, "MCP-06", "feature false, tab ausente e /api/mcp=404 após reinício", "mcp-06.json", {
            "options": options, "tab_wait": tab_wait, "mcp_status": response.status,
        })
    except Exception as exc:
        _fail(ctx, "MCP-06", exc, "mcp-06-failure.json", {"options": options})


def _cleanup_server(ctx: RunContext) -> None:
    """Best-effort stop/delete, called even when an MCP item failed."""
    errors: list[str] = []
    try:
        response = ctx.api.get("/api/mcp")
        if response.status != 404:
            servers = _json(ctx.api, response)
            try:
                server = _server(servers)
            except HarnessError:
                server = None
            if server is not None:
                try:
                    ctx.api.post(f"/api/mcp/{MCP_SERVER_ID}/stop")
                    _wait(ctx.api, lambda value: _server(value).get("status", {}).get("running") is False)
                except Exception as exc:
                    errors.append(f"stop: {exc}")
            try:
                ctx.api.delete(f"/api/mcp/{MCP_SERVER_ID}")
            except Exception as exc:
                errors.append(f"delete: {exc}")
    except Exception as exc:
        errors.append(f"list: {exc}")
    orphans = fixture_pids(ctx.root)
    if orphans:
        errors.append(f"fixtures MCP órfãs: {orphans}")
    if errors:
        raise HarnessError("; ".join(errors))


def run(lifecycle: McpLifecycle) -> None:
    """Run the isolated enabled/disabled MCP lifecycle supplied by the runner."""
    lifecycle.close_browser_context()
    lifecycle.stop_backend()
    lifecycle.start_backend(dict(MCP_ENV))
    enabled: RunContext | None = None
    try:
        enabled = lifecycle.open_browser_context()
        lifecycle.install_mcp_guard_extensions(enabled.guard)
        _run_enabled(enabled)
    finally:
        cleanup_error: Exception | None = None
        if enabled is not None:
            try:
                _cleanup_server(enabled)
            except Exception as exc:
                cleanup_error = exc
        try:
            lifecycle.close_browser_context()
        finally:
            lifecycle.stop_backend()
        if cleanup_error is not None:
            raise cleanup_error

    lifecycle.start_backend(dict(MCP_DISABLED_ENV))
    disabled: RunContext | None = None
    try:
        disabled = lifecycle.open_browser_context()
        _run_disabled(disabled)
    finally:
        lifecycle.close_browser_context()
        lifecycle.stop_backend()
