"""Executor resiliente do llama-server com streaming de eventos.

Substitui run_server_resiliently de models.py. Em vez de imprimir tudo no
stdout, dispara callbacks tipados (`on_event`) — a camada HTTP plugada
encaminha cada evento como SSE pro frontend.

Eventos:
    {"type": "stdout",    "line": "..."}                       — linha bruta
    {"type": "load_ok",   "attempt": N}                        — load detectado
    {"type": "stabilized", "attempt": N}                       — sobreviveu STABILIZE_SECONDS
    {"type": "exit",      "attempt": N, "rc": int}             — processo encerrou
    {"type": "failure",   "category": "OOM_RUNTIME",
                          "excerpt": "...", "attempt": N}      — queda classificada
    {"type": "degrade",   "description": "...",  "config": {…}} — degrau aplicado
    {"type": "restart",   "attempt": N, "backoff": s}          — vai reiniciar (degrade)
    {"type": "manual_restart", "attempt": N}                   — restart pedido pela UI (mesma config)
    {"type": "giveup",    "reason": "…"}                       — escada esgotada / loop
    {"type": "done"}                                           — terminou normalmente
"""
import os
import json
import ipaddress
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, TextIO

from .builder import build_command_from_cfg, build_router_command, build_router_preset
from .config_store import append_fail_history, save_config
from .constants import STABILIZE_SECONDS
from .failure import Failure, classify_failure, next_degrade
from .amd import hip_env

EventCallback = Callable[[dict], None]

SERVER_MAX_RESTARTS  = 10
SERVER_RESTART_WINDOW = 120
SERVER_BACKOFF_START  = 2
SERVER_BACKOFF_MAX    = 30


class LaunchHandle:
    """Handle de um launch — permite cancelar de fora (UI cancel).

    Dois modos:
      - normal: spawneamos via Popen, `_proc` está setado.
      - attached: re-anexamos um llama-server órfão de uma sessão anterior do
        Python; só temos o PID. `_proc` é None, cancel usa kill_pid_tree.
    """
    def __init__(self, attached_pid: int | None = None):
        self._cancelled = False
        self._restart = False
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._attached_pid: int | None = attached_pid

    @property
    def pid(self) -> int | None:
        if self._proc is not None:
            return self._proc.pid
        return self._attached_pid

    @property
    def attached(self) -> bool:
        return self._proc is None and self._attached_pid is not None

    def cancel(self) -> None:
        # Import tardio pra evitar ciclo (running → runner não importa, OK).
        from . import running
        with self._lock:
            self._cancelled = True
            pid = self.pid
            if pid is None:
                return
            if self._proc is not None and self._proc.poll() is not None:
                return  # já encerrou sozinho
            running.kill_pid_tree(pid)
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    pass

    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def request_restart(self) -> bool:
        """Mata o processo atual mas NÃO desiste: o loop resiliente sobe de
        novo com a MESMA config (sem degradar). Usado pra destravar um server
        vivo-mas-preso e deixar o cliente (porta fixa) reconectar.

        Retorna False se já foi cancelado ou se não há processo pra reiniciar.
        """
        from . import running
        with self._lock:
            if self._cancelled:
                return False
            pid = self.pid
            if pid is None:
                return False
            self._restart = True
            # Se já morreu sozinho, o loop já vai reentrar — só sinaliza.
            if self._proc is not None and self._proc.poll() is not None:
                return True
            running.kill_pid_tree(pid)
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    pass
        return True

    def restart_requested(self) -> bool:
        with self._lock:
            return self._restart

    def consume_restart(self) -> bool:
        """Lê-e-limpa o flag de restart de forma atômica."""
        with self._lock:
            if self._restart:
                self._restart = False
                return True
            return False

    def _set_proc(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._proc = proc
            self._attached_pid = None


class PortOccupiedError(RuntimeError):
    """O llama-server não pode substituir outro serviço na porta reservada."""


def _popen_isolation_kwargs() -> dict[str, Any]:
    """Cria uma sessão/grupo próprio sem recorrer a shell wrapper."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _command_port(cmd: str) -> int | None:
    tokens = shlex.split(cmd)
    for i, token in enumerate(tokens):
        if token == "--port" and i + 1 < len(tokens):
            try:
                return int(tokens[i + 1])
            except ValueError:
                return None
        if token.startswith("--port="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _preflight_port(port: int | None) -> None:
    """Falha antes do spawn se qualquer listener já ocupa a porta do server."""
    if port is None:
        return
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # SO_REUSEADDR evita falso positivo por sockets TIME_WAIT deixados pelas
        # conexões de health-probe ao llama-server recém-SIGTERMed; um listener
        # ativo (LISTEN) continua falhando o bind normalmente.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 0.0.0.0 detecta conflito tanto com o bind público quanto com loopback
        # na configuração padrão do llama-server.
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        raise PortOccupiedError(
            f"porta do llama-server {port} já está ocupada; spawn recusado"
        ) from exc
    finally:
        sock.close()


def _health_host(host: str) -> str:
    host = (host or "").strip().strip("[]")
    if host in {"", "0.0.0.0", "::"}:
        return "127.0.0.1"
    try:
        if ipaddress.ip_address(host).is_loopback:
            return "127.0.0.1"
    except ValueError:
        pass
    return host


def _command_host(cmd: str) -> str:
    tokens = shlex.split(cmd)
    for i, token in enumerate(tokens):
        if token == "--host" and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith("--host="):
            return token.split("=", 1)[1]
    return "127.0.0.1"


class _HealthProbe:
    """Sonda /health sem bloquear a leitura de stdout/SSE do runner."""

    def __init__(self, proc: Any, host: str, port: int,
                 on_ready: Callable[[], None]):
        self.proc = proc
        health_host = _health_host(host)
        url_host = f"[{health_host}]" if ":" in health_host else health_host
        self.url = f"http://{url_host}:{port}/health"
        self.on_ready = on_ready
        self._stop = threading.Event()
        self._ready = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"llama-health-{self.proc.pid}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def _emit_ready_once(self) -> None:
        with self._lock:
            if self._ready or self.proc.poll() is not None:
                return
            self._ready = True
        try:
            self.on_ready()
        except Exception:
            # Readiness notification must never kill the process reader.
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.proc.poll() is not None:
                return
            try:
                req = urllib.request.Request(self.url, headers={"Accept": "application/json"})
                with self._opener.open(req, timeout=0.5) as response:
                    status = getattr(response, "status", response.getcode())
                    body = response.read(64 * 1024)
                if status == 503:
                    pass  # explicit loading state
                elif status == 200 and self.proc.poll() is None:
                    try:
                        data = json.loads(body.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        data = None
                    if isinstance(data, dict) and data.get("status") == "ok":
                        self._emit_ready_once()
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            self._stop.wait(0.2)


def _safe_log_line(log: TextIO | None, line: str) -> None:
    if log is None:
        return
    try:
        log.write(line)
        log.flush()
    except Exception:
        pass


def _run_capturing(cmd: str, handle: LaunchHandle, on_line: Callable[[str], None],
                   on_load: Callable[[], None],
                   on_proc_pid: Callable[[int], None] | None = None,
                   launch_log: TextIO | None = None,
                   ) -> tuple[int, str, float | None]:
    # Removido shell=True: no Windows, shell=True cria cmd.exe wrapper que
    # survive ao terminate(), deixando o llama-server rodando sozinho.
    port = _command_port(cmd)
    _preflight_port(port)
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        env=hip_env(),  # respeita HIP_VISIBLE_DEVICES sem reordenar GPUs AMD
        shell=False,
        **_popen_isolation_kwargs(),
    )
    handle._set_proc(proc)
    if on_proc_pid is not None:
        try:
            on_proc_pid(proc.pid)
        except Exception:
            pass
    chunks: list[str] = []
    load_ts: list[float | None] = [None]

    def mark_load() -> None:
        if load_ts[0] is None and proc.poll() is None:
            load_ts[0] = time.time()
            on_load()

    probe = _HealthProbe(proc, _command_host(cmd), port, mark_load) if port else None
    if probe is not None:
        probe.start()
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if handle.cancelled() or handle.restart_requested():
                try:
                    proc.terminate()
                except Exception:
                    pass
                break
            chunks.append(line)
            _safe_log_line(launch_log, line)
            on_line(line.rstrip("\n"))
    finally:
        if probe is not None:
            probe.stop()
    rc = proc.wait()
    return rc, "".join(chunks), load_ts[0]


def stop_lms_server_if_running() -> None:
    """Para o servidor do LM Studio (libera a porta 1234)."""
    from .constants import LMS
    if not LMS.exists():
        return
    try:
        status = subprocess.run(
            [str(LMS), "server", "status"],
            capture_output=True, text=True, encoding="utf-8", timeout=10,
        )
        out = ((status.stdout or "") + (status.stderr or "")).lower()
        if "running" in out and "not running" not in out:
            subprocess.run([str(LMS), "server", "stop"], timeout=10)
    except Exception:
        pass


# Preset do modo router — regravado a cada launch múltiplo. Vive ao lado do
# api_running.json (pasta api/), fora das pastas de modelos do usuário.
ROUTER_PRESET_FILE = Path(__file__).resolve().parent.parent / "router_preset.ini"


def run_router_server(
    cfgs: list[dict],
    router_cfg: dict,
    on_event: EventCallback,
    handle: LaunchHandle | None = None,
    stop_lms: bool = True,
    on_proc_pid: Callable[[int], None] | None = None,
    launch_log: TextIO | None = None,
) -> None:
    """Sobe o llama-server em MODO ROUTER com N modelos (um por seção do
    preset INI, todos com load-on-startup). Sem escada de auto-degrade: quem
    cai é o processo filho de UM modelo e o próprio router o gerencia/recarrega;
    se o ROUTER inteiro morrer, reportamos e desistimos — degradar uma config
    específica não faz sentido aqui. Cancel e restart manual funcionam igual
    ao launch normal (kill na árvore inteira mata router + filhos).
    """
    if handle is None:
        handle = LaunchHandle()
    if stop_lms:
        stop_lms_server_if_running()

    try:
        preset_text, names = build_router_preset(cfgs)
        ROUTER_PRESET_FILE.write_text(preset_text, encoding="utf-8")
        cmd = build_router_command(cfgs, ROUTER_PRESET_FILE)
    except Exception as e:
        on_event({"type": "giveup", "reason": f"build do router falhou: {e}"})
        return

    attempt = 0
    while True:
        if handle.cancelled():
            on_event({"type": "giveup", "reason": "cancelled"})
            return

        attempt += 1
        on_event({"type": "start", "attempt": attempt, "cmd": cmd,
                  "config": {**router_cfg, "model_ids": names}})
        on_event({"type": "stdout", "line": f"── preset: {ROUTER_PRESET_FILE} ──"})
        for line in preset_text.splitlines():
            on_event({"type": "stdout", "line": f"    {line}"})

        def on_load() -> None:
            on_event({"type": "load_ok", "attempt": attempt})

        def on_line(line: str) -> None:
            on_event({"type": "stdout", "line": line})

        try:
            rc, output, load_ts = _run_capturing(
                cmd, handle, on_line, on_load, on_proc_pid=on_proc_pid,
                launch_log=launch_log,
            )
        except PortOccupiedError as exc:
            on_event({"type": "giveup", "reason": f"port_occupied: {exc}"})
            return
        on_event({"type": "exit", "attempt": attempt, "rc": rc, "load_ts": load_ts})

        if handle.cancelled():
            on_event({"type": "giveup", "reason": "cancelled"})
            return

        # Restart manual (botão da UI): mata router + filhos e sobe de novo
        # com o mesmo preset.
        if handle.consume_restart():
            on_event({"type": "manual_restart", "attempt": attempt + 1})
            slept = 0.0
            while slept < 1.0:
                if handle.cancelled():
                    on_event({"type": "giveup", "reason": "cancelled"})
                    return
                time.sleep(0.25)
                slept += 0.25
            continue

        if rc == 0:
            on_event({"type": "done", "attempt": attempt})
            return

        excerpt = "\n".join(output.splitlines()[-15:])
        on_event({"type": "failure", "category": "ROUTER_EXIT",
                  "excerpt": excerpt, "attempt": attempt})
        on_event({"type": "giveup", "reason": "router_exit", "excerpt": excerpt})
        return


def run_server_resiliently(
    cfg: dict,
    on_event: EventCallback,
    handle: LaunchHandle | None = None,
    stop_lms: bool = True,
    on_proc_pid: Callable[[int], None] | None = None,
    launch_log: TextIO | None = None,
) -> None:
    """Loop: launch → detecta load → roda → se cair, classifica + degrada + reinicia.

    Promove a config a salva quando sobrevive STABILIZE_SECONDS pós-load.
    Encerra ao receber cancel via `handle`, ou rc==0, ou escada esgotada.
    """
    if handle is None:
        handle = LaunchHandle()
    if stop_lms:
        stop_lms_server_if_running()

    restarts: list[float] = []
    backoff = SERVER_BACKOFF_START
    attempt = 0

    while True:
        if handle.cancelled():
            on_event({"type": "giveup", "reason": "cancelled"})
            return

        attempt += 1
        try:
            cmd = build_command_from_cfg(cfg, mode="server")
        except Exception as e:
            on_event({"type": "giveup", "reason": f"build_command falhou: {e}"})
            return

        on_event({"type": "start", "attempt": attempt, "cmd": cmd, "config": dict(cfg)})

        attempt_cfg = dict(cfg)

        def on_load() -> None:
            save_config(attempt_cfg)
            on_event({"type": "load_ok", "attempt": attempt})

        def on_line(line: str) -> None:
            on_event({"type": "stdout", "line": line})

        try:
            rc, output, load_ts = _run_capturing(
                cmd, handle, on_line, on_load, on_proc_pid=on_proc_pid,
                launch_log=launch_log,
            )
        except PortOccupiedError as exc:
            on_event({"type": "giveup", "reason": f"port_occupied: {exc}"})
            return
        on_event({"type": "exit", "attempt": attempt, "rc": rc, "load_ts": load_ts})

        if handle.cancelled():
            on_event({"type": "giveup", "reason": "cancelled"})
            return

        # Restart manual (botão da UI): sobe de novo com a MESMA config, sem
        # classificar como falha nem aplicar degrade, e sem contar no limite de
        # too_many_restarts. Apenas zera o backoff e reentra no loop.
        if handle.consume_restart():
            backoff = SERVER_BACKOFF_START
            on_event({"type": "manual_restart", "attempt": attempt + 1})
            slept = 0.0
            while slept < 1.0:
                if handle.cancelled():
                    on_event({"type": "giveup", "reason": "cancelled"})
                    return
                time.sleep(0.25)
                slept += 0.25
            continue

        stabilized = load_ts is not None and (time.time() - load_ts) >= STABILIZE_SECONDS
        if rc == 0:
            if stabilized:
                save_config(attempt_cfg)
            on_event({"type": "done", "attempt": attempt})
            return

        failure, excerpt = classify_failure(output, rc, load_ts)
        on_event({"type": "failure", "category": failure, "excerpt": excerpt, "attempt": attempt})

        if failure == Failure.MMPROJ:
            append_fail_history(attempt_cfg, failure, excerpt, None, attempt)
            on_event({"type": "giveup", "reason": "mmproj_unsupported",
                      "hint": "lms_fallback_available"})
            return

        # Arquivo corrompido / shard faltando: auto-degrade não consegue
        # consertar. Dá giveup direto com hint pro usuário rebaixar o quant.
        if failure == Failure.MODEL_CORRUPTED:
            append_fail_history(attempt_cfg, failure, excerpt, None, attempt)
            on_event({"type": "giveup", "reason": "model_corrupted",
                      "excerpt": excerpt, "hint": "redownload_model"})
            return

        # Modo auto: nenhum knob da config vira flag, então todo degrau geraria
        # exatamente o mesmo comando — reiniciar seria loop até o teto. Para
        # aqui e devolve a evidência pro usuário decidir (desmarcar o auto e
        # configurar na mão, ou rebaixar o quant).
        if cfg.get("llama_auto", False):
            append_fail_history(attempt_cfg, failure, excerpt, None, attempt)
            on_event({"type": "giveup", "reason": "auto_mode_no_degrade",
                      "failure": failure, "excerpt": excerpt,
                      "hint": "desmarque 'llama.cpp decide' pra habilitar o auto-degrade"})
            return

        new_cfg, degrade_desc = next_degrade(cfg, failure, excerpt)
        append_fail_history(attempt_cfg, failure, excerpt, degrade_desc, attempt)

        if new_cfg is None:
            on_event({"type": "giveup", "reason": "ladder_exhausted",
                      "failure": failure, "excerpt": excerpt})
            return

        now = time.time()
        restarts = [t for t in restarts if now - t < SERVER_RESTART_WINDOW]
        restarts.append(now)
        if len(restarts) > SERVER_MAX_RESTARTS:
            on_event({"type": "giveup", "reason": "too_many_restarts",
                      "window_s": SERVER_RESTART_WINDOW,
                      "count": len(restarts)})
            return

        on_event({"type": "degrade", "description": degrade_desc, "config": dict(new_cfg)})
        on_event({"type": "restart", "attempt": attempt + 1, "backoff": backoff})

        slept = 0.0
        step = 0.25
        while slept < backoff:
            if handle.cancelled():
                on_event({"type": "giveup", "reason": "cancelled"})
                return
            time.sleep(step)
            slept += step

        cfg = new_cfg
        backoff = min(backoff * 2, SERVER_BACKOFF_MAX)
