"""Testes de isolamento, readiness e cancelamento do runner."""
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path

import pytest

_API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_ROOT))

from api.core import runner, running  # noqa: E402


def test_popen_isolation_flags_posix_and_windows(monkeypatch):
    monkeypatch.setattr(runner.sys, "platform", "linux")
    assert runner._popen_isolation_kwargs() == {"start_new_session": True}

    monkeypatch.setattr(runner.sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    assert runner._popen_isolation_kwargs() == {"creationflags": 512}


def test_kill_pid_tree_refuses_shared_process_group(monkeypatch):
    killed = []
    groups = []
    monkeypatch.setattr(running.sys, "platform", "linux")
    monkeypatch.setattr(running.os, "getpgid", lambda _pid: 77)
    monkeypatch.setattr(running.os, "getsid", lambda _pid: 77)
    monkeypatch.setattr(running.os, "getpgrp", lambda: 77)
    monkeypatch.setattr(running.os, "kill", lambda pid, _sig: killed.append(pid))
    monkeypatch.setattr(running.os, "killpg", lambda pgid, _sig: groups.append(pgid))

    assert running.kill_pid_tree(77) is True
    assert killed == [77]
    assert groups == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups only")
def test_isolated_process_and_descendant_die_without_killing_test_runner():
    code = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(child.pid, flush=True); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        **runner._popen_isolation_kwargs(),
    )
    assert proc.stdout is not None
    child_pid = int(proc.stdout.readline().strip())
    assert os.getpgid(proc.pid) == proc.pid
    test_pid = os.getpid()

    assert running.kill_pid_tree(proc.pid) is True
    proc.wait(timeout=5)
    for _ in range(30):
        if not running.pid_alive(child_pid):
            break
        time.sleep(0.05)
    assert not running.pid_alive(child_pid)
    assert running.pid_alive(test_pid)


class _FakeProc:
    def __init__(self, pid=3210):
        self.pid = pid
        self.dead = False

    def poll(self):
        return 0 if self.dead else None


def _health_server(statuses):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            status = statuses.pop(0) if statuses else 200
            body = b'{"status":"ok"}' if status == 200 else b'{"status":"loading"}'
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_health_503_then_200_emits_one_load_ok():
    server, thread = _health_server([503, 200, 200])
    proc = _FakeProc()
    loaded = []
    probe = runner._HealthProbe(proc, "127.0.0.1", server.server_port, lambda: loaded.append("ok"))
    probe.start()
    for _ in range(40):
        if loaded:
            break
        time.sleep(0.05)
    probe.stop()
    server.shutdown()
    thread.join(timeout=2)
    assert loaded == ["ok"]


def test_dead_process_before_health_never_emits_load_ok():
    server, thread = _health_server([200])
    proc = _FakeProc()
    proc.dead = True
    loaded = []
    probe = runner._HealthProbe(proc, "127.0.0.1", server.server_port, lambda: loaded.append("ok"))
    probe.start()
    time.sleep(0.25)
    probe.stop()
    server.shutdown()
    thread.join(timeout=2)
    assert loaded == []


def test_occupied_port_prevents_spawn(monkeypatch):
    import socket
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    called = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_k: called.append(True))
    try:
        with pytest.raises(runner.PortOccupiedError, match="já está ocupada"):
            runner._run_capturing(
                f"{sys.executable} -c pass --port {listener.getsockname()[1]}",
                runner.LaunchHandle(), lambda _line: None, lambda: None,
            )
    finally:
        listener.close()
    assert called == []


def test_stdout_persists_without_sse_callback():
    output = StringIO()
    runner._run_capturing(
        f'{sys.executable} -c "print(\\"persisted-line\\")"',
        runner.LaunchHandle(), lambda _line: None, lambda: None, launch_log=output,
    )
    assert "persisted-line" in output.getvalue()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups only")
def test_cancel_handle_kills_fixture_child_and_keeps_test_runner_alive():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        **runner._popen_isolation_kwargs(),
    )
    handle = runner.LaunchHandle()
    handle._set_proc(proc)
    handle.cancel()
    proc.wait(timeout=5)
    assert proc.poll() is not None
    assert running.pid_alive(os.getpid())
