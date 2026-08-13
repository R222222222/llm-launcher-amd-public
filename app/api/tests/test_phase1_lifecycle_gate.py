from __future__ import annotations

import threading
import uuid

import pytest
from fastapi import HTTPException

from api import server
from api.core import launch_events


@pytest.fixture(autouse=True)
def empty_session_registry():
    with server._SESSIONS_LOCK:
        previous = dict(server._SESSIONS)
        server._SESSIONS.clear()
    yield
    with server._SESSIONS_LOCK:
        server._SESSIONS.clear()
        server._SESSIONS.update(previous)


def test_replay_detects_internal_hole_after_stdout_control_merge():
    replay = launch_events.LaunchEventReplay()
    replay.publish({"type": "start"})       # seq 1: retained control
    replay.publish({"type": "stdout", "line": "old"})  # seq 2: evicted
    replay.publish({"type": "done"})        # seq 3: retained control
    for _ in range(launch_events.MAX_STDOUT_EVENTS):
        replay.publish({"type": "stdout", "line": "x"})

    batch = replay.wait_after(0, timeout=0)
    sequences = [record.seq for record in batch.events]
    assert batch.history_gap is True
    assert sequences == sorted(set(sequences))
    assert sequences[:2] == [1, 3]


def test_replay_gap_boundaries_are_exact():
    contiguous = launch_events.LaunchEventReplay()
    contiguous.publish({"type": "start"})
    contiguous.publish({"type": "done"})
    batch = contiguous.wait_after(0, timeout=0)
    assert [record.seq for record in batch.events] == [1, 2]
    assert batch.history_gap is False
    batch = contiguous.wait_after(1, timeout=0)
    assert [record.seq for record in batch.events] == [2]
    assert batch.history_gap is False

    boundary = launch_events.LaunchEventReplay()
    boundary.publish({"type": "start"})
    boundary.publish({"type": "stdout", "line": "old"})
    boundary.publish({"type": "done"})
    for _ in range(launch_events.MAX_STDOUT_EVENTS):
        boundary.publish({"type": "stdout", "line": "x"})
    batch = boundary.wait_after(1, timeout=0)
    assert batch.history_gap is True
    assert batch.events[0].seq == 3


def _ordinary_launch_patches(monkeypatch, sessions):
    monkeypatch.setattr(server, "_validated_config_paths", lambda data: data)
    monkeypatch.setattr(server.builder_mod, "build_command_from_cfg", lambda *args, **kwargs: "cmd")
    monkeypatch.setattr(server.config_store, "save_config", lambda data: data)
    monkeypatch.setattr(server, "LaunchSession", sessions)


def _router_launch_patches(monkeypatch, sessions):
    monkeypatch.setattr(server, "_validated_config_paths", lambda data: data)
    monkeypatch.setattr(server.config_store, "read_all_configs", lambda: [
        {"id": "one", "model": "one.gguf", "backend": "vanilla"},
        {"id": "two", "model": "two.gguf", "backend": "vanilla"},
    ])
    monkeypatch.setattr(server.builder_mod, "router_binary_for", lambda cfgs: None)
    monkeypatch.setattr(server.builder_mod, "build_router_preset", lambda cfgs: ("preset", ["one", "two"]))
    monkeypatch.setattr(server.backends_mod, "backend_help_text", lambda *args: "--models-preset")
    monkeypatch.setattr(server, "LaunchSession", sessions)


def _concurrent_calls(call):
    barrier = threading.Barrier(2)
    results = []

    def worker():
        barrier.wait()
        try:
            results.append(("ok", call()))
        except BaseException as exc:
            results.append(("error", exc))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    return threads, results


@pytest.mark.parametrize("kind", ["ordinary", "router"])
def test_admission_reservation_allows_only_one_spawn_concurrently(monkeypatch, kind):
    spawned = []
    first_spawned = threading.Event()
    release = threading.Event()

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            self.id = uuid.uuid4().hex[:12]
            self.done = False

        def start(self):
            spawned.append(self.id)
            first_spawned.set()
            release.wait(2)

    if kind == "ordinary":
        _ordinary_launch_patches(monkeypatch, FakeSession)
        call = lambda: server.launch(server.LaunchConfig(model="model.gguf"))
    else:
        _router_launch_patches(monkeypatch, FakeSession)
        call = lambda: server.launch_router(server.RouterLaunchRequest(ids=["one", "two"]))

    threads, results = _concurrent_calls(call)
    assert first_spawned.wait(2)
    release.set()
    for thread in threads:
        thread.join(2)

    assert len(spawned) == 1
    assert sum(kind_ == "ok" for kind_, _ in results) == 1
    conflicts = [value for kind_, value in results if kind_ == "error"]
    assert len(conflicts) == 1
    assert isinstance(conflicts[0], HTTPException)
    assert conflicts[0].status_code == 409


@pytest.mark.parametrize("kind", ["ordinary", "router"])
def test_admission_failure_cleanup_allows_retry(monkeypatch, kind):
    starts = []

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            self.id = uuid.uuid4().hex[:12]
            self.done = False

        def start(self):
            starts.append(self.id)
            if len(starts) == 1:
                raise RuntimeError("startup failed")

    if kind == "ordinary":
        _ordinary_launch_patches(monkeypatch, FakeSession)
        call = lambda: server.launch(server.LaunchConfig(model="model.gguf"))
    else:
        _router_launch_patches(monkeypatch, FakeSession)
        call = lambda: server.launch_router(server.RouterLaunchRequest(ids=["one", "two"]))

    with pytest.raises(RuntimeError, match="startup failed"):
        call()
    result = call()

    assert len(starts) == 2
    assert result["launch_id"]
