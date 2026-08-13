"""Download admission/session tests use only local fakes."""
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

_API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_ROOT))

from api import server  # noqa: E402


def test_hf_resolve_public_contract_uses_revision_not_branch():
    sha = "c287502cd9e278dac8eed805c112cce5d0081e0b"
    response = TestClient(server.app).post(
        "/api/hf/resolve",
        json={"url": f"https://huggingface.co/owner/repo/resolve/{sha}/model.gguf"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "repo_id": "owner/repo", "filename": "model.gguf", "revision": sha,
    }


def test_hf_resolve_refs_heads_and_plain_url_are_canonical():
    client = TestClient(server.app)
    refs = client.post(
        "/api/hf/resolve",
        json={"url": "https://huggingface.co/owner/repo/blob/refs/heads/main/model.gguf"},
    )
    assert refs.status_code == 200
    assert refs.json() == {
        "repo_id": "owner/repo", "filename": "model.gguf", "revision": "refs/heads/main",
    }
    plain = client.post(
        "/api/hf/resolve", json={"url": "https://huggingface.co/owner/repo"},
    )
    assert plain.status_code == 200
    assert plain.json() == {"repo_id": "owner/repo", "filename": None, "revision": "main"}


def test_hf_download_requires_revision_and_does_not_fallback_to_main():
    response = TestClient(server.app).post(
        "/api/hf/download",
        json={
            "repo_id": "owner/repo",
            "branch": "main",
            "rel_paths": ["model.gguf"],
            "base_dir": "/tmp/models",
        },
    )
    assert response.status_code == 422


def test_hf_download_rejects_64_hex_revision():
    response = TestClient(server.app).post(
        "/api/hf/download",
        json={
            "repo_id": "owner/repo", "revision": "a" * 64,
            "rel_paths": ["model.gguf"], "base_dir": "/tmp/models",
        },
    )
    assert response.status_code == 400


def _plan(tmp_path, name="m.gguf"):
    dest = tmp_path / name
    return {"base_dir": str(tmp_path), "root": str(tmp_path), "items": [{"dest": str(dest)}]}


def test_session_emits_one_terminal_event_then_eof(monkeypatch, tmp_path):
    def fake_stream(plan, *, control):
        yield {"type": "file_done", "rel": "m.gguf"}

    monkeypatch.setattr(server.hf, "stream_download", fake_stream)
    session = server.DownloadSession(_plan(tmp_path))
    session.start()
    assert session.thread is not None
    session.thread.join(2)
    events = [session.queue.get_nowait() for _ in range(3)]
    assert events[1]["type"] == "done"
    assert events[2] == {"type": "_eof"}
    assert session.done
    assert not session.thread.daemon


def test_destination_reservation_rejects_overlap(monkeypatch, tmp_path):
    server._DOWNLOADS.clear()
    server._DOWNLOAD_DESTINATIONS.clear()
    first = server.DownloadSession(_plan(tmp_path))
    second = server.DownloadSession(_plan(tmp_path))
    server._reserve_download(first)
    try:
        try:
            server._reserve_download(second)
        except Exception as exc:
            assert getattr(exc, "status_code", None) == 409
        else:
            raise AssertionError("overlap was admitted")
    finally:
        server._DOWNLOADS.clear()
        server._DOWNLOAD_DESTINATIONS.clear()


def test_cancel_active_worker_returns_joined_only_after_worker_exits(monkeypatch, tmp_path):
    session = server.DownloadSession(_plan(tmp_path))
    gate = threading.Event()

    def worker():
        gate.wait(2)

    session.thread = threading.Thread(target=worker)
    session._state = server.DownloadState.RUNNING
    with server._DOWNLOADS_LOCK:
        server._DOWNLOADS[session.id] = session
    monkeypatch.setattr(server, "DOWNLOAD_JOIN_TIMEOUT", 1.0)
    try:
        session.control.cancel = lambda: gate.set()  # type: ignore[method-assign]
        session.thread.start()
        result = server.hf_download_cancel(session.id)
        assert result == {"ok": True, "joined": True}
        assert not session.thread.is_alive()
    finally:
        with server._DOWNLOADS_LOCK:
            server._DOWNLOADS.pop(session.id, None)


def test_cancel_terminal_session_without_thread_returns_joined():
    session = server.DownloadSession(_plan(Path("/tmp")))
    session._state = server.DownloadState.CANCELLED
    with server._DOWNLOADS_LOCK:
        server._DOWNLOADS[session.id] = session
    try:
        assert server.hf_download_cancel(session.id) == {"ok": True, "joined": True}
    finally:
        with server._DOWNLOADS_LOCK:
            server._DOWNLOADS.pop(session.id, None)


def test_cancel_timeout_returns_503_without_joined_success(monkeypatch, tmp_path):
    session = server.DownloadSession(_plan(tmp_path))
    session.thread = threading.Thread(target=time.sleep, args=(5,))
    session._state = server.DownloadState.RUNNING
    with server._DOWNLOADS_LOCK:
        server._DOWNLOADS[session.id] = session
    monkeypatch.setattr(server, "DOWNLOAD_JOIN_TIMEOUT", 0.001)
    try:
        session.control.cancel = lambda: None  # type: ignore[method-assign]
        session.thread.start()
        with pytest.raises(HTTPException) as exc:
            server.hf_download_cancel(session.id)
        assert exc.value.status_code == 503
        assert "joined" not in (exc.value.detail if isinstance(exc.value.detail, dict) else str(exc.value.detail))
    finally:
        session.thread.join(6)
        with server._DOWNLOADS_LOCK:
            server._DOWNLOADS.pop(session.id, None)
