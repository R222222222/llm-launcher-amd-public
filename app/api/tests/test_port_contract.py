"""Contratos de portas reservadas para web e llama-server."""
import socket
import sys
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_ROOT))

from api.core import builder, constants  # noqa: E402


def _cfg():
    return {
        "model": "/tmp/test-model.gguf",
        "backend": "vanilla",
        "context_window": 4096,
        "kv_cache": "q5_0",
        "gpu_layers": 99,
        "parallel_slots": 1,
        "flash_attn": True,
        "batch_size": 512,
        "ubatch_size": 128,
        "threads_gen": 2,
        "threads_batch": 2,
        "reasoning_budget": None,
        "mlock": False,
        "max_tokens": 8,
        "cache_ram": 2048,
        "ctx_checkpoints": 0,
    }


def test_server_and_router_builders_use_reserved_8421_not_1234(tmp_path, monkeypatch):
    cfg = _cfg()
    # Fake binário real (is_file()==True): o contrato testado é a porta/host,
    # não a resolução do binário em vendor/llama.cpp (ausente em clone limpo).
    fake_server = tmp_path / "llama-server"
    fake_server.write_bytes(b"")
    fake_cli = tmp_path / "llama-cli"
    fake_cli.write_bytes(b"")
    monkeypatch.setattr(
        builder, "backend_binary",
        lambda backend, mode: fake_server if mode == "server" else fake_cli,
    )
    command = builder.build_command_from_cfg(cfg, mode="server")
    router = builder.build_router_command([cfg], tmp_path / "router.ini")
    auto = builder.build_auto_server_command(Path(cfg["model"]), "vanilla")

    assert constants.LLAMA_SERVER_PORT == 8421
    assert "--port 8421" in command and "1234" not in command
    assert "--port 8421" in router and "1234" not in router
    assert "--port 8421" in auto and "1234" not in auto
    assert "--host 127.0.0.1" in command
    assert "--host 127.0.0.1" in router
    assert "--host 127.0.0.1" in auto


def test_web_port_selector_skips_reserved_llama_port(monkeypatch):
    from api import server

    base = 39000
    monkeypatch.setattr(server.constants, "LLAMA_SERVER_PORT", base + 1)
    first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    first.bind(("127.0.0.1", base))
    try:
        port, sock = server._bind_first_available_port("127.0.0.1", base)
        try:
            assert port == base + 2
        finally:
            sock.close()
    finally:
        first.close()
