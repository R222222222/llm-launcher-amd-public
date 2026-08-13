"""FastAPI app — backend HTTP do launcher web headless.

Sobe na primeira porta livre a partir de 8420, exclusivamente no endereço
Tailscale configurado para este fork.

Endpoints (prefixo /api):
  GET    /options            — listas de choices (kv, ctx, parallel, etc.)
  GET    /backends           — status dos backends (binários, KV types)
  GET    /system             — VRAM total/livre, RAM total
  GET    /gpu                — status detalhado das GPUs AMD

  GET    /models             — lista de modelos .gguf disponíveis
  POST   /models/meta        — GGUF meta de um path

  GET    /configs            — todas as configs salvas
  POST   /configs            — salva uma config
  DELETE /configs            — remove uma config (body: model, backend)

  POST   /estimate           — estima VRAM/RAM pra uma config
  POST   /build-command      — devolve a string do comando (server|cli)
  POST   /suggest/n-cpu-moe  — sugestão de --n-cpu-moe
  POST   /suggest/n-gpu-layers — sugestão de -ngl

  POST   /launch             — inicia o launch resiliente; devolve {launch_id}
                               (409 se já houver launch ativo — um por vez)
  POST   /launch-router      — sobe N configs de uma vez (modo router do
                               llama-server, preset INI); body {ids: [...]}
  GET    /launch/{id}/events — SSE de eventos
  POST   /launch/{id}/cancel — cancela o launch
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import socket
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Permitir `python api/server.py` direto (sem -m) — adiciona o pai do api/
# no sys.path pra que `from api.core import ...` resolva.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from api.core import (  # noqa: E402
    backends as backends_mod,
    builder as builder_mod,
    config_store,
    constants,
    defaults,
    estimator,
    gguf,
    hf,
    lms,
    mcp_runner,
    mcp_store,
    models_repo,
    amd,
    runner,
    running,
    sampling,
    settings_store,
    updates,
    path_policy,
    mcp_config,
    launch_events as launch_events_core,
)

_LOG = logging.getLogger(__name__)
MCP_ENABLED = os.getenv("LLM_LAUNCHER_ENABLE_MCP", "0") == "1"
MCP_ALLOW_REMOTE = os.getenv("LLM_LAUNCHER_ALLOW_REMOTE_MCP", "0") == "1"
if MCP_ENABLED and MCP_ALLOW_REMOTE:
    _LOG.warning(
        "SECURITY WARNING: remote MCP is explicitly enabled; MCP commands use "
        "the configured shell boundary and are reachable beyond loopback"
    )


def get_bind_host(environ: dict[str, str] | None = None) -> str:
    """Return the trusted bind address for this process.

    The environment is deployment configuration, not browser input.  Empty
    values intentionally fall back to loopback rather than the historical
    Tailscale address.
    """
    env = os.environ if environ is None else environ
    return (env.get("LLM_LAUNCHER_HOST") or "127.0.0.1").strip() or "127.0.0.1"


def _request_is_loopback(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    try:
        addr = ipaddress.ip_address(client.host.strip("[]"))
        mapped = getattr(addr, "ipv4_mapped", None)
        return bool(addr.is_loopback or (mapped is not None and mapped.is_loopback))
    except ValueError:
        return False


def _require_loopback(request: Request) -> None:
    if not _request_is_loopback(request):
        raise HTTPException(403, "esta operação só é permitida em loopback")


def _mcp_guard(request: Request) -> None:
    if not _request_is_loopback(request) and not MCP_ALLOW_REMOTE:
        raise HTTPException(403, "MCP remoto está desabilitado")


def _path_http(exc: path_policy.PathPolicyError) -> HTTPException:
    if isinstance(exc, path_policy.OutsideRoot):
        return HTTPException(403, str(exc))
    if isinstance(exc, path_policy.MissingPath):
        return HTTPException(404, str(exc))
    return HTTPException(400, str(exc))


def _canonical_roots(download: bool = False) -> list[Path]:
    paths = (settings_store.get_configured_paths() if download
             else settings_store.get_scan_paths())
    try:
        return list(path_policy.canonical_roots(paths))
    except path_policy.PathPolicyError as exc:
        raise _path_http(exc) from exc


def _scan_roots() -> list[Path]:
    """Best-effort scan roots: one bad/missing setting must not break the grid."""
    paths = settings_store.get_scan_paths()
    return list(path_policy.canonical_roots(paths, strict=False))


def _validated_model(model: str | Path, mmproj: str | Path | None = None) -> tuple[Path, Path | None]:
    try:
        return path_policy.validate_model_pair(model, mmproj, _canonical_roots())
    except path_policy.PathPolicyError as exc:
        raise _path_http(exc) from exc


def _validated_config_paths(data: dict) -> dict:
    model, mmproj = _validated_model(data.get("model", ""), data.get("mmproj"))
    data = dict(data)
    data["model"] = str(model)
    data["mmproj"] = str(mmproj) if mmproj else None
    mcp_servers_config = data.get("mcp_servers_config")
    if mcp_servers_config:
        try:
            validated_mcp, _schema = mcp_config.validate(mcp_servers_config)
        except mcp_config.McpConfigError as exc:
            status = 404 if "não existe" in str(exc) else 400
            raise HTTPException(status, str(exc)) from exc
        data["mcp_servers_config"] = str(validated_mcp)
    else:
        data["mcp_servers_config"] = None
    return data

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if MCP_ENABLED and MCP_ALLOW_REMOTE:
        _LOG.warning(
            "SECURITY WARNING: LLM_LAUNCHER_ALLOW_REMOTE_MCP=1 exposes MCP "
            "beyond loopback; review the configured shell commands"
        )
    # Re-anexa llama-server órfãos da sessão anterior antes de aceitar requests.
    _attach_orphan_servers()
    try:
        yield
    finally:
        _shutdown_downloads()


app = FastAPI(title="llm-launcher API", version="0.1.0", lifespan=_lifespan)
if os.getenv("LLM_LAUNCHER_DEV_CORS") == "1":
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

# ─── modelos Pydantic ─────────────────────────────────────────────────────────

class LaunchConfig(BaseModel):
    """Subset persistido em last_config.json — espelha o schema de save_config."""
    id: str | None = None   # identidade estável; backend gera no 1º save se ausente
    model: str
    backend: str = "turbo"
    context_window: int = 65_536
    kv_cache: str = "turbo4"
    flash_attn: bool = True
    gpu_layers: int = 99
    n_cpu_moe: int = 0
    parallel_slots: int = 1
    reasoning_budget: int | None = None
    preserve_thinking: bool = False
    mlock: bool = False
    max_tokens: int = -1
    batch_size: int = 2_048
    ubatch_size: int = 512
    threads_gen: int = constants.CPU_THREADS_GEN
    threads_batch: int = constants.CPU_THREADS_BATCH
    cache_ram: int = 8_192
    ctx_checkpoints: int = 8
    spec_draft_n_max: int = 2
    mmproj: str | None = None
    mcp_servers_config: str | None = None
    verbose: bool = False
    # "llama.cpp decide": comando mínimo (-m/--mmproj/--alias + HTTP). As demais
    # chaves continuam persistidas, só não viram flag enquanto isto estiver on.
    llama_auto: bool = False
    # Samplers. None = resolvido por modelo (generation_config do autor > preset do
    # chat template). sampler_source == "manual" trava os valores abaixo: é o usuário
    # mandando, e nada os sobrescreve.
    temp: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    sampler_source: str | None = None
    # Multi-GPU (opcionais; None = default do llama.cpp = layer split entre todas)
    split_mode: str | None = None      # "none" | "layer" | "row"
    tensor_split: str | None = None    # ex.: "1,1" (meio a meio entre 2 GPUs)
    main_gpu: int | None = None        # índice base-0 da GPU principal
    mode: str = "server"   # "server" | "cli" — só usado por build/estimate, não persiste


class EstimateRequest(BaseModel):
    model: str
    backend: str = "turbo"
    context_window: int
    kv_cache: str
    parallel_slots: int = 1
    gpu_layers: int = 99
    n_cpu_moe: int = 0
    mmproj: str | None = None
    cache_ram: int = 8_192
    mode: str = "server"


class SuggestRequest(BaseModel):
    model: str
    backend: str = "turbo"
    context_window: int
    kv_cache: str
    parallel_slots: int = 1
    gpu_layers: int = 99
    mmproj: str | None = None
    cache_ram: int = 8_192
    mode: str = "server"


class DeleteConfigRequest(BaseModel):
    id: str


# ─── routes: opções e status ──────────────────────────────────────────────────

@app.get("/api/options")
def get_options(request: Request):
    mcp_meta = {"path": str(constants.MCP_CONFIG_FILE),
                "exists": constants.MCP_CONFIG_FILE.exists(),
                "valid": False}
    if mcp_meta["exists"]:
        try:
            mcp_config.validate(constants.MCP_CONFIG_FILE)
            mcp_meta["valid"] = True
        except mcp_config.McpConfigError:
            pass
    return {
        "kv_cache":         constants.KV_CACHE_OPTIONS,
        "context_window":   constants.CONTEXT_OPTIONS,
        "reasoning_budget": constants.REASONING_BUDGET_OPTIONS,
        "parallel_slots":   constants.PARALLEL_OPTIONS,
        "max_tokens":       constants.MAX_TOKENS_OPTIONS,
        "batch_size":       constants.BATCH_OPTIONS,
        "ubatch_size":      constants.UBATCH_OPTIONS,
        "cache_ram":        constants.CACHE_RAM_OPTIONS,
        "ctx_checkpoints":  constants.CTX_CHECKPOINTS_OPTIONS,
        "spec_draft_n_max": constants.SPEC_DRAFT_N_MAX_OPTIONS,
        "cpu_threads_gen":   constants.CPU_THREADS_GEN,
        "cpu_threads_batch": constants.CPU_THREADS_BATCH,
        "sampler_presets":   sampling.PRESETS,
        "features": {
            "mcp": MCP_ENABLED and (_request_is_loopback(request) or MCP_ALLOW_REMOTE),
        },
        "mcp_runtime_config": mcp_meta,
    }


@app.get("/api/backends")
def get_backends():
    return backends_mod.backend_status()


@app.get("/api/system")
def get_system():
    total, avail = amd._ram_mib_pair()
    return {
        "vram_total_mib": amd.gpu_total_mib(),
        "vram_free_mib":  amd.gpu_free_mib(),
        "ram_total_mib":  total,
        "ram_avail_mib":  avail,
        "gpu_count":      amd.gpu_count(),
    }


@app.get("/api/gpu")
def get_gpu():
    return amd.amd_status()


# ─── routes: modelos ──────────────────────────────────────────────────────────

@app.get("/api/models")
def list_models():
    roots = _scan_roots()
    return [models_repo.describe_model(m, roots) for m in models_repo.collect_models(roots)]


@app.get("/api/models/updates")
def models_updates(deep: bool = False):
    """Confere cada modelo contra o HF. `deep=true` autoriza hashear os ainda
    não verificados (custo alto — reservado a botão manual). Sem deep, roda com
    tamanho + oid registrado/cacheado, barato o bastante pra chamada de startup.

    Um repo é consultado uma vez só (remote_cache), e falha de rede num modelo
    não derruba os outros — cada um vira status 'unknown' isolado.
    """
    roots = _scan_roots()
    remote_cache: dict[str, list[dict]] = {}
    results = [
        updates.check_model(m, roots, deep=deep, remote_cache=remote_cache)
        for m in models_repo.collect_models(roots)
    ]
    return {"results": results}


class CheckUpdateRequest(BaseModel):
    model: str
    deep: bool = True


@app.post("/api/models/check-update")
def model_check_update(req: CheckUpdateRequest):
    """Checagem de um modelo só — default deep=true (hasheia pra confirmar)."""
    roots = _canonical_roots()
    model_path, _ = _validated_model(req.model)
    return updates.check_model(model_path, roots, deep=req.deep)


# ─── routes: app settings (paths das pastas de modelos, etc.) ─────────────────

class SettingsPayload(BaseModel):
    model_paths:   list[str]      = Field(default_factory=list)
    backend_paths: dict[str, str] = Field(default_factory=dict)


@app.get("/api/settings")
def get_settings():
    return settings_store.read_settings()


@app.post("/api/settings")
def update_settings(request: Request, payload: SettingsPayload):
    _require_loopback(request)
    try:
        saved = settings_store.save_settings(payload.model_dump())
    except path_policy.PathPolicyError as exc:
        raise HTTPException(400, str(exc)) from exc
    # Os caches de --help / kv_types do backends_mod são por (backend, mode);
    # se o usuário trocou o diretório do binário, os caches ficam stale.
    backends_mod.invalidate_caches()
    return saved


class MetaRequest(BaseModel):
    model: str


@app.post("/api/models/meta")
def model_meta(req: MetaRequest):
    model, _ = _validated_model(req.model)
    meta = gguf.read_meta(model)
    if meta is None:
        raise HTTPException(404, "GGUF metadata não pôde ser lido")
    return meta


class ContextOptionsRequest(BaseModel):
    model: str


@app.post("/api/models/context-options")
def model_context_options(req: ContextOptionsRequest):
    model, _ = _validated_model(req.model)
    meta = gguf.read_meta(model)
    if meta is None:
        raise HTTPException(404, "GGUF metadata não pôde ser lido")
    n_ctx_train = meta.get("n_ctx_train")
    if n_ctx_train is None:
        return {"options": constants.CONTEXT_OPTIONS}
    options = [{"value": n_ctx_train, "label": f"{n_ctx_train:,} tokens [modelo]"}]
    options.extend(constants.CONTEXT_OPTIONS)
    return {"options": options}


class DefaultsRequest(BaseModel):
    model: str
    backend: str = "vanilla"
    mode: str = "server"


@app.post("/api/models/defaults")
def model_defaults(req: DefaultsRequest):
    """Config recomendada pro modelo: o que o botão "defaults do modelo" preenche.

    Deriva tudo que dá pra derivar (GGUF + VRAM + samplers do autor) e devolve as
    notas explicando cada escolha — palpite sem justificativa não é configurável.
    """
    try:
        model, _ = _validated_model(req.model)
        return defaults.suggest_config(model, req.backend, req.mode)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"não consegui derivar defaults: {e}")


class SamplingRequest(BaseModel):
    model: str
    config: dict | None = None


@app.post("/api/models/sampling")
def model_sampling(req: SamplingRequest):
    """Samplers que este modelo receberia agora, e de onde vieram.

    `source` é o ponto todo: "generation_config" = números do autor; "template" =
    preset derivado do chat template; "default" = chute; "manual" = o usuário fixou.
    A UI mostra isso pra um preset genérico nunca passar por recomendação oficial.
    """
    model, _ = _validated_model(req.model)
    return sampling.resolve(model, req.config)


# ─── routes: configs ──────────────────────────────────────────────────────────

@app.get("/api/configs")
def list_configs():
    return config_store.read_all_configs()


@app.post("/api/configs")
def save_cfg(cfg: LaunchConfig):
    data = cfg.model_dump(exclude={"mode"})
    data = _validated_config_paths(data)
    saved = config_store.save_config(data)   # atribui id se ausente (mutação)
    return {"ok": True, "config": saved}


@app.delete("/api/configs")
def delete_cfg(req: DeleteConfigRequest):
    removed = config_store.delete_config_by_id(req.id)
    return {"ok": True, "removed": removed}


# ─── routes: estimate / build / suggest ───────────────────────────────────────

@app.post("/api/estimate")
def estimate(req: EstimateRequest):
    model, mmproj = _validated_model(req.model, req.mmproj)
    return estimator.estimate_memory(
        model, req.backend, req.context_window, req.parallel_slots,
        req.kv_cache, req.gpu_layers, mmproj, req.cache_ram, req.mode,
        n_cpu_moe=req.n_cpu_moe,
    )


class EstimateManyRequest(BaseModel):
    items: list[EstimateRequest]


@app.post("/api/estimate-many")
def estimate_many(req: EstimateManyRequest):
    """Estimativa em lote — alimenta a coluna de status do grid sem N round-trips."""
    out: list[dict] = []
    for it in req.items:
        try:
            model, mmproj = _validated_model(it.model, it.mmproj)
            est = estimator.estimate_memory(
                model, it.backend, it.context_window, it.parallel_slots,
                it.kv_cache, it.gpu_layers, mmproj, it.cache_ram, it.mode,
                n_cpu_moe=it.n_cpu_moe,
            )
            out.append({"ok": True, "estimate": est, "model": it.model, "backend": it.backend})
        except Exception as e:
            out.append({"ok": False, "error": str(e), "model": it.model, "backend": it.backend})
    return out


@app.post("/api/build-command")
def build_command(cfg: LaunchConfig):
    try:
        data = cfg.model_dump()
        data = _validated_config_paths(data)
        mode = data.pop("mode", "server")
        cmd = builder_mod.build_command_from_cfg(data, mode=mode)
        return {"command": cmd, "mode": mode}
    except builder_mod.BackendBinaryMissing as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/suggest/n-cpu-moe")
def suggest_ncmoe(req: SuggestRequest):
    model, mmproj = _validated_model(req.model, req.mmproj)
    return {
        "n_cpu_moe": estimator.suggest_n_cpu_moe(
            model, req.backend, req.context_window, req.parallel_slots,
            req.kv_cache, req.gpu_layers, mmproj, req.cache_ram, req.mode,
        )
    }


@app.post("/api/suggest/n-gpu-layers")
def suggest_ngl(req: SuggestRequest):
    model, mmproj = _validated_model(req.model, req.mmproj)
    return {
        "n_gpu_layers": estimator.suggest_n_gpu_layers(
            model, req.backend, req.context_window, req.parallel_slots,
            req.kv_cache, mmproj, req.cache_ram, req.mode,
        )
    }


# ─── launches: fila de eventos por id, runner em thread, SSE drain ────────────

class LaunchSession:
    def __init__(self, cfg: dict, *, attached_pid: int | None = None, launch_id: str | None = None,
                 member_cfgs: list[dict] | None = None):
        self.id = launch_id or uuid.uuid4().hex[:12]
        self.cfg = cfg
        # Modo router: cfg é sintética ({"router": True, ...}) e member_cfgs
        # tem as configs reais que viram seções do preset INI.
        self.member_cfgs = member_cfgs
        self.handle = runner.LaunchHandle(attached_pid=attached_pid)
        self.events = launch_events_core.LaunchEventReplay()
        self.done = False
        self.done_at: float | None = None
        self.thread: threading.Thread | None = None
        self.attached = attached_pid is not None
        if self.attached:
            # Sessão sintética: alimenta a UI com 2 eventos pra parecer um
            # launch normal já carregado. Stop ainda funciona porque o
            # LaunchHandle conhece o PID.
            self.on_event({"type": "start", "attempt": 1, "cmd": "(reanexado)", "config": cfg})
            self.on_event({"type": "load_ok", "attempt": 1})

    def on_event(self, ev: dict) -> None:
        self.events.publish(ev)
        if ev.get("type") in ("done", "giveup"):
            self.done = True
            self.done_at = self.done_at or time.time()

    def _register_pid(self, pid: int) -> None:
        running.register(self.id, pid, self.cfg)

    def start(self) -> None:
        if self.attached:
            # Nada pra rodar — só aguarda cancel/morte do PID.
            return

        def _runner_target():
            launch_log = None
            try:
                log_dir = Path(__file__).resolve().parents[2] / "logs" / "launches"
                log_dir.mkdir(parents=True, exist_ok=True)
                launch_log = (log_dir / f"{self.id}.log").open(
                    "a", encoding="utf-8", buffering=1,
                )
            except Exception as e:
                # Runtime logging is best-effort and must never block a launch.
                self.on_event({"type": "stdout", "line": f"launch log unavailable: {e}"})
            try:
                if self.cfg.get("router") and self.member_cfgs:
                    runner.run_router_server(
                        self.member_cfgs, self.cfg, self.on_event, self.handle,
                        on_proc_pid=self._register_pid, launch_log=launch_log,
                    )
                else:
                    runner.run_server_resiliently(
                        self.cfg, self.on_event, self.handle,
                        on_proc_pid=self._register_pid, launch_log=launch_log,
                    )
            except Exception as e:
                self.on_event({"type": "giveup", "reason": f"runner crash: {e}"})
            finally:
                if launch_log is not None:
                    try:
                        launch_log.close()
                    except Exception:
                        pass
                self.done = True
                self.done_at = self.done_at or time.time()
                running.deregister(self.id)
                self.events.close()

        self.thread = threading.Thread(target=_runner_target, daemon=True)
        self.thread.start()


_SESSIONS: dict[str, Any] = {}
_SESSIONS_LOCK = threading.Lock()

SESSION_RETAIN_SECONDS = 15 * 60
SESSION_MAX_COMPLETED = 32


class _LaunchReservation:
    """Short-lived admission marker held in ``_SESSIONS``."""

    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.done = False


def _prune_sessions() -> None:
    now = time.time()
    with _SESSIONS_LOCK:
        completed = []
        for sid, session in list(_SESSIONS.items()):
            if session.done and session.done_at is not None:
                if now - session.done_at > SESSION_RETAIN_SECONDS:
                    _SESSIONS.pop(sid, None)
                else:
                    completed.append(session)
        completed.sort(key=lambda s: s.done_at or 0, reverse=True)
        for session in completed[SESSION_MAX_COMPLETED:]:
            _SESSIONS.pop(session.id, None)


def _active_launch_id() -> str | None:
    """id do launch vivo (single ou router), se houver. Um por vez: a porta
    reservada do llama-server é única e dois processos disputando GPU só geram
    OOM confuso."""
    _prune_sessions()
    with _SESSIONS_LOCK:
        for s in _SESSIONS.values():
            if not s.done:
                return s.id
    return None


def _reserve_launch(conflict_detail: str) -> _LaunchReservation:
    """Atomically admit one launch before doing work outside the lock."""
    _prune_sessions()
    with _SESSIONS_LOCK:
        for session in _SESSIONS.values():
            if not session.done:
                raise HTTPException(409, conflict_detail)
        reservation = _LaunchReservation()
        _SESSIONS[reservation.id] = reservation
        return reservation


def _install_reserved_session(
    reservation: _LaunchReservation, session: LaunchSession,
) -> None:
    """Replace a reservation with its live session under the admission lock."""
    with _SESSIONS_LOCK:
        if _SESSIONS.get(reservation.id) is not reservation:
            raise RuntimeError("launch admission was lost")
        existing = _SESSIONS.get(session.id)
        if existing is not None and existing is not reservation:
            raise RuntimeError("launch id already exists")
        _SESSIONS.pop(reservation.id, None)
        _SESSIONS[session.id] = session


def _release_launch_admission(
    reservation: _LaunchReservation, session: LaunchSession | None = None,
) -> None:
    """Remove only this request's reservation/session after a failure."""
    with _SESSIONS_LOCK:
        if _SESSIONS.get(reservation.id) is reservation:
            _SESSIONS.pop(reservation.id, None)
        session_id = getattr(session, "id", None)
        if session_id is not None and _SESSIONS.get(session_id) is session:
            _SESSIONS.pop(session_id, None)


@app.post("/api/launch")
def launch(cfg: LaunchConfig):
    reservation = _reserve_launch(
        "Já existe um launch ativo — pare o servidor atual antes de subir "
        "outro modelo (ou use a seleção múltipla pra rodar vários juntos).",
    )
    session = None
    try:
        data = _validated_config_paths(cfg.model_dump(exclude={"mode"}))
        try:
            # Required-binary and MCP boundaries must run before persistence
            # and before creating a session/thread.
            builder_mod.build_command_from_cfg(data, mode="server")
        except (builder_mod.BackendBinaryMissing, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        config_store.save_config(data)
        session = LaunchSession(data)
        _install_reserved_session(reservation, session)
        session.start()
        return {"launch_id": session.id}
    except BaseException:
        _release_launch_admission(reservation, session)
        raise


class RouterLaunchRequest(BaseModel):
    ids: list[str]


@app.post("/api/launch-router")
def launch_router(req: RouterLaunchRequest):
    """Sobe VÁRIOS modelos de uma vez no modo router do llama-server: um
    preset INI com uma seção por config selecionada, todas com
    load-on-startup. O client aponta pra porta reservada do llama-server e escolhe o modelo pelo
    campo "model" do request."""
    reservation = _reserve_launch(
        "Já existe um launch ativo — pare-o antes de subir o modo router."
    )
    session = None
    try:
        if len(req.ids) < 2:
            raise HTTPException(400, "Selecione pelo menos 2 configs pro modo router.")

        by_id = {c.get("id"): c for c in config_store.read_all_configs()}
        missing = [i for i in req.ids if i not in by_id]
        if missing:
            raise HTTPException(404, f"Configs não encontradas: {', '.join(missing)}")
        cfgs = [by_id[i] for i in req.ids]
        if any(c.get("mcp_servers_config") for c in cfgs):
            raise HTTPException(
                400,
                "O modo router não suporta configs com mcp_servers_config; "
                "selecione configs sem MCP.",
            )
        try:
            cfgs = [_validated_config_paths(c) for c in cfgs]
        except HTTPException:
            raise

        try:
            builder_mod.router_binary_for(cfgs)  # binário único + existente
        except builder_mod.BackendBinaryMissing as e:
            raise HTTPException(400, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

        # O binário precisa ser recente o bastante pra ter o modo router.
        probe_backend = cfgs[0].get("backend", "turbo")
        help_text = backends_mod.backend_help_text(probe_backend, "server")
        if help_text and "--models-preset" not in help_text:
            raise HTTPException(
                400,
                f"O llama-server do backend '{probe_backend}' não suporta o modo "
                "router (--models-preset). Atualize/recompile o binário.",
            )

        _, model_ids = builder_mod.build_router_preset(cfgs)
        router_cfg = {
            "router":     True,
            "backend":    probe_backend,
            "config_ids": req.ids,
            "model_ids":  model_ids,
            "models":     [c.get("model", "") for c in cfgs],
        }
        session = LaunchSession(router_cfg, member_cfgs=cfgs)
        _install_reserved_session(reservation, session)
        session.start()
        return {"launch_id": session.id, "config": router_cfg}
    except BaseException:
        _release_launch_admission(reservation, session)
        raise


@app.get("/api/launch/{launch_id}/events")
async def launch_events(launch_id: str, request: Request):
    _prune_sessions()
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(launch_id)
    if session is None or isinstance(session, _LaunchReservation):
        raise HTTPException(404, "launch_id desconhecido")

    def _cursor(raw: str | None, field: str) -> int:
        if raw is None or raw == "":
            return 0
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"{field} inválido") from exc
        if value < 0 or str(value) != str(raw).strip():
            raise HTTPException(400, f"{field} inválido")
        return value

    after = _cursor(request.query_params.get("after"), "after")
    if request.query_params.get("after") is None:
        after = _cursor(request.headers.get("last-event-id"), "Last-Event-ID")
    subscriber = session.events.subscribe(after)

    async def generator():
        gap_sent = False
        while True:
            if await request.is_disconnected():
                return
            cursor_before = subscriber.cursor
            batch = await asyncio.to_thread(subscriber.wait_after, None, 0.5)
            if batch.history_gap and not gap_sent:
                gap_sent = True
                yield f"id: {cursor_before}\nevent: history_gap\ndata: {json.dumps({'type': 'history_gap'}, ensure_ascii=False)}\n\n"
            if not batch.events and batch.closed:
                return
            if not batch.events:
                # keepalive comment line (SSE)
                yield ": keepalive\n\n"
                continue
            for record in batch.events:
                yield f"id: {record.seq}\ndata: {json.dumps(record.event, ensure_ascii=False)}\n\n"
            if batch.closed:
                return

    return StreamingResponse(generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/api/launch/{launch_id}/cancel")
def launch_cancel(launch_id: str):
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(launch_id)
    if session is None or isinstance(session, _LaunchReservation):
        raise HTTPException(404, "launch_id desconhecido")
    session.handle.cancel()
    if session.attached:
        # Sessão re-anexada não tem thread observando — fecha a sessão na mão.
        session.done = True
        session.done_at = time.time()
        running.deregister(session.id)
        session.on_event({"type": "giveup", "reason": "cancelled"})
        session.events.close()
    return {"ok": True}


@app.post("/api/launch/{launch_id}/restart")
def launch_restart(launch_id: str):
    """Restart 'soft': mata o llama-server atual e o loop resiliente sobe de
    novo com a MESMA config (porta fixa → cliente reconecta). Não desiste,
    não degrada. Serve pra destravar um server vivo-mas-preso."""
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(launch_id)
    if session is None or isinstance(session, _LaunchReservation):
        raise HTTPException(404, "launch_id desconhecido")
    if session.done:
        raise HTTPException(400, "launch já encerrado — use Launch pra subir de novo")
    if session.attached:
        # Órfão reanexado: não há thread/loop pra ressubir, só o PID.
        raise HTTPException(
            400, "sessão reanexada não pode reiniciar (sem config/loop ativo)")
    if session.member_cfgs:
        try:
            members = [_validated_config_paths(member) for member in session.member_cfgs]
            builder_mod.router_binary_for(members)
            builder_mod.build_router_preset(members)
        except (builder_mod.BackendBinaryMissing, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        session.member_cfgs = members
        session.cfg = dict(session.cfg)
        session.cfg["models"] = [member["model"] for member in members]
    else:
        try:
            candidate = _validated_config_paths(session.cfg)
            builder_mod.build_command_from_cfg(candidate, mode="server")
        except (builder_mod.BackendBinaryMissing, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        session.cfg = candidate
    ok = session.handle.request_restart()
    if not ok:
        raise HTTPException(409, "não foi possível reiniciar (cancelado ou sem processo)")
    return {"ok": True}


# ─── routes: launches ativos (sobreviventes a restart do Python) ──────────────

@app.get("/api/launches")
def list_launches():
    """Lista launches ativos — usado no boot do frontend pra recuperar
    botão de Stop de servidores órfãos da sessão anterior."""
    _prune_sessions()
    out = []
    with _SESSIONS_LOCK:
        for s in _SESSIONS.values():
            if s.done or isinstance(s, _LaunchReservation):
                continue
            out.append({
                "launch_id": s.id,
                "attached":  s.attached,
                "pid":       s.handle.pid,
                "config":    s.cfg,
            })
    return out


def _attach_orphan_servers() -> None:
    """Boot-time: re-anexa processos llama-server órfãos da sessão anterior.

    Chamado uma vez no startup. Reconcilia o api_running.json filtrando
    PIDs mortos / reciclados, depois cria LaunchSession sintética pra
    cada sobrevivente — assim a UI consegue dar Stop neles.
    """
    survivors = running.reconcile()
    for entry in survivors:
        sid = entry.get("launch_id")
        pid = int(entry.get("pid", 0) or 0)
        cfg = entry.get("cfg") or {}
        if not sid or not pid or not cfg:
            continue
        session = LaunchSession(cfg, attached_pid=pid, launch_id=sid)
        with _SESSIONS_LOCK:
            _SESSIONS[session.id] = session




# ─── routes: HuggingFace download ─────────────────────────────────────────────

class HfUrlRequest(BaseModel):
    url: str


@app.post("/api/hf/resolve")
def hf_resolve(req: HfUrlRequest):
    try:
        repo_id, filename, revision = hf.parse_hf_url(req.url)
        path_policy.validate_repo_id(repo_id)
        if filename is not None:
            path_policy.validate_relative_gguf(filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"repo_id": repo_id, "filename": filename, "revision": revision}


class HfRepoRequest(BaseModel):
    repo_id: str
    revision: str = "main"


@app.post("/api/hf/list")
def hf_list(req: HfRepoRequest):
    try:
        path_policy.validate_repo_id(req.repo_id)
        files, revision = hf.hf_list_with_revision(req.repo_id, req.revision)
    except Exception as e:
        raise HTTPException(400, f"falha ao listar {req.repo_id}: {e}")
    grouped = hf.group_gguf_files(files)
    return {
        "repo_id": req.repo_id,
        "requested_revision": req.revision,
        "revision": revision,
        "files": files,
        **grouped,
    }


class HfSearchRequest(BaseModel):
    query: str
    limit: int = 25


@app.post("/api/hf/search")
def hf_search(req: HfSearchRequest):
    try:
        return {"results": hf.hf_search_gguf(req.query, req.limit)}
    except Exception as e:
        raise HTTPException(400, f"busca falhou: {e}")


class HfDownloadRequest(BaseModel):
    repo_id: str
    revision: str
    rel_paths: list[str]
    subdir: str | None = None
    base_dir: str
    # Re-baixa mesmo se o arquivo já existir com o tamanho certo. Usado pelo
    # botão "Atualizar", onde a divergência foi detectada por sha256.
    force: bool = False


class DownloadState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DownloadSession:
    def __init__(self, plan: dict):
        self.id = uuid.uuid4().hex[:12]
        self.plan = plan
        self.queue: Queue[dict] = Queue()
        self.thread: threading.Thread | None = None
        self.control = hf.DownloadControl()
        self._state = DownloadState.CREATED
        self._state_lock = threading.Lock()
        self.done_at: float | None = None
        destinations = {
            Path(item["dest"]) for item in plan.get("items", [])
        }
        destinations.update(
            path.with_name(path.name + ".part") for path in tuple(destinations)
        )
        base_dir = Path(plan["base_dir"])
        destinations.update({base_dir / "origin.json", base_dir / "sampling.json"})
        self.destinations = frozenset(destinations)

    @property
    def state(self) -> DownloadState:
        with self._state_lock:
            return self._state

    @property
    def done(self) -> bool:
        return self.state in {
            DownloadState.CANCELLED, DownloadState.COMPLETED, DownloadState.FAILED,
        }

    def _transition(self, expected: DownloadState, new: DownloadState) -> bool:
        with self._state_lock:
            if self._state != expected:
                return False
            self._state = new
            if new in {DownloadState.CANCELLED, DownloadState.COMPLETED, DownloadState.FAILED}:
                self.done_at = self.done_at or time.time()
            return True

    def cancel(self) -> bool:
        with self._state_lock:
            if self._state == DownloadState.CREATED:
                self._state = DownloadState.CANCELLING
            elif self._state == DownloadState.RUNNING:
                self._state = DownloadState.CANCELLING
            elif self._state in {
                DownloadState.CANCELLING, DownloadState.CANCELLED,
                DownloadState.COMPLETED, DownloadState.FAILED,
            }:
                already = self._state != DownloadState.CANCELLING
                self.control.cancel()
                return not already
        self.control.cancel()
        return True

    def start(self) -> None:
        if not self._transition(DownloadState.CREATED, DownloadState.RUNNING):
            return

        def _target():
            terminal = DownloadState.COMPLETED
            error = None
            try:
                for ev in hf.stream_download(self.plan, control=self.control):
                    self.control.checkpoint()
                    self.queue.put(ev)
            except hf.DownloadCancelled as exc:
                terminal = DownloadState.CANCELLED
                error = str(exc)
            except Exception as e:
                terminal = DownloadState.CANCELLED if self.control.cancelled else DownloadState.FAILED
                error = str(e)
            finally:
                if terminal == DownloadState.COMPLETED and self.control.cancelled:
                    terminal = DownloadState.CANCELLED
                if terminal == DownloadState.CANCELLED:
                    self._transition(DownloadState.CANCELLING, DownloadState.CANCELLED)
                    self._transition(DownloadState.RUNNING, DownloadState.CANCELLED)
                    event = {"type": "cancelled"}
                elif terminal == DownloadState.FAILED:
                    self._transition(DownloadState.RUNNING, DownloadState.FAILED)
                    event = {"type": "error", "rel": "", "message": error or "download failed"}
                else:
                    self._transition(DownloadState.RUNNING, DownloadState.COMPLETED)
                    event = {"type": "done"}
                self.queue.put(event)
                self.queue.put({"type": "_eof"})
                _release_download_reservation(self)

        # Exactly one worker owns the whole transfer; it is non-daemon so shutdown
        # can prove that no writes remain in flight.
        self.thread = threading.Thread(target=_target, daemon=False, name=f"hf-download-{self.id}")
        self.thread.start()


_DOWNLOADS: dict[str, DownloadSession] = {}
_DOWNLOADS_LOCK = threading.Lock()
_DOWNLOAD_DESTINATIONS: dict[Path, str] = {}
DOWNLOAD_RETAIN_SECONDS = 15 * 60
DOWNLOAD_MAX_COMPLETED = 32
DOWNLOAD_JOIN_TIMEOUT = 2.0


def _prune_downloads() -> None:
    now = time.time()
    with _DOWNLOADS_LOCK:
        done = [s for s in _DOWNLOADS.values() if s.done and s.done_at is not None]
        done.sort(key=lambda s: s.done_at or 0, reverse=True)
        for session in done:
            if (now - (session.done_at or now) > DOWNLOAD_RETAIN_SECONDS
                    or done.index(session) >= DOWNLOAD_MAX_COMPLETED):
                _DOWNLOADS.pop(session.id, None)


def _reserve_download(session: DownloadSession) -> None:
    with _DOWNLOADS_LOCK:
        if any(path in _DOWNLOAD_DESTINATIONS for path in session.destinations):
            raise HTTPException(409, "já existe um download para um destino sobreposto")
        for path in session.destinations:
            _DOWNLOAD_DESTINATIONS[path] = session.id
        _DOWNLOADS[session.id] = session


def _release_download_reservation(session: DownloadSession) -> None:
    with _DOWNLOADS_LOCK:
        for path in session.destinations:
            if _DOWNLOAD_DESTINATIONS.get(path) == session.id:
                _DOWNLOAD_DESTINATIONS.pop(path, None)


def _shutdown_downloads() -> None:
    with _DOWNLOADS_LOCK:
        sessions = list(_DOWNLOADS.values())
    for session in sessions:
        if not session.done:
            session.cancel()
    for session in sessions:
        thread = session.thread
        if thread is not None and thread.is_alive():
            thread.join(DOWNLOAD_JOIN_TIMEOUT)


@app.post("/api/hf/download")
def hf_download(req: HfDownloadRequest):
    configured = settings_store.get_configured_paths()
    if not configured:
        raise HTTPException(
            400,
            "Nenhuma pasta de modelos cadastrada. Vá em Settings e adicione "
            "pelo menos um caminho antes de baixar.",
        )
    try:
        configured_roots = _canonical_roots(download=True)
        chosen = Path(req.base_dir)
        chosen_canonical = chosen.resolve(strict=True)
    except HTTPException as exc:
        # A download is only meaningful with an existing configured root; a
        # missing/invalid configured set is a client-side 400, not a model 404.
        raise HTTPException(400, f"Nenhuma pasta de download válida: {exc.detail}") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, f"base_dir inválido: {exc}") from exc
    if str(chosen_canonical) not in {str(root) for root in configured_roots}:
        raise HTTPException(
            400,
            f"Caminho '{req.base_dir}' não está cadastrado. Caminhos válidos: "
            + ", ".join(str(p) for p in configured),
        )
    try:
        if hf.normalize_commit(req.revision) is None:
            raise HTTPException(400, "revision precisa ser o commit SHA-1 completo")
        plan = hf.plan_download(req.repo_id, req.revision, req.rel_paths, req.subdir,
                                chosen_canonical, force=req.force, require_metadata=True)
    except path_policy.PathPolicyError as exc:
        raise _path_http(exc) from exc
    except hf.IntegrityError as exc:
        raise HTTPException(400, str(exc)) from exc
    session = DownloadSession(plan)
    _prune_downloads()
    _reserve_download(session)
    session.start()
    return {"download_id": session.id, "plan": plan}


@app.get("/api/hf/download/{download_id}/events")
async def hf_download_events(download_id: str, request: Request):
    with _DOWNLOADS_LOCK:
        session = _DOWNLOADS.get(download_id)
    if session is None:
        raise HTTPException(404, "download_id desconhecido")

    async def generator():
        while True:
            if await request.is_disconnected():
                return
            try:
                # to_thread: ver comentário no SSE de launch — get bloqueante
                # no event loop congelava a API inteira durante o download.
                ev = await asyncio.to_thread(session.queue.get, True, 0.5)
            except Empty:
                yield ": keepalive\n\n"
                continue
            if ev.get("type") == "_eof":
                return
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/api/hf/download/{download_id}/cancel")
def hf_download_cancel(download_id: str):
    with _DOWNLOADS_LOCK:
        session = _DOWNLOADS.get(download_id)
    if session is None:
        raise HTTPException(404, "download_id desconhecido")
    session.cancel()
    thread = session.thread
    if thread is not None and thread.is_alive():
        thread.join(DOWNLOAD_JOIN_TIMEOUT)
        if thread.is_alive():
            raise HTTPException(503, "download não encerrou dentro do prazo")
    return {"ok": True, "joined": True}


# ─── routes: delete model ─────────────────────────────────────────────────────

class DeleteModelRequest(BaseModel):
    model: str
    confirm: bool = False


@app.post("/api/models/plan-delete")
def model_plan_delete(req: DeleteModelRequest):
    """Lista o que seria removido sem apagar nada."""
    roots = _canonical_roots()
    try:
        model_path, _ = _validated_model(req.model)
        files = models_repo.plan_delete_model(model_path, roots=roots)
    except path_policy.PathPolicyError as exc:
        raise _path_http(exc) from exc
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except Exception:
            pass
    return {
        "files": [str(f) for f in files],
        "total_bytes": total,
        "count": len(files),
    }


@app.delete("/api/models")
def model_delete(req: DeleteModelRequest):
    if not req.confirm:
        raise HTTPException(400, "confirm=true é obrigatório")
    roots = _canonical_roots()
    try:
        model_path, _ = _validated_model(req.model)
        result = models_repo.delete_model(model_path, roots=roots)
    except path_policy.PathPolicyError as exc:
        raise _path_http(exc) from exc
    # Também remove as configs salvas do modelo (todos os backends)
    config_store.delete_config(str(model_path), backend=None)
    return result


# ─── routes: LM Studio fallback ───────────────────────────────────────────────

@app.get("/api/lms/status")
def lms_status_route():
    return lms.lms_status()


class LmsLoadRequest(BaseModel):
    model: str
    context_window: int = 65_536
    parallel_slots: int = 1


@app.post("/api/lms/load")
def lms_load(req: LmsLoadRequest):
    model, _ = _validated_model(req.model)
    return lms.load_via_lms(model, req.context_window, req.parallel_slots)


# ─── routes: MCP servers ──────────────────────────────────────────────────────

class McpServerPayload(BaseModel):
    id: str | None = None
    name: str = ""
    cwd: str = ""
    command: str = ""
    enabled: bool = False


@app.get("/api/mcp")
def mcp_list(request: Request):
    """Lista servers + status vivo de cada um."""
    _mcp_guard(request)
    servers = mcp_store.list_servers()
    statuses = mcp_runner.list_status()
    out = []
    for s in servers:
        sid = s.get("id", "")
        out.append({**s, "status": statuses.get(sid, mcp_runner.status_for(sid))})
    return out


@app.post("/api/mcp")
def mcp_upsert(request: Request, payload: McpServerPayload):
    _mcp_guard(request)
    data = payload.model_dump()
    entry = mcp_store.upsert_server(data)
    return {"ok": True, "server": entry, "status": mcp_runner.status_for(entry["id"])}


@app.delete("/api/mcp/{server_id}")
def mcp_delete(request: Request, server_id: str):
    _mcp_guard(request)
    # Para o processo antes de remover do disco, pra não deixar órfão.
    try:
        mcp_runner.stop(server_id)
    except Exception:
        pass
    removed = mcp_store.delete_server(server_id)
    return {"ok": removed}


@app.post("/api/mcp/{server_id}/start")
def mcp_start(request: Request, server_id: str):
    _mcp_guard(request)
    try:
        return mcp_runner.start(server_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/mcp/{server_id}/stop")
def mcp_stop(request: Request, server_id: str):
    _mcp_guard(request)
    return mcp_runner.stop(server_id)


@app.get("/api/mcp/{server_id}/logs")
def mcp_logs(request: Request, server_id: str, limit: int = 500):
    _mcp_guard(request)
    return {"logs": mcp_runner.get_logs(server_id, limit=limit)}


@app.get("/api/mcp/{server_id}/events")
async def mcp_events(request: Request, server_id: str):
    _mcp_guard(request)
    inst = mcp_runner.get_instance(server_id)
    if inst is None:
        raise HTTPException(404, "servidor MCP não está em execução")
    q = inst.subscribe()

    async def generator():
        try:
            # Backfill: manda o buffer atual antes de seguir streaming.
            for ev in list(inst.log):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    # to_thread: mesmo motivo dos outros SSE — não bloquear o loop.
                    ev = await asyncio.to_thread(q.get, True, 0.5)
                except Empty:
                    yield ": keepalive\n\n"
                    continue
                if ev.get("kind") == "_eof":
                    return
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            inst.unsubscribe(q)

    return StreamingResponse(generator(), media_type="text/event-stream")


# Keep MCP entirely out of the default route table/OpenAPI.  The functions
# above remain ordinary Python callables for focused tests, but the HTTP surface
# is absent unless explicitly enabled in the trusted process environment.
if not MCP_ENABLED:
    app.router.routes[:] = [
        route for route in app.router.routes
        if not getattr(route, "path", "").startswith("/api/mcp")
    ]


# ─── frontend estático ─────────────────────────────────────────────────────────

# O mount vem depois das rotas /api para não capturar endpoints existentes.
_FRONTEND_DIR = _HERE.parent / "dist"
if _FRONTEND_DIR.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(_FRONTEND_DIR), html=True),
        name="frontend",
    )


# ─── bootstrap: primeira porta livre + bind Tailscale ──────────────────────────

_BIND_HOST = get_bind_host()
_FIRST_PORT = 8420


def _bind_first_available_port(
    host: str | None = None, first_port: int = _FIRST_PORT,
) -> tuple[int, socket.socket]:
    """Reserve a primeira porta livre sem consultar comandos do sistema."""
    host = get_bind_host() if host is None else host
    for port in range(first_port, 65_536):
        if port == constants.LLAMA_SERVER_PORT:
            continue
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return port, sock
        except OSError:
            sock.close()
    raise RuntimeError(f"Nenhuma porta livre a partir de {first_port} em {host}")


def main():
    host = get_bind_host()
    port, sock = _bind_first_available_port(host)
    try:
        print(f"HOST={host} PORT={port}", flush=True)
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
        )
        uvicorn.Server(config).run(sockets=[sock])
    finally:
        sock.close()


if __name__ == "__main__":
    main()
