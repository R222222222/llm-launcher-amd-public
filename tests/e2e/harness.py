"""Safety, lifecycle and evidence primitives for the Phase 6 runner.

This module intentionally does not launch anything on import.  Scenario modules
receive :class:`RunContext` and use its guarded API instead of raw mutations.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse

from checklist import Checklist


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_BASE = "http://127.0.0.1:8420"
INFERENCE_BASE = "http://127.0.0.1:8421"
LOCK_PATH = Path("/tmp/llm-launcher-amd-e2e.lock")
ALLOWED_MODEL_RELATIVE = Path("runtime/fase4-models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
E2E_MODEL_PATH_ENV = "E2E_MODEL_PATH"
PROFILE_MANIFEST_RELATIVE = Path("docs/profiles/seed-profiles.json")
DOWNLOAD_RELATIVE = Path("runtime/e2e-downloads")
DOWNLOAD_SENTINEL = ".llm-launcher-amd-e2e"
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,80}$")
PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# Audited against the server routes: these POSTs derive/validate/read data and
# do not write launcher state.  Do not add check-update: it may update OID cache.
SIDE_EFFECT_FREE_POSTS = frozenset({
    "/api/estimate",
    "/api/estimate-many",
    "/api/build-command",
    "/api/models/meta",
    "/api/models/context-options",
    "/api/models/sampling",
    "/api/models/defaults",
    "/api/suggest/n-cpu-moe",
    "/api/suggest/n-gpu-layers",
    "/api/hf/resolve",
    "/api/hf/list",
    "/api/hf/search",
    "/api/models/plan-delete",
})

SNAPSHOT_RELATIVE = (
    Path("last_config.json"),
    Path("app_settings.json"),
    Path("fail_history.jsonl"),
    Path("oid_cache.json"),
    Path("mcp_servers.json"),
    Path("app/api/api_running.json"),
    Path("app/api/router_preset.ini"),
)


class HarnessError(RuntimeError):
    """Expected, evidence-worthy harness failure."""


class GuardViolation(HarnessError):
    """A mutation was outside the E2E allowlist."""


class PortOccupied(HarnessError):
    """A port is already in use; the runner never adopts or kills it."""


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def allowed_model_path(root: Path = REPO_ROOT) -> Path:
    """Allowlisted launch model: default 1.5B, overridable via E2E_MODEL_PATH.

    The override may be relative (resolved against ``root``) or absolute.
    Symlinks, missing files and non-GGUF paths are always rejected.
    """
    override = os.environ.get(E2E_MODEL_PATH_ENV, "").strip()
    relative = ALLOWED_MODEL_RELATIVE if not override else Path(override)
    candidate = relative if relative.is_absolute() else root / relative
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise HarnessError(f"modelo E2E ausente: {candidate}") from exc
    if resolved != candidate or not resolved.is_file() or resolved.suffix.lower() != ".gguf":
        raise HarnessError(f"modelo E2E não é arquivo GGUF regular: {candidate}")
    return resolved


def _url(path: str, base: str = CONTROL_BASE) -> str:
    return path if path.startswith("http://") else f"{base}{path}"


def _json_body(data: object | None) -> dict[str, Any] | list[Any] | None:
    if data is None:
        return None
    if isinstance(data, (dict, list)):
        return data
    if isinstance(data, (str, bytes)):
        try:
            raw = data.decode("utf-8") if isinstance(data, bytes) else data
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, (dict, list)) else None
    return None


class MutationGuard:
    """Central allowlist for every mutating API request made by the harness."""

    def __init__(self, root: Path, run_id: str):
        if not RUN_ID_RE.fullmatch(run_id):
            raise GuardViolation(f"run_id inválido: {run_id}")
        self.root = root.resolve()
        self.run_id = run_id
        self.expected_model = str(allowed_model_path(self.root))
        self.download_root = (self.root / DOWNLOAD_RELATIVE / run_id).resolve()
        settings_path = self.root / "app_settings.json"
        self.baseline_model_paths: tuple[str, ...] = ()
        self.baseline_backend_paths: dict[str, str] = {}
        if settings_path.exists():
            if settings_path.is_symlink() or not settings_path.is_file():
                raise GuardViolation(f"baseline de settings inválido: {settings_path}")
            try:
                baseline = json.loads(settings_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GuardViolation(f"baseline de settings não é JSON válido: {settings_path}") from exc
            if not isinstance(baseline, dict):
                raise GuardViolation("baseline de settings não é objeto")
            model_paths = baseline.get("model_paths", [])
            backend_paths = baseline.get("backend_paths", {})
            if not isinstance(model_paths, list) or not all(isinstance(path, str) for path in model_paths):
                raise GuardViolation("baseline model_paths inválido")
            if not isinstance(backend_paths, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in backend_paths.items()
            ):
                raise GuardViolation("baseline backend_paths inválido")
            self.baseline_model_paths = tuple(model_paths)
            self.baseline_backend_paths = dict(backend_paths)
        self.allowed_launch_ids: set[str] = set()
        self.owned_launches: dict[str, tuple[str, ...]] = {}
        self.owned_configs: dict[str, str] = {}
        self.expected_config_id: str | None = None
        self.expected_config_fields: dict[str, Any] = {}
        self.ui_launch_requests: list[dict[str, Any]] = []
        self.download_ids: set[str] = set()
        # Seeded profiles (Fase 3): launched through their own guarded path.
        # The 1.5B allowlist, the e2e config namespace and the delete/download
        # guards above stay untouched — profiles only ever launch and stop.
        self.profile_configs: dict[str, dict[str, Any]] = {}
        self.owned_profile_launches: dict[str, str] = {}

    def expect_config(self, config_id: str, fields: dict[str, Any]) -> None:
        self._validate_run_config_id(config_id)
        self.expected_config_id = config_id
        self.expected_config_fields = dict(fields)

    def _validate_run_config_id(self, config_id: object) -> str:
        if not isinstance(config_id, str) or not config_id.startswith("e2e-"):
            raise GuardViolation(f"config id fora do namespace E2E: {config_id}")
        if config_id != f"e2e-{self.run_id}" and not config_id.endswith(f"-{self.run_id}"):
            raise GuardViolation(f"config id fora do namespace desta run: {config_id}")
        return config_id

    def register_config(self, config_id: str, model: str | None = None) -> None:
        config_id = self._validate_run_config_id(config_id)
        exact_model = self.expected_model if model is None else model
        if exact_model != self.expected_model:
            raise GuardViolation("config owned usa modelo diferente do allowlist")
        self.owned_configs[config_id] = exact_model

    def register_launch(self, launch_id: str, config_ids: Iterable[str] | str | None = None) -> None:
        if not isinstance(launch_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}", launch_id):
            raise GuardViolation(f"launch_id opaco inválido: {launch_id!r}")
        if config_ids is None:
            config_ids = (self.expected_config_id,) if self.expected_config_id is not None else ()
        if isinstance(config_ids, str):
            config_ids = (config_ids,)
        owned = tuple(config_ids)
        if not owned or any(config_id not in self.owned_configs for config_id in owned):
            raise GuardViolation(f"launch {launch_id} referencia config não owned: {owned}")
        self.allowed_launch_ids.add(launch_id)
        self.owned_launches[launch_id] = owned

    def register_config_response(self, response: Any) -> str:
        payload = response.json()
        config = payload.get("config") if isinstance(payload, dict) else None
        if not isinstance(config, dict):
            raise GuardViolation("POST /api/configs sem objeto config na resposta")
        config_id = self._validate_run_config_id(config.get("id"))
        self.register_config(config_id, config.get("model"))
        return config_id

    def register_launch_response(self, response: Any, config_ids: Iterable[str] | str) -> str:
        payload = response.json()
        launch_id = payload.get("launch_id") if isinstance(payload, dict) else None
        if not isinstance(launch_id, str):
            raise GuardViolation("resposta de launch sem launch_id opaco")
        self.register_launch(launch_id, config_ids)
        return launch_id

    def register_download(self, download_id: str) -> None:
        """Register the opaque ID returned by the guarded download start."""
        if not isinstance(download_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", download_id):
            raise GuardViolation(f"download_id inválido: {download_id!r}")
        self.download_ids.add(download_id)

    def load_profile_configs(self) -> dict[str, dict[str, Any]]:
        """Load seeded profiles from the versioned manifest.

        Model/mmproj paths are resolved against the repo root and must be
        regular, non-symlink GGUF files inside the repo. Every field present
        in the manifest becomes the expected value for launch validation.
        """
        manifest = self.root / PROFILE_MANIFEST_RELATIVE
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GuardViolation(f"manifest de perfis inválido: {manifest}: {exc}") from exc
        raw_profiles = payload.get("profiles") if isinstance(payload, dict) else None
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise GuardViolation("manifest de perfis sem profiles")
        profiles: dict[str, dict[str, Any]] = {}
        for raw in raw_profiles:
            if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
                raise GuardViolation("perfil do manifest sem id")
            profile_id = raw["id"]
            if not PROFILE_ID_RE.fullmatch(profile_id) or profile_id in profiles:
                raise GuardViolation(f"id de perfil inválido/duplicado: {profile_id!r}")
            expected = dict(raw)
            expected.pop("id", None)
            for field in ("model", "mmproj"):
                value = expected.get(field)
                if value is None:
                    expected[field] = None
                    continue
                expected[field] = str(self._resolve_profile_gguf(value, profile_id, field))
            if not expected.get("model"):
                raise GuardViolation(f"perfil {profile_id} sem modelo")
            profiles[profile_id] = expected
        self.profile_configs = profiles
        return profiles

    def _resolve_profile_gguf(self, value: object, profile_id: str, field: str) -> Path:
        if not isinstance(value, str) or not value:
            raise GuardViolation(f"perfil {profile_id}: {field} inválido")
        raw = Path(value)
        if raw.is_absolute() or ".." in raw.parts:
            raise GuardViolation(f"perfil {profile_id}: {field} precisa ser relativo ao repo")
        resolved = (self.root / raw).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise GuardViolation(f"perfil {profile_id}: {field} fora do repo") from exc
        if resolved.suffix.lower() != ".gguf" or resolved.is_symlink() or not resolved.is_file():
            raise GuardViolation(f"perfil {profile_id}: {field} não é GGUF regular: {resolved}")
        return resolved

    def register_profile_launch_response(self, response: Any, profile_id: str) -> str:
        """Register the launch id of a seeded profile for guarded stop."""
        payload = response.json()
        launch_id = payload.get("launch_id") if isinstance(payload, dict) else None
        if not isinstance(launch_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}", launch_id):
            raise GuardViolation("resposta de launch de perfil sem launch_id opaco")
        if profile_id not in self.profile_configs:
            raise GuardViolation(f"launch de perfil não registrado: {profile_id!r}")
        self.owned_profile_launches[launch_id] = profile_id
        return launch_id

    def _validate_profile_launch(self, body: dict, profile_id: str) -> None:
        expected = self.profile_configs[profile_id]
        mismatched = {
            field: {"expected": expected[field], "actual": body.get(field)}
            for field in expected
            if body.get(field) != expected[field]
        }
        if mismatched:
            raise GuardViolation(f"launch do perfil {profile_id} diverge do seed: {mismatched}")

    def prepare_download_root(self) -> Path:
        self.download_root.mkdir(parents=True, exist_ok=False)
        sentinel = self.download_root / DOWNLOAD_SENTINEL
        sentinel.write_text(f"run_id={self.run_id}\n", encoding="utf-8")
        return self.download_root

    def _inside_download_root(self, value: object) -> bool:
        if not isinstance(value, str) or not value:
            return False
        candidate = Path(value)
        if not candidate.is_absolute():
            return False
        try:
            return candidate.resolve(strict=False) == self.download_root and (self.download_root / DOWNLOAD_SENTINEL).is_file()
        except OSError:
            return False

    def validate(self, method: str, url: str, data: object | None = None) -> None:
        method = method.upper()
        parsed = urlparse(_url(url))
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise GuardViolation(f"host não-loopback bloqueado: {url}")
        if method not in MUTATING_METHODS:
            return
        body = _json_body(data)
        path = parsed.path
        if parsed.port == 8421:
            if method == "POST" and path == "/v1/chat/completions":
                return
            raise GuardViolation(f"mutação de inferência não permitida: {method} {path}")
        if parsed.port != 8420:
            raise GuardViolation(f"porta mutável não permitida: {url}")
        if method == "POST" and path in SIDE_EFFECT_FREE_POSTS:
            return
        if method == "POST" and path == "/api/configs":
            if not isinstance(body, dict) or body.get("model") != self.expected_model:
                raise GuardViolation("config fora do modelo E2E allowlisted")
            config_id = body.get("id")
            if not isinstance(config_id, str):
                raise GuardViolation("config E2E exige id owned explícito")
            self._validate_run_config_id(config_id)
            if config_id not in self.owned_configs and config_id != self.expected_config_id:
                raise GuardViolation("config id não foi preparada/owned por esta run")
            if self.expected_config_id is not None and body.get("id") != self.expected_config_id:
                raise GuardViolation("config id não é o esperado pelo caminho crítico")
            for field, expected in self.expected_config_fields.items():
                if body.get(field) != expected:
                    raise GuardViolation(f"campo divergente na config E2E: {field}")
            if body.get("mmproj") or body.get("mcp_servers_config"):
                raise GuardViolation("config E2E não pode usar mmproj/MCP")
            return
        if method == "POST" and path == "/api/settings":
            if not isinstance(body, dict):
                raise GuardViolation("settings E2E exige JSON objeto")
            model_paths = body.get("model_paths")
            backend_paths = body.get("backend_paths")
            expected_model_paths = list(self.baseline_model_paths)
            if str(self.download_root) not in expected_model_paths:
                expected_model_paths.append(str(self.download_root))
            if model_paths != expected_model_paths:
                raise GuardViolation("settings E2E deve preservar baseline e adicionar somente a raiz de download")
            if not isinstance(backend_paths, dict) or set(backend_paths) - (set(self.baseline_backend_paths) | {"custom"}):
                raise GuardViolation("settings E2E contém backend fora do baseline/custom")
            for name, value in self.baseline_backend_paths.items():
                if name != "custom" and backend_paths.get(name) != value:
                    raise GuardViolation(f"backend baseline alterado: {name}")
            custom = backend_paths.get("custom")
            if custom is not None and not isinstance(custom, str):
                raise GuardViolation("backend custom E2E inválido")
            allowed_custom = {None, "", self.baseline_backend_paths.get("custom"), str(self.download_root)}
            if custom not in allowed_custom:
                raise GuardViolation("backend custom E2E fora da raiz de download")
            if custom not in {None, "", self.baseline_backend_paths.get("custom")} and not self._inside_download_root(custom):
                raise GuardViolation("backend custom E2E fora da raiz de download")
            return
        if method == "DELETE" and path == "/api/configs":
            config_id = body.get("id") if isinstance(body, dict) else None
            if config_id not in self.owned_configs or self.owned_configs[config_id] != self.expected_model:
                raise GuardViolation("DELETE de config não owned pelo E2E")
            return
        if method == "POST" and path == "/api/launch":
            if not isinstance(body, dict):
                raise GuardViolation("launch E2E exige JSON objeto")
            profile_id = body.get("id")
            if profile_id in self.profile_configs:
                self._validate_profile_launch(body, profile_id)
                self.ui_launch_requests.append(json.loads(json.dumps(body, ensure_ascii=False)))
                return
            if body.get("model") != self.expected_model:
                raise GuardViolation("launch fora do modelo E2E allowlisted")
            if body.get("id") not in self.owned_configs:
                raise GuardViolation("POST /api/launch usa config não owned")
            self.ui_launch_requests.append(json.loads(json.dumps(body, ensure_ascii=False)))
            return
        if method == "POST" and path == "/api/launch-router":
            ids = body.get("ids") if isinstance(body, dict) else None
            if not isinstance(ids, list) or not ids or not all(i in self.owned_configs for i in ids):
                raise GuardViolation("launch-router contém config não owned")
            return
        match = re.fullmatch(r"/api/launch/([^/]+)/(cancel|restart)", path)
        if method == "POST" and match:
            launch_id = match.group(1)
            if launch_id not in self.owned_launches and launch_id not in self.owned_profile_launches:
                raise GuardViolation("operação em launch não registrado pelo E2E")
            return
        if method == "DELETE" and path == "/api/models":
            if not isinstance(body, dict) or body.get("model") != self.expected_model:
                raise GuardViolation("DELETE de modelo fora do allowlist literal")
            return
        if method == "POST" and path == "/api/hf/download":
            if not isinstance(body, dict) or not self._inside_download_root(body.get("base_dir")):
                raise GuardViolation("download fora da raiz E2E com sentinel")
            return
        match = re.fullmatch(r"/api/hf/download/([^/]+)/cancel", path)
        if method == "POST" and match:
            if match.group(1) not in self.download_ids:
                raise GuardViolation("cancel de download não registrado pelo E2E")
            return
        raise GuardViolation(f"mutação não autorizada pelo E2E: {method} {path}")

    def install_browser(self, context: Any) -> None:
        """Intercept UI mutations too; no page.evaluate/fetch bypass exists."""
        def handler(route: Any) -> None:
            request = route.request
            self.validate(request.method, request.url, request.post_data)
            route.continue_()

        context.route("**/api/**", handler)

    def write_ui_launch_evidence(self, path: Path) -> None:
        path.write_text(json.dumps(self.ui_launch_requests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class GuardedAPI:
    """Small sync API facade used by scenarios and backed by MutationGuard."""

    def __init__(self, request_context: Any, guard: MutationGuard):
        self.request_context = request_context
        self.guard = guard

    def request(self, method: str, path: str, *, base: str = CONTROL_BASE, data: object | None = None, **kwargs: Any) -> Any:
        url = _url(path, base)
        self.guard.validate(method, url, data)
        fn = getattr(self.request_context, method.lower())
        response = fn(url, data=data, **kwargs) if data is not None else fn(url, **kwargs)
        if getattr(response, "ok", False) and urlparse(url).port == 8420:
            route = urlparse(url).path
            body = _json_body(data)
            if method.upper() == "POST" and route == "/api/configs":
                self.guard.register_config_response(response)
            elif method.upper() == "POST" and route == "/api/launch" and isinstance(body, dict) and body.get("id") in self.guard.profile_configs:
                self.guard.register_profile_launch_response(response, str(body["id"]))
            elif method.upper() == "POST" and route == "/api/launch" and isinstance(body, dict):
                self.guard.register_launch_response(response, str(body["id"]))
            elif method.upper() == "POST" and route == "/api/launch-router" and isinstance(body, dict):
                self.guard.register_launch_response(response, body.get("ids", []))
        return response

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, data: object | None = None, *, base: str = CONTROL_BASE, **kwargs: Any) -> Any:
        return self.request("POST", path, base=base, data=data, **kwargs)

    def delete(self, path: str, data: object | None = None, **kwargs: Any) -> Any:
        return self.request("DELETE", path, data=data, **kwargs)

    @staticmethod
    def json(response: Any) -> Any:
        if not response.ok:
            raise HarnessError(f"HTTP {response.status}: {response.text()[:500]}")
        return response.json()


@dataclass
class FileState:
    path: Path
    existed: bool
    data: bytes = b""
    mode: int | None = None


class RuntimeState:
    """Byte/existence/mode snapshot for all mutable launcher state files."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.states: list[FileState] = []

    def snapshot(self) -> list[FileState]:
        self.states = []
        for relative in SNAPSHOT_RELATIVE:
            path = self.root / relative
            if path.is_symlink():
                raise HarnessError(f"estado é symlink, recusa snapshot: {path}")
            if path.exists():
                if not path.is_file():
                    raise HarnessError(f"estado não é arquivo: {path}")
                self.states.append(FileState(path, True, path.read_bytes(), path.stat().st_mode & 0o7777))
            else:
                self.states.append(FileState(path, False))
        return self.states

    def restore(self, backend_stopped: Callable[[], bool]) -> None:
        if not backend_stopped():
            raise HarnessError("restauração recusada enquanto backend E2E está vivo")
        for state in self.states:
            if state.path.is_symlink():
                raise HarnessError(f"recusa restore sobre symlink atual: {state.path}")
            if state.existed:
                state.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = tempfile.NamedTemporaryFile(
                    mode="wb", dir=state.path.parent, prefix=f".{state.path.name}.e2e-", delete=False,
                )
                temporary_name = Path(temporary.name)
                try:
                    with temporary:
                        temporary.write(state.data)
                        temporary.flush()
                        os.fsync(temporary.fileno())
                    if state.mode is not None:
                        os.chmod(temporary_name, state.mode)
                    os.replace(temporary_name, state.path)
                    self._fsync_directory(state.path.parent)
                finally:
                    if temporary_name.exists():
                        temporary_name.unlink()
            elif state.path.exists():
                state.path.unlink()
                self._fsync_directory(state.path.parent)
        expected = self._expected_inventory()
        actual = self.inventory()
        if actual != expected:
            raise HarnessError(f"restore não conferiu existence/hash/mode: expected={expected!r} actual={actual!r}")

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _expected_inventory(self) -> dict[str, dict[str, Any]]:
        return {
            str(state.path.relative_to(self.root)): {
                "exists": state.existed,
                "sha256": hashlib.sha256(state.data).hexdigest() if state.existed else None,
                "mode": state.mode if state.existed else None,
            }
            for state in self.states
        }

    def inventory(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for state in self.states:
            path = state.path
            if path.is_symlink():
                raise HarnessError(f"estado atual é symlink: {path}")
            exists = path.exists()
            if exists and not path.is_file():
                raise HarnessError(f"estado atual não é arquivo: {path}")
            result[str(path.relative_to(self.root))] = {
                "exists": exists,
                "sha256": _sha256(path) if exists else None,
                "mode": path.stat().st_mode & 0o7777 if exists else None,
            }
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GGUFInventory:
    """Hash manifest for production GGUFs, verified again during cleanup."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.items: dict[str, dict[str, Any]] = {}
        self.captured = False

    def capture(self) -> dict[str, dict[str, Any]]:
        production = self.root / "runtime" / "production-models"
        self.items = {}
        for path in sorted(production.rglob("*.gguf")):
            if not path.is_file() or path.is_symlink():
                raise HarnessError(f"GGUF de produção não é arquivo regular: {path}")
            self.items[str(path.relative_to(self.root))] = {
                "size": path.stat().st_size,
                "mode": path.stat().st_mode & 0o7777,
                "sha256": _sha256(path),
            }
        self.captured = True
        return self.items

    def verify(self) -> None:
        if not self.captured:
            return
        current = GGUFInventory(self.root).capture()
        if current != self.items:
            raise HarnessError("GGUF de runtime/production-models mudou durante o E2E")


def port_occupied(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def assert_ports_free() -> None:
    occupied = [str(port) for port in (8420, 8421) if port_occupied(port)]
    if occupied:
        raise PortOccupied(f"portas ocupadas: {', '.join(occupied)}; não matar/adotar PID")


def read_sysfs_gpu() -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    for device in sorted(Path("/sys/class/drm").glob("card*/device")):
        try:
            if (device / "vendor").read_text(encoding="utf-8").strip().lower() != "0x1002":
                continue
            total = int((device / "mem_info_vram_total").read_text().strip()) // (1024 * 1024)
            used = int((device / "mem_info_vram_used").read_text().strip()) // (1024 * 1024)
        except (OSError, ValueError):
            continue
        cards.append({"path": str(device), "total_mib": total, "used_mib": used, "free_mib": max(total - used, 0)})
    return {
        "gpu_count": len(cards),
        "vram_total_mib": sum(card["total_mib"] for card in cards),
        "vram_used_mib": sum(card["used_mib"] for card in cards),
        "cards": cards,
    }


def median_vram(values: list[int]) -> int:
    if len(values) != 3:
        raise ValueError("baseline exige exatamente três leituras")
    return int(median(values))


class BackendProcess:
    """Own exactly one process group; never kills/adopts an unknown PID."""

    def __init__(self, root: Path, backend_log: Path):
        self.root = root.resolve()
        self.backend_log = backend_log
        self.process: subprocess.Popen[bytes] | None = None
        self._stream: Any = None

    def start(self, env: Mapping[str, str] | None = None) -> None:
        assert_ports_free()
        python = self.root / "app" / ".venv" / "bin" / "python"
        script = self.root / "app" / "api" / "server.py"
        if not python.is_file() or not script.is_file():
            raise HarnessError("python/backend do launcher ausente")
        requested = dict(env or {})
        allowed_env_keys = {
            "LLM_LAUNCHER_ENABLE_MCP", "LLM_LAUNCHER_ALLOW_REMOTE_MCP",
            "LLM_LAUNCHER_HOST", "LLM_LAUNCHER_LLAMA_HOST",
        }
        unknown = set(requested) - allowed_env_keys
        if unknown:
            raise HarnessError(f"env MCP não allowlisted: {sorted(unknown)}")
        if any(str(value) not in {"0", "1"} for value in requested.values()):
            raise HarnessError("env MCP aceita somente valores 0/1")
        if str(requested.get("LLM_LAUNCHER_ALLOW_REMOTE_MCP", "0")) != "0":
            raise HarnessError("remote MCP deve permanecer desabilitado (LLM_LAUNCHER_ALLOW_REMOTE_MCP=0)")
        for key in ("LLM_LAUNCHER_HOST", "LLM_LAUNCHER_LLAMA_HOST"):
            if str(requested.get(key, "127.0.0.1")) != "127.0.0.1":
                raise HarnessError(f"{key} deve ser 127.0.0.1")
        process_env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "LLM_LAUNCHER_ENABLE_MCP": "0",
            "LLM_LAUNCHER_ALLOW_REMOTE_MCP": "0",
            "LLM_LAUNCHER_HOST": "127.0.0.1",
            "LLM_LAUNCHER_LLAMA_HOST": "127.0.0.1",
        }
        process_env.update({str(key): str(value) for key, value in requested.items()})
        # Remote MCP is a permanent safety invariant, including when a caller
        # requests the enabled local-MCP lifecycle.
        process_env["LLM_LAUNCHER_ALLOW_REMOTE_MCP"] = "0"
        self._stream = self.backend_log.open("ab")
        self.process = subprocess.Popen(
            [str(python), str(script)], cwd=self.root, stdin=subprocess.DEVNULL,
            stdout=self._stream, stderr=subprocess.STDOUT, start_new_session=True,
            env=process_env,
        )

    def wait_options(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise HarnessError(f"backend encerrou rc={self.process.returncode}")
            try:
                with urllib.request.urlopen(f"{CONTROL_BASE}/api/options", timeout=1) as response:
                    if response.status == 200:
                        return
                    last = f"HTTP {response.status}"
            except (OSError, urllib.error.URLError) as exc:
                last = str(exc)
            time.sleep(0.25)
        raise HarnessError(f"timeout esperando /api/options: {last}")

    def stopped(self) -> bool:
        if self.process is not None and self.process.poll() is None:
            return False
        if self.process is not None:
            try:
                os.killpg(self.process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                return False
        return not any(port_occupied(port) for port in (8420, 8421))

    def stop(self, timeout: float = 20.0) -> None:
        process = self.process
        if process is not None:
            try:
                # The server may have exited while a child still owns 8421;
                # the process-group check is therefore independent of poll().
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                time.sleep(0.1)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if not self.stopped():
            raise HarnessError("grupo/backend E2E ou portas 8420/8421 ainda persistem")
        if self._stream is not None:
            self._stream.close()
            self._stream = None


@dataclass
class RunContext:
    root: Path
    run_id: str
    evidence_dir: Path
    page: Any
    api: GuardedAPI
    guard: MutationGuard
    checklist: Checklist
    current_item: str = "CP-01"

    @property
    def model(self) -> Path:
        return allowed_model_path(self.root)

    @property
    def model_alias(self) -> str:
        return self.model.stem

    def evidence(self, name: str) -> Path:
        target = self.evidence_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
