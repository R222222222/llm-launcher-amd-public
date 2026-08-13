"""E2E scenario for the Models and HuggingFace Download tabs.

This scenario is intentionally conservative.  It does not manufacture model
rows, shard files, HF responses, or progress events.  In particular, an
external-HF run is only attempted when the runner has explicitly configured
the per-run guarded download directory as a model root.

Required API endpoints (all reads or side-effect-free POSTs except the two
explicitly listed mutations):

* ``GET /api/models``, ``GET /api/settings``;
* ``POST /api/hf/resolve``, ``/api/hf/list``, ``/api/hf/search``;
* ``POST /api/hf/download`` and ``/api/hf/download/{id}/cancel``;
* ``POST /api/models/plan-delete`` and ``DELETE /api/models``.

The harness registers the returned download ID and allows only the matching
scoped cancel route.  The download-root rule remains sentinel-backed; this
scenario never widens it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

from harness import ALLOWED_MODEL_RELATIVE, GuardViolation, HarnessError, RunContext


def _error_text(exc: BaseException) -> str:
    return str(exc) or repr(exc)


HF_REPO = "ggml-org/tinygemma3-GGUF"
HF_REQUESTED_REVISION = "c287502cd9e278dac8eed805c112cce5d0081e0b"
HF_RESOLVED_REVISION = HF_REQUESTED_REVISION
HF_SEARCH = HF_REPO
HF_QUANT = "tinygemma3-Q8_0"
HF_FILE = "tinygemma3-Q8_0.gguf"
HF_SIZE = 47227552
HF_OID = "7566ae7219c93ea2ecc692a931ee122d30c55261d0e2c3347acb8b939d2e9abd"
HF_URL = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REQUESTED_REVISION}/{HF_FILE}"
# Explicit aliases make the qualification fixture easy to audit from evidence
# and keep the size/OID names unambiguous in downstream harness extensions.
HF_COMMIT = HF_RESOLVED_REVISION
HF_EXPECTED_SIZE = HF_SIZE
HF_EXPECTED_OID = HF_OID
DOWNLOAD_SENTINEL = ".llm-launcher-amd-e2e"
RECOVERY_MANIFEST_SUFFIX = ".e2e-recovery.json"
CANCEL_QUIET_SECONDS = 15.0
COMPLETION_SUBDIR = "phase2-completion"
CANCELLATION_SUBDIR = "phase2-cancellation"
DOWNLOAD_IDS = tuple(f"DOWNLOAD-{n:02d}" for n in range(1, 10))
MODEL_IDS = tuple(f"MODELS-{n:02d}" for n in range(1, 5))
TERMINAL_EVENT_TYPES = frozenset({"done", "cancelled", "error"})


class DeletePlanViolation(HarnessError):
    """The server proposed a deletion outside the literal E2E allowlist."""


class HardlinkUnavailable(HarnessError):
    """The filesystem cannot make the mandatory recovery hardlink."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recovery_backup_path(model: Path) -> Path:
    """Return ``model.gguf.e2e-backup`` without changing its directory."""
    return model.with_name(f"{model.stem}.gguf.e2e-backup")


def recovery_manifest_path(model: Path) -> Path:
    """Return the adjacent, fixed-name crash-recovery sidecar."""
    return model.with_name(f"{model.name}{RECOVERY_MANIFEST_SUFFIX}")


def _fsync_dir(directory: Path) -> None:
    """Persist a directory entry; best effort only where the OS lacks O_DIRECTORY."""
    flags = getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_RDONLY", 0)
    fd = os.open(str(directory), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_dir(path.parent)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _regular_file(path: Path) -> os.stat_result:
    if path.is_symlink() or not path.is_file():
        raise HarnessError(f"arquivo regular obrigatório: {path}")
    return path.stat()


def _assert_literal_small_model(root: Path, model: Path) -> None:
    """Reject every GGUF except the runner's literal small-model path."""
    root = root.resolve(strict=True)
    literal = root / ALLOWED_MODEL_RELATIVE
    if literal.is_symlink():
        raise HarnessError(f"small model literal é symlink: {literal}")
    expected = literal.resolve(strict=False)
    if model.resolve(strict=False) != expected:
        raise HarnessError(
            f"recovery restrito ao modelo literal {expected}; recebido {model}"
        )


def create_hardlink_manifest(
    model: Path,
    manifest_path: Path | None = None,
    *,
    root: Path | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    """Create and validate the only recovery artifact used by this scenario.

    ``os.link`` is deliberately not wrapped by a copy fallback.  If hardlinks
    are unavailable, callers must record NV and must not delete anything.
    """
    model = model.resolve(strict=True)
    root = (root or model.parent).resolve(strict=True)
    _assert_literal_small_model(root, model)
    manifest_path = manifest_path or recovery_manifest_path(model)
    if manifest_path.resolve(strict=False) != recovery_manifest_path(model).resolve(strict=False):
        raise HarnessError("manifest de recovery deve ser sidecar adjacente ao modelo")
    model_stat = _regular_file(model)
    backup = recovery_backup_path(model)
    if backup.exists() or backup.is_symlink():
        raise HardlinkUnavailable(f"backup já existe, recusa sobrescrever: {backup}")
    if backup.parent != model.parent:
        raise HardlinkUnavailable("backup não está no mesmo diretório do modelo")
    try:
        os.link(model, backup, follow_symlinks=False)
        _fsync_dir(model.parent)
    except OSError as exc:
        raise HardlinkUnavailable(f"hardlink indisponível: {exc}") from exc

    try:
        backup_stat = _regular_file(backup)
        model_stat = _regular_file(model)
        if backup_stat.st_ino != model_stat.st_ino:
            raise HardlinkUnavailable("backup não compartilha inode com o modelo")
        if backup_stat.st_size != model_stat.st_size:
            raise HardlinkUnavailable("backup tem tamanho divergente")
        digest = sha256_file(model)
        if sha256_file(backup) != digest or model_stat.st_nlink < 2:
            raise HardlinkUnavailable("hash/nlink inválido no backup hardlink")
        payload = {
            "state": "prepared",
            "root": str(root),
            "model": str(model),
            "backup": str(backup),
            "inode": model_stat.st_ino,
            "size": model_stat.st_size,
            "sha256": digest,
            "hash": digest,
            "nlink": model_stat.st_nlink,
            "run_id": run_id,
            "created_at": time.time(),
        }
        _atomic_json(manifest_path, payload)
        return payload
    except Exception:
        # The failed safety preparation must not leave a recovery-looking file.
        try:
            if backup.exists() or backup.is_symlink():
                backup.unlink()
        except OSError:
            pass
        raise


def _manifest_identity(manifest: dict[str, Any], root: Path) -> tuple[Path, Path]:
    if not isinstance(manifest, dict):
        raise HarnessError("manifest de recovery não é objeto")
    required = ("state", "root", "model", "backup", "inode", "size", "sha256", "hash", "nlink", "run_id")
    if any(key not in manifest for key in required):
        raise HarnessError("manifest de recovery incompleto")
    root = root.resolve(strict=True)
    declared_root = Path(str(manifest["root"])).resolve(strict=False)
    model = Path(str(manifest["model"])).resolve(strict=False)
    backup = Path(str(manifest["backup"])).resolve(strict=False)
    if declared_root != root:
        raise HarnessError("manifest de recovery aponta para root diferente")
    if not model.is_absolute() or not model.is_relative_to(root):
        raise HarnessError("modelo do manifest está fora do root")
    _assert_literal_small_model(root, model)
    if backup != recovery_backup_path(model).resolve(strict=False):
        raise HarnessError("paths do manifest não são o GGUF/backup fixos")
    if recovery_manifest_path(model).resolve(strict=False) != Path(str(manifest.get("manifest", recovery_manifest_path(model)))).resolve(strict=False):
        # Older manifests have no explicit field; the adjacent filename remains
        # authoritative.  A supplied field must agree with it.
        raise HarnessError("manifest sidecar não é adjacente ao modelo")
    if not isinstance(manifest["inode"], int) or not isinstance(manifest["size"], int) or manifest["size"] < 0:
        raise HarnessError("inode/size inválidos no manifest")
    if not isinstance(manifest["sha256"], str) or manifest["hash"] != manifest["sha256"] or len(manifest["sha256"]) != 64:
        raise HarnessError("hash inválido no manifest")
    if not isinstance(manifest["nlink"], int) or manifest["nlink"] < 2:
        raise HarnessError("nlink original inválido no manifest")
    if manifest["state"] not in {"prepared", "restored"}:
        raise HarnessError("estado inválido no manifest")
    return model, backup


def validate_hardlink_manifest(manifest: dict[str, Any], model: Path, backup: Path | None = None) -> None:
    """Validate manifest identity and current bytes before a destructive step."""
    model = model.resolve(strict=True)
    backup = backup or recovery_backup_path(model)
    expected_model, expected_backup = _manifest_identity(manifest, Path(str(manifest["root"])))
    if expected_model != model or expected_backup != backup.resolve(strict=False):
        raise HarnessError("manifest aponta para modelo/backup diferente")
    ms = _regular_file(model)
    bs = _regular_file(backup)
    if ms.st_ino != manifest.get("inode") or bs.st_ino != ms.st_ino:
        raise HarnessError("inode do recovery manifest divergente")
    if ms.st_size != manifest.get("size") or bs.st_size != ms.st_size:
        raise HarnessError("tamanho do recovery manifest divergente")
    if ms.st_nlink < 2 or bs.st_nlink < 2:
        raise HarnessError("nlink menor que dois durante recovery")
    if sha256_file(model) != manifest.get("sha256") or sha256_file(backup) != manifest.get("sha256"):
        raise HarnessError("hash do recovery manifest divergente")


def _remove_recovery_sidecars(manifest_path: Path, backup: Path) -> None:
    """Remove recovery artifacts only after the original inode is validated."""
    if manifest_path.exists() or manifest_path.is_symlink():
        manifest_path.unlink()
        _fsync_dir(manifest_path.parent)
    if backup.exists() or backup.is_symlink():
        backup.unlink()
        _fsync_dir(backup.parent)


def recover_pending_model(root: Path) -> dict[str, Any]:
    """Recover a crashed delete without deleting on an inconsistent record.

    The function is intentionally runner-callable: it only considers adjacent
    fixed-name sidecars whose manifest explicitly names this exact root.
    """
    root = root.resolve(strict=True)
    manifests = sorted(
        path for path in root.rglob(f"*{RECOVERY_MANIFEST_SUFFIX}")
        if path.is_file() and not path.is_symlink()
    )
    if not manifests:
        return {"status": "none", "recovered": False}
    if len(manifests) != 1:
        raise HarnessError(f"múltiplos recovery manifests pendentes: {manifests}")
    manifest_path = manifests[0]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarnessError(f"recovery manifest ilegível: {manifest_path}") from exc
    model, backup = _manifest_identity(manifest, root)
    if manifest_path.resolve(strict=False) != recovery_manifest_path(model).resolve(strict=False):
        raise HarnessError("recovery manifest não é sidecar do modelo declarado")
    if model.exists() or model.is_symlink():
        if manifest["state"] == "restored" and not (backup.exists() or backup.is_symlink()):
            # A crash after the WAL state update and backup unlink is safe.
            manifest_path.unlink()
            _fsync_dir(manifest_path.parent)
            return {"status": "cleaned", "recovered": False, "model": str(model)}
        backup_stat = _regular_file(backup)
        if backup_stat.st_ino != manifest["inode"] or backup_stat.st_size != manifest["size"] or sha256_file(backup) != manifest["sha256"]:
            raise HarnessError("backup pendente inconsistente; nada foi apagado")
        current = _regular_file(model)
        if current.st_ino != manifest["inode"] or current.st_size != manifest["size"] or sha256_file(model) != manifest["sha256"]:
            raise HarnessError("modelo existente diverge do recovery manifest; nada foi apagado")
        _remove_recovery_sidecars(manifest_path, backup)
        return {"status": "cleaned", "recovered": False, "model": str(model)}

    backup_stat = _regular_file(backup)
    if backup_stat.st_ino != manifest["inode"] or backup_stat.st_size != manifest["size"] or sha256_file(backup) != manifest["sha256"]:
        raise HarnessError("backup pendente inconsistente; nada foi apagado")
    # The link is the recovery commit.  Keep the manifest and backup until the
    # new model is fully checked and the WAL state is durable.
    os.link(backup, model, follow_symlinks=False)
    _fsync_dir(model.parent)
    restored = _regular_file(model)
    if restored.st_ino != manifest["inode"] or restored.st_size != manifest["size"] or sha256_file(model) != manifest["sha256"]:
        raise HarnessError("modelo restaurado pelo manifest não passou a validação")
    manifest["state"] = "restored"
    _atomic_json(manifest_path, manifest)
    _remove_recovery_sidecars(manifest_path, backup)
    return {"status": "recovered", "recovered": True, "model": str(model)}


def restore_hardlink_manifest(manifest: dict[str, Any]) -> None:
    """Restore the model from its hardlink, then remove only that backup."""
    model = Path(str(manifest["model"]))
    backup = Path(str(manifest["backup"]))
    expected_inode = manifest["inode"]
    expected_size = manifest["size"]
    expected_hash = manifest["sha256"]
    _regular_file(backup)
    if backup.stat().st_ino != expected_inode or backup.stat().st_size != expected_size:
        raise HarnessError("backup não é o inode/tamanho do manifest")
    if sha256_file(backup) != expected_hash:
        raise HarnessError("backup mudou antes da restauração")

    if model.exists() or model.is_symlink():
        current = _regular_file(model)
        if (
            current.st_ino != expected_inode
            or current.st_size != expected_size
            or sha256_file(model) != expected_hash
        ):
            raise HarnessError("delete deixou um modelo divergente; não sobrescrevo")
    else:
        os.link(backup, model, follow_symlinks=False)
        _fsync_dir(model.parent)
        restored = _regular_file(model)
        if restored.st_ino != expected_inode or restored.st_size != expected_size or sha256_file(model) != expected_hash:
            raise HarnessError("hardlink de restauração não foi validado")

    final = _regular_file(model)
    if final.st_ino != expected_inode or final.st_size != expected_size or final.st_nlink != 2:
        raise HarnessError("modelo restaurado não passou a validação final")
    manifest_path = recovery_manifest_path(model)
    manifest["state"] = "restored"
    _atomic_json(manifest_path, manifest)
    # The backup is an E2E artifact, not a second model.  Remove it only after
    # the original path is proven to contain the original inode and bytes.
    _remove_recovery_sidecars(manifest_path, backup)
    final = _regular_file(model)
    if final.st_ino != expected_inode or final.st_size != expected_size or final.st_nlink != 1:
        raise HarnessError("modelo restaurado não passou a validação final")


def validate_delete_plan(
    plan: dict[str, Any], model: Path, production_root: Path,
) -> tuple[Path, ...]:
    """Allow only the literal small GGUF; ``.part`` is never deletable."""
    model = model.resolve(strict=True)
    production_root = production_root.resolve(strict=False)
    raw_files = plan.get("files") if isinstance(plan, dict) else None
    if not isinstance(raw_files, list):
        raise DeletePlanViolation("plan-delete sem lista de arquivos")
    allowed = {model}
    seen: set[Path] = set()
    result: list[Path] = []
    for raw in raw_files:
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise DeletePlanViolation(f"path não absoluto no plan-delete: {raw!r}")
        candidate = Path(raw)
        resolved = candidate.resolve(strict=False)
        if resolved == Path(f"{model}.part") or candidate.name.endswith(".part"):
            raise DeletePlanViolation(f"plan-delete contém .part, que nunca é apagado: {raw}")
        if resolved == production_root or resolved.is_relative_to(production_root):
            raise DeletePlanViolation(f"plan-delete toca production-models: {raw}")
        if resolved not in allowed:
            raise DeletePlanViolation(f"plan-delete fora do GGUF small exato: {raw}")
        if candidate.is_symlink():
            raise DeletePlanViolation(f"plan-delete contém symlink: {raw}")
        if resolved in seen:
            raise DeletePlanViolation(f"plan-delete contém path duplicado: {raw}")
        seen.add(resolved)
        result.append(resolved)
    if model not in seen:
        raise DeletePlanViolation("plan-delete não contém o modelo exato")
    return tuple(result)


def _json_evidence(ctx: RunContext, name: str, payload: Any) -> str:
    path = ctx.evidence(name)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return str(path.relative_to(ctx.evidence_dir))


def _record(
    ctx: RunContext, item_id: str, *, status: str, observed: str = "", reason: str = "", payload: Any = None,
) -> None:
    ctx.current_item = item_id
    evidence: list[str] = []
    if payload is not None:
        evidence = [_json_evidence(ctx, f"models-download/{item_id}.json", payload)]
    ctx.checklist.record(item_id, status, observed=observed, reason=reason, evidence=evidence)


def _error(ctx: RunContext, item_id: str, exc: BaseException, payload: Any = None) -> None:
    message = _error_text(exc)
    details = dict(payload) if isinstance(payload, dict) else {"payload": payload} if payload is not None else {}
    if not details.get("error"):
        details["error"] = message
    details["error_fallback"] = repr(exc)
    _record(ctx, item_id, status="FAIL", observed="assertion raised", reason=message, payload=details)


def _api_json(ctx: RunContext, response: Any) -> Any:
    return ctx.api.json(response)


def _model_row(ctx: RunContext, models: Iterable[dict[str, Any]]) -> dict[str, Any]:
    target = str(ctx.model)
    matches = [m for m in models if isinstance(m, dict) and m.get("path") == target]
    if len(matches) != 1:
        raise HarnessError(f"/api/models tem {len(matches)} ocorrências do modelo E2E")
    return matches[0]


def _find_ui_model_row(ctx: RunContext) -> Any:
    return _find_ui_model_row_for_path(ctx.page, str(ctx.model))


def _find_ui_model_row_for_path(page: Any, path: str) -> Any:
    rows = page.locator("tbody tr")
    for index in range(rows.count()):
        row = rows.nth(index)
        titles = row.locator("[title]").all()
        if any(node.get_attribute("title") == path for node in titles):
            return row
    raise HarnessError(f"linha UI do modelo não encontrada: {path}")


def evaluate_models03_flags(
    models: Iterable[dict[str, Any]], ui_flags: dict[str, dict[str, bool]],
) -> tuple[str, str, dict[str, Any]]:
    """Aggregate flag assertion; false-only small-model flags never PASS."""
    required = ("thinking", "mtp", "vision")
    api_flags: dict[str, dict[str, bool]] = {}
    mismatches: list[dict[str, Any]] = []
    for model in models:
        path = str(model.get("path", ""))
        api_flags[path] = {
            "thinking": model.get("is_thinking") is True,
            "mtp": model.get("is_mtp") is True,
            "vision": bool(model.get("mmproj")),
        }
        if path in ui_flags and api_flags[path] != ui_flags[path]:
            mismatches.append({"path": path, "api": api_flags[path], "ui": ui_flags[path]})
    fixtures = {flag: any(values[flag] for values in api_flags.values()) for flag in required}
    ui_fixtures = {flag: any(values.get(flag) is True for values in ui_flags.values()) for flag in required}
    evidence = {"api_flags": api_flags, "ui_flags": ui_flags, "fixtures": fixtures, "ui_fixtures": ui_fixtures, "mismatches": mismatches}
    if mismatches:
        return "FAIL", "flags API/UI divergentes por title", evidence
    if not all(fixtures.values()) or not all(ui_fixtures.values()):
        missing = [flag for flag in required if not fixtures[flag] or not ui_fixtures[flag]]
        return "NÃO VERIFICADO", f"NV: faltam fixtures reais API+UI para flags: {', '.join(missing)}", evidence
    return "PASS", "flags thinking/MTP/vision conferidas por API e title UI", evidence


def _models_observe(ctx: RunContext) -> None:
    models = _api_json(ctx, ctx.api.get("/api/models"))
    row = _model_row(ctx, models)
    _record(ctx, "MODELS-01", status="PASS", observed="modelo E2E listado uma vez pela API real", payload={"models": models})

    size = ctx.model.stat().st_size
    if row.get("size_mib") != size // (1024 * 1024):
        raise HarnessError(f"tamanho divergente: API={row.get('size_mib')} local={size}")
    _record(ctx, "MODELS-01", status="PASS", observed="modelo listado e tamanho local/API conferido", payload={"row": row, "local_bytes": size})

    # The app exposes the aggregate size only when a real split model exists.
    # Never create synthetic shards to make this criterion pass.
    split = re.match(r"^(?P<base>.+)-\d{5}-of-\d{5}\.gguf$", ctx.model.name, re.IGNORECASE)
    shard_base = split.group("base") if split else ctx.model.stem
    shards = sorted(ctx.model.parent.glob(f"{shard_base}-*-of-*.gguf")) if split else []
    if not shards:
        _record(ctx, "MODELS-02", status="NÃO VERIFICADO", reason="NV: nenhum shard real do modelo E2E; fixture sintético é proibido")
    else:
        parts = [ctx.model, *shards]
        total = sum(_regular_file(part).st_size for part in parts)
        if row.get("size_mib") != total // (1024 * 1024):
            raise HarnessError(f"agregação de shards divergente: API={row.get('size_mib')} local={total}")
        _record(ctx, "MODELS-02", status="PASS", observed="soma dos shards reais confere com a listagem", payload={"shards": [str(p) for p in parts], "total": total})

    flags = {key: row.get(key) for key in ("mmproj", "is_thinking", "is_mtp")}
    if not all(isinstance(row.get(key), (bool, type(None))) for key in ("is_thinking", "is_mtp")):
        raise HarnessError(f"flags não booleanas: {flags!r}")
    ctx.page.get_by_test_id("tab-models").click()
    filter_box = ctx.page.get_by_placeholder("filtrar…")
    filter_box.fill(ctx.model_alias)
    ui_row = _find_ui_model_row(ctx)
    ui_text = ui_row.inner_text()
    if ctx.model_alias not in ui_text:
        raise HarnessError("filtro de alias não manteve o modelo E2E")
    filter_box.fill(ctx.model.name)
    _find_ui_model_row(ctx)
    filter_screenshot = ctx.evidence("screenshots/models-filter.png")
    ctx.page.screenshot(path=str(filter_screenshot), full_page=True)
    filter_box.fill("")
    screenshot = ctx.evidence("screenshots/models-list.png")
    ctx.page.screenshot(path=str(screenshot), full_page=True)
    ui_flags: dict[str, dict[str, bool]] = {}
    for model in models:
        path = str(model.get("path", ""))
        model_row = _find_ui_model_row_for_path(ctx.page, path)
        titles = {node.get_attribute("title") for node in model_row.locator("[title]").all()}
        ui_flags[path] = {
            "thinking": "modelo thinking" in titles,
            "mtp": "modelo MTP" in titles,
            "vision": "tem mmproj" in titles,
        }
    flag_status, flag_observed, flag_evidence = evaluate_models03_flags(models, ui_flags)
    flag_evidence.update({
        "target_flags": flags,
        "filter_ui_text": ui_text,
        "screenshots": [str(filter_screenshot.relative_to(ctx.evidence_dir)), str(screenshot.relative_to(ctx.evidence_dir))],
    })
    if flag_status == "PASS":
        _record(ctx, "MODELS-03", status="PASS", observed=f"{flag_observed}; filtros por alias/nome conferidos", payload=flag_evidence)
    elif flag_status == "NÃO VERIFICADO":
        _record(ctx, "MODELS-03", status=flag_status, reason=flag_observed, payload=flag_evidence)
    else:
        _record(ctx, "MODELS-03", status="FAIL", observed="assertion de flags API/UI falhou", reason=flag_observed, payload=flag_evidence)


def _download_root(ctx: RunContext) -> Path:
    root = ctx.guard.download_root.resolve(strict=False)
    sentinel = root / DOWNLOAD_SENTINEL
    if not root.is_dir() or sentinel.is_symlink() or not sentinel.is_file():
        raise HarnessError("download_root sem diretório/sentinel E2E")
    if sentinel.read_text(encoding="utf-8") != f"run_id={ctx.run_id}\n":
        raise HarnessError("sentinel do download_root divergente")
    return root


def _mark_downloads_external_off(ctx: RunContext) -> None:
    """Mark all download checks NV without touching page, API, or filesystem."""
    for item_id in DOWNLOAD_IDS:
        _record(ctx, item_id, status="NÃO VERIFICADO", reason="NV: external_hf=False; download HF externo desabilitado, sem efeitos")


def _full_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))


def validate_fixture_listing(listing: dict[str, Any]) -> dict[str, Any]:
    """Validate the pinned listing without trusting a mutable branch/tag."""
    if listing.get("repo_id") != HF_REPO:
        raise HarnessError(f"repo divergente na listagem: {listing.get('repo_id')!r}")
    if listing.get("requested_revision") != HF_REQUESTED_REVISION:
        raise HarnessError("requested_revision não é o commit fixado")
    if listing.get("revision") != HF_RESOLVED_REVISION:
        raise HarnessError("revision resolvida não é o commit fixado")
    files = [f for f in listing.get("files", []) if isinstance(f, dict) and f.get("path") == HF_FILE]
    if len(files) != 1:
        raise HarnessError(f"listagem não contém exatamente o GGUF fixado: {files!r}")
    file = files[0]
    if file.get("size") != HF_SIZE or file.get("oid", "").lower() != HF_OID:
        raise HarnessError(f"metadados do GGUF divergentes: {file!r}")
    return file


def validate_download_request(request: dict[str, Any], *, subdir: str) -> None:
    """Validate the exact immutable request sent to each independent session."""
    if request.get("repo_id") != HF_REPO or request.get("revision") != HF_RESOLVED_REVISION:
        raise HarnessError(f"request de download não está pinado: {request!r}")
    if request.get("rel_paths") != [HF_FILE] or request.get("subdir") != subdir:
        raise HarnessError(f"request de download tem destino/arquivo inesperado: {request!r}")
    expected = request.get("expected_files")
    if expected is not None and expected != [{"rel": HF_FILE, "expected_size": HF_SIZE, "expected_oid": HF_OID}]:
        raise HarnessError(f"expected_files divergente: {expected!r}")


def _validate_download_plan(plan: dict[str, Any], root: Path) -> None:
    if plan.get("repo_id") != HF_REPO:
        raise HarnessError(f"repo divergente no plano: {plan.get('repo_id')!r}")
    if plan.get("download_revision", plan.get("revision")) != HF_RESOLVED_REVISION:
        raise HarnessError(f"revision do plano divergente: {plan!r}")
    if plan.get("requested_revision", HF_REQUESTED_REVISION) != HF_REQUESTED_REVISION:
        raise HarnessError("requested_revision do plano divergente")
    if len(plan.get("items", [])) != 1:
        raise HarnessError(f"plano não contém exatamente um arquivo: {plan!r}")
    for item in plan.get("items", []):
        dest = Path(str(item.get("dest", "")))
        if dest.is_symlink() or not dest.is_absolute() or not dest.resolve(strict=False).is_relative_to(root):
            raise HarnessError(f"destino de download fora do sentinel: {dest}")
        if (root / DOWNLOAD_SENTINEL).resolve() == dest.resolve(strict=False):
            raise HarnessError("download plan tenta sobrescrever o sentinel")
        if item.get("rel") != HF_FILE or item.get("expected_size") != HF_SIZE or str(item.get("expected_oid", "")).lower() != HF_OID:
            raise HarnessError(f"metadados do plano divergentes: {item!r}")
        if f"/resolve/{HF_RESOLVED_REVISION}/{HF_FILE}" not in str(item.get("url", "")):
            raise HarnessError(f"URL do plano não usa commit completo: {item!r}")


def _download_snapshots(plan: dict[str, Any], root: Path) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for item in plan.get("items", []):
        dest = Path(str(item.get("dest", "")))
        for path in (dest, Path(f"{dest}.part")):
            if path.exists() and not path.is_symlink() and path.is_file():
                stat = path.stat()
                if not path.resolve(strict=False).is_relative_to(root):
                    raise HarnessError(f"artefato de download fora do sentinel: {path}")
                result[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return result


def _wait_download_done(ctx: RunContext, plan: dict[str, Any], root: Path, timeout: float = 900.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _validate_download_plan(plan, root)
        last = _download_snapshots(plan, root)
        done_text = ctx.page.get_by_text("concluído", exact=False).count() > 0
        complete = all(Path(str(item["dest"])).is_file() and not Path(f"{item['dest']}.part").exists() for item in plan.get("items", []))
        if done_text and complete:
            return {"done_text": True, "artifacts": last}
        time.sleep(0.5)
    raise HarnessError(f"download não finalizou sem .part: {last}")


def _wait_cancel_quiet(ctx: RunContext, plan: dict[str, Any], root: Path, timeout: float = CANCEL_QUIET_SECONDS) -> dict[str, Any]:
    """Observe a complete fixed quiet window; cancellation UI is not enough."""
    first = _download_snapshots(plan, root)
    deadline = time.monotonic() + timeout
    samples: list[dict[str, tuple[int, int]]] = [first]
    while time.monotonic() < deadline:
        current = _download_snapshots(plan, root)
        samples.append(current)
        new_artifacts = set(current) - set(first)
        if new_artifacts:
            raise HarnessError(f"artefato novo após cancel: {sorted(new_artifacts)}")
        for name, value in current.items():
            if value[0] != first.get(name, value)[0]:
                raise HarnessError(f"worker continuou escrevendo após cancel: {name}")
        for item in plan.get("items", []):
            dest = str(item["dest"])
            if dest in current and dest not in first:
                raise HarnessError(f"worker finalizou após cancel: {dest}")
        time.sleep(0.5)
    return {"samples": samples, "quiet_seconds": timeout, "quiet": True}


def _download_events(ctx: RunContext, download_id: str) -> list[dict[str, Any]]:
    """Drain the replay endpoint after the worker has joined."""
    response = ctx.api.get(f"/api/hf/download/{download_id}/events")
    if not response.ok:
        raise HarnessError(f"eventos do download HTTP {response.status}: {response.text()[:500]}")
    events: list[dict[str, Any]] = []
    for line in response.text().splitlines():
        if not line.startswith("data:"):
            continue
        try:
            value = json.loads(line[5:].strip())
        except json.JSONDecodeError as exc:
            raise HarnessError(f"evento SSE inválido: {line!r}") from exc
        if isinstance(value, dict):
            events.append(value)
    terminals = [event for event in events if event.get("type") in TERMINAL_EVENT_TYPES]
    if len(terminals) != 1:
        raise HarnessError(f"download sem exatamente um evento terminal: {events!r}")
    if terminals[0]["type"] == "error":
        raise HarnessError(f"download terminou com erro terminal: {terminals[0]!r}")
    return events


def validate_cancel_response(payload: dict[str, Any]) -> None:
    """The cancel HTTP response must explicitly prove the bounded join."""
    if payload.get("ok") is not True or payload.get("joined") is not True:
        raise HarnessError(f"cancel não confirmou join do worker: {payload!r}")
    if payload.get("state") not in {None, "CANCELLED"}:
        raise HarnessError(f"estado após cancel divergente: {payload!r}")


def assert_independent_destinations(plans: Iterable[dict[str, Any]]) -> None:
    paths: set[Path] = set()
    for plan in plans:
        current = {Path(str(item["dest"])).resolve(strict=False) for item in plan.get("items", [])}
        if paths & current:
            raise HarnessError(f"sessões de download compartilham destino: {paths & current}")
        paths.update(current)


def _abort_after_cancel(ctx: RunContext, reason: str) -> bool:
    """Cancel path never starts a second download or model deletion."""
    for item_id in ("DOWNLOAD-08", "DOWNLOAD-09"):
        if ctx.checklist.results[item_id].status == "NÃO VERIFICADO":
            _record(ctx, item_id, status="NÃO VERIFICADO", reason=f"NV: {reason}")
    return False


def _run_downloads_legacy(ctx: RunContext) -> bool:
    """Run DOWNLOAD-* and return whether it is safe to continue to deletion."""
    root = _download_root(ctx)
    settings = _api_json(ctx, ctx.api.get("/api/settings"))
    configured = settings.get("model_paths", []) if isinstance(settings, dict) else []
    if str(root) not in {str(Path(p).resolve()) for p in configured if isinstance(p, str)}:
        reason = "NV: guard.download_root não está entre as raízes de modelos configuradas; preflight recusa efeitos"
        for item_id in DOWNLOAD_IDS:
            _record(ctx, item_id, status="NÃO VERIFICADO", reason=reason)
        return True
    _record(ctx, "DOWNLOAD-01", status="PASS", observed="sentinel download_root selecionável e cadastrado", payload={"root": str(root), "settings": settings})

    resolved = _api_json(ctx, ctx.api.post("/api/hf/resolve", {"url": HF_URL}))
    if resolved.get("repo_id") != HF_REPO:
        raise HarnessError(f"resolve HF divergente: {resolved!r}")
    _record(ctx, "DOWNLOAD-02", status="PASS", observed="URL real resolvida para owner/repo fixture", payload=resolved)

    listing = _api_json(ctx, ctx.api.post("/api/hf/list", {"repo_id": HF_REPO}))
    groups = listing.get("quants", {}) if isinstance(listing, dict) else {}
    matching = [key for key in groups if HF_QUANT.lower() in str(key).lower()]
    if matching != [next((key for key in matching if str(key).lower() == HF_FILE), matching[0] if matching else "")]:
        raise HarnessError(f"quant fixture Q4_K_M não determinístico: {matching!r}")
    quant_key = matching[0]
    quant_files = groups[quant_key]
    if len(quant_files) != 1 or quant_files[0].get("path") != HF_FILE:
        raise HarnessError(f"fixture não é um único arquivo estável: {quant_files!r}")
    search = _api_json(ctx, ctx.api.post("/api/hf/search", {"query": HF_SEARCH, "limit": 25}))
    results = search.get("results", []) if isinstance(search, dict) else []
    if not any(r.get("repo_id") == HF_REPO for r in results if isinstance(r, dict)):
        raise HarnessError(f"busca HF não encontrou fixture: {results!r}")
    _record(ctx, "DOWNLOAD-03", status="PASS", observed="busca real retornou o owner/repo fixture", payload={"query": HF_SEARCH, "results": results})
    _record(ctx, "DOWNLOAD-04", status="PASS", observed="agrupamento real contém exatamente Q4_K_M e um GGUF", payload={"quant": quant_key, "files": quant_files})

    ctx.page.get_by_test_id("tab-download").click()
    ctx.page.get_by_placeholder("https://huggingface.co/owner/repo  ou  owner/repo").fill(HF_URL)
    ctx.page.get_by_role("button", name="inspecionar", exact=True).click()
    ctx.page.get_by_text(HF_REPO, exact=True).wait_for(timeout=30_000)
    ctx.page.get_by_text(quant_key, exact=True).click()
    for checkbox in ctx.page.get_by_role("checkbox").all():
        if checkbox.is_checked():
            checkbox.uncheck()
    with ctx.page.expect_response(lambda r: r.url.endswith("/api/hf/download") and r.request.method == "POST") as response_info:
        ctx.page.get_by_role("button", name="baixar", exact=True).click()
    download_response = response_info.value
    download_payload = download_response.json()
    plan = download_payload.get("plan")
    download_id = download_payload.get("download_id")
    if not isinstance(plan, dict) or not download_id:
        raise HarnessError(f"download sem plan/id: {download_payload!r}")
    ctx.guard.register_download(str(download_id))
    _validate_download_plan(plan, root)

    progress_evidence: dict[str, Any] = {"plan": plan, "download_id": download_id}
    progress_seen = False
    for _ in range(30):
        text = ctx.page.locator("body").inner_text()
        if "MB/s" in text:
            progress_seen = True
            progress_evidence["ui_text"] = text
            break
        time.sleep(0.5)
    if progress_seen:
        _record(ctx, "DOWNLOAD-05", status="PASS", observed="UI exibiu progresso e velocidade MB/s", payload=progress_evidence)
    else:
        _record(ctx, "DOWNLOAD-05", status="FAIL", reason="FAIL: nenhum progresso/velocidade observável na UI", payload=progress_evidence)

    # The unextended harness must not even attempt this mutation.  This check
    # is local and has no backend effect; a future guard extension enables the
    # real UI cancellation assertion.
    cancel_allowed = True
    try:
        ctx.guard.validate("POST", f"http://127.0.0.1:8420/api/hf/download/{download_id}/cancel", {})
    except GuardViolation as exc:
        cancel_allowed = False
        _record(ctx, "DOWNLOAD-06", status="NÃO VERIFICADO", reason=f"NV: guard extension ausente para cancel scoped: {_error_text(exc)}")

    ctx.page.get_by_test_id("tab-models").click()
    ctx.page.get_by_test_id("tab-download").click()
    tab_evidence = ctx.evidence("models-download/tab-switch.png")
    ctx.page.screenshot(path=str(tab_evidence), full_page=True)
    if ctx.page.get_by_text("Progresso · destino:", exact=False).count() == 0:
        raise HarnessError("progresso não sobreviveu à troca de aba")
    _record(ctx, "DOWNLOAD-07", status="PASS", observed="progresso persistiu após troca Models/Download", payload={"screenshot": str(tab_evidence.relative_to(ctx.evidence_dir)), "download_id": download_id})

    if cancel_allowed:
        cancel_button = ctx.page.get_by_role("button", name="cancelar", exact=True)
        if cancel_button.count() == 0:
            _record(ctx, "DOWNLOAD-06", status="NÃO VERIFICADO", reason="NV: fixture terminou antes de existir janela cancelável")
        else:
            cancel_button.click()
            try:
                quiet = _wait_cancel_quiet(ctx, plan, root)
            except Exception as exc:
                _error(ctx, "DOWNLOAD-06", exc, {"download_id": download_id, "snapshots": _download_snapshots(plan, root)})
                return _abort_after_cancel(ctx, "worker/artefato observado após cancelamento")
            _record(
                ctx,
                "DOWNLOAD-06",
                status="FAIL",
                observed="janela quiet fixa sem artefato novo/crescimento, mas worker não expõe terminação autoritativa",
                reason="FAIL: backend atual sinaliza cancelamento sem prova de que a thread de download terminou",
                payload={"download_id": download_id, **quiet},
            )
            return _abort_after_cancel(ctx, "cancel sem prova de terminação do worker; não iniciar segundo download no mesmo destino")

    try:
        finished = _wait_download_done(ctx, plan, root)
    except Exception as exc:
        _error(ctx, "DOWNLOAD-08", exc, {"plan": plan})
        return False
    _record(ctx, "DOWNLOAD-05", status="PASS", observed="progresso concluído sem artefato .part", payload={"plan": plan, **finished})

    integrity: dict[str, Any] = {"files": []}
    for item in plan.get("items", []):
        dest = Path(str(item["dest"]))
        stat = _regular_file(dest)
        expected_size = int(item.get("expected_size") or 0)
        if expected_size and stat.st_size != expected_size:
            _error(ctx, "DOWNLOAD-08", HarnessError(f"size divergente em {dest}: {stat.st_size} != {expected_size}"), integrity)
            return False
        digest = sha256_file(dest)
        oid = item.get("expected_oid")
        if oid and digest.lower() != str(oid).lower():
            _error(ctx, "DOWNLOAD-08", HarnessError(f"OID/hash divergente em {dest}"), {"dest": str(dest), "oid": oid, "sha256": digest})
            return False
        integrity["files"].append({"dest": str(dest), "size": stat.st_size, "sha256": digest, "oid": oid})
    _record(ctx, "DOWNLOAD-08", status="PASS", observed="size validado e hash SHA-256 conferido quando HF forneceu OID", payload=integrity)

    sidecars = {Path(str(item["dest"])).parent / "sampling.json" for item in plan.get("items", [])}
    sampling_files = [path for path in sidecars if path.is_file() and not path.is_symlink()]
    if not sampling_files:
        _error(ctx, "DOWNLOAD-09", HarnessError("sampling.json não foi gerado pelo download"), {"sidecars": [str(p) for p in sidecars]})
    else:
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sampling_files]
        if not all(isinstance(p, dict) and any(k in p for k in ("temp", "temperature", "top_p")) for p in payloads):
            _error(ctx, "DOWNLOAD-09", HarnessError("sampling.json sem valores de sampling"), {"files": [str(p) for p in sampling_files], "payloads": payloads})
        else:
            _record(ctx, "DOWNLOAD-09", status="PASS", observed="sampling.json real gerado e contém valores", payload={"files": [str(p) for p in sampling_files], "payloads": payloads})
    return True


def _wait_download_files(plan: dict[str, Any], root: Path, timeout: float = 900.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, tuple[int, int]] = {}
    while time.monotonic() < deadline:
        _validate_download_plan(plan, root)
        last = _download_snapshots(plan, root)
        if all(
            Path(str(item["dest"])).is_file()
            and not Path(f"{item['dest']}.part").exists()
            for item in plan.get("items", [])
        ):
            return {"artifacts": last}
        time.sleep(0.5)
    raise HarnessError(f"download não finalizou sem .part: {last}")


def _terminal(events: list[dict[str, Any]], expected: str) -> dict[str, Any]:
    terminals = [event for event in events if event.get("type") in TERMINAL_EVENT_TYPES]
    if len(terminals) != 1:
        raise HarnessError(f"eventos terminais não determinísticos: {events!r}")
    event = terminals[0]
    if event.get("type") != expected:
        raise HarnessError(f"terminal esperado={expected}, observado={event!r}")
    return event


def _origin_evidence(plan: dict[str, Any], expected_dest: Path) -> dict[str, Any]:
    origin = expected_dest.parent / "origin.json"
    if origin.is_symlink() or not origin.is_file():
        raise HarnessError(f"origin.json ausente ou symlink: {origin}")
    payload = json.loads(origin.read_text(encoding="utf-8"))
    entry = payload.get("files", {}).get(HF_FILE, {})
    if (
        payload.get("repo_id") != HF_REPO
        or payload.get("requested_revision") != HF_REQUESTED_REVISION
        or payload.get("revision") != HF_RESOLVED_REVISION
        or entry.get("size") != HF_SIZE
        or str(entry.get("oid", "")).lower() != HF_OID
    ):
        raise HarnessError(f"origin.json não é procedência verificada: {payload!r}")
    digest = sha256_file(expected_dest)
    if digest != HF_OID or entry.get("oid", "").lower() != digest:
        raise HarnessError(f"origin não contém o digest efetivamente verificado: {payload!r}")
    return {"path": str(origin), "payload": payload, "actual_sha256": digest}


def _run_downloads(ctx: RunContext) -> bool:
    """Run pinned completion and cancellation as independent sessions.

    The deterministic/local cancellation contract is covered by the pure helper
    tests below; live qualification deliberately uses two fresh guarded
    destinations so a cancellation cannot invalidate completion evidence.
    """
    root = _download_root(ctx)
    settings = _api_json(ctx, ctx.api.get("/api/settings"))
    configured = settings.get("model_paths", []) if isinstance(settings, dict) else []
    if str(root) not in {str(Path(p).resolve()) for p in configured if isinstance(p, str)}:
        reason = "NV: guard.download_root não está entre as raízes configuradas; nenhum efeito HF iniciado"
        for item_id in DOWNLOAD_IDS:
            _record(ctx, item_id, status="NÃO VERIFICADO", reason=reason)
        return True
    _record(ctx, "DOWNLOAD-01", status="PASS", observed="sentinel download_root selecionável e cadastrado", payload={"root": str(root), "settings": settings})

    resolved = _api_json(ctx, ctx.api.post("/api/hf/resolve", {"url": HF_URL}))
    if resolved.get("repo_id") != HF_REPO or resolved.get("revision") != HF_RESOLVED_REVISION:
        raise HarnessError(f"resolve HF não retornou o commit fixado: {resolved!r}")
    _record(ctx, "DOWNLOAD-02", status="PASS", observed="URL pinned resolvida para repo e commit completo", payload=resolved)

    listing = _api_json(ctx, ctx.api.post("/api/hf/list", {"repo_id": HF_REPO, "revision": HF_REQUESTED_REVISION}))
    file_meta = validate_fixture_listing(listing)
    _record(ctx, "DOWNLOAD-03", status="PASS", observed="list aceitou requested_revision e devolveu resolved revision completa", payload={"listing": listing})
    search = _api_json(ctx, ctx.api.post("/api/hf/search", {"query": HF_SEARCH, "limit": 25}))
    if not any(item.get("repo_id") == HF_REPO for item in search.get("results", []) if isinstance(item, dict)):
        raise HarnessError(f"busca HF não encontrou fixture: {search!r}")
    _record(ctx, "DOWNLOAD-04", status="PASS", observed="GGUF pinado tem tamanho/OID exatos e busca retorna o repo", payload={"file": file_meta, "search": search})

    completion_subdir = f"{COMPLETION_SUBDIR}-{ctx.run_id}"
    cancel_subdir = f"{CANCELLATION_SUBDIR}-{ctx.run_id}"
    common = {
        "repo_id": HF_REPO,
        "revision": HF_RESOLVED_REVISION,
        "rel_paths": [HF_FILE],
        "expected_files": [{"rel": HF_FILE, "expected_size": HF_SIZE, "expected_oid": HF_OID}],
        "base_dir": str(root),
        "force": True,
    }

    completion_request = {**common, "subdir": completion_subdir}
    validate_download_request(completion_request, subdir=completion_subdir)
    completion_response = ctx.api.post("/api/hf/download", completion_request)
    completion_payload = _api_json(ctx, completion_response)
    completion_plan = completion_payload.get("plan")
    completion_id = completion_payload.get("download_id")
    if not isinstance(completion_plan, dict) or not isinstance(completion_id, str):
        raise HarnessError(f"completion sem plan/id: {completion_payload!r}")
    ctx.guard.register_download(completion_id)
    _validate_download_plan(completion_plan, root)

    cancel_request = {**common, "subdir": cancel_subdir}
    validate_download_request(cancel_request, subdir=cancel_subdir)
    cancel_response = ctx.api.post("/api/hf/download", cancel_request)
    cancel_payload = _api_json(ctx, cancel_response)
    cancel_id = cancel_payload.get("download_id")
    cancel_plan = cancel_payload.get("plan")
    if not isinstance(cancel_id, str) or not isinstance(cancel_plan, dict):
        raise HarnessError(f"cancel session sem plan/id: {cancel_payload!r}")
    ctx.guard.register_download(cancel_id)
    _validate_download_plan(cancel_plan, root)
    assert_independent_destinations((completion_plan, cancel_plan))
    _record(ctx, "DOWNLOAD-07", status="PASS", observed="completion e cancellation usam sessões e destinos independentes", payload={"completion": completion_plan, "cancellation": cancel_plan})

    cancel_dest = Path(str(cancel_plan["items"][0]["dest"]))
    cancel_failure: BaseException | None = None
    cancel_result: dict[str, Any] = {}
    cancel_events: list[dict[str, Any]] = []
    quiet: dict[str, Any] = {}
    try:
        cancel_result = _api_json(ctx, ctx.api.post(f"/api/hf/download/{cancel_id}/cancel", {}))
        validate_cancel_response(cancel_result)
        cancel_events = _download_events(ctx, cancel_id)
        _terminal(cancel_events, "cancelled")
        if cancel_dest.exists() or (cancel_dest.parent / "origin.json").exists():
            raise HarnessError("cancel deixou final/origin.json")
        quiet = _wait_cancel_quiet(ctx, cancel_plan, root)
        if cancel_dest.exists() or (cancel_dest.parent / "origin.json").exists():
            raise HarnessError("final/origin apareceu durante a janela quiet de cancelamento")
        _record(ctx, "DOWNLOAD-06", status="PASS", observed="cancel response confirmou join, terminal cancelled e quiet window sem final/origin/escritas posteriores", payload={"response": cancel_result, "events": cancel_events, **quiet})
    except Exception as exc:
        # Keep the completion session independent: cancellation evidence fails,
        # but its failure must never prevent completion/hash/origin evidence.
        cancel_failure = exc
        _error(ctx, "DOWNLOAD-06", exc, {"response": cancel_result, "events": cancel_events, "quiet": quiet})

    finished = _wait_download_files(completion_plan, root)
    completion_events = _download_events(ctx, completion_id)
    _terminal(completion_events, "done")
    completion_dest = Path(str(completion_plan["items"][0]["dest"]))
    if completion_dest.stat().st_size != HF_SIZE or sha256_file(completion_dest) != HF_OID:
        raise HarnessError("completion não produziu size/hash exatos")
    if Path(f"{completion_dest}.part").exists():
        raise HarnessError("completion deixou .part")
    origin = _origin_evidence(completion_plan, completion_dest)
    _record(ctx, "DOWNLOAD-05", status="PASS", observed="sessão pinned completou com terminal done, final sem .part e hash exato", payload={"plan": completion_plan, "events": completion_events, **finished, "origin": origin})
    _record(ctx, "DOWNLOAD-08", status="PASS", observed="size/OID final e digest computado conferem", payload={"dest": str(completion_dest), "size": HF_SIZE, "sha256": HF_OID})

    sampling_files = [completion_dest.parent / "sampling.json"]
    existing_sampling = [path for path in sampling_files if path.is_file() and not path.is_symlink()]
    if existing_sampling:
        _record(ctx, "DOWNLOAD-09", status="PASS", observed="sampling.json best-effort gerado após completion", payload={"files": [str(path) for path in existing_sampling]})
    else:
        _record(ctx, "DOWNLOAD-09", status="NÃO VERIFICADO", reason="NV: sampling.json é best-effort e não foi produzido")
    return cancel_failure is None


def _delete_model_via_ui(ctx: RunContext) -> None:
    model = ctx.model.resolve(strict=True)
    manifest: dict[str, Any] | None = None
    try:
        try:
            manifest = create_hardlink_manifest(model, root=ctx.root, run_id=ctx.run_id)
            _json_evidence(ctx, "models-download/recovery-manifest.json", manifest)
        except HardlinkUnavailable as exc:
            _record(ctx, "MODELS-04", status="NÃO VERIFICADO", reason=f"NV: sem hardlink obrigatório; nenhuma cópia fallback foi usada ({_error_text(exc)})")
            return
        validate_hardlink_manifest(manifest, model)

        ctx.page.get_by_test_id("tab-models").click()
        ctx.page.get_by_placeholder("filtrar…").fill(ctx.model_alias)
        row = _find_ui_model_row(ctx)
        with ctx.page.expect_response(lambda r: r.url.endswith("/api/models/plan-delete") and r.request.method == "POST") as plan_info:
            row.get_by_title("apagar do disco").click()
        plan_response = plan_info.value.json()
        allowed = validate_delete_plan(
            plan_response, model, ctx.root / "runtime" / "production-models",
        )
        pending_part = Path(f"{model}.part")
        if pending_part.exists() or pending_part.is_symlink():
            raise DeletePlanViolation(".part presente antes da confirmação; delete abortado sem apagar .part")
        delete_screenshot = ctx.evidence("screenshots/models-delete-plan.png")
        ctx.page.screenshot(path=str(delete_screenshot), full_page=True)
        plan_evidence = {
            "response": plan_response,
            "allowed": [str(p) for p in allowed],
            "screenshot": str(delete_screenshot.relative_to(ctx.evidence_dir)),
        }
        _json_evidence(ctx, "models-download/delete-plan.json", plan_evidence)
        with ctx.page.expect_response(lambda r: r.url.endswith("/api/models") and r.request.method == "DELETE") as delete_info:
            ctx.page.get_by_role("button", name="apagar definitivamente", exact=True).click()
        delete_response = delete_info.value.json()
        if any(Path(str(path)).resolve(strict=False) not in set(allowed) for path in delete_response.get("removed", [])):
            raise DeletePlanViolation(f"DELETE retornou path fora do plano: {delete_response!r}")
        if model.exists() or model.is_symlink():
            raise HarnessError("UI delete não removeu o modelo exato")
        _record(ctx, "MODELS-04", status="PASS", observed="delete confirmado somente após plan allowlisted e remoção UI", payload={**plan_evidence, "delete": delete_response})
    except DeletePlanViolation as exc:
        _error(ctx, "MODELS-04", exc)
        raise
    except Exception as exc:
        _error(ctx, "MODELS-04", exc)
    finally:
        if manifest is not None:
            try:
                restore_hardlink_manifest(manifest)
                _json_evidence(ctx, "models-download/recovery-complete.json", {"model": str(model), "restored": True, "sha256": sha256_file(model)})
            except Exception as exc:
                _error(ctx, "MODELS-04", exc, {"manifest": manifest, "recovery": "failed"})
                raise


def run(ctx: RunContext, *, external_hf: bool) -> None:
    """Execute MODELS-01..04 and DOWNLOAD-01..09 using the supplied context."""
    try:
        _models_observe(ctx)
    except Exception as exc:
        for item_id in MODEL_IDS[:3]:
            if ctx.checklist.results[item_id].status == "NÃO VERIFICADO":
                _error(ctx, item_id, exc)

    download_safe = True
    if external_hf:
        try:
            download_safe = _run_downloads(ctx)
        except Exception as exc:
            current = ctx.current_item if ctx.current_item in DOWNLOAD_IDS else "DOWNLOAD-01"
            _error(ctx, current, exc)
            for item_id in DOWNLOAD_IDS:
                if ctx.checklist.results[item_id].status == "NÃO VERIFICADO":
                    _record(
                        ctx,
                        item_id,
                        status="NÃO VERIFICADO",
                        reason=f"NV: não executado após falha de preflight/download ({_error_text(exc)})",
                    )
            download_safe = False
    else:
        _mark_downloads_external_off(ctx)

    # A live downloader is never allowed to overlap the destructive model
    # action.  Failed/cancelled worker observation therefore aborts deletion.
    if not download_safe:
        _record(ctx, "MODELS-04", status="NÃO VERIFICADO", reason="NV: delete não executado enquanto worker de download não foi provado inativo")
        return
    _delete_model_via_ui(ctx)
