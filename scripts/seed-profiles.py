#!/usr/bin/env python3
"""Seed the versioned Phase 1 profiles through the local launcher API."""

from __future__ import annotations

import json
import sys
from http.client import HTTPConnection, HTTPException
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_RESOLVED = REPO_ROOT.resolve()
MANIFEST = REPO_ROOT / "docs" / "profiles" / "seed-profiles.json"


def _post_config(payload: dict) -> object:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    connection = HTTPConnection("127.0.0.1", 8420, timeout=30)
    response = None
    try:
        connection.request("POST", "/api/configs", body=body,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read().decode("utf-8")
        if not 200 <= response.status < 300:
            raise RuntimeError(
                f"POST /api/configs retornou HTTP {response.status}: {raw!r}"
            )
    except (HTTPException, OSError) as exc:
        raise RuntimeError(f"POST /api/configs falhou: {exc}") from exc
    finally:
        if response is not None:
            response.close()
        connection.close()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"POST /api/configs retornou JSON inválido: {raw!r}") from exc


def _get_configs() -> list[dict]:
    connection = HTTPConnection("127.0.0.1", 8420, timeout=30)
    response = None
    try:
        connection.request("GET", "/api/configs")
        response = connection.getresponse()
        raw = response.read().decode("utf-8")
        if not 200 <= response.status < 300:
            raise RuntimeError(
                f"GET /api/configs retornou HTTP {response.status}: {raw!r}"
            )
    except (HTTPException, OSError) as exc:
        raise RuntimeError(f"GET /api/configs falhou: {exc}") from exc
    finally:
        if response is not None:
            response.close()
        connection.close()
    try:
        configs = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GET /api/configs retornou JSON inválido: {raw!r}") from exc
    if not isinstance(configs, list) or not all(isinstance(item, dict) for item in configs):
        raise RuntimeError("GET /api/configs retornou schema inválido")
    return configs


def _verify_persisted(profiles: list[dict]) -> list[dict]:
    configs = _get_configs()
    persisted: list[dict] = []
    for expected in profiles:
        profile_id = expected["id"]
        matches = [item for item in configs if item.get("id") == profile_id]
        if len(matches) != 1:
            raise RuntimeError(
                f"id {profile_id} tem {len(matches)} ocorrências em GET /api/configs"
            )
        actual = matches[0]
        missing = [field for field in expected if field not in actual]
        mismatched = {
            field: {"expected": expected[field], "actual": actual.get(field)}
            for field in expected
            if field in actual and actual[field] != expected[field]
        }
        if missing or mismatched:
            raise RuntimeError(
                f"persistência divergente para {profile_id}: "
                f"missing={missing} mismatched={mismatched}"
            )
        persisted.append(actual)
        print(json.dumps({"id": profile_id, "count": 1, "fields": "match"}, ensure_ascii=False))
    return persisted


def _resolve_repo_gguf(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{field} inválido")
    raw_path = Path(value)
    if raw_path.is_absolute():
        raise RuntimeError(f"{field} deve ser relativo ao repo: {raw_path}")
    resolved = (REPO_ROOT / raw_path).resolve()
    try:
        resolved.relative_to(REPO_ROOT_RESOLVED)
    except ValueError as exc:
        raise RuntimeError(f"{field} resolve fora do repo: {resolved}") from exc
    if ".." in raw_path.parts:
        raise RuntimeError(f"{field} contém traversal: {raw_path}")
    if resolved.suffix.lower() != ".gguf":
        raise RuntimeError(f"{field} precisa terminar em .gguf: {resolved}")
    if not resolved.is_file():
        raise RuntimeError(f"{field} inexistente: {resolved}")
    return resolved


def _load_profiles() -> list[dict]:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"manifest inválido: {MANIFEST}: {exc}") from exc
    if (not isinstance(manifest, dict) or manifest.get("version") != 1
            or not isinstance(manifest.get("profiles"), list)
            or not manifest["profiles"]):
        raise RuntimeError("manifest precisa ter version=1 e profiles como lista")
    profiles = []
    ids: set[str] = set()
    for raw in manifest["profiles"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise RuntimeError("cada perfil precisa ter id")
        profile = dict(raw)
        profile_id = profile["id"]
        if profile_id in ids:
            raise RuntimeError(f"id duplicado no manifest: {profile_id}")
        ids.add(profile_id)
        model_path = _resolve_repo_gguf(profile.get("model"), "modelo GGUF")
        profile["model"] = str(model_path)
        if profile.get("mmproj") is not None:
            mmproj_path = _resolve_repo_gguf(profile["mmproj"], "mmproj GGUF")
            profile["mmproj"] = str(mmproj_path)
        profiles.append(profile)
    return profiles


def main() -> int:
    if len(sys.argv) != 1:
        print(f"uso: {Path(sys.argv[0]).name}", file=sys.stderr)
        return 2
    try:
        profiles = _load_profiles()
        for profile in profiles:
            result = _post_config(profile)
            if not isinstance(result, dict) or result.get("ok") is not True:
                raise RuntimeError(f"POST /api/configs retornou schema inesperado: {result!r}")
            saved = result.get("config")
            if not isinstance(saved, dict) or saved.get("id") != profile["id"]:
                raise RuntimeError(f"config salva sem o id esperado: {result!r}")
            print(json.dumps({"id": profile["id"], "ok": True}, ensure_ascii=False))
        _verify_persisted(profiles)
    except (KeyError, TypeError, RuntimeError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
