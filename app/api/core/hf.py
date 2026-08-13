"""HuggingFace download utilities — portado de models.py.

Inclui parse de URLs, listagem de arquivos via API, busca por mirrors GGUF,
e download com progresso streamável (yield em vez de print).
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, NewType

from . import path_policy, sampling
from .constants import MODELS_DIR
from .models_repo import SPLIT_RE, gguf_group_key, is_mmproj


def hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "llm-launcher/0.1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_hf_url(url: str) -> tuple[str, str | None, str]:
    """Extract ``(repo_id, filename, requested_revision)``.

    Suporta:
      - https://huggingface.co/{owner}/{repo}
      - https://huggingface.co/{owner}/{repo}?show_file_info={file}
      - https://huggingface.co/{owner}/{repo}/(resolve|blob|tree)/{revision}[/{file.gguf}]
      - refs/heads and refs/tags revisions in either encoded or path form
    """
    url = url.strip().strip('"').strip("'")
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc not in ("huggingface.co", "www.huggingface.co"):
        raise ValueError("A URL precisa ser do huggingface.co")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("Não consegui identificar o repositório na URL")

    owner, repo = parts[0], parts[1]
    repo_id = f"{owner}/{repo}"
    revision = "main"
    filename: str | None = None

    if len(parts) >= 4 and parts[2] in ("resolve", "blob", "tree"):
        ref_parts = urllib.parse.unquote(parts[3]).split("/")
        rest_start = 4
        if ref_parts == ["refs"] and len(parts) >= 6 and parts[4] in ("heads", "tags"):
            ref_parts.extend((parts[4], parts[5]))
            rest_start = 6
        elif ref_parts and ref_parts[0] == "refs" and len(ref_parts) >= 3:
            rest_start = 4
        revision = "/".join(ref_parts)
        rest = "/".join(parts[rest_start:])
        filename = rest or None
    else:
        qs = urllib.parse.parse_qs(parsed.query)
        for key in ("show_file_info", "show-file-info", "filename"):
            if qs.get(key):
                filename = qs[key][0]
                break

    return repo_id, (urllib.parse.unquote(filename) if filename else None), revision


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_OID_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
RepoCommit = NewType("RepoCommit", str)
LfsOid = NewType("LfsOid", str)


class IntegrityError(ValueError):
    """The Hub metadata or a downloaded object contradicts its manifest."""


class DownloadCancelled(Exception):
    """Raised at a cancellation checkpoint, including while reading a stream."""


@dataclass(frozen=True)
class HfFileMetadata:
    path: str
    size: int
    oid: str


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size: int
    oid: str
    skipped: bool = False


# Descriptive names retained for callers that treat the manifest/result as
# public typed values rather than raw plan dictionaries.
DownloadMetadata = HfFileMetadata
DownloadFileResult = DownloadResult


class DownloadControl:
    """Cancellation token plus the currently active urllib response.

    Closing the response is deliberately done without holding the token lock:
    urllib implementations and test doubles are allowed to call back into the
    token while unwinding their read.
    """

    def __init__(self) -> None:
        self._cancel = threading.Event()
        self._response_lock = threading.Lock()
        self._response = None

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()
        with self._response_lock:
            response = self._response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    def checkpoint(self) -> None:
        if self.cancelled:
            raise DownloadCancelled("download cancelado")

    def attach_response(self, response) -> None:
        with self._response_lock:
            self._response = response
        if self.cancelled:
            try:
                response.close()
            except Exception:
                pass

    def detach_response(self, response) -> None:
        with self._response_lock:
            if self._response is response:
                self._response = None


def normalize_commit(value: object) -> RepoCommit | None:
    text = str(value or "").strip().lower().strip('"\'')
    return RepoCommit(text) if _COMMIT_RE.fullmatch(text) else None


def normalize_oid(value: object) -> LfsOid | None:
    text = str(value or "").strip().lower().strip('"\'')
    if text.startswith("sha256:"):
        text = text[7:].strip('"\'')
    return LfsOid(text) if _OID_RE.fullmatch(text) else None


def _requested_revision(value: object) -> str:
    text = str(value or "").strip()
    if not text or "\\" in text or '"' in text or "\x00" in text:
        raise IntegrityError("revision inválida")
    if _OID_RE.fullmatch(text.strip('"\'')):
        raise IntegrityError("revision não pode ser SHA-256 de 64 hex")
    return text


def _response_header(response, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return str(value).strip() if value is not None else None


def _revision_url(repo_id: str, requested_revision: str) -> str:
    return (
        f"https://huggingface.co/api/models/{repo_id}/revision/"
        f"{urllib.parse.quote(requested_revision, safe='')}?blobs=true"
    )


def _metadata_oid(lfs: object) -> str | None:
    if not isinstance(lfs, Mapping):
        return None
    values = [normalize_oid(lfs.get(key)) for key in ("sha256", "oid")]
    values = [value for value in values if value is not None]
    if len(set(values)) > 1:
        raise IntegrityError("metadados LFS têm SHA-256 contraditórios")
    return values[0] if values else None


def _metadata_size(
    entry: Mapping[str, object], lfs: object, *, allow_pointer_size: bool = False,
) -> int:
    raw_entry_value = entry.get("size")
    raw_lfs_value = lfs.get("size") if isinstance(lfs, Mapping) else None
    raw_entry = int(raw_entry_value) if isinstance(raw_entry_value, (int, str)) and str(raw_entry_value).strip() else 0
    raw_lfs = int(raw_lfs_value) if isinstance(raw_lfs_value, (int, str)) and str(raw_lfs_value).strip() else 0
    if raw_entry > 0 and raw_lfs > 0 and raw_entry != raw_lfs and not allow_pointer_size:
        raise IntegrityError(f"tamanhos contraditórios para {entry.get('rfilename') or entry.get('path')}")
    return raw_lfs or raw_entry


def _normalize_file_entry(
    entry: Mapping[str, object], revision: str, *, strict_gguf: bool = True,
    allow_pointer_size: bool = False,
) -> dict:
    path = entry.get("rfilename") or entry.get("path")
    if not isinstance(path, str) or not path:
        raise IntegrityError("sibling sem caminho")
    lfs = entry.get("lfs") or {}
    size = _metadata_size(entry, lfs, allow_pointer_size=allow_pointer_size)
    oid = _metadata_oid(lfs)
    if strict_gguf and path.lower().endswith(".gguf") and (size <= 0 or oid is None):
        raise IntegrityError(f"metadados GGUF incompletos para {path}")
    return {"path": path, "size": size, "oid": oid, "revision": revision}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Return the first response; never follow a signed Hub/CDN location."""

    def http_error_301(self, req, fp, code, msg, headers):
        return fp

    http_error_302 = http_error_301
    http_error_303 = http_error_301
    http_error_307 = http_error_301
    http_error_308 = http_error_301


def _urlopen_no_redirect(request, *, timeout: int):
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def _hub_origin_probe(
    url: str,
    *,
    revision: str,
    expected_size: int,
    expected_oid: str,
) -> str | None:
    """Probe Hub before transfer; Location is observed but never persisted."""
    request = urllib.request.Request(url, headers=hf_headers(), method="HEAD")
    response = None
    try:
        response = _urlopen_no_redirect(request, timeout=30)
        _validate_hub_headers(
            response, revision=revision, expected_size=expected_size,
            expected_oid=expected_oid,
        )
        return _response_header(response, "Location")
    except urllib.error.HTTPError as exc:
        # Some Hub-compatible deployments reject HEAD.  A complete pinned API
        # manifest remains authoritative, so an unavailable optional probe is
        # not fatal; any headers actually returned were already checked above.
        if exc.code not in {405, 501}:
            raise
        return None
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def _fetch_revision_metadata(
    repo_id: str, requested_revision: str,
) -> tuple[list[dict], str]:
    requested_revision = _requested_revision(requested_revision)
    req = urllib.request.Request(_revision_url(repo_id, requested_revision), headers=hf_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise IntegrityError("resposta de revisão do Hub não é um objeto")
    revision = normalize_commit(payload.get("sha"))
    if revision is None:
        raise IntegrityError("Hub não informou commit SHA-1 completo na API de revisão")
    requested_sha = normalize_commit(requested_revision)
    if requested_sha is not None and requested_sha != revision:
        raise IntegrityError("sha da API de revisão diverge do commit solicitado")
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise IntegrityError("API de revisão não informou siblings")
    return [
        _normalize_file_entry(entry, revision)
        for entry in siblings
        if isinstance(entry, Mapping)
    ], revision


def _fetch_tree_metadata(repo_id: str, revision: str) -> list[dict]:
    """Optional recursive detail, always pinned and never header-authoritative."""
    path_policy.validate_repo_id(repo_id)
    qs = urllib.parse.urlencode({"recursive": "true", "expand": "true"})
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo_id}/tree/{revision}?{qs}",
        headers=hf_headers(),
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        entries = json.loads(resp.read().decode("utf-8"))
    if not isinstance(entries, list):
        raise IntegrityError("tree do Hub não é uma lista")
    return [
        _normalize_file_entry(
            entry, revision, strict_gguf=False, allow_pointer_size=True,
        )
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("type") == "file"
    ]


def _cross_check_metadata(primary: list[dict], tree: list[dict]) -> None:
    by_path = {item["path"]: item for item in primary}
    for item in tree:
        base = by_path.get(item["path"])
        if base is None:
            continue
        if item["size"] > 0 and item["size"] != base["size"]:
            raise IntegrityError(f"tree contradiz tamanho de {item['path']}")
        if item["oid"] is not None and item["oid"] != base["oid"]:
            raise IntegrityError(f"tree contradiz OID de {item['path']}")


def resolve_hf_revision(repo_id: str, requested_revision: str = "main") -> str:
    """Resolve any ref through the Hub revision body, not response headers."""
    path_policy.validate_repo_id(repo_id)
    if not isinstance(requested_revision, str) or not requested_revision.strip():
        raise IntegrityError("revision inválida")
    _files, revision = _fetch_revision_metadata(repo_id, requested_revision)
    return revision


def hf_list_with_revision(repo_id: str, requested_revision: str = "main") -> tuple[list[dict], str]:
    """Lista os arquivos do repo. Retorna dicts {path, size}.

    Usa o endpoint `tree?recursive=true` em vez do `siblings` do model card:
    o último retorna `size=0` pra arquivos LFS (limitação conhecida da API),
    e a app precisa de tamanho real pra validar integridade do download.
    """
    path_policy.validate_repo_id(repo_id)
    if not isinstance(requested_revision, str) or not requested_revision.strip():
        raise IntegrityError("revision inválida")
    primary, revision = _fetch_revision_metadata(repo_id, requested_revision)
    # The tree endpoint is detail only.  Its missing X-Repo-Commit header is
    # acceptable because the revision body above is authoritative.
    _cross_check_metadata(primary, _fetch_tree_metadata(repo_id, revision))
    return primary, revision


def hf_list_files(repo_id: str, requested_revision: str = "main") -> list[dict]:
    """Compatibility wrapper returning the resolved revision on each entry."""
    return hf_list_with_revision(repo_id, requested_revision)[0]


def hf_search_gguf(query: str, limit: int = 25) -> list[dict]:
    """Busca repositórios com GGUF batendo `query`. Ordenado por downloads."""
    qs = urllib.parse.urlencode({
        "search": query, "filter": "gguf",
        "sort": "downloads", "direction": -1, "limit": limit,
    })
    req = urllib.request.Request(
        f"https://huggingface.co/api/models?{qs}", headers=hf_headers()
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        results = json.loads(resp.read().decode("utf-8"))
    out = []
    for r in results:
        rid = r.get("id") or r.get("modelId") or ""
        if "/" in rid:
            out.append({
                "repo_id":   rid,
                "downloads": r.get("downloads", 0) or 0,
                "likes":     r.get("likes", 0) or 0,
            })
    return out


# ─── generation_config.json: os samplers que o AUTOR publicou ───────────────
# Fonte machine-readable de temperature/top_p/top_k/repetition_penalty. É o que
# permite configurar um modelo novo sem depender de alguém ler o README.

def _get_json(
    url: str, timeout: int = 15, *, control: DownloadControl | None = None,
) -> dict | None:
    try:
        if control:
            control.checkpoint()
        req = urllib.request.Request(url, headers=hf_headers())
        resp = urllib.request.urlopen(req, timeout=timeout)
        if control:
            control.attach_response(resp)
        try:
            if control:
                control.checkpoint()
            data = json.loads(resp.read().decode("utf-8"))
            if control:
                control.checkpoint()
            return data
        finally:
            if control:
                control.detach_response(resp)
            resp.close()
    except Exception:
        return None


_QUANT_SUFFIX_RE = re.compile(r"[-_.](?:gguf|awq|gptq|mlx|exl2|i1)$", re.IGNORECASE)


def _base_models(
    repo_id: str, *, control: DownloadControl | None = None,
) -> list[str]:
    """Repos onde procurar o generation_config quando o repo baixado não tem.

    Repo de quant (bartowski, lmstudio-community, unsloth) quase nunca carrega
    generation_config.json — mas aponta pro original. Duas trilhas, porque nenhuma
    cobre sozinha:
      1. `base_model` do card — é o que os quantizadores conhecidos preenchem;
      2. o nome sem o sufixo de quant, no mesmo owner — é o caso do autor que
         publica ele mesmo o GGUF (deepreinforce-ai/Ornith-1.0-35B-GGUF →
         deepreinforce-ai/Ornith-1.0-35B) e não preenche base_model nenhum.
    Candidato inexistente só vira um 404 barato, então tentar os dois sai de graça.
    """
    out: list[str] = []
    info = _get_json(f"https://huggingface.co/api/models/{repo_id}", control=control)
    card = (info or {}).get("cardData") or {}
    base = card.get("base_model")
    if isinstance(base, str):
        base = [base]
    if isinstance(base, list):
        out += [b for b in base if isinstance(b, str) and "/" in b]

    owner, _, name = repo_id.partition("/")
    stripped = _QUANT_SUFFIX_RE.sub("", name)
    if stripped != name:
        out.append(f"{owner}/{stripped}")

    seen: set[str] = set()
    return [r for r in out if r != repo_id and not (r in seen or seen.add(r))]


def fetch_generation_config(
    repo_id: str, branch: str = "main", *, control: DownloadControl | None = None,
) -> dict | None:
    """Samplers do autor: {..., "source": "generation_config", "from_repo": repo}.

    Tenta o próprio repo e, se ele não tiver, os base_model do card (um nível —
    quant → original é a cadeia real; mais que isso vira caça ao tesouro).
    None quando ninguém publicou nada de sampling.
    """
    candidates = [(repo_id, branch)] + [
        (b, "main") for b in _base_models(repo_id, control=control)
    ]
    for rid, br in candidates:
        if control:
            control.checkpoint()
        gc = _get_json(
            f"https://huggingface.co/{rid}/resolve/{br}/generation_config.json",
            control=control,
        )
        vals = sampling.from_generation_config(gc) if gc else None
        if vals:
            return {
                **vals,
                "source":    "generation_config",
                "from_repo": rid,
                "raw":       gc,
            }
    return None


def group_gguf_files(repo_files: list[dict]) -> dict:
    """Agrupa os GGUFs do repo em quants + mmprojs.

    Retorna {
      "mmprojs":  [{"path": ..., "size": ...}],
      "quants":   {"<base>": [{"path", "size"}, ...]},
    }
    """
    gguf = [f for f in repo_files if f["path"].lower().endswith(".gguf")]
    mmprojs = [f for f in gguf if is_mmproj(f["path"])]
    quants: dict[str, list[dict]] = {}
    for f in gguf:
        if is_mmproj(f["path"]):
            continue
        key = gguf_group_key(f["path"])
        quants.setdefault(key, []).append(f)
    return {"mmprojs": mmprojs, "quants": quants}


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class IncompleteDownload(Exception):
    """Levantada quando o arquivo termina menor que o esperado.

    Mantém o `.part` em disco pra próxima tentativa retomar — o caller
    decide se reentra ou reporta pro usuário.
    """


_COMMIT_LOCKS: dict[Path, threading.Lock] = {}
_COMMIT_LOCKS_GUARD = threading.Lock()


def _commit_lock(dest: Path) -> threading.Lock:
    with _COMMIT_LOCKS_GUARD:
        return _COMMIT_LOCKS.setdefault(dest, threading.Lock())


def _invalidate_before_commit(dest: Path, root: Path | None) -> None:
    """Invalidate provenance/cache while the destination commit is serialized."""
    try:
        from . import updates
        updates.remove_origin(dest.parent, root=root)
        updates.invalidate_cached_oid(dest)
    except Exception:
        pass


def _remove_part(tmp: Path) -> None:
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass


def _hash_file(path: Path, control: DownloadControl | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if control:
                control.checkpoint()
            chunk = f.read(1 << 20)
            if not chunk:
                break
            if control:
                control.checkpoint()
            h.update(chunk)
    if control:
        control.checkpoint()
    return h.hexdigest()


def _validate_hub_headers(
    response,
    *,
    revision: str | None,
    expected_size: int,
    expected_oid: str | None,
) -> None:
    """Validate only Hub provenance headers; CDN ETag is never consulted."""
    if revision:
        raw_revision = _response_header(response, "X-Repo-Commit")
        got_revision = normalize_commit(raw_revision)
        if raw_revision is not None and got_revision != revision.lower():
            raise IntegrityError("X-Repo-Commit não corresponde à revisão resolvida")
    if expected_size > 0:
        raw_size = _response_header(response, "X-Linked-Size")
        if raw_size is not None:
            try:
                linked_size = int(raw_size)
            except ValueError as exc:
                raise IntegrityError("X-Linked-Size inválido") from exc
            if linked_size != expected_size:
                raise IntegrityError("X-Linked-Size não corresponde ao tamanho esperado")
    if expected_oid:
        raw_oid = _response_header(response, "X-Linked-Etag")
        linked_oid = normalize_oid(raw_oid)
        if raw_oid is not None and linked_oid != expected_oid.lower():
            raise IntegrityError("X-Linked-Etag não corresponde ao OID esperado")


def _content_range(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", value.strip(), re.I)
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    total = -1 if match.group(3) == "*" else int(match.group(3))
    if end < start:
        return None
    return start, end, total


def download_file_stream(
    url: str,
    dest: Path,
    on_progress: Callable[[int, int, float], None] | None = None,
    expected_size: int = 0,
    *,
    root: Path | None = None,
    expected_oid: str | None = None,
    revision: str | None = None,
    control: DownloadControl | None = None,
) -> DownloadResult:
    """Baixa `url` para `dest` com retomada via .part. `on_progress` recebe
    (downloaded_bytes, total_bytes, speed_bps) periodicamente.

    Se `expected_size > 0`, valida que o resultado tem exatamente esse tamanho
    antes de renomear `.part` → dest. Caso contrário levanta IncompleteDownload
    sem renomear, preservando os bytes baixados pra retomada.
    """
    control = control or DownloadControl()
    control.checkpoint()
    if expected_size <= 0:
        raise IntegrityError("tamanho esperado precisa ser positivo")
    expected_oid = normalize_oid(expected_oid)
    if expected_oid is None:
        raise IntegrityError("OID esperado precisa ser um SHA-256 completo")
    if revision is not None and normalize_commit(revision) is None:
        raise IntegrityError("revision precisa ser um commit SHA-1 completo")
    if root is not None:
        path_policy.validate_write_destination(root, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if root is not None:
        path_policy.validate_write_destination(root, dest)
    tmp = dest.with_name(dest.name + ".part")

    resume_pos = tmp.stat().st_size if tmp.exists() else 0
    if resume_pos > expected_size:
        _remove_part(tmp)
        resume_pos = 0
    elif resume_pos == expected_size:
        digest = _hash_file(tmp, control)
        if digest == expected_oid:
            control.checkpoint()
            if root is not None:
                path_policy.validate_write_destination(root, dest)
            with _commit_lock(dest):
                control.checkpoint()
                _invalidate_before_commit(dest, root)
                tmp.replace(dest)
            return DownloadResult(dest, expected_size, digest, skipped=True)
        _remove_part(tmp)
        resume_pos = 0

    headers = hf_headers()
    if resume_pos:
        headers["Range"] = f"bytes={resume_pos}-"
    headers["Accept-Encoding"] = "identity"

    origin_url = url
    try:
        origin_url = urllib.parse.urlunparse((*urllib.parse.urlparse(url)[:2],
            urllib.parse.urlparse(url).path, "", "", ""))
    except Exception:
        origin_url = url
    if urllib.parse.urlparse(origin_url).netloc in {
        "huggingface.co", "www.huggingface.co",
    }:
        _hub_origin_probe(
            origin_url, revision=revision or "", expected_size=expected_size,
            expected_oid=expected_oid,
        )
    # Body retrieval is a separate request.  urllib may follow the signed CDN
    # redirect, but no Authorization header is ever copied cross-host by this
    # explicit request path, and final CDN ETag is ignored.
    req = urllib.request.Request(url, headers=headers)
    resp = None
    try:
        control.checkpoint()
        resp = urllib.request.urlopen(req, timeout=60)
        control.attach_response(resp)
        control.checkpoint()
        raw_status = getattr(resp, "status", None)
        if raw_status is None and hasattr(resp, "getcode"):
            raw_status = resp.getcode()
        status = int(raw_status or 200)
        if status == 416 and resume_pos:
            # A 416 is acceptable only when the local part is already a
            # complete, verified object.  Anything else is reset, never appended.
            if resume_pos == expected_size and _hash_file(tmp, control) == expected_oid:
                with _commit_lock(dest):
                    control.checkpoint()
                    _invalidate_before_commit(dest, root)
                    tmp.replace(dest)
                return DownloadResult(dest, expected_size, expected_oid, skipped=True)
            _remove_part(tmp)
            raise IntegrityError("resposta 416 para part incompleto ou corrompido")

        if resume_pos and status == 206:
            cr = _content_range(_response_header(resp, "Content-Range"))
            if cr is None or cr[0] != resume_pos or cr[2] not in (-1, expected_size):
                raise IntegrityError("Content-Range inválido para retomada")
            file_mode = "ab"
        elif resume_pos and status == 200:
            # A server that ignored Range requires a clean restart.
            file_mode = "wb"
            resume_pos = 0
        elif status == 200:
            file_mode = "wb"
        else:
            raise IntegrityError(f"status HTTP inesperado: {status}")

        _validate_hub_headers(
            resp, revision=revision, expected_size=expected_size,
            expected_oid=expected_oid,
        )
        downloaded = resume_pos
        start = time.time()
        last = 0.0
        if root is not None:
            path_policy.validate_write_destination(root, dest)
        # Hash the retained prefix before opening the response.  This also makes
        # cancellation during a large prefix deterministic and non-destructive.
        digest = hashlib.sha256()
        if resume_pos:
            with open(tmp, "rb") as prefix:
                while True:
                    control.checkpoint()
                    chunk = prefix.read(1 << 20)
                    if not chunk:
                        break
                    control.checkpoint()
                    digest.update(chunk)
        control.checkpoint()
        with open(tmp, file_mode) as f:
            while True:
                control.checkpoint()
                chunk = resp.read(1 << 20)
                control.checkpoint()
                if not chunk:
                    break
                control.checkpoint()
                digest.update(chunk)
                control.checkpoint()
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if on_progress and now - last >= 0.4:
                    last = now
                    speed = (downloaded - resume_pos) / max(now - start, 1e-6)
                    on_progress(downloaded, expected_size, speed)
            control.checkpoint()
            f.flush()
            os.fsync(f.fileno())
        control.checkpoint()

    finally:
        if resp is not None:
            control.detach_response(resp)
            try:
                resp.close()
            except Exception:
                pass

    # A clean EOF is still a truncation; retain it for the next attempt.
    if downloaded != expected_size:
        raise IncompleteDownload(
            f"download truncado: recebeu {downloaded} bytes, esperava {expected_size} "
            f"(faltam {expected_size - downloaded} bytes). Arquivo .part mantido pra retomada."
        )

    control.checkpoint()
    actual_oid = _hash_file(tmp, control)
    if actual_oid != expected_oid:
        _remove_part(tmp)
        raise IntegrityError("sha256 do download não confere")
    control.checkpoint()
    if root is not None:
        path_policy.validate_write_destination(root, dest)
    with _commit_lock(dest):
        control.checkpoint()
        if root is not None:
            path_policy.validate_write_destination(root, dest)
        _invalidate_before_commit(dest, root)
        tmp.replace(dest)
    if on_progress:
        on_progress(downloaded, expected_size, 0.0)  # final tick
    return DownloadResult(dest, downloaded, actual_oid)


def plan_download(
    repo_id: str,
    requested_revision: str,
    rel_paths: list[str],
    subdir: str | None = None,
    base_root: Path | None = None,
    force: bool = False,
    require_metadata: bool = False,
) -> dict:
    """Devolve o plano: arquivos esperados + caminhos de destino + tamanhos.

    base_root é a pasta raiz onde criar o subdir owner/repo. Quando None, cai
    pro MODELS_DIR — só pra manter compat de chamadas antigas; o caminho novo
    sempre passa base_root explícito.

    Resolve `expected_size` via hf_list_files (endpoint tree?recursive=true),
    que devolve o tamanho real do blob LFS — o downloader usa pra validar
    integridade no fim. Falha silenciosa de tamanho => não rename.

    `force` marca os itens pra re-baixar mesmo se já existirem com o tamanho
    certo. É o caso do botão "Atualizar": o detector já comparou sha256 e sabe
    que o conteúdo mudou; sem isso o downloader pularia por tamanho igual e o
    "update" terminaria sem trocar byte nenhum.
    """
    path_policy.validate_repo_id(repo_id)
    if (
        not isinstance(requested_revision, str) or not requested_revision
        or "\\" in requested_revision or '"' in requested_revision
        or "\x00" in requested_revision
    ):
        raise path_policy.MalformedPath("revision inválida")
    safe_subdir = path_policy.validate_subdir(subdir)
    owner, repo = repo_id.split("/", 1)
    root = path_policy.canonical_roots([base_root or MODELS_DIR])[0]
    base_dir = root / owner / repo
    if safe_subdir:
        base_dir = base_dir / Path(*safe_subdir.split("/"))

    sizes: dict[str, int] = {}
    oids: dict[str, str | None] = {}
    resolved_revision = normalize_commit(requested_revision)
    try:
        for f in hf_list_files(repo_id, requested_revision):
            sizes[f["path"]] = int(f.get("size") or 0)
            oids[f["path"]] = f.get("oid")
            resolved_revision = resolved_revision or normalize_commit(f.get("revision"))
    except Exception:
        # Preserve the old planning helper's best-effort behavior for callers
        # that only use it to render a form.  The HTTP admission path below
        # rejects a plan without complete metadata.
        pass

    items = []
    for rel in rel_paths:
        safe_rel = path_policy.validate_relative_gguf(rel)
        dest = base_dir / Path(safe_rel).name
        path_policy.validate_write_destination(root, dest)
        if require_metadata and safe_rel.lower().endswith(".gguf"):
            oid = normalize_oid(oids.get(rel))
            size = sizes.get(rel, 0)
            if size <= 0 or oid is None or resolved_revision is None:
                raise IntegrityError(
                    f"metadados incompletos para GGUF {rel}: revision, size e OID são obrigatórios"
                )
        items.append({
            "rel":           safe_rel,
            "dest":          str(dest),
            "exists":        dest.exists(),
            "size_disk":     dest.stat().st_size if dest.exists() else 0,
            "expected_size": sizes.get(rel, 0),
            "expected_oid":  oids.get(rel),
            "force":         force,
            "url":           (
                f"https://huggingface.co/{repo_id}/resolve/{resolved_revision}/{rel}"
                f"?download=true"
            ),
        })
    return {
        "repo_id":  repo_id,
        "requested_revision": requested_revision,
        "revision": resolved_revision,
        "download_revision": resolved_revision,
        "root":     str(root),
        "base_dir": str(base_dir),
        "items":    items,
    }


def skip_existing(
    actual: int, expected: int, force: bool = False,
    *, local_oid: str | None = None, expected_oid: str | None = None,
) -> bool:
    """O arquivo já em disco dispensa o download?

    Só quando o tamanho bate (ou a API não devolveu tamanho — aí confia no
    disco) E ninguém pediu `force`. Tamanho errado = truncado/desatualizado;
    `force` = o caller já sabe por sha256 que o conteúdo divergiu.
    """
    if force:
        return False
    if expected <= 0:
        return expected_oid is None
    if actual != expected:
        return False
    if expected_oid is None:
        # Compatibility for callers which only have the historical size hint;
        # admission/download paths always provide an OID and use strict mode.
        return True
    return normalize_oid(local_oid) == normalize_oid(expected_oid)


def stream_download(
    plan: dict, *, control: DownloadControl | None = None,
) -> Iterator[dict]:
    """Generator de eventos do download:
        {"type": "file_start", "rel", "dest", "index", "total"}
        {"type": "progress",   "rel", "downloaded", "total", "speed"}
        {"type": "file_done",  "rel", "dest"}
        {"type": "file_skip",  "rel", "dest", "reason"}
        {"type": "sampling",   "found", "source", "from_repo", "values"}
        {"type": "error",      "rel", "message"}
        {"type": "done"}
    """
    control = control or DownloadControl()
    items = plan["items"]
    total = len(items)
    committed: list[dict] = []
    had_failure = False
    for idx, it in enumerate(items, 1):
        control.checkpoint()
        if committed:
            try:
                from . import updates
                updates.remove_origin(Path(plan["base_dir"]), root=Path(plan["root"]))
            except Exception:
                pass
        rel  = it["rel"]
        dest = Path(it["dest"])
        expected = int(it.get("expected_size") or 0)
        expected_oid = normalize_oid(it.get("expected_oid"))
        if expected <= 0 or expected_oid is None or normalize_commit(plan.get("revision")) is None:
            raise IntegrityError(f"metadados incompletos para GGUF {rel}")
        # Revalidate before even stat'ing an existing destination: a symlink
        # could have appeared after plan_download returned.
        path_policy.validate_write_destination(Path(plan["root"]), dest)
        if dest.exists():
            actual = dest.stat().st_size
            local_oid = _hash_file(dest, control) if actual == expected else None
            if skip_existing(
                actual, expected, bool(it.get("force")),
                local_oid=local_oid, expected_oid=expected_oid,
            ):
                yield {"type": "file_skip", "rel": rel, "dest": str(dest),
                       "reason": "already_exists", "size": actual,
                       "oid": local_oid}
                committed.append({"rel": rel, "size": actual, "oid": local_oid})
                continue
            # Não apagamos o arquivo atual aqui: o download vai pro `.part` e só
            # substitui o dest no fim (download_file_stream). Apagar antes trocava
            # um arquivo velho por nenhum arquivo se o download falhasse no meio.
            note = (f"arquivo em disco tem tamanho errado ({actual} != {expected}) — re-baixando"
                    if expected and actual != expected
                    else "re-baixando por cima (conteúdo divergiu do repositório)")
            yield {"type": "file_start", "rel": rel, "dest": str(dest),
                   "index": idx, "total": total, "note": note}
        else:
            yield {"type": "file_start", "rel": rel, "dest": str(dest),
                   "index": idx, "total": total}
        try:
            _progress: list[tuple[int, int, float]] = []
            result = download_file_stream(
                it["url"], dest,
                on_progress=lambda d, t, s: _progress.append((d, t, s)),
                expected_size=expected,
                expected_oid=expected_oid,
                revision=plan["revision"],
                root=Path(plan["root"]),
                control=control,
            )
            for d, tot, speed in _progress:
                yield {"type": "progress", "rel": rel,
                       "downloaded": int(d), "total": int(tot),
                       "speed": float(speed)}
            committed.append({"rel": rel, "size": result.size, "oid": result.oid})
            yield {"type": "file_done", "rel": rel, "dest": str(dest), "oid": result.oid}
        except DownloadCancelled:
            try:
                from . import updates
                updates.remove_origin(Path(plan["base_dir"]), root=Path(plan["root"]))
            except Exception:
                pass
            raise
        except Exception:
            had_failure = True
            try:
                from . import updates
                updates.remove_origin(Path(plan["base_dir"]), root=Path(plan["root"]))
            except Exception:
                pass
            raise

    control.checkpoint()
    # Origin is written only after every requested file is committed and verified.
    try:
        from . import updates
        if committed and not had_failure:
            updates.write_origin(
                Path(plan["base_dir"]), plan["repo_id"],
                plan["requested_revision"], committed,
                requested_revision=plan.get("requested_revision"),
                resolved_revision=plan.get("revision"),
                root=Path(plan["root"]),
            )
    except Exception:
        pass

    # Sidecar de sampling: puxa o generation_config.json do repo (ou do base_model)
    # e grava sampling.json ao lado do .gguf. É o que faz um modelo recém-baixado
    # já subir com os samplers do autor em vez de um preset genérico. Best-effort:
    # rede caindo aqui não pode invalidar um download que terminou.
    try:
        vals = fetch_generation_config(
            plan["repo_id"], plan["requested_revision"],
            control=control,
        )
        if vals:
            control.checkpoint()
            sampling.write_sidecar(
                Path(plan["base_dir"]), vals, root=Path(plan["root"]),
            )
            control.checkpoint()
            yield {
                "type":      "sampling",
                "found":     True,
                "source":    vals["source"],
                "from_repo": vals.get("from_repo"),
                "values":    {k: vals[k] for k in sampling.SAMPLER_KEYS if k in vals},
            }
        else:
            yield {"type": "sampling", "found": False}
    except Exception as e:
        yield {"type": "sampling", "found": False, "error": str(e)}
