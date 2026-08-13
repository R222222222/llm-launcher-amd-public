"""Small, shared filesystem boundary for model reads and downloads.

The API receives paths from a browser, so a path is not trusted merely because
it was produced by one of our own settings files.  This module deliberately
keeps policy separate from the model/download implementations and exposes
typed failures that the HTTP layer can map to 400/403/404.
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Iterable


class PathPolicyError(ValueError):
    """Base class for a rejected path."""


class MalformedPath(PathPolicyError):
    """The value is syntactically invalid for the operation."""


class MissingPath(PathPolicyError):
    """A required root or existing file/directory does not exist."""


class OutsideRoot(PathPolicyError):
    """The resolved path is outside every allowed root."""


class SymlinkEscape(OutsideRoot):
    """A symlink resolves outside the configured boundary."""


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SEGMENT_RE = re.compile(r'^[^\\/:"\x00-\x1f\x7f]+$')


def _as_path(value: str | Path, *, field: str) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        if not value or _CONTROL_RE.search(value):
            raise MalformedPath(f"{field} inválido")
        path = Path(value)
    else:
        raise MalformedPath(f"{field} inválido")
    if "\x00" in str(path) or '"' in str(path) or _CONTROL_RE.search(str(path)):
        raise MalformedPath(f"{field} inválido")
    return path


def canonical_roots(
    roots: Iterable[str | Path], *, strict: bool = True,
) -> tuple[Path, ...]:
    """Return existing, canonical directory roots, preserving order."""
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in roots:
        try:
            path = _as_path(raw, field="root")
        except PathPolicyError:
            if not strict:
                continue
            raise
        try:
            root = path.resolve(strict=True)
        except FileNotFoundError as exc:
            if not strict:
                continue
            raise MissingPath(f"root não existe: {path}") from exc
        except OSError as exc:
            if not strict:
                continue
            raise MissingPath(f"root não pôde ser resolvida: {path}") from exc
        if not root.is_dir():
            if not strict:
                continue
            raise MalformedPath(f"root não é diretório: {path}")
        if root not in seen:
            seen.add(root)
            out.append(root)
    return tuple(out)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _root_for(path: Path, roots: tuple[Path, ...]) -> Path:
    for root in roots:
        if _inside(path, root):
            return root
    raise OutsideRoot(f"caminho fora das raízes permitidas: {path}")


def _validate_gguf_suffix(path: Path, *, field: str) -> None:
    if path.suffix.lower() != ".gguf":
        raise MalformedPath(f"{field} precisa terminar em .gguf")


def validate_existing_gguf(
    value: str | Path,
    roots: Iterable[str | Path],
    *,
    optional: bool = False,
) -> Path | None:
    """Validate an absolute, regular, existing GGUF under a canonical root."""
    if value is None and optional:
        return None
    path = _as_path(value, field="GGUF")
    if not path.is_absolute():
        raise MalformedPath("GGUF precisa ser absoluto")
    _validate_gguf_suffix(path, field="GGUF")
    root_tuple = canonical_roots(roots)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise MissingPath(f"GGUF não existe: {path}") from exc
    except OSError as exc:
        raise MissingPath(f"GGUF não pôde ser resolvido: {path}") from exc
    if not path.is_file() or not resolved.is_file():
        raise MissingPath(f"GGUF não é arquivo regular: {path}")
    root = _root_for(resolved, root_tuple)
    # A symlink that lands inside is safe for reads; one that lands outside was
    # rejected by _root_for.  Returning the canonical file prevents later code
    # from accidentally following a different spelling of the same path.
    return resolved


def validate_existing_json_config(
    value: str | Path,
    roots: Iterable[str | Path],
    *,
    optional: bool = False,
) -> Path | None:
    """Validate the canonical ``config/mcp/servers.json`` stdio config.

    The MCP config has one repository-relative authority; configured model
    roots are never valid locations for this file.
    """
    if value is None and optional:
        return None
    # MCP launcher configs have one canonical location.  The roots argument is
    # retained for source compatibility, but model roots are never authority
    # for this file.
    from . import mcp_config
    try:
        return mcp_config.validate(value)[0]
    except mcp_config.McpConfigError as exc:
        if "não existe" in str(exc):
            raise MissingPath(str(exc)) from exc
        raise MalformedPath(str(exc)) from exc


def validate_model_pair(
    model: str | Path,
    mmproj: str | Path | None,
    roots: Iterable[str | Path],
) -> tuple[Path, Path | None]:
    model_path = validate_existing_gguf(model, roots)
    mm_path = validate_existing_gguf(mmproj, roots, optional=True) if mmproj else None
    assert model_path is not None
    return model_path, mm_path


def validate_repo_id(repo_id: str) -> str:
    """Validate exactly ``owner/repo`` with safe, single path segments."""
    if not isinstance(repo_id, str) or not repo_id or repo_id.count("/") != 1:
        raise MalformedPath("repo_id precisa ser exatamente owner/repo")
    owner, repo = repo_id.split("/")
    if not owner or not repo or owner in {".", ".."} or repo in {".", ".."}:
        raise MalformedPath("repo_id precisa ser exatamente owner/repo")
    if not _SEGMENT_RE.fullmatch(owner) or not _SEGMENT_RE.fullmatch(repo):
        raise MalformedPath("repo_id contém segmento inválido")
    return repo_id


def validate_subdir(subdir: str | None) -> str | None:
    """Validate a nested relative POSIX destination directory."""
    if subdir is None:
        return None
    if not isinstance(subdir, str) or not subdir:
        raise MalformedPath("subdir vazio ou inválido")
    if "\\" in subdir or ":" in subdir or _CONTROL_RE.search(subdir):
        raise MalformedPath("subdir precisa usar caminho POSIX seguro")
    if subdir.startswith("/") or subdir.endswith("/") or "//" in subdir:
        raise MalformedPath("subdir precisa ser relativo e sem segmentos vazios")
    pure = PurePosixPath(subdir)
    if pure.is_absolute() or not pure.parts:
        raise MalformedPath("subdir precisa ser relativo")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise MalformedPath("subdir contém segmento inválido")
    if any(not _SEGMENT_RE.fullmatch(part) for part in pure.parts):
        raise MalformedPath("subdir contém segmento inválido")
    return "/".join(pure.parts)


def validate_relative_gguf(value: str) -> str:
    """Validate a remote relative GGUF path without allowing traversal."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise MalformedPath("rel_path inválido")
    if ":" in value or _CONTROL_RE.search(value) or value.startswith("/"):
        raise MalformedPath("rel_path precisa ser POSIX relativo")
    parts = value.split("/")
    if any(not p or p in {".", ".."} for p in parts):
        raise MalformedPath("rel_path contém traversal")
    if any(not _SEGMENT_RE.fullmatch(p) for p in parts):
        raise MalformedPath("rel_path contém segmento inválido")
    if Path(parts[-1]).suffix.lower() != ".gguf":
        raise MalformedPath("rel_path precisa terminar em .gguf")
    return "/".join(parts)


def _check_no_symlink_components(root: Path, path: Path) -> None:
    """Reject existing symlinks from root through path (including path)."""
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise OutsideRoot(f"destino fora da raiz: {path}") from exc
    if any(part in {".", ".."} for part in rel.parts):
        raise MalformedPath(f"caminho não pode conter . ou ..: {path}")
    current = root
    for part in rel.parts:
        current = current / part
        try:
            if current.is_symlink():
                raise SymlinkEscape(f"symlink no destino: {current}")
        except OSError as exc:
            raise MalformedPath(f"não foi possível validar destino: {current}") from exc


def validate_write_destination(
    root: str | Path,
    destination: str | Path,
    *,
    suffix: str = ".gguf",
) -> tuple[Path, Path]:
    """Validate destination and its adjacent ``.part`` before a write.

    Nonexistent final components are allowed (downloads create them), but all
    existing components must be real directories/files, never symlinks.  Callers
    intentionally call this again immediately before open and replace.
    """
    root_path = canonical_roots([root])[0]
    dest = _as_path(destination, field="destino")
    if not dest.is_absolute():
        raise MalformedPath("destino precisa ser absoluto")
    if suffix and dest.suffix.lower() != suffix.lower():
        raise MalformedPath(f"destino precisa terminar em {suffix}")
    # Do not resolve destination: a not-yet-created parent must be checked
    # component-by-component, and an existing symlink must be rejected.
    lexical = Path(*dest.parts)
    if not _inside(lexical, root_path):
        raise OutsideRoot(f"destino fora da raiz: {dest}")
    _check_no_symlink_components(root_path, lexical.parent)
    _check_no_symlink_components(root_path, lexical)
    part = dest.with_name(dest.name + ".part")
    _check_no_symlink_components(root_path, part)
    return dest, part


def validate_write_sidecar(
    root: str | Path,
    parent: str | Path,
    name: str,
) -> tuple[Path, Path]:
    """Validate a JSON sidecar destination and its adjacent ``.part``.

    Parent directories may not exist yet, but every existing component must be
    inside the canonical root and free of symlinks.  Callers repeat this check
    after mkdir and immediately before open/replace.
    """
    root_path = canonical_roots([root])[0]
    parent_path = _as_path(parent, field="sidecar parent")
    if not parent_path.is_absolute():
        raise MalformedPath("sidecar parent precisa ser absoluto")
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise MalformedPath("sidecar name inválido")
    if (
        Path(name).name != name or not _SEGMENT_RE.fullmatch(name)
        or Path(name).suffix.lower() != ".json"
    ):
        raise MalformedPath("sidecar name inválido")
    destination = parent_path / name
    if not _inside(parent_path, root_path):
        raise OutsideRoot(f"sidecar parent fora da raiz: {parent_path}")
    _check_no_symlink_components(root_path, parent_path)
    _check_no_symlink_components(root_path, destination)
    part = destination.with_name(destination.name + ".part")
    _check_no_symlink_components(root_path, part)
    return destination, part


def validate_existing_sidecar(root: str | Path, sidecar: str | Path) -> Path:
    """Validate an existing sidecar without following a symlink boundary."""
    root_path = canonical_roots([root])[0]
    path = _as_path(sidecar, field="sidecar")
    if not path.is_absolute():
        raise MalformedPath("sidecar precisa ser absoluto")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise MissingPath(f"sidecar não existe: {path}") from exc
    if not path.is_file() or not resolved.is_file():
        raise MissingPath(f"sidecar não é arquivo regular: {path}")
    _check_no_symlink_components(root_path, path)
    return resolved


def validate_delete_targets(
    model: str | Path,
    roots: Iterable[str | Path],
    targets: Iterable[str | Path],
) -> tuple[Path, Path, list[Path]]:
    """Validate a model deletion plan and return (model, root, targets)."""
    root_tuple = canonical_roots(roots)
    model_path = validate_existing_gguf(model, root_tuple)
    assert model_path is not None
    root = _root_for(model_path, root_tuple)
    validated: list[Path] = []
    seen: set[Path] = set()
    for raw in targets:
        path = _as_path(raw, field="delete target")
        if not path.is_absolute():
            raise MalformedPath("delete target precisa ser absoluto")
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            continue
        if not resolved.is_file() or not _inside(resolved, root):
            raise OutsideRoot(f"delete target fora da raiz: {path}")
        # Existing symlink targets are never unlinked through this policy.
        _check_no_symlink_components(root, path)
        if resolved not in seen:
            seen.add(resolved)
            validated.append(path)
    return model_path, root, validated


# Descriptive aliases used by callers/tests that prefer verb-oriented names.
validate_root = canonical_roots
canonical_existing_roots = canonical_roots
validate_gguf = validate_existing_gguf
validate_model_path = validate_existing_gguf
validate_mmproj = validate_existing_gguf
validate_mmproj_path = validate_existing_gguf
validate_repo = validate_repo_id
validate_subdir_path = validate_subdir
validate_remote_rel_path = validate_relative_gguf
validate_download_destination = validate_write_destination
validate_write_path = validate_write_destination
validate_sidecar = validate_write_sidecar
