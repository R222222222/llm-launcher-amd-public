"""Listagem e classificação de modelos GGUF na pasta do LM Studio.

Funções puras — não imprimem nada, não chamam exit, não interagem com TTY.
Foi extraído do bloco no topo de models.py (collect_models, find_mmproj,
make_alias, is_thinking_model, is_mtp_model, _SPLIT_RE).
"""
import re
from pathlib import Path

from . import gguf, path_policy, sampling
from .constants import MODELS_DIR, THINKING_KEYWORDS

SPLIT_RE = re.compile(
    r"^(?P<base>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)


def is_primary_part(f: Path) -> bool:
    """True se for o arquivo "principal" (sem split, ou parte 00001 do split)."""
    if "mmproj" in f.name.lower():
        return False
    m = SPLIT_RE.match(f.name)
    return not m or m.group("part") == "00001"


def gguf_group_key(name: str) -> str:
    """Conjunto a que o arquivo pertence: o quant, sem o sufixo -00001-of-00005."""
    base = Path(name).name
    m = SPLIT_RE.match(base)
    if m:
        return m.group("base")
    return base[:-5] if base.lower().endswith(".gguf") else base


def is_mmproj(name: str) -> bool:
    return "mmproj" in Path(name).name.lower()


# Nome do repo do mmproj deve compartilhar com o arquivo do modelo ao menos
# este número de tokens (split em não-alfanumérico, case-insensitive). Nos
# fixtures de produção o repo DavidAU compartilha 10 tokens com o 27B MTP
# (Ornith: 2; 27B não-MTP: 9) — o limiar evita anexar mmproj de visão a
# modelo de texto ou à variante errada do mesmo modelo.
_MIN_MMPROJ_NAME_OVERLAP = 10


def _name_tokens(name: str) -> set[str]:
    # "gguf" (extensão do modelo e sufixo de repo tipo "…-MTP-GGUF") é ruído:
    # poluiria a sobreposição com 1 token para todo candidato.
    return {token.lower() for token in re.split(r"[^0-9A-Za-z]+", name) if token and token.lower() != "gguf"}


def find_mmproj(
    folder: Path,
    *,
    include_parent: bool = True,
    roots: list[Path] | tuple[Path, ...] | None = None,
    model_name: str | None = None,
) -> Path | None:
    """Acha o mmproj associado ao modelo (mesma pasta ou pasta-pai).

    Se a busca direta falhar, faz uma varredura recursiva limitada a
    *mmproj*.gguf e prefere o candidato cujo diretório-pai compartilha mais
    tokens com o nome do arquivo do modelo. Sem isso, um mmproj aninhado
    (ex.: production-models/DavidAU/…/mmproj-F16.gguf, 2 níveis abaixo da raiz)
    seria anexado a qualquer modelo da raiz, inclusive variantes erradas.
    """
    if not folder.exists():
        return None
    def _candidate(f: Path) -> bool:
        if '"' in f.name:
            return False
        if not (f.is_file() and "mmproj" in f.name.lower() and f.suffix.lower() == ".gguf"):
            return False
        if f.is_symlink():
            try:
                resolved = f.resolve(strict=True)
            except OSError:
                return False
            allowed = (
                any(path_policy._inside(resolved, root.resolve()) for root in roots)
                if roots is not None
                else path_policy._inside(resolved, f.parent.resolve())
            )
            if not allowed:
                return False
        return True

    for f in folder.iterdir():
        if _candidate(f):
            return f
    if include_parent:
        parent = folder.parent
        if parent != folder and parent.exists():
            try:
                for f in parent.iterdir():
                    if _candidate(f):
                        return f
            except OSError:
                pass

    # Fallback recursivo limitado a *mmproj*.gguf, com preferência por nome.
    model_tokens = _name_tokens(model_name or folder.name)
    best: Path | None = None
    best_score = 0
    try:
        for f in folder.rglob("*mmproj*.gguf"):
            if not _candidate(f):
                continue
            score = len(_name_tokens(f.parent.name) & model_tokens)
            if score > best_score:
                best, best_score = f, score
    except OSError:
        pass
    if best is not None and best_score >= _MIN_MMPROJ_NAME_OVERLAP:
        return best
    return None


def make_alias(model_path: Path) -> str:
    m = SPLIT_RE.match(model_path.name)
    if m:
        return m.group("base")
    return model_path.stem


def is_thinking_model(model_path: Path) -> bool:
    """Decide pelo chat template do GGUF; keywords no nome são só o fallback.

    A lista de keywords era a detecção principal e errava calado em todo modelo
    com nome fora do padrão (ornith, glm, minimax…): o modelo pensava, o app o
    tratava como não-thinking, escondia o reasoning-budget e servia o preset de
    sampling errado. Só cai no nome quando o GGUF não pôde ser lido.
    """
    meta = gguf.read_meta(model_path)
    if meta is not None:
        return bool(meta.get("has_thinking_template"))
    name = model_path.name.lower()
    return any(kw in name for kw in THINKING_KEYWORDS)


_MTP_PATTERN = re.compile(r"(?:^|[-_.])mtp(?:[-_.]|$)", re.IGNORECASE)


def is_mtp_model(model_path: Path) -> bool:
    """Detecta tag MTP no nome ou em qualquer pasta ancestral."""
    parts = [model_path.name, *(p.name for p in model_path.parents)]
    return any(_MTP_PATTERN.search(part) for part in parts)


def model_disk_size_mib(model_path: Path) -> int:
    """Tamanho total em MiB, somando partes split."""
    total = 0
    m = SPLIT_RE.match(model_path.name)
    if m:
        key = m.group("base")
        for f in sorted(model_path.parent.glob("*.gguf")):
            if not is_mmproj(f.name) and gguf_group_key(f.name) == key:
                try:
                    total += f.stat().st_size
                except Exception:
                    pass
    else:
        try:
            total = model_path.stat().st_size
        except Exception:
            pass
    return total // (1024 * 1024)


def collect_models(models_dir: Path | list[Path] = MODELS_DIR) -> list[Path]:
    """Lista todos os .gguf primários (sem mmproj, sem partes 00002+).

    Aceita um único Path (compat) ou uma lista — varre todas as pastas, dedup
    por caminho resolvido e devolve em ordem alfabética estável.
    """
    roots = [models_dir] if isinstance(models_dir, Path) else list(models_dir)
    found: dict[Path, Path] = {}
    canonical: list[Path] = []
    for raw_root in roots:
        try:
            root = raw_root.resolve(strict=True)
        except OSError:
            continue
        if not root.is_dir():
            continue
        canonical.append(root)
        for f in root.rglob("*.gguf"):
            try:
                if '"' in f.name or '"' in f.relative_to(root).as_posix():
                    continue
            except ValueError:
                continue
            if not is_primary_part(f):
                continue
            try:
                key = f.resolve(strict=True)
            except OSError:
                continue
            if not path_policy._inside(key, root) or not key.is_file():
                # In particular, do not expose a symlink which points outside
                # the configured scan root.
                continue
            found.setdefault(key, f)
    return sorted(found.values())


def plan_delete_model(
    model_path: Path,
    roots: list[Path] | tuple[Path, ...] | None = None,
) -> list[Path]:
    """Lista todos os arquivos que seriam removidos (não remove nada).

    Inclui o(s) .gguf do quant, mmproj associado e arquivos .part de download
    interrompido. Sempre dedup, preserva ordem, só lista o que existe.
    """
    folder = model_path.parent
    canonical = path_policy.canonical_roots(roots) if roots is not None else None
    if canonical is not None:
        model_path = path_policy.validate_existing_gguf(model_path, canonical)  # type: ignore[assignment]
        assert model_path is not None
        root = path_policy._root_for(model_path, canonical)
        folder = model_path.parent
    to_remove: list[Path] = []
    m = SPLIT_RE.match(model_path.name)
    if m:
        key = m.group("base")
        for f in sorted(folder.glob("*.gguf")):
            if not is_mmproj(f.name) and gguf_group_key(f.name) == key:
                to_remove.append(f)
    else:
        to_remove.append(model_path)

    mm = find_mmproj(folder, include_parent=False, roots=canonical, model_name=model_path.name)
    if mm:
        to_remove.append(mm)

    for f in list(to_remove):
        part = f.with_name(f.name + ".part")
        if part.exists():
            to_remove.append(part)

    seen: set[Path] = set()
    existing = [f for f in to_remove if f.exists() and not (f in seen or seen.add(f))]
    if canonical is not None:
        # Validate every split/mmproj/.part target before the caller can unlink
        # any of them.  A symlink target is rejected even when it resolves inside.
        _, _, existing = path_policy.validate_delete_targets(model_path, canonical, existing)
    return existing


def rmdir_if_empty(path: Path, stop_at: Path) -> None:
    """Sobe removendo diretórios vazios até (sem incluir) stop_at."""
    try:
        stop_at = stop_at.resolve(strict=True)
        path = path.resolve(strict=False)
        if not path.is_relative_to(stop_at):
            return
        while path != stop_at and path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            path = path.parent
    except Exception:
        pass


def delete_model(
    model_path: Path,
    models_dir: Path = MODELS_DIR,
    *,
    roots: list[Path] | tuple[Path, ...] | None = None,
) -> dict:
    """Apaga os arquivos listados por plan_delete_model + diretórios vazios.

    Retorna {removed: [paths], errors: [{path, error}], total_bytes}.
    """
    targets = plan_delete_model(model_path, roots=roots)
    removed: list[str] = []
    errors: list[dict] = []
    total = 0
    for f in targets:
        try:
            total += f.stat().st_size
        except Exception:
            pass
        try:
            f.unlink()
            removed.append(str(f))
        except Exception as e:
            errors.append({"path": str(f), "error": str(e)})
    stop_at = (roots[0] if roots else models_dir)
    rmdir_if_empty(model_path.parent, stop_at)
    return {"removed": removed, "errors": errors, "total_bytes": total}


def describe_model(model_path: Path, models_dir: Path | list[Path] = MODELS_DIR) -> dict:
    """Dict serializável com infos básicas — alimenta a lista no frontend.

    `models_dir` pode ser uma raiz só ou várias — usamos a primeira que contém
    o arquivo pra montar o caminho relativo. Cai pro path absoluto se nenhuma
    raiz for ancestral.
    """
    mmproj = find_mmproj(model_path.parent, roots=models_dir if isinstance(models_dir, list) else None, model_name=model_path.name)
    roots = [models_dir] if isinstance(models_dir, Path) else list(models_dir)
    rel = str(model_path)
    for root in roots:
        try:
            rel = model_path.relative_to(root).as_posix()
            break
        except ValueError:
            continue
    return {
        "path":           str(model_path),
        "alias":          make_alias(model_path),
        "relative_path":  rel,
        "filename":       model_path.name,
        "mmproj":         str(mmproj) if mmproj else None,
        "size_mib":       model_disk_size_mib(model_path),
        "is_thinking":    is_thinking_model(model_path),
        "is_mtp":         is_mtp_model(model_path),
        "sampling":       sampling.resolve(model_path),
    }
