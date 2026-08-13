"""Testes da lógica pura do downloader do HuggingFace."""
import json
import sys
from pathlib import Path

import pytest

_API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_ROOT))

from api.core import hf  # noqa: E402


# ─── skip_existing ────────────────────────────────────────────────────────────

def test_skip_when_size_matches():
    assert hf.skip_existing(actual=100, expected=100) is True


def test_no_skip_when_size_differs():
    # Truncado ou desatualizado: re-baixa.
    assert hf.skip_existing(actual=50, expected=100) is False


def test_skip_when_remote_size_unknown():
    # API não devolveu tamanho — confia no que está em disco.
    assert hf.skip_existing(actual=100, expected=0) is True


def test_force_beats_matching_size():
    # Caso do botão "Atualizar": mesmo tamanho, sha256 diferente. Sem isto o
    # update terminava "concluído" sem trocar byte nenhum.
    assert hf.skip_existing(actual=100, expected=100, force=True) is False


def test_force_beats_unknown_size():
    assert hf.skip_existing(actual=100, expected=0, force=True) is False


# ─── canonical requested revision parsing ─────────────────────────────────────

def test_parse_pinned_resolve_url_returns_full_sha_as_revision():
    sha = "c287502cd9e278dac8eed805c112cce5d0081e0b"
    assert hf.parse_hf_url(
        f"https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/{sha}/tinygemma3-Q8_0.gguf"
    ) == ("ggml-org/tinygemma3-GGUF", "tinygemma3-Q8_0.gguf", sha)


def test_parse_refs_heads_url_preserves_canonical_requested_ref():
    assert hf.parse_hf_url(
        "https://huggingface.co/owner/repo/blob/refs/heads/main/model.gguf"
    ) == ("owner/repo", "model.gguf", "refs/heads/main")


def test_parse_plain_repository_url_explicitly_defaults_listing_ref_to_main():
    assert hf.parse_hf_url("https://huggingface.co/owner/repo") == (
        "owner/repo", None, "main",
    )


def test_commit_and_oid_validators_are_distinct():
    commit = "c287502cd9e278dac8eed805c112cce5d0081e0b"
    oid = "7566ae7219c93ea2ecc692a931ee122d30c55261d0e2c3347acb8b939d2e9abd"
    assert hf.normalize_commit(commit) == commit
    assert hf.normalize_oid(oid) == oid
    assert hf.normalize_commit(oid) is None
    assert hf.normalize_oid(commit) is None
    assert hf.normalize_oid(f'sha256:"{oid}"') == oid


class _JsonResponse:
    def __init__(self, payload, headers=None):
        self.payload = payload
        self.headers = headers or {}

    def read(self, _size=-1):
        return json.dumps(self.payload).encode()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _revision_payload(sha: str, *, oid: str | None = "a" * 64, size: int = 8):
    lfs = {"size": size, "sha256": oid} if oid is not None else {}
    return {"sha": sha, "siblings": [{"rfilename": "model.gguf", "lfs": lfs}]}


def test_revision_api_body_authorizes_main_and_full_sha(monkeypatch):
    sha = "c287502cd9e278dac8eed805c112cce5d0081e0b"
    oid = "7566ae7219c93ea2ecc692a931ee122d30c55261d0e2c3347acb8b939d2e9abd"
    seen: list[str] = []

    def urlopen(request, timeout=30):
        seen.append(request.full_url)
        if "/tree/" in request.full_url:
            return _JsonResponse([{
                "type": "file", "path": "model.gguf", "size": 8,
                "lfs": {"size": 8, "oid": oid},
            }])
        return _JsonResponse(_revision_payload(sha, oid=oid))

    monkeypatch.setattr(hf.urllib.request, "urlopen", urlopen)
    files, resolved = hf.hf_list_with_revision("owner/repo", "main")
    assert resolved == sha and files[0]["size"] == 8 and files[0]["oid"] == oid
    files, resolved = hf.hf_list_with_revision("owner/repo", sha)
    assert resolved == sha and files[0]["oid"] == oid
    assert "/revision/main?blobs=true" in seen[0]
    assert f"/revision/{sha}?blobs=true" in seen[2]


def test_revision_api_rejects_malformed_or_mismatched_sha(monkeypatch):
    monkeypatch.setattr(
        hf.urllib.request, "urlopen",
        lambda *args, **kwargs: _JsonResponse(_revision_payload("not-a-sha")),
    )
    with pytest.raises(hf.IntegrityError, match="SHA-1"):
        hf.hf_list_with_revision("owner/repo", "main")

    requested = "b" * 40
    monkeypatch.setattr(
        hf.urllib.request, "urlopen",
        lambda *args, **kwargs: _JsonResponse(_revision_payload("a" * 40)),
    )
    with pytest.raises(hf.IntegrityError, match="diverge"):
        hf.hf_list_with_revision("owner/repo", requested)


def test_revision_api_requires_gguf_lfs_metadata(monkeypatch):
    payload = {"sha": "a" * 40, "siblings": [{"rfilename": "model.gguf", "size": 8}]}
    monkeypatch.setattr(hf.urllib.request, "urlopen", lambda *args, **kwargs: _JsonResponse(payload))
    with pytest.raises(hf.IntegrityError, match="incompletos"):
        hf.hf_list_with_revision("owner/repo", "main")


def test_tree_without_repo_commit_header_succeeds_but_contradiction_fails(monkeypatch):
    sha = "a" * 40
    primary = _revision_payload(sha)
    tree = [{"type": "file", "path": "model.gguf", "size": 8,
             "lfs": {"size": 8, "oid": "a" * 64}}]
    responses = iter([_JsonResponse(primary), _JsonResponse(tree)])
    monkeypatch.setattr(hf.urllib.request, "urlopen", lambda *args, **kwargs: next(responses))
    files, resolved = hf.hf_list_with_revision("owner/repo", "main")
    assert resolved == sha and files[0]["oid"] == "a" * 64

    bad_tree = [{"type": "file", "path": "model.gguf", "size": 9,
                 "lfs": {"size": 9, "oid": "a" * 64}}]
    responses = iter([_JsonResponse(primary), _JsonResponse(bad_tree)])
    monkeypatch.setattr(hf.urllib.request, "urlopen", lambda *args, **kwargs: next(responses))
    with pytest.raises(hf.IntegrityError, match="contradiz tamanho"):
        hf.hf_list_with_revision("owner/repo", "main")
