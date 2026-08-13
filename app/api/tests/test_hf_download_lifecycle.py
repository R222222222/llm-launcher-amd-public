"""Deterministic transfer/lifecycle coverage; no live Hub access."""
import hashlib
import sys
from pathlib import Path

import pytest

_API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_ROOT))

from api.core import hf  # noqa: E402


class Response:
    def __init__(self, body=b"", status: int = 200, headers=None, **extra_headers: str):
        self.body = body
        self.status = status
        self.headers = dict(headers or {})
        self.headers.update(extra_headers)
        self.closed = False

    def read(self, size=-1):
        if self.closed:
            raise OSError("closed")
        if size < 0:
            size = len(self.body)
        body, self.body = self.body[:size], self.body[size:]
        return body

    def close(self):
        self.closed = True


def _headers(body, revision="a" * 40, oid=None):
    return {
        "X-Repo-Commit": revision,
        "X-Linked-Size": str(len(body)),
        "X-Linked-Etag": oid or hashlib.sha256(body).hexdigest(),
        "ETag": "cdn-is-not-integrity",
    }


def test_optional_origin_headers_absent_still_use_pinned_authority(monkeypatch, tmp_path):
    body = b"authority"
    response = Response(body)
    _download(monkeypatch, tmp_path, body, response)
    assert (tmp_path / "m.gguf").read_bytes() == body


def test_present_origin_header_mismatch_fails_closed(monkeypatch, tmp_path):
    body = b"authority"
    response = Response(body, headers=_headers(body, revision="b" * 40))
    monkeypatch.setattr(hf, "_hub_origin_probe", lambda *a, **k: (_ for _ in ()).throw(hf.IntegrityError("mismatch")))
    with pytest.raises(hf.IntegrityError):
        _download(monkeypatch, tmp_path, body, response, probe=False)


def test_hub_origin_probe_uses_head_without_redirect_and_captures_location(monkeypatch):
    seen = {}

    class ProbeResponse(Response):
        def __init__(self):
            super().__init__(headers={
                "X-Repo-Commit": "a" * 40,
                "X-Linked-Size": "8",
                "X-Linked-Etag": hashlib.sha256(b"authority").hexdigest(),
                "Location": "https://cdn.example/signed",
            })

    def opener(request, timeout):
        seen["method"] = request.get_method()
        seen["headers"] = dict(request.headers)
        return ProbeResponse()

    monkeypatch.setattr(hf, "_urlopen_no_redirect", opener)
    location = hf._hub_origin_probe(
        "https://huggingface.co/o/r/resolve/a" + "a" * 63 + "/m.gguf",
        revision="a" * 40, expected_size=8,
        expected_oid=hashlib.sha256(b"authority").hexdigest(),
    )
    assert seen["method"] == "HEAD"
    assert location == "https://cdn.example/signed"


def _download(monkeypatch, tmp_path, body, response, *, probe=True):
    oid = hashlib.sha256(body).hexdigest()
    if probe:
        monkeypatch.setattr(hf, "_hub_origin_probe", lambda *a, **k: None)
    monkeypatch.setattr(hf.urllib.request, "urlopen", lambda *a, **k: response)
    return hf.download_file_stream(
        "https://huggingface.co/o/r/resolve/" + "a" * 40 + "/m.gguf",
        tmp_path / "m.gguf", expected_size=len(body), expected_oid=oid,
        revision="a" * 40, root=tmp_path,
    )


def test_resume_206_and_identity_header(monkeypatch, tmp_path):
    body = b"abcdefgh"
    (tmp_path / "m.gguf.part").write_bytes(body[:3])
    response = Response(body[3:], status=206, headers=_headers(body))
    response.headers["Content-Range"] = "bytes 3-7/8"
    _download(monkeypatch, tmp_path, body, response)
    assert (tmp_path / "m.gguf").read_bytes() == body


def test_range_200_restarts_without_appending(monkeypatch, tmp_path):
    body = b"new-content"
    (tmp_path / "m.gguf.part").write_bytes(b"old")
    _download(monkeypatch, tmp_path, body, Response(body, headers=_headers(body)))
    assert (tmp_path / "m.gguf").read_bytes() == body


def test_malformed_content_range_does_not_append(monkeypatch, tmp_path):
    body = b"abcdefgh"
    part = tmp_path / "m.gguf.part"
    part.write_bytes(body[:3])
    response = Response(body[3:], status=206, headers=_headers(body))
    response.headers["Content-Range"] = "bytes 1-7/8"
    monkeypatch.setattr(hf, "_hub_origin_probe", lambda *a, **k: None)
    monkeypatch.setattr(hf.urllib.request, "urlopen", lambda *a, **k: response)
    with pytest.raises(hf.IntegrityError):
        hf.download_file_stream("https://example.test/m.gguf", tmp_path / "m.gguf",
                                expected_size=8, expected_oid=hashlib.sha256(body).hexdigest(),
                                revision="a" * 40, root=tmp_path)
    assert part.read_bytes() == body[:3]


def test_same_size_corrupt_part_is_reset(monkeypatch, tmp_path):
    body = b"correct"
    (tmp_path / "m.gguf.part").write_bytes(b"corrupt")
    _download(monkeypatch, tmp_path, body, Response(body, headers=_headers(body)))
    assert (tmp_path / "m.gguf").read_bytes() == body


def test_cdn_etag_is_ignored_but_linked_metadata_is_required(monkeypatch, tmp_path):
    body = b"verified"
    response = Response(body, headers=_headers(body))
    response.headers["ETag"] = '"wrong-cdn-etag"'
    _download(monkeypatch, tmp_path, body, response)
    assert (tmp_path / "m.gguf").read_bytes() == body
