from __future__ import annotations

import json
import logging

import httpx
import pytest
from conftest import FAKE_KEY

TID = "11111111-1111-1111-1111-111111111111"
FID = "22222222-2222-2222-2222-222222222222"
BASE = "https://api.soniox.com"


class Fake:
    """Deterministic Soniox async STT server."""

    def __init__(self, statuses=("queued", "processing", "completed"), text="你好世界",
                 transcript=None, fail=None, status_body=None):
        self.statuses = list(statuses)
        self.text = text
        self.transcript = transcript
        self.fail = fail or {}
        self.calls = []
        self.requests = []
        self.status_body = status_body

    def __call__(self, req: httpx.Request):
        key = f"{req.method} {req.url.path}"
        self.calls.append(key)
        self.requests.append(req)
        assert req.headers["authorization"] == f"Bearer {FAKE_KEY}"
        if key in self.fail:
            return self.fail[key]
        if key == "POST /v1/files":
            assert b'name="file"' in req.content and b"AUDIO-BYTES" in req.content
            return httpx.Response(201, json={"id": FID, "filename": "a.ogg", "size": 11})
        if key == "POST /v1/transcriptions":
            return httpx.Response(201, json={"id": TID, "status": "queued"})
        if key == f"GET /v1/transcriptions/{TID}":
            st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            body = self.status_body or {"id": TID, "status": st,
                                        "error_type": "bad_audio" if st == "error" else None,
                                        "error_message": "cannot decode" if st == "error" else None}
            return httpx.Response(200, json=body)
        if key == f"GET /v1/transcriptions/{TID}/transcript":
            return httpx.Response(200, json=self.transcript or {"id": TID, "text": self.text, "tokens": []})
        if key in (f"DELETE /v1/transcriptions/{TID}", f"DELETE /v1/files/{FID}"):
            return httpx.Response(204)
        return httpx.Response(404, json={"error_type": "not_found"})


@pytest.fixture
def audio(tmp_path):
    p = tmp_path / "a.ogg"
    p.write_bytes(b"AUDIO-BYTES")
    return str(p)


def _stt(mod, fake, monkeypatch, cfg=None, **kw):
    clock = {"t": 0.0}
    kw.setdefault("sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    kw.setdefault("monotonic", lambda: clock["t"])
    monkeypatch.setattr(mod.providers, "load_section", lambda name: (cfg or {}).get(name, {}))
    return mod.providers.SonioxTranscriptionProvider(transport=httpx.MockTransport(fake), **kw)


def _create_body(fake):
    r = next(r for r in fake.requests if r.method == "POST" and r.url.path == "/v1/transcriptions")
    return json.loads(r.content)


def test_success_lifecycle_and_cleanup(mod, monkeypatch, audio):
    f = Fake()
    out = _stt(mod, f, monkeypatch).transcribe(audio)
    assert out == {"success": True, "transcript": "你好世界", "provider": "soniox"}
    assert f.calls[:2] == ["POST /v1/files", "POST /v1/transcriptions"]
    assert f.calls[-2:] == [f"DELETE /v1/transcriptions/{TID}", f"DELETE /v1/files/{FID}"]
    assert f.calls.count(f"GET /v1/transcriptions/{TID}") == 3
    assert _create_body(f) == {"model": "stt-async-v5", "file_id": FID, "language_hints": ["zh"]}


def test_model_language_prompt(mod, monkeypatch, audio):
    f = Fake()
    out = _stt(mod, f, monkeypatch).transcribe(audio, model="stt-async-v5", language="en-US",
                                               prompt="Soniox, Hermes", ignored=1)
    assert out["success"]
    body = _create_body(f)
    assert body["language_hints"] == ["en"]
    assert body["context"] == {"text": "Soniox, Hermes"}


def test_config_fallbacks(mod, monkeypatch, audio):
    f = Fake()
    cfg = {"stt": {"soniox": {"model": "stt-async-v5", "language": "ja"}}}
    _stt(mod, f, monkeypatch, cfg).transcribe(audio)
    assert _create_body(f)["language_hints"] == ["ja"]


def test_missing_key_never_raises(mod, monkeypatch, audio):
    monkeypatch.delenv("SONIOX_API_KEY")
    f = Fake()
    out = _stt(mod, f, monkeypatch).transcribe(audio)
    assert out["success"] is False and out["transcript"] == "" and out["provider"] == "soniox"
    assert "SONIOX_API_KEY" in out["error"] and f.calls == []


def test_failed_terminal_status(mod, monkeypatch, audio):
    f = Fake(statuses=("processing", "error"))
    out = _stt(mod, f, monkeypatch).transcribe(audio)
    assert out["success"] is False and out["transcript"] == ""
    assert "bad_audio" in out["error"] and "cannot decode" in out["error"]
    assert f.calls[-2:] == [f"DELETE /v1/transcriptions/{TID}", f"DELETE /v1/files/{FID}"]
    assert not any(c.endswith("/transcript") for c in f.calls)


def test_poll_timeout_cleans_up(mod, monkeypatch, audio):
    f = Fake(statuses=("processing",))
    out = _stt(mod, f, monkeypatch, poll_timeout=20).transcribe(audio)
    assert out["success"] is False and "timed out" in out["error"]
    assert f.calls[-2:] == [f"DELETE /v1/transcriptions/{TID}", f"DELETE /v1/files/{FID}"]
    assert 2 < f.calls.count(f"GET /v1/transcriptions/{TID}") < 15  # backoff, bounded


def test_unknown_status_keeps_polling_until_timeout(mod, monkeypatch, audio):
    f = Fake(status_body={"id": TID, "status": "downloading"})
    out = _stt(mod, f, monkeypatch, poll_timeout=20).transcribe(audio)
    assert out["success"] is False and "timed out" in out["error"]
    assert f.calls.count(f"GET /v1/transcriptions/{TID}") > 2
    assert f"DELETE /v1/files/{FID}" in f.calls


@pytest.mark.parametrize("resp", [
    httpx.Response(201, json={"nope": 1}),
    httpx.Response(201, content=b"<html>"),
])
def test_malformed_upload(mod, monkeypatch, audio, resp):
    f = Fake(fail={"POST /v1/files": resp})
    out = _stt(mod, f, monkeypatch).transcribe(audio)
    assert out["success"] is False and out["transcript"] == ""
    assert not any(c.startswith("DELETE") for c in f.calls)  # nothing was created


def test_malformed_create_still_deletes_file(mod, monkeypatch, audio):
    f = Fake(fail={"POST /v1/transcriptions": httpx.Response(201, json={})})
    out = _stt(mod, f, monkeypatch).transcribe(audio)
    assert out["success"] is False
    assert f.calls[-1] == f"DELETE /v1/files/{FID}"
    assert not any(c.startswith("DELETE /v1/transcriptions") for c in f.calls)


@pytest.mark.parametrize("tr", [{"id": TID}, {"id": TID, "text": 5}, "not json"])
def test_malformed_transcript(mod, monkeypatch, audio, tr):
    resp = (httpx.Response(200, content=b"not json") if tr == "not json"
            else httpx.Response(200, json=tr))
    f = Fake(fail={f"GET /v1/transcriptions/{TID}/transcript": resp})
    out = _stt(mod, f, monkeypatch).transcribe(audio)
    assert out["success"] is False and "transcript" in out["error"].lower()
    assert f.calls[-2:] == [f"DELETE /v1/transcriptions/{TID}", f"DELETE /v1/files/{FID}"]


def test_token_fallback_when_text_absent(mod, monkeypatch, audio):
    f = Fake(transcript={"id": TID, "tokens": [{"text": "你"}, {"text": "好"}]})
    out = _stt(mod, f, monkeypatch).transcribe(audio)
    assert out["success"] and out["transcript"] == "你好"


def test_cleanup_errors_do_not_mask_success(mod, monkeypatch, audio, caplog):
    f = Fake(fail={f"DELETE /v1/transcriptions/{TID}": httpx.Response(409, json={}),
                   f"DELETE /v1/files/{FID}": httpx.Response(500, json={})})
    with caplog.at_level(logging.DEBUG):
        out = _stt(mod, f, monkeypatch).transcribe(audio)
    assert out["success"] and out["transcript"] == "你好世界"
    assert f.calls[-1] == f"DELETE /v1/files/{FID}"  # second delete still attempted
    assert FAKE_KEY not in caplog.text


def test_cleanup_exception_does_not_mask_failure(mod, monkeypatch, audio):
    class Boom(Fake):
        def __call__(self, req):
            if req.method == "DELETE":
                raise httpx.ConnectError("net down", request=req)
            return super().__call__(req)

    f = Boom(statuses=("error",))
    out = _stt(mod, f, monkeypatch).transcribe(audio)
    assert out["success"] is False and "bad_audio" in out["error"]


def test_api_error_redacts_key(mod, monkeypatch, audio):
    f = Fake(fail={"POST /v1/files": httpx.Response(
        401, json={"error_type": "unauthenticated", "error_message": f"bad {FAKE_KEY}"})})
    out = _stt(mod, f, monkeypatch).transcribe(audio)
    assert out["success"] is False and "401" in out["error"] and FAKE_KEY not in out["error"]


def test_missing_local_file(mod, monkeypatch, tmp_path):
    f = Fake()
    out = _stt(mod, f, monkeypatch).transcribe(str(tmp_path / "nope.ogg"))
    assert out["success"] is False and f.calls == []


def test_unexpected_exception_never_raises(mod, monkeypatch, audio):
    f = Fake()
    p = _stt(mod, f, monkeypatch)
    monkeypatch.setattr(mod.providers, "SonioxClient", lambda **kw: 1 / 0)
    out = p.transcribe(audio)
    assert out["success"] is False and out["provider"] == "soniox"


def test_unicode_filename_upload(mod, monkeypatch, tmp_path):
    audio = tmp_path / "录音 café.wav"
    audio.write_bytes(b"AUDIO-BYTES")
    f = Fake()
    out = _stt(mod, f, monkeypatch).transcribe(str(audio))
    assert out["success"] is True, out
