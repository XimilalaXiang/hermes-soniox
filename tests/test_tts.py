from __future__ import annotations

import json

import httpx
import pytest
from conftest import FAKE_KEY

AUDIO = b"ID3\x00fake-audio-bytes\xff\x00"


def _provider(mod, handler, monkeypatch, cfg=None):
    prov = mod.providers.SonioxTTSProvider(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(mod.providers, "load_section", lambda name: (cfg or {}).get(name, {}))
    return prov


def _capture(status=200, content=AUDIO):
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        return httpx.Response(status, content=content)

    return handler, seen


def test_defaults_and_exact_bytes(mod, monkeypatch, tmp_path):
    h, seen = _capture()
    out = tmp_path / "o.mp3"
    path = _provider(mod, h, monkeypatch).synthesize("你好", str(out))
    assert path == str(out) and out.read_bytes() == AUDIO
    req = seen[0]
    assert str(req.url) == "https://tts-rt.soniox.com/tts"
    assert req.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(req.content)
    assert body == {"model": "tts-rt-v2", "language": "zh", "voice": "Meilin",
                    "audio_format": "mp3", "text": "你好"}


def test_config_language_voice_model_speed(mod, monkeypatch, tmp_path):
    h, seen = _capture()
    p = _provider(mod, h, monkeypatch, {"tts": {"soniox": {"language": "en-US"}}})
    p.synthesize("hi", str(tmp_path / "a.wav"), voice="Adrian", model="tts-rt-v2",
                 speed=2.5, format="wav")
    body = json.loads(seen[0].content)
    assert body["language"] == "en" and body["voice"] == "Adrian"
    assert body["audio_format"] == "wav" and body["speed"] == 1.3


@pytest.mark.parametrize("speed,expected", [(0.25, 0.7), (0.9, 0.9), (4.0, 1.3)])
def test_speed_clamped(mod, monkeypatch, tmp_path, speed, expected):
    h, seen = _capture()
    _provider(mod, h, monkeypatch).synthesize("x", str(tmp_path / "a.mp3"), speed=speed)
    assert json.loads(seen[0].content)["speed"] == expected


@pytest.mark.parametrize("speed", [None, float("nan"), "fast", True])
def test_bad_speed_omitted(mod, monkeypatch, tmp_path, speed):
    h, seen = _capture()
    _provider(mod, h, monkeypatch).synthesize("x", str(tmp_path / "a.mp3"), speed=speed)
    assert "speed" not in json.loads(seen[0].content)


def test_non_soniox_model_falls_back(mod, monkeypatch, tmp_path):
    h, seen = _capture()
    _provider(mod, h, monkeypatch).synthesize("x", str(tmp_path / "a.mp3"), model="gpt-4o-mini-tts")
    assert json.loads(seen[0].content)["model"] == "tts-rt-v2"


@pytest.mark.parametrize("fmt", ["mp3", "wav", "flac"])
def test_native_formats(mod, monkeypatch, tmp_path, fmt):
    h, seen = _capture()
    out = tmp_path / f"a.{fmt}"
    assert _provider(mod, h, monkeypatch).synthesize("x", str(out), format=fmt) == str(out)
    assert json.loads(seen[0].content)["audio_format"] == fmt


def test_ogg_ogg_container_keeps_ogg_ext(mod, monkeypatch, tmp_path):
    h, seen = _capture(content=b"OggS\x00data")
    out = _provider(mod, h, monkeypatch).synthesize("x", str(tmp_path / "a.mp3"), format="ogg")
    assert out.endswith(".ogg") and json.loads(seen[0].content)["audio_format"] == "opus"
    assert open(out, "rb").read() == b"OggS\x00data"


def test_opus_non_ogg_uses_opus_ext(mod, monkeypatch, tmp_path):
    h, _ = _capture(content=b"rawopus")
    out = _provider(mod, h, monkeypatch).synthesize("x", str(tmp_path / "a.mp3"), format="opus")
    assert out.endswith(".opus") and open(out, "rb").read() == b"rawopus"


def test_unknown_format_falls_back_to_mp3(mod, monkeypatch, tmp_path):
    h, seen = _capture()
    _provider(mod, h, monkeypatch).synthesize("x", str(tmp_path / "a.mp3"), format="weird")
    assert json.loads(seen[0].content)["audio_format"] == "mp3"


@pytest.mark.parametrize("text", ["", "   \n"])
def test_empty_input(mod, monkeypatch, tmp_path, text):
    h, seen = _capture()
    with pytest.raises(ValueError):
        _provider(mod, h, monkeypatch).synthesize(text, str(tmp_path / "a.mp3"))
    assert not seen


def test_too_long(mod, monkeypatch, tmp_path):
    h, seen = _capture()
    with pytest.raises(ValueError, match="5000"):
        _provider(mod, h, monkeypatch).synthesize("a" * 5001, str(tmp_path / "a.mp3"))
    assert not seen


def test_missing_key(mod, monkeypatch, tmp_path):
    monkeypatch.delenv("SONIOX_API_KEY")
    h, seen = _capture()
    with pytest.raises(RuntimeError, match="SONIOX_API_KEY"):
        _provider(mod, h, monkeypatch).synthesize("x", str(tmp_path / "a.mp3"))
    assert not seen


def test_api_error_safe_and_redacted(mod, monkeypatch, tmp_path):
    body = {"error_type": "invalid_request", "error_message": f"bad voice {FAKE_KEY}",
            "request_id": "r1"}
    h, _ = _capture(400, json.dumps(body).encode())
    with pytest.raises(Exception) as ei:
        _provider(mod, h, monkeypatch).synthesize("x", str(tmp_path / "a.mp3"))
    msg = str(ei.value)
    assert "400" in msg and "invalid_request" in msg and FAKE_KEY not in msg
    assert not (tmp_path / "a.mp3").exists()


def test_empty_audio_body_is_error(mod, monkeypatch, tmp_path):
    h, _ = _capture(200, b"")
    with pytest.raises(Exception, match="empty"):
        _provider(mod, h, monkeypatch).synthesize("x", str(tmp_path / "a.mp3"))
    assert not (tmp_path / "a.mp3").exists()


def test_timeout_is_clear_error(mod, monkeypatch, tmp_path):
    def h(req):
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(Exception, match="timed out") as ei:
        _provider(mod, h, monkeypatch).synthesize("x", str(tmp_path / "a.mp3"))
    assert FAKE_KEY not in str(ei.value)


def test_transport_error_redacted(mod, monkeypatch, tmp_path):
    def h(req):
        raise httpx.ConnectError(f"boom {FAKE_KEY}", request=req)

    with pytest.raises(Exception) as ei:
        _provider(mod, h, monkeypatch).synthesize("x", str(tmp_path / "a.mp3"))
    assert FAKE_KEY not in str(ei.value)


def test_non_ascii_key_error_does_not_echo_key(mod, monkeypatch, tmp_path):
    secret = "kéy-secret"
    monkeypatch.setenv("SONIOX_API_KEY", secret)
    h, _ = _capture()
    with pytest.raises(mod.client.SonioxError) as ei:
        _provider(mod, h, monkeypatch).synthesize("hi", str(tmp_path / "o.mp3"))
    assert secret not in str(ei.value) and "invalid API key characters" in str(ei.value)
