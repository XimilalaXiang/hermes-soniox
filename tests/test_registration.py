from __future__ import annotations

from pathlib import Path

from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider

ROOT = Path(__file__).resolve().parents[1]


class Ctx:
    def __init__(self):
        self.tts, self.stt = [], []

    def register_tts_provider(self, p):
        self.tts.append(p)

    def register_transcription_provider(self, p):
        self.stt.append(p)


def test_registers_both(pkg):
    ctx = Ctx()
    pkg.register(ctx)
    assert len(ctx.tts) == 1 and len(ctx.stt) == 1
    assert isinstance(ctx.tts[0], TTSProvider) and ctx.tts[0].name == "soniox"
    assert isinstance(ctx.stt[0], TranscriptionProvider) and ctx.stt[0].name == "soniox"


def test_manifest():
    text = (ROOT / "plugin.yaml").read_text()
    for needle in ("name: hermes-soniox", "version: 0.1.0", 'requires_hermes: ">=0.20.0"',
                   "SONIOX_API_KEY", "license: MIT", "kind: standalone",
                   "https://github.com/XimilalaXiang/hermes-soniox"):
        assert needle in text


def test_availability_and_metadata(mod, monkeypatch):
    p = mod.providers.SonioxTTSProvider()
    s = mod.providers.SonioxTranscriptionProvider()
    assert p.is_available() and s.is_available()
    monkeypatch.delenv("SONIOX_API_KEY")
    assert not p.is_available() and not s.is_available()
    assert p.display_name == s.display_name == "Soniox"
    assert p.default_model() == "tts-rt-v2" and p.default_voice() == "Meilin"
    assert s.default_model() == "stt-async-v5"
    assert any(v["id"] == "Meilin" for v in p.list_voices())
    assert s.list_models()[0]["id"] == "stt-async-v5"
    assert p.voice_compatible is True
    assert p.get_setup_schema()["env_vars"][0]["key"] == "SONIOX_API_KEY"
