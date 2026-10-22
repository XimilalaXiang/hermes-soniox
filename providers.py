"""Soniox TTS and STT providers for Hermes Agent."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import DEFAULT_OUTPUT_FORMAT, TTSProvider

from .client import SonioxClient, SonioxError
from .config import (
    API_KEY_ENV,
    get_api_key,
    language_hints,
    load_section,
    normalize_language,
    provider_section,
)

logger = logging.getLogger(__name__)

PROVIDER_ID = "soniox"
DEFAULT_TTS_MODEL = "tts-rt-v2"
DEFAULT_TTS_LANGUAGE = "zh"
DEFAULT_TTS_VOICE = "Meilin"
DEFAULT_STT_MODEL = "stt-async-v5"
DEFAULT_STT_LANGUAGE = "zh"

TTS_MAX_CHARS = 5000
SPEED_MIN, SPEED_MAX = 0.7, 1.3
CONTEXT_MAX_CHARS = 10000  # Soniox caps context at ~8000 tokens

# Hermes format -> Soniox ``audio_format``. Hermes "ogg" and "opus" both ask for Soniox opus.
_FORMAT_MAP = {"mp3": "mp3", "wav": "wav", "flac": "flac", "opus": "opus", "ogg": "opus"}

_SETUP_ENV = [{"key": API_KEY_ENV, "prompt": "Soniox API key", "url": "https://console.soniox.com"}]

# Voice ids documented by Soniox plus the plugin default; the live list is at GET /v1/tts-models.
_VOICES = [
    {"id": "Meilin", "display": "Meilin"},
    {"id": "Adrian", "display": "Adrian"},
    {"id": "Daniel", "display": "Daniel"},
    {"id": "Maya", "display": "Maya"},
    {"id": "Noah", "display": "Noah"},
    {"id": "Meera", "display": "Meera"},
]


def _clamp_speed(speed: Any) -> float | None:
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed):
        return None
    return round(max(SPEED_MIN, min(SPEED_MAX, float(speed))), 3)


class SonioxTTSProvider(TTSProvider):
    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport

    @property
    def name(self) -> str:
        return PROVIDER_ID

    @property
    def display_name(self) -> str:
        return "Soniox"

    @property
    def voice_compatible(self) -> bool:
        # Hermes converts non-.ogg output to Opus with ffmpeg for voice bubbles.
        return True

    def is_available(self) -> bool:
        return get_api_key() is not None

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": DEFAULT_TTS_MODEL, "display": "Soniox TTS Realtime v2"}]

    def default_model(self) -> str | None:
        return DEFAULT_TTS_MODEL

    def list_voices(self) -> list[dict[str, Any]]:
        return [dict(v) for v in _VOICES]

    def default_voice(self) -> str | None:
        return DEFAULT_TTS_VOICE

    def get_setup_schema(self) -> dict[str, Any]:
        return {"name": "Soniox", "badge": "BYOK", "tag": "Soniox TTS (SONIOX_API_KEY)",
                "env_vars": [dict(e) for e in _SETUP_ENV]}

    def synthesize(self, text: str, output_path: str, *, voice: str | None = None,
                   model: str | None = None, speed: float | None = None,
                   format: str = DEFAULT_OUTPUT_FORMAT, **extra: Any) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Soniox TTS requires non-empty text")
        if len(text) > TTS_MAX_CHARS:
            raise ValueError(f"Soniox TTS text exceeds the {TTS_MAX_CHARS} character limit")
        key = get_api_key()
        if not key:
            raise RuntimeError(f"{API_KEY_ENV} is not set")

        fmt = str(format or DEFAULT_OUTPUT_FORMAT).strip().lower()
        audio_format = _FORMAT_MAP.get(fmt, "mp3")
        language = (normalize_language(provider_section(load_section("tts")).get("language"))
                    or DEFAULT_TTS_LANGUAGE)
        payload: dict[str, Any] = {
            # tts.model is shared across providers; ignore stale non-Soniox values.
            "model": model if isinstance(model, str) and model.startswith("tts-") else DEFAULT_TTS_MODEL,
            "language": language,
            "voice": voice if isinstance(voice, str) and voice.strip() else DEFAULT_TTS_VOICE,
            "audio_format": audio_format,
            "text": text,
        }
        clamped = _clamp_speed(speed)
        if clamped is not None:
            payload["speed"] = clamped

        audio = SonioxClient(key, transport=self._transport).synthesize(payload)
        target = self._target_path(output_path, audio_format, audio)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{target}.part"
        try:
            with open(tmp, "wb") as fh:
                fh.write(audio)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return target

    @staticmethod
    def _target_path(output_path: str, audio_format: str, audio: bytes) -> str:
        if audio_format == "opus":
            ext = ".ogg" if audio.startswith(b"OggS") else ".opus"
        else:
            ext = f".{audio_format}"
        p = Path(output_path)
        return str(p) if p.suffix.lower() == ext else str(p.with_suffix(ext))


class SonioxTranscriptionProvider(TranscriptionProvider):
    def __init__(self, *, transport: httpx.BaseTransport | None = None,
                 poll_timeout: float = 300.0,
                 sleep: Callable[[float], None] | None = None,
                 monotonic: Callable[[], float] | None = None) -> None:
        self._transport = transport
        self._poll_timeout = poll_timeout
        self._clock: dict[str, Any] = {}
        if sleep:
            self._clock["sleep"] = sleep
        if monotonic:
            self._clock["monotonic"] = monotonic

    @property
    def name(self) -> str:
        return PROVIDER_ID

    @property
    def display_name(self) -> str:
        return "Soniox"

    def is_available(self) -> bool:
        return get_api_key() is not None

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": DEFAULT_STT_MODEL, "display": "Soniox STT Async v5"}]

    def default_model(self) -> str | None:
        return DEFAULT_STT_MODEL

    def get_setup_schema(self) -> dict[str, Any]:
        return {"name": "Soniox", "badge": "BYOK", "tag": "Soniox async STT (SONIOX_API_KEY)",
                "env_vars": [dict(e) for e in _SETUP_ENV]}

    def _fail(self, message: str) -> dict[str, Any]:
        return {"success": False, "transcript": "", "provider": PROVIDER_ID, "error": message}

    def transcribe(self, file_path: str, *, model: str | None = None,
                   language: str | None = None, **extra: Any) -> dict[str, Any]:
        try:
            return self._transcribe(file_path, model, language, extra.get("prompt"))
        except SonioxError as exc:
            return self._fail(str(exc))
        except Exception as exc:  # contract: never raise
            logger.debug("Soniox transcription crashed: %s", type(exc).__name__)
            return self._fail(f"Soniox transcription failed ({type(exc).__name__})")

    def _transcribe(self, file_path: str, model: str | None, language: str | None,
                    prompt: Any) -> dict[str, Any]:
        key = get_api_key()
        if not key:
            return self._fail(f"{API_KEY_ENV} is not set")
        if not os.path.isfile(file_path):
            return self._fail("Audio file not found")

        stt_cfg = provider_section(load_section("stt"))
        body: dict[str, Any] = {
            "model": model or stt_cfg.get("model") or DEFAULT_STT_MODEL,
            "language_hints": language_hints(language or stt_cfg.get("language"),
                                             default=DEFAULT_STT_LANGUAGE),
        }
        if isinstance(prompt, str) and prompt.strip():
            body["context"] = {"text": prompt.strip()[:CONTEXT_MAX_CHARS]}

        client = SonioxClient(key, transport=self._transport, poll_timeout=self._poll_timeout,
                              **self._clock)
        file_id = transcription_id = None
        try:
            file_id = client.upload_file(file_path)
            body["file_id"] = file_id
            transcription_id = client.create_transcription(body)
            client.wait_for_completion(transcription_id)
            text = client.get_transcript(transcription_id)
            return {"success": True, "transcript": text, "provider": PROVIDER_ID}
        finally:
            client.best_effort_cleanup(transcription_id, file_id)
