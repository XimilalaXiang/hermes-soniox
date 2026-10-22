"""Hermes Soniox — native BYOK TTS and STT providers (SONIOX_API_KEY)."""

from __future__ import annotations

__all__ = ["register"]


def register(ctx) -> None:
    """Register the Soniox TTS and transcription providers."""
    from .providers import SonioxTranscriptionProvider, SonioxTTSProvider

    ctx.register_tts_provider(SonioxTTSProvider())
    ctx.register_transcription_provider(SonioxTranscriptionProvider())
