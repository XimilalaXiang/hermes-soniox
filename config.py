"""Config and language helpers. Reads standard Hermes config; never reads or stores secrets."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

API_KEY_ENV = "SONIOX_API_KEY"


def get_api_key() -> str | None:
    """The Soniox key from the environment only (never from config files)."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def _load_config() -> dict[str, Any]:
    from hermes_cli.config import load_config

    cfg = load_config()
    return cfg if isinstance(cfg, dict) else {}


def load_section(name: str) -> dict[str, Any]:
    """Top-level Hermes config section (``tts`` / ``stt``), ``{}`` on any failure."""
    try:
        section = _load_config().get(name)
    except Exception:  # Hermes config unavailable (tests, import-time probes)
        logger.debug("Hermes config unavailable for section %r", name)
        return {}
    return section if isinstance(section, dict) else {}


def provider_section(section: dict[str, Any], provider: str = "soniox") -> dict[str, Any]:
    sub = section.get(provider)
    return sub if isinstance(sub, dict) else {}


def normalize_language(value: Any) -> str | None:
    """``zh-CN`` / ``EN_us`` -> ``zh`` / ``en`` (Soniox takes bare ISO language codes)."""
    if not isinstance(value, str):
        return None
    code = value.strip().replace("_", "-").split("-")[0].lower()
    return code or None


def language_hints(value: str | list[str] | None, default: str | None = None) -> list[str]:
    """Normalized, de-duplicated hint list from a string (comma separated) or list."""
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = []
    hints: list[str] = []
    for item in items:
        code = normalize_language(item)
        if code and code not in hints:
            hints.append(code)
    if not hints and default:
        hints.append(default)
    return hints
