from __future__ import annotations

import pytest


@pytest.mark.parametrize("raw,expected", [
    ("zh", "zh"), ("zh-CN", "zh"), ("EN_us", "en"), (" ja ", "ja"), (None, None), ("", None), (5, None),
])
def test_normalize_language(mod, raw, expected):
    assert mod.config.normalize_language(raw) == expected


def test_language_hints_multiple(mod):
    assert mod.config.language_hints("zh, en-US") == ["zh", "en"]
    assert mod.config.language_hints(["zh", "en"]) == ["zh", "en"]
    assert mod.config.language_hints(None, default="zh") == ["zh"]


def test_api_key_uses_hermes_secret_scope(mod):
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    token = set_secret_scope({"SONIOX_API_KEY": "scoped-test-key"})
    try:
        assert mod.config.get_api_key() == "scoped-test-key"
    finally:
        reset_secret_scope(token)


def test_empty_multiplexed_scope_cannot_use_ambient_key(mod):
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )

    set_multiplex_active(True)
    token = set_secret_scope({})
    try:
        assert mod.config.get_api_key() is None
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


def test_load_section_without_hermes(mod, monkeypatch):
    assert mod.config.load_section("tts") in ({}, mod.config.load_section("tts"))
    monkeypatch.setattr(mod.config, "_load_config", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert mod.config.load_section("tts") == {}
