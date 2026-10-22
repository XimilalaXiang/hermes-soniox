from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "hermes_soniox_testpkg"
FAKE_KEY = "sk-test-SECRET-1234567890"

_agent = Path(os.environ.get("HERMES_AGENT_PATH", Path.home() / ".hermes" / "hermes-agent"))
if (_agent / "agent" / "tts_provider.py").exists() and str(_agent) not in sys.path:
    sys.path.insert(0, str(_agent))


def _load_package():
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    spec = importlib.util.spec_from_file_location(
        PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def pkg():
    return _load_package()


@pytest.fixture
def mod(pkg):
    """Submodules by short name: mod.client, mod.providers, mod.config."""
    import importlib

    class NS:
        client = importlib.import_module(f"{PACKAGE}.client")
        providers = importlib.import_module(f"{PACKAGE}.providers")
        config = importlib.import_module(f"{PACKAGE}.config")

    return NS


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("SONIOX_API_KEY", FAKE_KEY)
    monkeypatch.setenv("HERMES_HOME", "/nonexistent-hermes-home")


@pytest.fixture
def make_client(mod):
    def factory(handler, **kw):
        kw.setdefault("sleep", lambda s: None)
        return mod.client.SonioxClient(
            api_key=FAKE_KEY, transport=httpx.MockTransport(handler), **kw
        )

    return factory
