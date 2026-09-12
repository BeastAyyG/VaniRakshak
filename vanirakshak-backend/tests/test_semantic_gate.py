"""The semantic layer must not ship caller audio abroad unless explicitly enabled.

docs/10-quality-and-hardening-plan.md G6. With the layer on, every 4.04 s
window is uploaded to Groq (third-party, US-hosted), which contradicts the
zero-retention claim made under DPDP Act 2023 s.8(7). Decision: off by default,
and disclosed on every compliance surface when on.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from vanirakshak import config as config_module
from vanirakshak.config import SemanticConfig

app_module = importlib.import_module("vanirakshak.server.app")


@pytest.fixture
def semantic(request):
    """Swap settings.semantic (a frozen dataclass field) for the test, then restore."""
    enabled = request.param
    original = config_module.settings.semantic
    object.__setattr__(config_module.settings, "semantic", SemanticConfig(enabled=enabled))
    yield enabled
    object.__setattr__(config_module.settings, "semantic", original)


def _fresh_signal():
    from vanirakshak.semantic.signal import SemanticSignal
    return SemanticSignal()


def test_disabled_by_default():
    assert SemanticConfig().enabled is False or config_module.settings.semantic.enabled is False


@pytest.mark.parametrize("semantic", [False], indirect=True)
def test_groq_key_alone_does_not_enable_upload(semantic, monkeypatch):
    """The key lives in .env for the SACH and IndicConformer prototypes.

    Its mere presence must not start sending caller audio to a third party.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk-present-for-another-prototype")
    sig = _fresh_signal()
    assert sig._ensure_loaded() is False
    assert sig.available is False
    assert "VR_SEMANTIC_ENABLED" in sig.load_error


@pytest.mark.parametrize("semantic", [False], indirect=True)
def test_disabled_layer_degrades_with_full_shape(semantic):
    """Degrading must keep the documented key set (docs/PROTOCOL.md s.3.1)."""
    import numpy as np
    out = _fresh_signal().analyze_window(np.ones(64000, dtype=np.int16))
    assert out["available"] is False
    assert out["semantic_risk"] == 0.0
    assert "VR_SEMANTIC_ENABLED" in out["reason"]


@pytest.mark.parametrize("semantic", [False], indirect=True)
def test_health_claims_zero_retention_when_disabled(semantic):
    models = TestClient(app_module.app).get("/v1/health").json()["models"]
    assert models["zero_retention"] is True
    assert models["third_party_audio_processors"] == []


@pytest.mark.parametrize("semantic", [True], indirect=True)
def test_health_discloses_transfer_when_enabled(semantic):
    """Fails toward disclosure: declared from config, before any call happens."""
    models = TestClient(app_module.app).get("/v1/health").json()["models"]
    assert models["zero_retention"] is False
    assert models["third_party_audio_processors"] == ["groq"]


@pytest.mark.parametrize("semantic", [True], indirect=True)
def test_processor_list_names_the_recipient(semantic):
    assert app_module._third_party_processors() == ["groq"]


@pytest.mark.parametrize("semantic", [False], indirect=True)
def test_processor_list_empty_when_disabled(semantic):
    assert app_module._third_party_processors() == []
