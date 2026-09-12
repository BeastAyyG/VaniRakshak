"""Every route must reject an unauthenticated caller once keys are configured.

docs/10-quality-and-hardening-plan.md G5. Until this existed, anyone who could
reach the port could enroll a voiceprint under someone else's ``user_id``,
stream audio, or read the audit chain (docs/09 P0 #1).

The app builds its ``Authenticator`` at import time from ``settings``, so these
tests rebuild it on the module under test rather than re-importing the app.
"""

from __future__ import annotations

import importlib
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

# vanirakshak/server/__init__.py re-exports the FastAPI instance as `app`,
# which shadows the submodule name, so import the module explicitly.
app_module = importlib.import_module("vanirakshak.server.app")
from vanirakshak.server.auth import ApiKey, AuthError, Authenticator, RateLimiter, extract_secret

KEYS = (
    "demo:s3cret:*:*,"                      # all scopes, any user
    "narrow:n4rrow:analyze:alice,"          # analyze only, only user 'alice'
    "noaudit:no4udit"                       # default scopes: stream|analyze|enroll
)
AUTHED = {"Authorization": "Bearer s3cret"}


@pytest.fixture
def client(monkeypatch):
    """Auth ON, with rate limits high enough not to interfere."""
    monkeypatch.setattr(
        app_module,
        "AUTH",
        Authenticator(raw_keys=KEYS, require_auth=False,
                      rate_limits={"analyze": 0, "enroll": 0}),
    )
    return TestClient(app_module.app)


def _wav_bytes(seconds: float = 1.0, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.zeros(int(sr * seconds)) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


# --------------------------------------------------------------------------
# Unauthenticated access
# --------------------------------------------------------------------------

PROTECTED_GETS = ["/v1/enroll", "/v1/audit"]


@pytest.mark.parametrize("path", PROTECTED_GETS)
def test_protected_get_requires_key(client, path):
    assert client.get(path).status_code == 401


def test_enroll_requires_key(client):
    r = client.post("/v1/enroll", data={"user_id": "alice"},
                    files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 401


def test_analyze_requires_key(client):
    r = client.post("/v1/analyze",
                    files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 401


def test_verify_speaker_requires_key(client):
    r = client.post("/v1/verify-speaker",
                    files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 401


def test_sessions_requires_key(client):
    assert client.post("/v1/sessions", json={"sample_rate": 16000}).status_code == 401


def test_health_stays_open(client):
    """Health is an unauthenticated liveness probe on purpose.

    A load balancer cannot hold a credential, and the endpoint exposes only
    backend names and whether an anchor loaded -- no per-caller data.
    """
    assert client.get("/v1/health").status_code == 200


def test_bad_key_rejected(client):
    r = client.get("/v1/audit", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_api_key_header_also_accepted(client):
    assert client.get("/v1/enroll", headers={"X-API-Key": "s3cret"}).status_code == 200


# --------------------------------------------------------------------------
# Scopes
# --------------------------------------------------------------------------

def test_audit_not_in_default_scopes(client):
    """The audit chain is a log of who was scored and when."""
    r = client.get("/v1/audit", headers={"Authorization": "Bearer no4udit"})
    assert r.status_code == 403
    assert "audit" in r.json()["detail"]


def test_wildcard_key_reaches_audit(client):
    assert client.get("/v1/audit", headers=AUTHED).status_code == 200


def test_scope_limited_key_cannot_enroll(client):
    r = client.post("/v1/enroll", data={"user_id": "alice"},
                    files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
                    headers={"Authorization": "Bearer n4rrow"})
    assert r.status_code == 403


# --------------------------------------------------------------------------
# The identity attack: enrolling under someone else's user_id
# --------------------------------------------------------------------------

def test_key_cannot_act_as_another_user(client):
    r = client.post("/v1/analyze",
                    files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
                    data={"claimed_identity": "bob"},
                    headers={"Authorization": "Bearer n4rrow"})
    assert r.status_code == 403
    assert "bob" in r.json()["detail"]


def test_key_may_act_as_its_own_user(client):
    r = client.post("/v1/analyze",
                    files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
                    data={"claimed_identity": "alice"},
                    headers={"Authorization": "Bearer n4rrow"})
    assert r.status_code != 403


# --------------------------------------------------------------------------
# WebSocket
# --------------------------------------------------------------------------

def test_ws_rejects_unauthenticated_handshake(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text('{"sample_rate": 16000}')
        with pytest.raises(Exception):
            ws.receive_json()


def test_ws_accepts_key_in_handshake(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text('{"sample_rate": 16000, "api_key": "s3cret"}')
        ack = ws.receive_json()
        assert ack["type"] == "session_ack"


# --------------------------------------------------------------------------
# Auth disabled -- the demo path must be untouched
# --------------------------------------------------------------------------

def test_disabled_auth_leaves_everything_open(monkeypatch):
    monkeypatch.setattr(app_module, "AUTH", Authenticator(raw_keys="", require_auth=False))
    c = TestClient(app_module.app)
    assert c.get("/v1/audit").status_code == 200
    assert c.get("/v1/enroll").status_code == 200


def test_require_auth_without_keys_refuses_to_start():
    with pytest.raises(RuntimeError, match="refusing to start"):
        Authenticator(raw_keys="", require_auth=True)


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def test_rate_limiter_allows_then_blocks():
    rl = RateLimiter()
    for _ in range(3):
        rl.check("k", "analyze", 3)
    with pytest.raises(AuthError) as e:
        rl.check("k", "analyze", 3)
    assert e.value.status == 429


def test_rate_limiter_is_per_key_and_per_route():
    rl = RateLimiter()
    rl.check("a", "analyze", 1)
    rl.check("b", "analyze", 1)   # different key, own budget
    rl.check("a", "enroll", 1)    # different route, own budget
    with pytest.raises(AuthError):
        rl.check("a", "analyze", 1)


def test_rate_limit_zero_disables():
    rl = RateLimiter()
    for _ in range(50):
        rl.check("k", "analyze", 0)


def test_malformed_key_entries_are_skipped_not_widened():
    a = Authenticator(raw_keys="bad, :nokeyid, keyid:, good:secret", require_auth=False)
    assert list(a.keys) == ["good"]


def test_extract_secret_forms():
    assert extract_secret({"authorization": "Bearer abc"}) == "abc"
    assert extract_secret({"x-api-key": "abc"}) == "abc"
    assert extract_secret({}) is None


def test_default_scopes_exclude_audit():
    a = Authenticator(raw_keys="k:s", require_auth=False)
    assert a.keys["k"].may("enroll") and not a.keys["k"].may("audit")


def test_enroll_rate_limit_enforced(monkeypatch):
    monkeypatch.setattr(
        app_module, "AUTH",
        Authenticator(raw_keys=KEYS, require_auth=False,
                      rate_limits={"analyze": 0, "enroll": 2}),
    )
    c = TestClient(app_module.app)
    codes = [
        c.post("/v1/enroll", data={"user_id": "alice"},
               files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
               headers=AUTHED).status_code
        for _ in range(4)
    ]
    assert 429 in codes, f"rate limit never fired: {codes}"
