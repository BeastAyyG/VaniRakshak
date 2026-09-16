"""Tests for G5 Authentication and Authorization (Product P0).

Verifies:
  1. Unauthenticated request to every route returns 401 Unauthorized.
  2. Invalid credentials return 401 Unauthorized.
  3. Privilege separation: POST /v1/enroll and GET /v1/audit return 403 Forbidden
     when accessed with a standard client key.
  4. Both 'Authorization: Bearer <key>' and 'X-API-Key: <key>' headers are accepted.
  5. WebSocket stream rejects unauthenticated/invalid connections with code 1008.
  6. Rate limiting returns 429 Too Many Requests when limits are exceeded.
"""

from __future__ import annotations

import io
import json
import wave
from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vanirakshak.config import settings
from vanirakshak.server.app import app
from vanirakshak.server.auth import RATE_LIMITER


def _dummy_wav(duration_s: float = 1.0, sr: int = 16000) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = (np.zeros(int(sr * duration_s), dtype=np.int16)).tobytes()
        w.writeframes(frames)
    return bio.getvalue()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    RATE_LIMITER.reset()
    yield
    RATE_LIMITER.reset()


def test_unauthenticated_requests_return_401():
    """Criterion 1: An unauthenticated request to every route returns 401."""
    client = TestClient(app)
    wav_bytes = _dummy_wav(1.0)

    # 1. GET /v1/health
    r = client.get("/v1/health")
    assert r.status_code == 401, f"GET /v1/health gave {r.status_code}"

    # 2. GET /v1/enroll
    r = client.get("/v1/enroll")
    assert r.status_code == 401, f"GET /v1/enroll gave {r.status_code}"

    # 3. POST /v1/enroll
    r = client.post(
        "/v1/enroll",
        data={"user_id": "test_user"},
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert r.status_code == 401, f"POST /v1/enroll gave {r.status_code}"

    # 4. POST /v1/analyze
    r = client.post(
        "/v1/analyze",
        data={"caller_id": "+91-1234567890"},
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert r.status_code == 401, f"POST /v1/analyze gave {r.status_code}"

    # 5. POST /v1/verify-speaker
    r = client.post(
        "/v1/verify-speaker",
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert r.status_code == 401, f"POST /v1/verify-speaker gave {r.status_code}"

    # 6. POST /v1/sessions
    r = client.post(
        "/v1/sessions",
        json={"caller_id": "+91-1234567890"},
    )
    assert r.status_code == 401, f"POST /v1/sessions gave {r.status_code}"

    # 7. POST /v1/sessions/{sid}/challenge
    r = client.post(
        "/v1/sessions/nonexistent/challenge",
        json={"response_text": "1 2 3", "response_latency_ms": 500},
    )
    assert r.status_code == 401, f"POST /v1/sessions/challenge gave {r.status_code}"

    # 8. GET /v1/audit
    r = client.get("/v1/audit")
    assert r.status_code == 401, f"GET /v1/audit gave {r.status_code}"


def test_invalid_token_returns_401():
    """Invalid token returns 401 Unauthorized."""
    client = TestClient(app)
    bad_headers = {"Authorization": "Bearer bad-token-xyz"}
    r = client.get("/v1/health", headers=bad_headers)
    assert r.status_code == 401


def test_client_key_privilege_separation():
    """Client key can access standard endpoints, but gets 403 Forbidden on admin endpoints."""
    client = TestClient(app)
    client_headers = {"Authorization": f"Bearer {settings.auth.api_key}"}
    wav_bytes = _dummy_wav(1.0)

    # Standard endpoint allowed
    r = client.get("/v1/health", headers=client_headers)
    assert r.status_code == 200

    # Admin endpoints forbidden with 403
    r = client.get("/v1/audit", headers=client_headers)
    assert r.status_code == 403, f"Expected 403 on /v1/audit with client key, got {r.status_code}"

    r = client.post(
        "/v1/enroll",
        headers=client_headers,
        data={"user_id": "attacker_spoof"},
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert r.status_code == 403, f"Expected 403 on /v1/enroll with client key, got {r.status_code}"


def test_admin_key_access():
    """Admin key can access all endpoints including /v1/audit and /v1/enroll."""
    client = TestClient(app)
    admin_headers = {"Authorization": f"Bearer {settings.auth.admin_key}"}

    r = client.get("/v1/health", headers=admin_headers)
    assert r.status_code == 200

    r = client.get("/v1/audit", headers=admin_headers)
    assert r.status_code == 200
    assert "chain_valid" in r.json()


def test_x_api_key_header_support():
    """Both Authorization: Bearer and X-API-Key headers are supported."""
    client = TestClient(app)
    r = client.get("/v1/health", headers={"X-API-Key": settings.auth.api_key})
    assert r.status_code == 200

    r = client.get("/v1/audit", headers={"X-API-Key": settings.auth.admin_key})
    assert r.status_code == 200


def test_websocket_stream_auth():
    """WebSocket stream requires valid credentials in handshake or query param."""
    client = TestClient(app)

    # 1. Without auth token -> closes with code 1008
    with client.websocket_connect("/ws/stream") as ws:
        handshake_no_auth = {
            "session_id": "no-auth-check",
            "sample_rate": 16000,
        }
        ws.send_text(json.dumps(handshake_no_auth))
        # After handshake without auth, server closes socket
        with pytest.raises(Exception):
            ws.receive_json()

    # 2. With valid auth_token in handshake -> succeeds and receives session_ack
    with client.websocket_connect("/ws/stream") as ws:
        handshake_with_auth = {
            "session_id": "auth-check",
            "sample_rate": 16000,
            "auth_token": settings.auth.api_key,
        }
        ws.send_text(json.dumps(handshake_with_auth))
        ack = ws.receive_json()
        assert ack.get("type") == "session_ack"
        assert ack.get("session_id") == "auth-check"


def test_rate_limiting():
    """Rate limit enforces 429 Too Many Requests when request limit is reached."""
    client = TestClient(app)
    admin_headers = {"Authorization": f"Bearer {settings.auth.admin_key}"}
    wav_bytes = _dummy_wav(0.2)  # Short audio

    # Max enroll requests is 10 by default
    limit = settings.auth.rate_limit_enroll
    # Trigger `limit` requests
    for i in range(limit):
        client.post(
            "/v1/enroll",
            headers=admin_headers,
            data={"user_id": f"user_{i}"},
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )

    # The (limit + 1)th request must be rejected with 429
    r_exceeded = client.post(
        "/v1/enroll",
        headers=admin_headers,
        data={"user_id": "user_overflow"},
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert r_exceeded.status_code == 429
    assert "Retry-After" in r_exceeded.headers
