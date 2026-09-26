"""Tests for DPDP Act 2023 Sec 12 Biometric Erasure: DELETE /v1/enroll/{user_id}."""

from __future__ import annotations

import importlib
import io
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient

app_module = importlib.import_module("vanirakshak.server.app")
from vanirakshak.server.auth import Authenticator

KEYS = (
    "admin:adm1n:*:*,"                       # all scopes, any user
    "alice_key:s3cret:enroll:alice,"         # enroll scope, user alice only
    "bob_key:b0bsec:enroll:bob,"             # enroll scope, user bob only
    "readonly:r3ad:analyze:*"                # analyze only
)
AUTH_ADMIN = {"Authorization": "Bearer adm1n"}
AUTH_ALICE = {"Authorization": "Bearer s3cret"}
AUTH_BOB = {"Authorization": "Bearer b0bsec"}
AUTH_RO = {"Authorization": "Bearer r3ad"}


@pytest.fixture
def client(monkeypatch):
    """Auth configured with test keys, bypass QA for quick audio tests."""
    monkeypatch.setattr(
        app_module,
        "AUTH",
        Authenticator(raw_keys=KEYS, require_auth=False, rate_limits={"analyze": 0, "enroll": 0}),
    )
    monkeypatch.setenv("VR_ENROLL_BYPASS_QA", "1")
    return TestClient(app_module.app)


def _wav_bytes(seconds: float = 1.0, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.zeros(int(sr * seconds)) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def test_delete_enrollment_unauthenticated_rejected(client):
    """Calling DELETE /v1/enroll/{id} without auth header returns 401."""
    res = client.delete("/v1/enroll/alice")
    assert res.status_code == 401


def test_delete_enrollment_scope_check(client):
    """Key without enroll scope returns 403."""
    res = client.delete("/v1/enroll/alice", headers=AUTH_RO)
    assert res.status_code == 403


def test_delete_enrollment_user_isolation(client):
    """Bob's key cannot delete Alice's voiceprint."""
    res = client.delete("/v1/enroll/alice", headers=AUTH_BOB)
    assert res.status_code == 403


def test_delete_enrollment_not_found(client):
    """Deleting a non-enrolled user returns 404."""
    # Ensure user is not enrolled
    app_module.PIPELINE.unenroll("nonexistent_user")
    res = client.delete("/v1/enroll/nonexistent_user", headers=AUTH_ADMIN)
    assert res.status_code == 404
    assert "not found" in res.json()["detail"]


def test_enroll_then_delete_success(client):
    """Enroll a user, verify enrolled, then erase via DELETE /v1/enroll/{user_id}."""
    user = "alice_dpdp_test"

    # 1. Enroll user
    enroll_res = client.post(
        "/v1/enroll",
        data={"user_id": user},
        files={"audio": ("enroll.wav", _wav_bytes(1.0), "audio/wav")},
        headers=AUTH_ADMIN,
    )
    assert enroll_res.status_code == 200
    assert app_module.PIPELINE.is_enrolled(user)

    # 2. Delete enrollment using user-scoped key
    del_res = client.delete(f"/v1/enroll/{user}", headers=AUTH_ADMIN)
    assert del_res.status_code == 200
    data = del_res.json()
    assert data["status"] == "deleted"
    assert data["user_id"] == user
    assert "receipt_hash" in data
    assert len(data["receipt_hash"]) == 64

    # 3. Verify user is no longer enrolled
    assert not app_module.PIPELINE.is_enrolled(user)

    # 4. Verify audit log contains biometric erasure receipt and chain is valid
    audit_res = client.get("/v1/audit", headers=AUTH_ADMIN)
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["chain_valid"] is True
    erasure_events = [
        ev for ev in audit_data["events"]
        if ev["claimed_identity"] == user and ev.get("tier") == "ERASED"
    ]
    assert len(erasure_events) >= 1
    assert erasure_events[-1]["features"].get("biometric_erasure") == 1.0


def test_delete_enrollment_api_v1_alias(client):
    """The /api/v1/enroll/{id} alias also functions correctly."""
    user = "bob"

    client.post(
        "/api/v1/enroll",
        data={"user_id": user},
        files={"audio": ("enroll.wav", _wav_bytes(1.0), "audio/wav")},
        headers=AUTH_BOB,
    )
    assert app_module.PIPELINE.is_enrolled(user)

    del_res = client.delete(f"/api/v1/enroll/{user}", headers=AUTH_BOB)
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"
    assert not app_module.PIPELINE.is_enrolled(user)
