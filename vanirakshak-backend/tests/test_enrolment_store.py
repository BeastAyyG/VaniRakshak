"""docs/15 N4: persistent enrolment store and erasure that reaches live sessions.

Exit criterion: the restart test keeps enrolments; the erasure test proves the
record is gone from disk and the live session is closed.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vanirakshak.privacy.enrolment_store import (
    EnrolmentRecord,
    EnrolmentStore,
    EnrolmentStoreError,
    parse_key,
    store_from_env,
)
from vanirakshak.server.auth import Authenticator

app_module = importlib.import_module("vanirakshak.server.app")

KEY = bytes(range(32))
KEY_HEX = KEY.hex()
BACKEND_DIR = Path(__file__).resolve().parents[1]


def _record(user="alice", backend="synthetic_asv_v1") -> EnrolmentRecord:
    return EnrolmentRecord(
        user_id=user,
        embedding=np.linspace(-1, 1, 192, dtype=np.float32),
        asv_backend=backend,
        calibration={"mid": 0.97, "scale": 0.004},
        qa={"voiced_s": 12.5, "snr_db": 24.0, "cm_llr": 3.1},
        consent_ref="consent/2026-09-28/alice",
    )


def _wav_bytes(seconds: float = 2.0, sr: int = 16000) -> bytes:
    rng = np.random.default_rng(7)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((rng.normal(0, 3000, int(sr * seconds))).astype("<i2").tobytes())
    return buf.getvalue()


# --- the store itself --------------------------------------------------------


def test_round_trip_keeps_every_field(tmp_path):
    store = EnrolmentStore(tmp_path, KEY)
    rec = _record()
    store.save(rec)
    back = store.load("alice")
    assert np.array_equal(back.embedding, rec.embedding)
    assert back.calibration == rec.calibration
    assert back.qa == rec.qa
    assert back.consent_ref == rec.consent_ref
    assert back.asv_backend == rec.asv_backend


def test_disk_holds_no_plaintext(tmp_path):
    """Neither the file name nor its bytes reveal who is enrolled."""
    store = EnrolmentStore(tmp_path, KEY)
    store.save(_record("alice_plaintext_probe"))
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert "alice" not in files[0].name
    blob = files[0].read_bytes()
    assert b"alice_plaintext_probe" not in blob
    assert b"consent/" not in blob


def test_wrong_key_is_unreadable_not_fatal(tmp_path):
    EnrolmentStore(tmp_path, KEY).save(_record())
    report = EnrolmentStore(tmp_path, bytes(32)).load_all()
    assert report.records == []
    assert len(report.unreadable) == 1


def test_erase_removes_the_file(tmp_path):
    store = EnrolmentStore(tmp_path, KEY)
    store.save(_record())
    assert store.erase("alice") is True
    assert list(tmp_path.glob("*.vre")) == []
    assert store.load("alice") is None
    assert store.erase("alice") is False


def test_path_without_key_refuses_to_start(tmp_path):
    with pytest.raises(EnrolmentStoreError):
        store_from_env(str(tmp_path), "")
    assert store_from_env("", "") is None


@pytest.mark.parametrize("encoded", [
    KEY_HEX,
    base64.b64encode(KEY).decode(),
    base64.urlsafe_b64encode(KEY).decode(),
])
def test_parse_key_accepts_hex_and_base64(encoded):
    assert parse_key(encoded) == KEY


def test_parse_key_rejects_short_keys():
    with pytest.raises(EnrolmentStoreError):
        parse_key(base64.b64encode(b"too short").decode())


# --- restart: two real processes ---------------------------------------------

_CHILD = r"""
import json, sys
from fastapi.testclient import TestClient
import importlib
m = importlib.import_module("vanirakshak.server.app")

mode = sys.argv[1]
if mode == "enrol":
    wav = open(sys.argv[2], "rb").read()
    r = TestClient(m.app).post(
        "/v1/enroll",
        data={"user_id": "restart_user", "consent_ref": "consent-001"},
        files={"audio": ("a.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200, r.text
emb = m.PIPELINE.asv.get_enrolment("restart_user")
print(json.dumps({
    "enrolled": m.PIPELINE.is_enrolled("restart_user"),
    "embedding": emb[0].tolist() if emb else None,
    "calibration": emb[1] if emb else None,
    "store": m.ENROL_STORE_STATE,
}))
"""


def _child(mode: str, store_dir: Path, audit: Path, *args: str) -> dict:
    env = dict(os.environ)
    env.update({
        "VR_ENROL_STORE_PATH": str(store_dir),
        "VR_ENROL_STORE_KEY": KEY_HEX,
        "VR_ENROLL_BYPASS_QA": "1",
        "VR_AUDIT_PATH": str(audit),
        "PYTHONPATH": os.pathsep.join([str(BACKEND_DIR), str(BACKEND_DIR.parent)]),
    })
    out = subprocess.run(
        [sys.executable, "-c", _CHILD, mode, *args],
        cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_restart_keeps_enrolments(tmp_path):
    store_dir, audit = tmp_path / "store", tmp_path / "audit.jsonl"
    wav = tmp_path / "enrol.wav"
    wav.write_bytes(_wav_bytes())
    first = _child("enrol", store_dir, audit, str(wav))
    assert first["enrolled"] is True
    assert first["store"]["persistent"] is True

    second = _child("check", store_dir, audit)  # a fresh process: a restart
    assert second["enrolled"] is True
    assert second["store"]["loaded"] >= 1
    assert second["embedding"] == first["embedding"]
    assert second["calibration"] == first["calibration"]


# --- erasure reaches disk and live sessions ----------------------------------


@pytest.fixture
def store_app(tmp_path, monkeypatch):
    store = EnrolmentStore(tmp_path / "store", KEY)
    monkeypatch.setattr(app_module, "ENROL_STORE", store)
    monkeypatch.setattr(
        app_module, "AUTH",
        Authenticator(raw_keys="", require_auth=False, rate_limits={"analyze": 0, "enroll": 0}),
    )
    monkeypatch.setenv("VR_ENROLL_BYPASS_QA", "1")
    return store, TestClient(app_module.app)


def test_erasure_deletes_record_and_closes_live_session(store_app):
    store, client = store_app
    user = "n4_erasure_user"
    r = client.post("/v1/enroll", data={"user_id": user, "consent_ref": "c-9"},
                    files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert store.contains(user)
    assert store.load(user).consent_ref == "c-9"

    handshake = {"session_id": "n4-live", "sample_rate": 16000, "claimed_identity": user}
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps(handshake))
        assert ws.receive_json()["type"] == "session_ack"
        assert "n4-live" in app_module.SESSIONS

        receipt = client.delete(f"/v1/enroll/{user}").json()
        assert receipt["store_record_erased"] is True
        assert "n4-live" in receipt["sessions_closed"]

        # Gone from disk, from memory, and from the session table.
        assert list((store.root).glob("*.vre")) == []
        assert not app_module.PIPELINE.is_enrolled(user)
        assert "n4-live" not in app_module.SESSIONS

        # The stream's next message closes it instead of scoring.
        ws.send_bytes(np.zeros(16000, dtype=np.int16).tobytes())
        msg = ws.receive_json()
        assert msg == {"type": "session_closed", "session_id": "n4-live",
                       "reason": "biometric_erasure"}


def test_erasure_of_record_only_on_disk(store_app):
    """A record left on disk (e.g. from another backend) can still be erased."""
    store, client = store_app
    store.save(_record("disk_only_user", backend="some_other_asv"))
    assert not app_module.PIPELINE.is_enrolled("disk_only_user")
    receipt = client.delete("/v1/enroll/disk_only_user")
    assert receipt.status_code == 200
    assert receipt.json()["store_record_erased"] is True
    assert not store.contains("disk_only_user")


def test_revoked_session_stops_scoring():
    from vanirakshak.session import CallSession, SessionContext

    sess = CallSession("r", SessionContext(claimed_identity="x"))
    sess.revoke("biometric_erasure")
    for _ in range(6):
        assert sess.push_chunk(np.zeros(16000, dtype=np.int16)) is None
    assert sess.revoked == "biometric_erasure"


def test_health_reports_store_state():
    store = TestClient(app_module.app).get("/v1/health").json()["enrolment_store"]
    assert set(store) == {"persistent", "loaded", "skipped_other_backend", "unreadable"}
