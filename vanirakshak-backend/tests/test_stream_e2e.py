"""End-to-end check of the WebSocket contract in docs/PROTOCOL.md.

Replays what the Next.js dashboard does in `frontend/app/page.tsx`:
open the socket, send the JSON text handshake, then stream int16 PCM and
read verdicts.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

# Allow direct standalone execution via `python tests/test_stream_e2e.py`
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from fastapi.testclient import TestClient

from vanirakshak.server.app import app

SAMPLE_RATE = 16000

# Keys the dashboard's normalizeResult() reads. If any is missing the UI
# silently shows zeros, which is exactly the class of bug this test catches.
REQUIRED_TOP_LEVEL = {
    "risk_score",
    "p_spoof",
    "tier",
    "metrics",
    "interlock_active",
}
REQUIRED_METRICS = {"asv_consistency", "snr_db"}


def pcm_chunk(seconds: float = 1.5, freq: float = 220.0) -> bytes:
    """A quiet sine tone as mono int16 PCM, like the AudioWorklet produces."""
    frames = int(seconds * SAMPLE_RATE)
    return b"".join(
        struct.pack(
            "<h",
            int(8000 * math.sin(2 * math.pi * freq * (i / SAMPLE_RATE))),
        )
        for i in range(frames)
    )


def test_websocket_stream_e2e() -> None:
    client = TestClient(app)
    handshake = {
        "session_id": "e2e-protocol-check",
        "sample_rate": SAMPLE_RATE,
        "caller_id": "",
        "claimed_identity": "",
        "transaction_value_inr": 0,
        "origin_country": "IN",
        "prior_fraud_score": 0,
    }

    with client.websocket_connect("/ws/stream") as ws:
        # 1. Text handshake FIRST — the server closes with 1003 otherwise.
        ws.send_text(json.dumps(handshake))

        # 1b. Handshake ack FIRST — carries the resolved session_id.
        ack = ws.receive_json()
        assert ack.get("type") == "session_ack", f"expected session_ack, got: {ack}"
        assert ack.get("session_id") == "e2e-protocol-check", f"ack echoed wrong id: {ack}"

        # 2. Stream audio: 3 chunks (each 1.5s @ 16kHz = 4.5s > 4.04s window)
        chunk = pcm_chunk(1.5)
        for _ in range(3):
            ws.send_bytes(chunk)

        verdict = ws.receive_json()

    assert verdict is not None, "Server did not return a verdict after 4.5s of audio"
    missing_top = REQUIRED_TOP_LEVEL - verdict.keys()
    assert not missing_top, f"Missing top-level keys: {sorted(missing_top)}"

    metrics = verdict.get("metrics") or {}
    missing_metrics = REQUIRED_METRICS - metrics.keys()
    assert not missing_metrics, f"Missing metrics keys: {sorted(missing_metrics)}"

    assert verdict.get("tier") in {"ALLOW", "CHALLENGE", "BLOCK"}, (
        f"Invalid tier: {verdict.get('tier')!r}"
    )
    assert isinstance(verdict.get("risk_score"), (int, float))
    assert isinstance(verdict.get("p_spoof"), (int, float))
    assert isinstance(verdict.get("interlock_active"), bool)
    assert "receipt_hash" in verdict, "receipt_hash missing from verdict"
    assert "timestamp_ms" in verdict, "timestamp_ms missing from verdict"

    # Protocol enforcement check: binary frame before handshake must be closed with 1003
    with client.websocket_connect("/ws/stream") as ws_bad:
        ws_bad.send_bytes(chunk)
        close_msg = ws_bad.receive()
        assert close_msg.get("type") == "websocket.close" and close_msg.get("code") == 1003


def make_wav_bytes(seconds: float = 5.0, freq: float = 440.0) -> bytes:
    """Generate in-memory mono 16-bit 16kHz WAV byte payload."""
    import io
    import wave

    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        frames = int(seconds * SAMPLE_RATE)
        raw = b"".join(
            struct.pack(
                "<h",
                int(8000 * math.sin(2 * math.pi * freq * (i / SAMPLE_RATE))),
            )
            for i in range(frames)
        )
        w.writeframes(raw)
    return bio.getvalue()


def test_rest_health_endpoint() -> None:
    client = TestClient(app)
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data.get("ok") is True
    assert "hardware" in data
    assert "models" in data
    assert data["hardware"].get("device_name")


def test_rest_analyze_endpoint() -> None:
    client = TestClient(app)
    wav = make_wav_bytes(5.0)

    # 1. Test POST /api/v1/analyze/audio with "file" form field (Next.js dashboard contract)
    res = client.post(
        "/api/v1/analyze/audio",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    assert res.status_code == 200, f"Analysis failed: {res.text}"
    data = res.json()
    assert data["status"] == "success"
    assert "overall_risk_score" in data
    assert "verdict" in data
    assert "windows" in data
    assert len(data["windows"]) > 0
    assert "dpdp_compliance" in data
    assert data["dpdp_compliance"]["zero_retention_verified"] is True
    assert data["dpdp_compliance"]["receipt_sha256"]

    # 2. Test POST /v1/analyze with "audio" form field (backward compatibility)
    res2 = client.post(
        "/v1/analyze",
        files={"audio": ("test2.wav", wav, "audio/wav")},
    )
    assert res2.status_code == 200, f"Legacy analysis failed: {res2.text}"
    data2 = res2.json()
    assert "calibrated_risk" in data2
    assert "tier" in data2
    assert "breakdown" in data2


def main() -> int:
    try:
        test_websocket_stream_e2e()
        test_rest_health_endpoint()
        test_rest_analyze_endpoint()
        print("PASS: all stream and REST endpoint tests succeeded")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
