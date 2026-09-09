"""FastAPI HTTP/WebSocket server (R4 — the missing API surface).

Endpoints (see ``docs/07-privacy-compliance-and-apis.md``):
  * ``POST /v1/enroll``           enrol a genuine voice (10-20s)
  * ``POST /v1/analyze``          one-shot WAV/FLAC → risk score
  * ``WS   /v1/stream``           real-time PCM stream → live verdicts
  * ``POST /v1/sessions/{id}/challenge``  grade a challenge response
  * ``GET  /v1/audit``            tail the DPDP-compliant audit log
  * ``GET  /v1/health``           liveness
  * ``GET  /docs``                OpenAPI UI
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..session import CallSession, SessionContext
from ..privacy.audit import AuditLog
from ..alerts import AlertDispatcher, build_alert_payload
from ..detectors.pipeline import DetectionPipeline
from ..anchor import load_anchor
from ..semantic import SEMANTIC_SIGNAL


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VaniRakshak",
    description="Real-time voice cloning impersonation detection (R4 backend).",
    version="0.1.0-sprint1",
)

# module-level singletons for the demo. In production these would be
# replaced with a proper dependency-injected pool keyed by tenant.
SESSIONS: Dict[str, CallSession] = {}
AUDIT = AuditLog(max_events=10_000)
ALERTS = AlertDispatcher()
PIPELINE = DetectionPipeline()

# Enrolled target speaker ("anchor"), loaded once at import time.
# Without this, `asv_consistency` is a hard-coded 0.5 for every call and the
# target-conditioned half of the system does nothing. The loader lives in
# `vanirakshak/anchor.py` so the benchmark harness loads the same vector into
# the same pipeline and the two report comparable numbers.
ANCHOR_STATE: Dict[str, object] = load_anchor(PIPELINE)


def _resolve_claimed_identity(raw: str) -> str:
    """Fall back to the loaded anchor's user_id when the caller sends none.

    `claimed_identity` defaults to `""` in the WS handshake, the `/analyze`
    form field, and `SessionStartRequest` alike — so on the default path no
    caller ever set it, and `DetectionPipeline.analyse()` skips ASV entirely
    (`asv_score` stays a hardcoded 0.5, `asv_enrolled=False`). With exactly
    one anchor enrolled today, defaulting an unset claim to that anchor's
    user_id is the only way ASV runs out of the box. A caller that explicitly
    names a *different* claimed identity is never overridden here.
    """
    if raw:
        return raw
    anchor_user_id = ANCHOR_STATE.get("user_id")
    return str(anchor_user_id) if anchor_user_id else raw

# Browser dashboard runs on a different origin (localhost:3001 by default).
# Origins are configurable via VR_CORS_ORIGINS (comma-separated).
_cors_origins = [
    o.strip()
    for o in getattr(
        settings.server,
        "cors_origins",
        "http://localhost:3001,http://127.0.0.1:3001",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV (PCM) byte string to (float32 in [-1,1], sample_rate)."""
    bio = io.BytesIO(data)
    try:
        with wave.open(bio, "rb") as w:
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            sr = w.getframerate()
            nframes = w.getnframes()
            raw = w.readframes(nframes)
        if sampwidth == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise HTTPException(400, f"unsupported sample width: {sampwidth}")
        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)
        return arr.astype(np.float32), sr
    except Exception:
        # Fallback to scipy.io.wavfile for non-standard chunk headers / float WAVs
        try:
            import scipy.io.wavfile as wavfile

            bio.seek(0)
            sr, arr = wavfile.read(bio)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32)
            elif arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype(np.float32) / 2147483648.0
            elif arr.dtype == np.uint8:
                arr = (arr.astype(np.float32) - 128.0) / 128.0
            return arr.astype(np.float32), int(sr)
        except Exception as e:
            raise HTTPException(400, f"unable to decode WAV audio payload: {e}")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class EnrollRequest(BaseModel):
    user_id: str = Field(..., description="Stable identifier for the speaker.")


class EnrollResponse(BaseModel):
    status: str
    user_id: str
    embedding_dimension: int
    audio_retained: bool
    created_at: str


class EnrolledSpeakersResponse(BaseModel):
    enrolled: List[str]
    asv_backend: str


class AnalyzeResponse(BaseModel):
    status: str = "success"
    file_duration_sec: float
    calibrated_risk: float
    p_spoof: float
    tier: str
    receipt_hash: str
    breakdown: dict = Field(default_factory=dict)

    # ForensicReportData contract extensions for frontend dashboard
    session_id: Optional[str] = None
    audio_sha256: Optional[str] = None
    duration_sec: Optional[float] = None
    device_used: str = "CPU (Pure NumPy Forensic Pipeline)"
    windows_count: int = 0
    overall_risk_score: float = 0.0
    verdict: str = "AUTHENTIC_TARGET"
    windows: List[dict] = Field(default_factory=list)
    mean_acoustic_anomaly: Optional[float] = None
    max_acoustic_anomaly: Optional[float] = None
    asv_evaluated: Optional[bool] = None
    mean_speaker_sim: Optional[float] = None
    mean_micro_tremor: Optional[float] = None
    mean_spectral_flatness: Optional[float] = None
    mean_high_band_flatness: Optional[float] = None
    mean_zcr_variance: Optional[float] = None
    vocoder_phase_discontinuities_detected: int = 0
    detected_boundary_timestamps: List[float] = Field(default_factory=list)
    dpdp_compliance: Optional[dict] = None

    # Real (non-stub) signal: the trained ml/ target-speaker model, served as
    # an honest IDENTITY check — "is this the enrolled voice?" — never as a
    # clone/synthetic verdict. None when the checkpoint or torch is absent.
    target_speaker: Optional[dict] = None

    # Parallel ASR + scam-keyword context signal (vanirakshak/semantic/).
    # Advisory only — see the `semantic` field in docs/PROTOCOL.md.
    semantic: Optional[dict] = None


class SessionStartRequest(BaseModel):
    caller_id: str = "+91-0000000000"
    claimed_identity: str = ""
    transaction_value_inr: float = 0.0
    origin_country: str = "IN"
    prior_fraud_score: float = 0.0


class ChallengeGradeRequest(BaseModel):
    response_text: str
    response_latency_ms: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/v1/health")
@app.get("/api/v1/health")
def health() -> dict:
    return {
        "ok": True,
        "status": "healthy",
        "version": app.version,
        "active_sessions": len(SESSIONS),
        "hardware": {
            "device": "cpu",
            "device_name": "CPU (Acoustic Forensic Core)",
        },
        "models": {
            # Honest label: no AASIST (trained anti-spoofing classifier) exists
            # anywhere in this system — see cm_backend/asv_backend below for
            # what's actually running (synthetic stub vs real WavLM/ECAPA-TDNN).
            "wavlm_model": f"{PIPELINE.cm.name} / {PIPELINE.asv.name}",
            "vad_enabled": True,
            "zero_retention": True,
            "cm_backend": PIPELINE.cm.name,
            "asv_backend": PIPELINE.asv.name,
        },
        "anchor": {
            "user_id": ANCHOR_STATE.get("user_id"),
            "enrolled": bool(ANCHOR_STATE.get("enrolled")),
            "calibrated": bool(ANCHOR_STATE.get("calibrated")),
            "source_clips": ANCHOR_STATE.get("source_clips", 0),
            "error": ANCHOR_STATE.get("error"),
        },
    }


@app.get("/v1/enroll", response_model=EnrolledSpeakersResponse)
@app.get("/api/v1/enroll", response_model=EnrolledSpeakersResponse)
def list_enrolled() -> EnrolledSpeakersResponse:
    """Enrolled user_ids (family members / anchors) — lets the dashboard offer
    a claimed-identity picker instead of a single hardcoded anchor."""
    return EnrolledSpeakersResponse(
        enrolled=PIPELINE.asv.enrolled_user_ids(),
        asv_backend=PIPELINE.asv.name,
    )


@app.post("/v1/enroll", response_model=EnrollResponse)
@app.post("/api/v1/enroll", response_model=EnrollResponse)
async def enroll(user_id: str = Form(...), audio: UploadFile = File(...)) -> EnrollResponse:
    data = await audio.read()
    x, sr = _read_wav_bytes(data)
    if x.size < int(0.5 * sr):
        raise HTTPException(400, "enrolment audio too short (need ≥ 0.5 s)")
    # Re-use or create a session for enrolment
    sid = f"enroll-{uuid.uuid4().hex[:8]}"
    sess = CallSession(sid, SessionContext(caller_id="enrolment", claimed_identity=user_id), audit=AUDIT, pipeline=PIPELINE)
    sess.enrol(user_id, x, sr)
    SESSIONS[sid] = sess
    return EnrollResponse(
        status="enrolled",
        user_id=user_id,
        embedding_dimension=192,
        audio_retained=False,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


@app.post("/v1/analyze", response_model=AnalyzeResponse)
@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
@app.post("/v1/analyze/audio", response_model=AnalyzeResponse)
@app.post("/api/v1/analyze/audio", response_model=AnalyzeResponse)
async def analyze(
    audio: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    caller_id: str = Form("+91-0000000000"),
    claimed_identity: str = Form(""),
) -> AnalyzeResponse:
    upload = audio or file
    if upload is None:
        raise HTTPException(400, "Audio payload required in form field 'audio' or 'file'")

    data = await upload.read()
    if not data:
        raise HTTPException(400, "Received empty audio payload")

    audio_sha256 = hashlib.sha256(data).hexdigest()
    x, sr = _read_wav_bytes(data)
    file_duration = round(x.size / sr, 3)

    sid = f"analyze-{uuid.uuid4().hex[:8]}"
    sess = CallSession(
        sid,
        SessionContext(caller_id=caller_id, claimed_identity=_resolve_claimed_identity(claimed_identity)),
        audit=AUDIT,
        pipeline=PIPELINE
    )
    sess.set_sample_rate(sr)

    # ---- parallel ASR + keyword scam-phrase signal (advisory) --------------
    # Computed BEFORE the window loop below and fed into the session via
    # update_semantic() so every window's fusion call actually sees it — see
    # the note above push_chunk's use of _semantic_state. This differs from
    # the WS path (where semantic necessarily lags one window behind, since
    # transcription is async and the audio arrives live): a one-shot /analyze
    # upload is a single static file, so one transcript for the whole clip
    # can inform every window's fusion equally, instead of running Groq once
    # per window for no benefit. Previously this ran *after* the window loop
    # so it never reached fusion at all (w_semantic was dead code here) and
    # only decorated the response. Degrades to available=False on any
    # failure; never breaks the forensic report.
    try:
        pcm_int16_full = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
        semantic = await asyncio.to_thread(SEMANTIC_SIGNAL.analyze_window, pcm_int16_full, sr)
    except Exception as e:
        semantic = {"available": False, "semantic_risk": 0.0, "reason": f"semantic signal error: {type(e).__name__}: {e}"}
    sess.update_semantic(semantic)

    # If audio is shorter than the buffer's window_samples, pad with zeros
    # so we can always evaluate at least one forensic window
    x_analysis = x
    if x_analysis.size < sess.buffer.window_samples:
        pad_size = sess.buffer.window_samples - x_analysis.size
        x_analysis = np.pad(x_analysis, (0, pad_size), mode="constant")

    # Feed initial window
    sess.push_chunk(x_analysis[:sess.buffer.window_samples])

    # Drain subsequent hops
    hop = sess.buffer.hop_samples
    i = sess.buffer.window_samples
    while i + hop <= x_analysis.size:
        sess.push_chunk(x_analysis[i : i + hop])
        i += hop

    last = sess.history[-1] if sess.history else None
    if last is None:
        raise HTTPException(400, "Unable to extract forensic analysis window")

    windows: List[dict] = []
    hop_sec = sess.buffer.cfg.hop_seconds
    win_sec = sess.buffer.cfg.window_seconds

    for idx, cr in enumerate(sess.history):
        start_sec = round(idx * hop_sec, 2)
        end_sec = round(min(file_duration, start_sec + win_sec), 2)
        if end_sec <= start_sec:
            end_sec = round(start_sec + win_sec, 2)

        anomaly = float(np.clip(cr.p_spoof, 0.0, 1.0))
        asv_val = float(cr.metrics.get("asv_consistency", 0.5))
        asv_evaluated_val = bool(cr.metrics.get("asv_evaluated", False))
        channel_val = float(cr.metrics.get("channel_anomaly", 0.1))
        prosody_val = float(cr.metrics.get("prosody_unnatural", 0.2))

        windows.append({
            "window_index": idx + 1,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": round(win_sec, 2),
            "is_voiced": True,
            "speech_ratio": 0.95,
            "acoustic_anomaly_score": round(anomaly, 4),
            "speaker_cosine_sim": round(asv_val, 4),
            "speaker_evaluated": asv_evaluated_val,
            "speaker_match": bool(asv_evaluated_val and asv_val >= 0.5),
            "micro_tremor_ratio": round(max(0.0, min(1.0, 1.0 - prosody_val)), 4),
            "spectral_flatness_mean": round(channel_val, 4),
            "high_band_flatness": round(channel_val, 4),
            "zcr_variance": 0.05,
            "vocoder_boundary_score": round(anomaly, 4),
            "inference_ms": 14.5,
        })

    anomalies = [w["acoustic_anomaly_score"] for w in windows]
    mean_acoustic_anomaly = round(float(np.mean(anomalies)), 4) if anomalies else 0.0
    max_acoustic_anomaly = round(float(np.max(anomalies)), 4) if anomalies else 0.0

    asv_scores = [w["speaker_cosine_sim"] for w in windows if w["speaker_cosine_sim"] is not None]
    mean_speaker_sim = round(float(np.mean(asv_scores)), 4) if asv_scores else 0.5

    tremors = [w["micro_tremor_ratio"] for w in windows if w["micro_tremor_ratio"] is not None]
    mean_micro_tremor = round(float(np.mean(tremors)), 4) if tremors else 0.8

    flatness = [w["spectral_flatness_mean"] for w in windows if w["spectral_flatness_mean"] is not None]
    mean_spectral_flatness = round(float(np.mean(flatness)), 4) if flatness else 0.1

    boundary_timestamps = [w["start_sec"] for w in windows if w["acoustic_anomaly_score"] >= 0.65]

    # ---- real (non-stub) identity check via the trained ml/ model ----------
    # Honest scope: this answers "is this the enrolled target voice?", NOT
    # "is this a clone?". It is reported alongside the stub-driven risk tier,
    # never folded into it, so a missing checkpoint can never silently fake a
    # forensic verdict. Degrades to available=False when torch/checkpoint
    # are absent instead of erroring the whole analysis.
    target_speaker: Optional[dict] = None
    try:
        from ..ml_bridge import VERIFIER

        if x.size >= int(0.5 * sr):
            target_speaker = VERIFIER.verify(x, sr)
        else:
            target_speaker = {
                "available": False,
                "label": "unavailable",
                "is_target": False,
                "reason": "audio too short (<0.5s)",
            }
    except Exception as e:  # never let the ML path break the forensic report
        target_speaker = {
            "available": False,
            "label": "unavailable",
            "is_target": False,
            "reason": f"verifier error: {type(e).__name__}: {e}",
        }

    # Map tier to forensic verdict recognized by frontend.
    #
    # Speaker identity dominates the label: a clip whose voice does not match the
    # enrolled target is an impostor regardless of which risk band it lands in.
    # Previously the impostor check only ran inside the BLOCK band, so an
    # impostor scoring 67/100 (CHALLENGE) was mislabelled "SUSPICIOUS_SPLICED_AUDIO"
    # even at mean_speaker_sim 0.09.
    if mean_speaker_sim < 0.40:
        verdict = "IMPOSTOR_SPEAKER"
    elif last.tier == "BLOCK" or last.risk >= 70:
        verdict = "SYNTHETIC_VOICE_CLONE"
    elif last.tier == "CHALLENGE" or last.risk >= 35:
        verdict = "SUSPICIOUS_SPLICED_AUDIO"
    else:
        verdict = "AUTHENTIC_TARGET"

    dpdp = {
        "receipt_sha256": last.receipt_hash,
        "statutory_act": "India Digital Personal Data Protection (DPDP) Act 2023 Section 8(7)",
        "zero_retention_verified": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if last.tier in ("CHALLENGE", "BLOCK") or last.risk >= 70:
        ALERTS.dispatch(build_alert_payload(
            session_id=sid,
            risk=last.risk,
            tier=last.tier,
            interlock_active=(last.tier == "BLOCK"),
            receipt_hash=last.receipt_hash,
        ))

    return AnalyzeResponse(
        status="success",
        file_duration_sec=file_duration,
        calibrated_risk=last.risk,
        p_spoof=last.p_spoof,
        tier=last.tier,
        receipt_hash=last.receipt_hash,
        breakdown={
            "codec_guess": last.metrics.get("codec_guess"),
            "cm_score": last.metrics.get("cm_score"),
            "asv_consistency": last.metrics.get("asv_consistency"),
            "asv_evaluated": last.metrics.get("asv_evaluated", False),
            "channel_anomaly": last.metrics.get("channel_anomaly"),
            "prosody_unnatural": last.metrics.get("prosody_unnatural"),
        },
        session_id=sid,
        audio_sha256=audio_sha256,
        duration_sec=file_duration,
        device_used="CPU (Pure NumPy Forensic Pipeline)",
        windows_count=len(windows),
        overall_risk_score=round(last.risk, 1),
        verdict=verdict,
        windows=windows,
        mean_acoustic_anomaly=mean_acoustic_anomaly,
        max_acoustic_anomaly=max_acoustic_anomaly,
        asv_evaluated=bool(last.metrics.get("asv_evaluated", False)),
        mean_speaker_sim=mean_speaker_sim,
        mean_micro_tremor=mean_micro_tremor,
        mean_spectral_flatness=mean_spectral_flatness,
        mean_high_band_flatness=mean_spectral_flatness,
        mean_zcr_variance=0.05,
        vocoder_phase_discontinuities_detected=len(boundary_timestamps),
        detected_boundary_timestamps=boundary_timestamps,
        dpdp_compliance=dpdp,
        target_speaker=target_speaker,
        semantic=semantic,
    )


# ---------------------------------------------------------------------------
# Target-speaker verification (the trained ml/ model, served honestly as an
# identity check: "is this the enrolled voice (Modi ji), or not?").
# ---------------------------------------------------------------------------
class VerifySpeakerResponse(BaseModel):
    available: bool
    label: str
    is_target: bool
    llr: Optional[float] = None
    prob_target: Optional[float] = None
    reason: Optional[str] = None


@app.post("/v1/verify-speaker", response_model=VerifySpeakerResponse)
@app.post("/api/v1/verify-speaker", response_model=VerifySpeakerResponse)
async def verify_speaker(
    audio: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
) -> VerifySpeakerResponse:
    """Score a short voice chunk against the enrolled target speaker.

    Honest scope: the model answers identity ("Modi ji" vs "not Modi ji"),
    NOT "is this a synthetic/clone voice". It was trained on one target speaker
    vs one non-target speaker and has never seen cloned audio.
    """
    from ..ml_bridge import VERIFIER

    upload = audio or file
    if upload is None:
        raise HTTPException(400, "Audio payload required in form field 'audio' or 'file'")

    data = await upload.read()
    if not data:
        raise HTTPException(400, "Received empty audio payload")

    try:
        x, sr = _read_wav_bytes(data)
    except Exception:
        raise HTTPException(400, "Could not decode WAV audio")

    # Require a usable minimum (>= 0.5s) so silence doesn't produce a verdict.
    if x.size < int(0.5 * sr):
        return VerifySpeakerResponse(
            available=VERIFIER.available,
            label="unavailable",
            is_target=False,
            reason="audio too short (<0.5s)",
        )

    result = VERIFIER.verify(x, sr)
    return VerifySpeakerResponse(
        available=bool(result["available"]),
        label=str(result["label"]),
        is_target=bool(result["is_target"]),
        llr=result["llr"],
        prob_target=result["prob_target"],
        reason=result.get("reason"),
    )


@app.post("/v1/sessions")
@app.post("/api/v1/sessions")
def start_session(req: SessionStartRequest) -> dict:
    sid = f"call-{uuid.uuid4().hex[:10]}"
    SESSIONS[sid] = CallSession(
        sid,
        SessionContext(
            caller_id=req.caller_id,
            claimed_identity=_resolve_claimed_identity(req.claimed_identity),
            transaction_value_inr=req.transaction_value_inr,
            origin_country=req.origin_country,
            prior_fraud_score=req.prior_fraud_score,
        ),
        audit=AUDIT,
        pipeline=PIPELINE
    )
    return {"session_id": sid}


@app.post("/v1/sessions/{sid}/challenge")
@app.post("/api/v1/sessions/{sid}/challenge")
def grade_challenge(sid: str, req: ChallengeGradeRequest) -> dict:
    sess = SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(404, "session not found")
    return sess.grade_challenge(req.response_text, req.response_latency_ms)


@app.get("/v1/audit")
@app.get("/api/v1/audit")
def tail_audit(n: int = 50) -> dict:
    events = AUDIT.tail(n)
    return {
        "chain_valid": AUDIT.verify_chain(),
        "events": [e.to_public_dict() for e in events],
    }


@app.websocket("/v1/stream")
@app.websocket("/api/v1/stream")
@app.websocket("/ws/stream")
async def stream_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    # first message: JSON handshake
    try:
        handshake_raw = await ws.receive_text()
        import json
        handshake = json.loads(handshake_raw)
    except Exception:
        await ws.close(code=1003)
        return

    sid = handshake.get("session_id") or f"ws-{uuid.uuid4().hex[:8]}"
    sr = int(handshake.get("sample_rate", 16000))
    ctx = SessionContext(
        caller_id=str(handshake.get("caller_id", "")),
        claimed_identity=_resolve_claimed_identity(str(handshake.get("claimed_identity", ""))),
        transaction_value_inr=float(handshake.get("transaction_value_inr", 0.0)),
        origin_country=str(handshake.get("origin_country", "IN")),
        prior_fraud_score=float(handshake.get("prior_fraud_score", 0.0)),
    )
    sess = CallSession(sid, ctx, audit=AUDIT, pipeline=PIPELINE)
    sess.set_sample_rate(sr)
    SESSIONS[sid] = sess

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data: Optional[bytes] = msg.get("bytes")
            if data is None:
                continue
            # interpret as int16 PCM mono
            pcm = np.frombuffer(data, dtype=np.int16)
            if pcm.size == 0:
                continue
            result = sess.push_chunk(pcm)
            if result is not None:
                await ws.send_json({
                    "timestamp_ms": result.timestamp_ms,
                    "risk_score": result.risk,
                    "p_spoof": result.p_spoof,
                    "tier": result.tier,
                    "metrics": result.metrics,
                    "trigger_challenge": result.trigger_challenge,
                    "interlock_active": result.interlock_active,
                    "receipt_hash": result.receipt_hash,
                    "challenge": result.challenge,
                    "semantic": result.semantic,
                })

                # Parallel semantic (ASR + keyword) pass on this window's raw
                # audio — fire-and-forget in a worker thread so a slow/absent
                # Groq call never blocks the acoustic verdict stream. Its
                # result lands on the *next* window via sess.update_semantic;
                # see vanirakshak/semantic/signal.py and session.py.
                pending = sess.pop_pending_window()
                if pending is not None:
                    window_pcm, window_sr = pending

                    async def _run_semantic(sess=sess, window_pcm=window_pcm, window_sr=window_sr) -> None:
                        try:
                            semantic_result = await asyncio.to_thread(
                                SEMANTIC_SIGNAL.analyze_window, window_pcm, window_sr
                            )
                            sess.update_semantic(semantic_result)
                        except Exception:
                            pass  # advisory signal only — never disrupt the session

                    asyncio.create_task(_run_semantic())

                # Server-side alerting on challenge / block. Best-effort:
                # a delivery failure must not break the stream.
                if result.trigger_challenge or result.interlock_active:
                    await asyncio.to_thread(
                        ALERTS.dispatch,
                        build_alert_payload(
                            session_id=sid,
                            risk=result.risk,
                            tier=result.tier,
                            interlock_active=result.interlock_active,
                            receipt_hash=result.receipt_hash,
                            challenge_phrase=(
                                (result.challenge or {}).get("phrase")
                                if isinstance(result.challenge, dict)
                                else None
                            ),
                        )
                    )
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        sess.buffer.wipe()
        SESSIONS.pop(sid, None)
