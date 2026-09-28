"""FastAPI HTTP/WebSocket server (R4 — the missing API surface).

Endpoints (see ``docs/07-privacy-compliance-and-apis.md``):
  * ``POST /v1/enroll``           enrol a genuine voice (10-20s)
  * ``DELETE /v1/enroll/{id}``     erase an enrolled voiceprint (DPDP Act 2023)
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
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..session import CallSession, SessionContext
from ..privacy.audit import AuditLog
from ..alerts import AlertDispatcher, build_alert_payload
from ..detectors.pipeline import DetectionPipeline
from ..detectors.voice import vad_stats, zcr_variance
from ..detectors.channel import detect as channel_detect
from ..anchor import load_anchor
from ..privacy.enrolment_store import EnrolmentRecord, store_from_env
from ..detectors.calibration import Cohort, fit_user_calibration
from ..semantic import SEMANTIC_SIGNAL
from .auth import ApiKey, AuthError, Authenticator, extract_secret


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
# Durable audit chain: VR_AUDIT_PATH (default data/audit_log.jsonl, gitignored)
# persists every event as JSONL and reloads it on restart. Empty path =
# in-memory ring only.
AUDIT = AuditLog(max_events=settings.audit.max_events, path=settings.audit.path or None)
ALERTS = AlertDispatcher()
PIPELINE = DetectionPipeline()

# Enrolled target speaker ("anchor"), loaded once at import time.
# Without this, `asv_consistency` is a hard-coded 0.5 for every call and the
# target-conditioned half of the system does nothing. The loader lives in
# `vanirakshak/anchor.py` so the benchmark harness loads the same vector into
# the same pipeline and the two report comparable numbers.
ANCHOR_STATE: Dict[str, object] = load_anchor(PIPELINE)


def _load_enrolment_store():
    """docs/15 N4: restore persisted enrolments into the running ASV model.

    A record made by another ASV backend is skipped, not loaded: a stub
    projection means nothing to ECAPA, and the reverse.
    """
    store = store_from_env(settings.enrolment.store_path, settings.enrolment.store_key)
    state: Dict[str, object] = {"persistent": store is not None, "loaded": 0,
                                "skipped_other_backend": 0, "unreadable": 0}
    if store is None:
        return None, state
    report = store.load_all()
    for rec in report.records:
        if rec.asv_backend != PIPELINE.asv.name:
            state["skipped_other_backend"] = int(state["skipped_other_backend"]) + 1
            continue
        PIPELINE.asv.restore_enrolment(rec.user_id, rec.embedding, rec.calibration)
        state["loaded"] = int(state["loaded"]) + 1
    state["unreadable"] = len(report.unreadable)
    if report.unreadable:
        print(f"[enrolment-store] {len(report.unreadable)} unreadable record(s) (wrong key?)")
    return store, state


ENROL_STORE, ENROL_STORE_STATE = _load_enrolment_store()
# Non-target cohort for per-user calibration (W5); clips load on first enrol.
ASV_COHORT = Cohort(Path(settings.enrolment.cohort_dir))

# API-key auth. Disabled (and loudly warned about) when VR_API_KEYS is unset,
# so the demo and integration_check keep working; enforced the moment keys
# exist. See server/auth.py for the key format and scope model.
AUTH = Authenticator(
    raw_keys=settings.auth.api_keys,
    require_auth=settings.auth.require_auth,
    rate_limits={
        "analyze": settings.auth.rate_limit_analyze,
        "enroll": settings.auth.rate_limit_enroll,
    },
)
AUTH.warn_if_open()

# W5: in production auth-required deployments, refuse to start on synthetic stub ASV
if settings.auth.require_auth and PIPELINE.asv_is_synthetic:
    raise RuntimeError("VR_REQUIRE_AUTH=1 requires real ASV backend (VR_ASV_BACKEND=ecapa)")


def _third_party_processors() -> List[str]:
    """Who receives caller audio beyond this process (G6). Empty = nobody."""
    return [settings.semantic.provider] if settings.semantic.uploads_audio else []


def _guard(
    request: Request, scope: str, route: str, acting_as: Optional[str] = None
) -> Optional[ApiKey]:
    """Authenticate + authorize, translating AuthError into HTTPException."""
    try:
        key = AUTH.authenticate(extract_secret(request.headers))
        AUTH.authorize(key, scope, route, acting_as=acting_as)
        return key
    except AuthError as e:
        raise HTTPException(status_code=e.status, detail=e.detail) from None


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
    # W5: "fitted_loo_vs_cohort", or "seeded_from_anchor" with a reason in
    # the stored QA summary; None when no calibration applies.
    calibration_method: Optional[str] = None


class EnrollDeleteResponse(BaseModel):
    status: str = "deleted"
    user_id: str
    erased_at: str
    receipt_hash: str
    # docs/15 N4. None when no persistent store is configured.
    store_record_erased: Optional[bool] = None
    # Live sessions claiming this identity, ended by the erasure.
    sessions_closed: List[str] = Field(default_factory=list)


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
    # G7: whether this run's ASV backend is the synthetic stub. Lets the
    # dashboard label mean_speaker_sim as a demo projection, not a voiceprint.
    asv_synthetic: Optional[bool] = None
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
    # Graded only under VR_CHALLENGE_CLIENT_TEXT_DRILL=1. Otherwise the server
    # transcribes the response audio itself (docs/15 N2) and ignores this.
    response_text: str = ""
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
            # G6: false whenever the semantic layer is enabled with a cloud
            # (or unrecognised) provider, because it then uploads caller audio
            # to a third party. A local provider keeps audio on the box. Declared from config rather
            # than from whether a call has happened yet, so it fails toward
            # disclosure: a server configured to transfer says so before the
            # first byte leaves.
            "zero_retention": not settings.semantic.uploads_audio,
            "third_party_audio_processors": _third_party_processors(),
            "cm_backend": PIPELINE.cm.name,
            "asv_backend": PIPELINE.asv.name,
            # G7: true when asv_backend is the synthetic random-projection stub,
            # i.e. every asv_consistency the dashboard renders is not a
            # voiceprint. Read from the running instance, so it stays truthful
            # even if VR_ASV_BACKEND=ecapa was requested but degraded to stub.
            "asv_synthetic": PIPELINE.asv_is_synthetic,
        },
        # docs/15 N2: where challenge digits are graded from. "server_asr"
        # transcribes the response audio with a local engine and fails closed
        # without one; "client_text_drill" trusts the client and is forgeable.
        "enrolment_store": dict(ENROL_STORE_STATE),
        "challenge_grading": {
            "source": "client_text_drill" if settings.challenge.client_text_drill else "server_asr",
            "asr_provider": settings.challenge.asr_provider,
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
def list_enrolled(request: Request) -> EnrolledSpeakersResponse:
    _guard(request, "enroll", "enroll_list")
    """Enrolled user_ids (family members / anchors) — lets the dashboard offer
    a claimed-identity picker instead of a single hardcoded anchor."""
    return EnrolledSpeakersResponse(
        enrolled=PIPELINE.asv.enrolled_user_ids(),
        asv_backend=PIPELINE.asv.name,
    )


@app.post("/v1/enroll", response_model=EnrollResponse)
@app.post("/api/v1/enroll", response_model=EnrollResponse)
async def enroll(
    request: Request,
    user_id: str = Form(...),
    audio: UploadFile = File(...),
    consent_ref: Optional[str] = Form(None),
) -> EnrollResponse:
    # acting_as=user_id is the important part: enrolling a voiceprint under
    # someone else's user_id is an identity attack, not a misconfiguration.
    _guard(request, "enroll", "enroll", acting_as=user_id)
    data = await audio.read()
    x, sr = _read_wav_bytes(data)
    if x.size < int(0.5 * sr):
        raise HTTPException(400, "enrolment audio too short (need ≥ 0.5 s)")

    # W5: Enrolment QA checks (voiced seconds ≥ 3.0s, SNR floor ≥ 10 dB, CM score ≥ 0.0)
    # Applied when authenticating or when real audio is supplied (bypass if VR_ENROLL_BYPASS_QA=1)
    bypass_qa = os.environ.get("VR_ENROLL_BYPASS_QA", "").strip().lower() in ("1", "true", "yes")
    qa: Dict[str, object] = {"bypassed": bypass_qa, "duration_s": round(float(x.size) / float(sr), 3)}
    if not bypass_qa and x.size >= int(0.5 * sr):
        v_stats = vad_stats(x, sr)
        total_duration = float(x.size) / float(sr)
        voiced_duration = v_stats["speech_ratio"] * total_duration
        if voiced_duration < 3.0 and total_duration >= 3.0:
            raise HTTPException(400, f"insufficient voiced speech in enrolment: {voiced_duration:.2f}s (need ≥ 3.0s)")

        ch_stats = channel_detect(x, sr)
        if ch_stats.get("snr_db", 0.0) < 10.0 and total_duration >= 3.0:
            raise HTTPException(400, f"enrolment audio SNR too low: {ch_stats.get('snr_db', 0.0):.1f} dB (need ≥ 10.0 dB)")

        cm_score = PIPELINE.cm.score(x, sr)
        if cm_score < 0.0:
            raise HTTPException(400, f"enrolment audio rejected: countermeasure score indicates synthetic voice (LLR={cm_score:.2f})")
        qa.update(voiced_s=round(voiced_duration, 3),
                  snr_db=round(float(ch_stats.get("snr_db", 0.0)), 2),
                  cm_llr=round(float(cm_score), 4))

    # Re-use or create a session for enrolment
    sid = f"enroll-{uuid.uuid4().hex[:8]}"
    sess = CallSession(sid, SessionContext(caller_id="enrolment", claimed_identity=user_id), audit=AUDIT, pipeline=PIPELINE)
    sess.enrol(user_id, x, sr)
    SESSIONS[sid] = sess

    # W5: per-user calibration, fitted from leave-one-out target scores
    # against a fixed non-target cohort. Falls back to the anchor's logistic
    # when there is too little audio or no cohort, and records which.
    cal, cal_info = await asyncio.to_thread(
        fit_user_calibration, PIPELINE.asv, user_id, x, sr, ASV_COHORT
    )
    if cal is not None:
        PIPELINE.asv._user_calibrations[user_id] = cal
    elif PIPELINE.asv.calibration:
        PIPELINE.asv._user_calibrations[user_id] = dict(PIPELINE.asv.calibration)
    else:
        cal_info["method"] = None
    qa["calibration"] = cal_info

    # docs/15 N4: persist, or undo. An enrolment that silently would not
    # survive a restart is the G-D defect this store exists to close.
    if ENROL_STORE is not None:
        emb, cal = PIPELINE.asv.get_enrolment(user_id)
        try:
            ENROL_STORE.save(EnrolmentRecord(
                user_id=user_id, embedding=emb, asv_backend=PIPELINE.asv.name,
                calibration=cal, qa=qa, consent_ref=consent_ref,
            ))
        except Exception as e:
            PIPELINE.unenroll(user_id)
            raise HTTPException(500, f"enrolment not persisted ({type(e).__name__}); nothing was kept")

    return EnrollResponse(
        status="enrolled",
        user_id=user_id,
        embedding_dimension=192,
        audio_retained=False,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        calibration_method=cal_info.get("method"),
    )


@app.delete("/v1/enroll/{user_id}", response_model=EnrollDeleteResponse)
@app.delete("/api/v1/enroll/{user_id}", response_model=EnrollDeleteResponse)
def delete_enrollment(request: Request, user_id: str) -> EnrollDeleteResponse:
    """DPDP Act 2023 Sec 12: Right to correction and erasure of personal biometric data.

    Permanently erases the voiceprint embedding and per-user calibration for
    user_id, from memory and from the persistent store, and ends every live
    session claiming that identity (docs/15 N4). Logs an immutable receipt to
    the cryptographic audit chain.
    """
    key = _guard(request, "enroll", "enroll", acting_as=user_id)
    in_store = ENROL_STORE is not None and ENROL_STORE.contains(user_id)
    if not PIPELINE.is_enrolled(user_id) and not in_store:
        raise HTTPException(404, f"enrolled voiceprint for user '{user_id}' not found")

    PIPELINE.unenroll(user_id)
    store_erased: Optional[bool] = None
    if ENROL_STORE is not None:
        ENROL_STORE.erase(user_id)
        store_erased = not ENROL_STORE.contains(user_id)
        if not store_erased:
            raise HTTPException(500, "voiceprint erased from memory but not from disk")

    # A session opened before erasure would otherwise keep scoring against
    # a claim that no longer exists (G-E). End it; a WS stream is closed on
    # its next message.
    closed = sorted(
        sid for sid, s in list(SESSIONS.items()) if s.context.claimed_identity == user_id
    )
    for sid in closed:
        sess = SESSIONS.pop(sid, None)
        if sess is not None:
            sess.revoke("biometric_erasure")

    # Immutable receipt chained into DPDP audit log
    ev = AUDIT.append(
        session_id=f"erase-{uuid.uuid4().hex[:8]}",
        caller_id=key.key_id if key else "direct_api",
        claimed_identity=user_id,
        risk_score=0.0,
        tier="ERASED",
        features={
            "biometric_erasure": 1.0,
            "dpdp_section": 12.0,
            "store_record_erased": -1.0 if store_erased is None else float(store_erased),
            "sessions_closed": float(len(closed)),
        },
    )

    return EnrollDeleteResponse(
        status="deleted",
        user_id=user_id,
        erased_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        receipt_hash=ev.receipt_hash,
        store_record_erased=store_erased,
        sessions_closed=closed,
    )


@app.post("/v1/analyze", response_model=AnalyzeResponse)
@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
@app.post("/v1/analyze/audio", response_model=AnalyzeResponse)
@app.post("/api/v1/analyze/audio", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    audio: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    caller_id: str = Form("+91-0000000000"),
    claimed_identity: str = Form(""),
) -> AnalyzeResponse:
    # claimed_identity may be "" here; the anchor fallback resolves it below,
    # and an empty string means "no identity claimed", not "any identity".
    _guard(request, "analyze", "analyze",
           acting_as=claimed_identity or None)
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

    # Feed initial window (+ drain subsequent hops), timing each push so
    # the per-window report carries a measured inference_ms instead of a
    # hardcoded constant. push_chunk returns None until a hop-window
    # accumulates, so only pushes that extend sess.history record a timing.
    hop = sess.buffer.hop_samples
    inference_ms: List[float] = []

    def _timed_push(chunk: np.ndarray) -> None:
        before = len(sess.history)
        t0 = time.perf_counter()
        sess.push_chunk(chunk)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if len(sess.history) > before:
            inference_ms.append(round(dt_ms, 2))

    _timed_push(x_analysis[:sess.buffer.window_samples])

    i = sess.buffer.window_samples
    while i + hop <= x_analysis.size:
        _timed_push(x_analysis[i : i + hop])
        i += hop

    if not sess.history:
        raise HTTPException(400, "Unable to extract forensic analysis window")

    # Configurable window pooling: 'last' (default), 'max' (peak risk window),
    # or 'top_k' (highest risk window among top windows).
    pooling_mode = getattr(settings.risk, "analyze_pooling", "last")
    if pooling_mode == "max":
        last = max(sess.history, key=lambda cr: cr.risk)
    elif pooling_mode.startswith("top_"):
        try:
            k = int(pooling_mode.split("_")[1])
        except (IndexError, ValueError):
            k = 3
        sorted_history = sorted(sess.history, key=lambda cr: cr.risk, reverse=True)
        # Select peak of top-k (or top-k representative)
        last = sorted_history[0]
    else:
        last = sess.history[-1]

    # A BLOCK anywhere in the file is a BLOCK for the file. The live WebSocket
    # path latches BLOCK one-way (session.py sets _blocked and never clears
    # it), so a call that blocked in any window stays blocked. /analyze runs the
    # same CallSession but used to report only the pooled window, which let a
    # single borderline window override decisive ones: demo_04_impostor under
    # the real ECAPA backend scored speaker similarity [0.044, 0.060, 0.055,
    # 0.067, 0.092, 0.135] -- five windows under VR_IMPOSTOR_BLOCK_BELOW=0.10 --
    # and was reported CHALLENGE because pooling='last' picked the 0.135 one.
    # Pooling still chooses the representative window when nothing blocked.
    if sess.blocked and last.tier != "BLOCK":
        last = max((cr for cr in sess.history if cr.tier == "BLOCK"), key=lambda cr: cr.risk)

    windows: List[dict] = []
    hop_sec = sess.buffer.cfg.hop_seconds
    win_sec = sess.buffer.cfg.window_seconds

    win_samples = sess.buffer.window_samples
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

        # Measure VAD + ZCR on this window's actual audio slice
        # (history[k] scores samples [k*hop, k*hop+window) of x_analysis,
        # padding included — the same signal the fusion saw).
        seg_start = idx * hop
        seg = x_analysis[seg_start : seg_start + win_samples]
        vad = vad_stats(seg, sr)
        zcr_var = zcr_variance(seg, sr)

        windows.append({
            "window_index": idx + 1,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": round(win_sec, 2),
            "is_voiced": vad["is_voiced"],
            "speech_ratio": vad["speech_ratio"],
            "acoustic_anomaly_score": round(anomaly, 4),
            "speaker_cosine_sim": round(asv_val, 4),
            "speaker_evaluated": asv_evaluated_val,
            "speaker_match": bool(asv_evaluated_val and asv_val >= 0.5),
            "micro_tremor_ratio": round(max(0.0, min(1.0, 1.0 - prosody_val)), 4),
            "spectral_flatness_mean": round(channel_val, 4),
            "high_band_flatness": round(channel_val, 4),
            "zcr_variance": zcr_var,
            "vocoder_boundary_score": round(anomaly, 4),
            "inference_ms": inference_ms[idx] if idx < len(inference_ms) else 0.0,
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

    zcr_vars = [w["zcr_variance"] for w in windows if w["zcr_variance"] is not None]
    mean_zcr_var = round(float(np.mean(zcr_vars)), 4) if zcr_vars else 0.0

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
    #
    # DECIDED — two different impostor cutoffs, deliberately: the 0.40 below is
    # the report-level identity *label* ("not the enrolled voice"), while the
    # per-window *interlock* override (VR_IMPOSTOR_BLOCK_BELOW, default 0.10
    # in config.py, enforced in risk/fusion.py) only fires on decisive
    # mismatch. A clip in [0.10, 0.40) is therefore labelled IMPOSTOR_SPEAKER
    # while its tier stays CHALLENGE — label states identity, tier states
    # interlock action. See docs/RUNBOOK.md Known Limitations.
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
        # G6: a claim of zero retention is only true when no audio left the
        # box. With a cloud semantic provider it went to a named third party.
        "zero_retention_verified": not settings.semantic.uploads_audio,
        "third_party_audio_processors": _third_party_processors(),
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
            "asv_synthetic": PIPELINE.asv_is_synthetic,
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
        asv_synthetic=PIPELINE.asv_is_synthetic,
        mean_speaker_sim=mean_speaker_sim,
        mean_micro_tremor=mean_micro_tremor,
        mean_spectral_flatness=mean_spectral_flatness,
        mean_high_band_flatness=mean_spectral_flatness,
        mean_zcr_variance=mean_zcr_var,
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
async def verify_speaker(request: Request, 
    audio: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
) -> VerifySpeakerResponse:
    _guard(request, "analyze", "verify_speaker")
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
def start_session(request: Request, req: SessionStartRequest) -> dict:
    _guard(request, "stream", "sessions")
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
def grade_challenge(request: Request, sid: str, req: ChallengeGradeRequest) -> dict:
    _guard(request, "stream", "challenge")
    sess = SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(404, "session not found")
    return sess.grade_challenge(req.response_text, req.response_latency_ms)


@app.get("/v1/audit")
@app.get("/api/v1/audit")
def tail_audit(request: Request, n: int = 50) -> dict:
    # 'audit' is deliberately absent from the default scope set: this is a log
    # of who was scored and when.
    _guard(request, "audit", "audit")
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

    # Auth on the handshake. A browser cannot set headers on a WebSocket, so
    # the key may travel in the handshake body as "api_key"; a header is still
    # accepted for non-browser clients. Closed with 1008 (policy violation),
    # distinct from the 1003 above, which means "first frame was not JSON".
    presented = handshake.get("api_key") or extract_secret(ws.headers)
    claimed = _resolve_claimed_identity(str(handshake.get("claimed_identity", "")))
    try:
        key = AUTH.authenticate(presented)
        AUTH.authorize(key, "stream", "stream", acting_as=claimed or None)
    except AuthError as e:
        # Never echo the detail of a 401 back over the socket beyond its reason
        # string; it is already non-specific about which key ids exist.
        await ws.close(code=1008, reason=e.detail[:120])
        return

    sid = handshake.get("session_id") or f"ws-{uuid.uuid4().hex[:8]}"
    sr = int(handshake.get("sample_rate", 16000))
    ctx = SessionContext(
        caller_id=str(handshake.get("caller_id", "")),
        claimed_identity=claimed,
        transaction_value_inr=float(handshake.get("transaction_value_inr", 0.0)),
        origin_country=str(handshake.get("origin_country", "IN")),
        prior_fraud_score=float(handshake.get("prior_fraud_score", 0.0)),
    )
    sess = CallSession(sid, ctx, audit=AUDIT, pipeline=PIPELINE)
    sess.set_sample_rate(sr)
    SESSIONS[sid] = sess

    # Handshake ack: always tell the client the resolved session_id (the
    # client may have omitted it, in which case sid is server-generated).
    # Without this, grading a challenge over REST afterwards is impossible.
    # Clients must accept this non-verdict message before verdicts arrive;
    # it carries no tier/risk_score so verdict parsers skip it by shape.
    await ws.send_json({"type": "session_ack", "session_id": sid, "sample_rate": sr})

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if sess.revoked is not None:
                # docs/15 N4: the claimed identity was erased mid-call.
                await ws.send_json({"type": "session_closed", "session_id": sid,
                                    "reason": sess.revoked})
                await ws.close(code=1000, reason=sess.revoked)
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
