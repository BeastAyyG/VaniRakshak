"""Session orchestrator.

A :class:`CallSession` is the single per-call state container: it owns
the audio buffer, the detection pipeline, the risk fusion engine, the
challenge engine, the audit log, and exposes a single
:meth:`CallSession.push_chunk` that the WebSocket / REST endpoints call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import AudioConfig, ChallengeConfig, RiskConfig, FusionConfig, resolve_risk_config, settings
from .privacy.audit import AuditLog
from .privacy.buffer import SlidingAudioBuffer
from .detectors.pipeline import DetectionPipeline
from .risk.fusion import RiskFusionEngine, RiskDecision
from .challenge.engine import ChallengeEngine, Challenge


@dataclass
class SessionContext:
    caller_id: str = ""
    claimed_identity: str = ""
    transaction_value_inr: float = 0.0
    origin_country: str = "IN"
    prior_fraud_score: float = 0.0  # 0-1, comes from CRM
    metadata: Dict[str, str] = field(default_factory=dict)

    def risk_weight(self) -> float:
        w = 0.0
        if self.origin_country and self.origin_country not in ("IN", "USA", "UK", "SG"):
            w += 0.4
        if self.transaction_value_inr >= 1_00_000:  # ≥ ₹1 lakh
            w += 0.5
        elif self.transaction_value_inr >= 10_000:  # ≥ ₹10k
            w += 0.2
        w += 0.8 * max(0.0, min(1.0, self.prior_fraud_score))
        return float(min(1.0, w))


_SEMANTIC_IDLE: Dict[str, object] = {
    "available": False,
    "transcript": "",
    "hinglish_text": "",
    "language": None,
    "matches": [],
    "categories_hit": [],
    "semantic_risk": 0.0,
    "keyword_risk": 0.0,
    "claim_risk": 0.0,
    "claim_findings": [],
    "provider": None,
    "reason": "not started",
}


@dataclass
class ChunkResult:
    timestamp_ms: int
    risk: float
    p_spoof: float
    tier: str
    metrics: Dict[str, object]
    trigger_challenge: bool
    interlock_active: bool
    receipt_hash: str
    challenge: Optional[dict] = None  # only present the first time a challenge fires
    semantic: Dict[str, object] = field(default_factory=lambda: dict(_SEMANTIC_IDLE))


class CallSession:
    """Per-call stateful orchestrator."""

    def __init__(
        self,
        session_id: str,
        context: SessionContext,
        *,
        audio_cfg: Optional[AudioConfig] = None,
        risk_cfg: Optional[RiskConfig] = None,
        fusion_cfg: Optional[FusionConfig] = None,
        pipeline: Optional[DetectionPipeline] = None,
        fusion_engine: Optional[RiskFusionEngine] = None,
        challenge_engine: Optional[ChallengeEngine] = None,
        audit: Optional[AuditLog] = None,
        challenge_cfg: Optional[ChallengeConfig] = None,
    ) -> None:
        self.session_id = session_id
        self.context = context
        self.buffer = SlidingAudioBuffer(audio_cfg or settings.audio)
        self.pipeline = pipeline or DetectionPipeline()
        if risk_cfg is None:
            risk_cfg = resolve_risk_config(
                settings.risk, getattr(getattr(self.pipeline, "asv", None), "impostor_threshold", None)
            )
        self.fusion = fusion_engine or RiskFusionEngine(risk_cfg, fusion_cfg or settings.fusion)
        self.challenge = challenge_engine or ChallengeEngine()
        self.audit = audit or AuditLog()
        self._active_challenge: Optional[Challenge] = None
        self._history: List[ChunkResult] = []
        self._triggered_challenge = False
        self._challenge_passed = False
        self._challenge_cfg = challenge_cfg or settings.challenge
        self._challenges_issued = 0
        # Set once the tier drops back to ALLOW after a passed challenge; a
        # re-arm needs that drop so a caller hovering in CHALLENGE isn't
        # re-challenged every window.
        self._rearm_ready = False
        # (cm_llr, asv_score, asv_enrolled) of each window scored while a
        # challenge is outstanding — the caller's response audio.
        self._challenge_windows: List[Tuple[float, float, bool]] = []
        self._challenge_streak = 0
        self._blocked = False
        self._sr = settings.audio.sample_rate
        # Parallel semantic (ASR + keyword) context signal — see
        # vanirakshak/semantic/. Updated asynchronously by the WS layer
        # (Groq calls are network I/O and must not block push_chunk), so a
        # given window's fusion sees the *previous* window's transcript.
        # That one-window lag is an accepted trade-off for keeping this a
        # non-blocking, advisory signal rather than a bottleneck.
        self._semantic_state: Dict[str, object] = dict(_SEMANTIC_IDLE)
        self._pending_window: Optional[np.ndarray] = None

    @property
    def blocked(self) -> bool:
        return self._blocked

    @property
    def sr(self) -> int:
        return self._sr

    def set_sample_rate(self, sr: int) -> None:
        if sr != self._sr:
            self._sr = sr
            # re-create buffer with new rate
            cfg = AudioConfig(sample_rate=sr, window_seconds=settings.audio.window_seconds, hop_seconds=settings.audio.hop_seconds)
            self.buffer = SlidingAudioBuffer(cfg)

    def enrol(self, user_id: str, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        sr = sr or self._sr
        return self.pipeline.enrol(user_id, audio, sr)

    def is_enrolled(self, user_id: str) -> bool:
        return self.pipeline.is_enrolled(user_id)

    def push_chunk(self, pcm: np.ndarray) -> Optional[ChunkResult]:
        """Push a chunk of int16 / float32 mono PCM. Returns a result
        once a new hop-window has accumulated, else None.
        """
        ready = self.buffer.push(pcm)
        if not ready:
            return None
        window = self.buffer.read_window()
        if window is None:
            return None

        # Hand this window's raw int16 audio to the WS layer so it can
        # schedule a non-blocking ASR pass (see pop_pending_window()).
        self._pending_window = window.copy()

        # detection
        wr = self.pipeline.analyse(
            window.astype(np.float32) / 32768.0,
            self._sr,
            claimed_user=self.context.claimed_identity or None,
        )
        # fuse — acoustic evidence stays primary; semantic_risk is the
        # advisory ASR/keyword term from the most recently completed window.
        decision = self.fusion.update(
            wr,
            context_risk=self.context.risk_weight(),
            semantic_risk=float(self._semantic_state.get("semantic_risk", 0.0)),
        )

        # challenge logic
        if self._active_challenge is not None:
            self._challenge_windows.append((float(wr.cm_score), float(wr.asv_score), bool(wr.asv_enrolled)))
        if decision.tier == "ALLOW" and self._challenge_passed:
            self._rearm_ready = True
        self._challenge_streak = self._challenge_streak + 1 if decision.tier == "CHALLENGE" else 0
        trigger = False
        first = not self._triggered_challenge
        rearm = (
            self._rearm_ready
            and not self._blocked
            and self._active_challenge is None
            and self._challenges_issued < self._challenge_cfg.max_per_session
        )
        if (first or rearm) and self._challenge_streak >= self._challenge_cfg.trigger_consecutive:
            ch = self.challenge.issue(lang="mixed")
            self._active_challenge = ch
            self._triggered_challenge = True
            self._challenges_issued += 1
            self._challenge_passed = False
            self._rearm_ready = False
            self._challenge_windows = []
            trigger = True

        # block logic: if BLOCK, latch
        if decision.tier == "BLOCK":
            self._blocked = True

        ev = self.audit.append(
            session_id=self.session_id,
            caller_id=self.context.caller_id,
            claimed_identity=self.context.claimed_identity,
            risk_score=decision.risk,
            tier=decision.tier,
            features={
                **decision.features,
                **({f"override_{decision.override}": 1.0} if decision.override else {}),
            },
        )

        cr = ChunkResult(
            timestamp_ms=int(time.time() * 1000),
            risk=decision.risk,
            p_spoof=decision.p_spoof,
            tier=decision.tier,
            metrics={
                "cm_score": wr.cm_score,
                "cm_genuine_prob": wr.features.get("cm_genuine_prob", 0.5),
                "asv_consistency": wr.asv_score,
                "asv_evaluated": wr.asv_enrolled,
                # Whether the ASV backend behind asv_consistency is the
                # synthetic random-projection stub rather than real biometrics
                # (G7). Always sent so the dashboard can label the number
                # honestly — never let a projection read as a voiceprint.
                "asv_synthetic": self.pipeline.asv_is_synthetic,
                "channel_anomaly": wr.channel_anomaly,
                "prosody_unnatural": wr.prosody_unnatural,
                "snr_db": wr.snr_db,
                "tier_override": decision.override,
            },
            trigger_challenge=trigger,
            interlock_active=self._blocked,
            receipt_hash=ev.receipt_hash,
            challenge=self._active_challenge.to_dict() if (trigger and self._active_challenge is not None) else None,
            semantic=dict(self._semantic_state),
        )
        cr.metrics["codec_guess"] = wr.codec_guess
        self._history.append(cr)
        return cr

    def pop_pending_window(self) -> Optional[tuple[np.ndarray, int]]:
        """Consume the raw audio for the window just scored, for async ASR.

        Returns ``None`` once already consumed for this window, so the WS
        layer never schedules the same window's transcription twice.
        """
        if self._pending_window is None:
            return None
        window, self._pending_window = self._pending_window, None
        return window, self._sr

    def update_semantic(self, result: Dict[str, object]) -> None:
        """Called by the WS layer once a background ASR pass completes."""
        self._semantic_state = result

    def grade_challenge(self, response_text: str, response_latency_ms: int) -> dict:
        if self._active_challenge is None:
            return {"ok": False, "reason": "no_active_challenge"}
        passed, reason, conf = self.challenge.grade(self._active_challenge, response_text, response_latency_ms)
        acoustic = self._score_challenge_audio()
        if passed and acoustic["verdict"] != "pass":
            passed, reason, conf = False, acoustic["verdict"], 0.05

        # DECIDED (Option B — one-shot gate, not a recovery path): a pass
        # only prevents an escalation that hasn't happened yet; it never
        # clears `_blocked` and never re-arms `_triggered_challenge`.
        # Rationale: grading checks digit order + 400-4000ms latency only
        # (challenge/engine.py::grade) — no voice biometrics — so it cannot
        # override strong acoustic evidence (risk>=70 or asv<0.10). Fail
        # escalates to BLOCK permanently; pass leaves the latch as-is.
        # The challenge is consumed on first grading to prevent oracle
        # probing; a second grade returns `no_active_challenge`.
        # AMENDED (docs/13 §1.2): the response audio's CM/ASV evidence is now
        # scored too (_score_challenge_audio), and VR_CHALLENGE_MAX_PER_SESSION
        # > 1 re-arms after a pass. A pass still never clears `_blocked`.
        if passed:
            self._challenge_passed = True
        else:
            # failed challenge => escalate to BLOCK
            self._blocked = True
        self._active_challenge = None
        self._challenge_windows = []

        last = self._history[-1] if self._history else None
        self.audit.append(
            session_id=self.session_id,
            caller_id=self.context.caller_id,
            claimed_identity=self.context.claimed_identity,
            risk_score=last.risk if last else 0.0,
            tier="BLOCK" if not passed else (last.tier if last else "ALLOW"),
            features={
                "challenge_passed": 1.0 if passed else 0.0,
                "challenge_confidence": float(conf),
                "response_latency_ms": float(response_latency_ms),
                "challenge_response_windows": float(acoustic["windows"]),
            },
        )
        return {
            "ok": passed,
            "reason": reason,
            "confidence": conf,
            "blocked": self._blocked,
            "acoustic": acoustic,
        }

    def _score_challenge_audio(self) -> dict:
        """Grade the response audio scored while the challenge was outstanding.

        Verdict is ``"pass"`` unless a configured threshold is crossed:
        ``response_voice_synthetic`` (mean CM LLR too low),
        ``response_voice_mismatch`` (mean ASV of enrolled windows too low) or
        ``no_response_audio`` (fewer than ``min_windows`` and audio required).
        """
        cfg = self._challenge_cfg
        wins = self._challenge_windows
        out: Dict[str, object] = {
            "windows": len(wins),
            "checked": False,
            "mean_cm_llr": None,
            "mean_asv": None,
            "verdict": "pass",
        }
        if len(wins) < cfg.min_windows:
            if cfg.require_audio:
                out["verdict"] = "no_response_audio"
            return out
        out["checked"] = True
        mean_llr = float(np.mean([w[0] for w in wins]))
        out["mean_cm_llr"] = round(mean_llr, 4)
        enrolled = [w[1] for w in wins if w[2]]
        mean_asv = float(np.mean(enrolled)) if enrolled else None
        out["mean_asv"] = round(mean_asv, 4) if mean_asv is not None else None
        if cfg.cm_llr_max is not None and mean_llr <= cfg.cm_llr_max:
            out["verdict"] = "response_voice_synthetic"
        elif cfg.asv_min is not None and mean_asv is not None and mean_asv < cfg.asv_min:
            out["verdict"] = "response_voice_mismatch"
        return out

    @property
    def history(self) -> List[ChunkResult]:
        return list(self._history)
