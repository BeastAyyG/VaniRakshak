"""Session simulator (docs/13 §1.4) — torch-free, no audio.

Friction and security are properties of a *call*, not of a 4 s window: one
low-CM window can flip a whole session to CHALLENGE, and the BLOCK latch is
one-way. This replays per-window detector scores through the real
:class:`~vanirakshak.session.CallSession` (fusion, EMA, impostor + clone
overrides, challenge issue/re-arm, BLOCK latch) by swapping the detection
pipeline for one that returns pre-computed scores. No logic is duplicated, so
a number from here is a number from the shipped decision layer.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import AudioConfig, ChallengeConfig, FusionConfig, RiskConfig, settings
from ..detectors.pipeline import WindowResult
from ..risk.fusion import RiskFusionEngine
from ..session import CallSession, SessionContext


@dataclass(frozen=True)
class WindowScore:
    cm_llr: float
    asv: float = 0.5
    asv_enrolled: bool = False
    channel_anomaly: float = 0.0
    prosody_unnatural: float = 0.0

    def to_result(self) -> WindowResult:
        return WindowResult(
            cm_score=float(self.cm_llr),
            asv_score=float(self.asv),
            asv_enrolled=bool(self.asv_enrolled),
            channel_anomaly=float(self.channel_anomaly),
            prosody_unnatural=float(self.prosody_unnatural),
            codec_guess="simulated",
            snr_db=30.0,
            features={"cm_genuine_prob": float(1.0 / (1.0 + np.exp(-self.cm_llr)))},
        )


class _ScriptedPipeline:
    """Stands in for DetectionPipeline; returns the next scripted window."""

    asv_is_synthetic = False

    def __init__(self, windows: Sequence[WindowScore]) -> None:
        self._it = iter(windows)

    def analyse(self, x, sr, claimed_user=None) -> WindowResult:  # noqa: ARG002
        return next(self._it).to_result()


@dataclass
class SessionOutcome:
    n_windows: int
    first_challenge_s: Optional[float]
    first_block_s: Optional[float]
    blocked: bool
    challenges_issued: int
    overrides: Dict[str, int]


def simulate_session(
    windows: Sequence[WindowScore],
    *,
    risk_cfg: Optional[RiskConfig] = None,
    fusion_cfg: Optional[FusionConfig] = None,
    challenge_cfg: Optional[ChallengeConfig] = None,
    audio_cfg: Optional[AudioConfig] = None,
    context: Optional[SessionContext] = None,
    challenge_responder: Optional[Callable[[int], bool]] = None,
    response_windows: int = 3,
) -> SessionOutcome:
    """Replay one call. Window *i* is reported at ``window_s + i * hop_s``.

    ``challenge_responder(issue_index) -> text_ok`` models whether the caller
    says the right digits; the answer is graded ``response_windows`` windows
    after issue, so those windows are scored as the response audio. ``None`` leaves challenges
    unanswered, which is what "P(challenge within T)" friction measures.
    """
    audio_cfg = audio_cfg or settings.audio
    # The scripted pipeline ignores the audio, so the buffer only has to count
    # hops; a 100 Hz buffer keeps the same window/hop timing ~100x cheaper.
    sr = 100
    buf_cfg = AudioConfig(sample_rate=sr, window_seconds=audio_cfg.window_seconds,
                          hop_seconds=audio_cfg.hop_seconds)
    hop = max(1, int(round(audio_cfg.hop_seconds * sr)))
    win = int(round(audio_cfg.window_seconds * sr))

    # The responder stands in for what the caller said, so digits are graded
    # from its text: there is no real audio here to transcribe (docs/15 N2).
    challenge_cfg = dataclasses.replace(challenge_cfg or settings.challenge, client_text_drill=True)
    sess = CallSession(
        "sim",
        context or SessionContext(),
        audio_cfg=buf_cfg,
        pipeline=_ScriptedPipeline(windows),  # type: ignore[arg-type]
        fusion_engine=RiskFusionEngine(risk_cfg or settings.risk, fusion_cfg or settings.fusion),
        challenge_cfg=challenge_cfg,
    )
    first_challenge: Optional[float] = None
    first_block: Optional[float] = None
    overrides: Dict[str, int] = {}
    pending_grade: Optional[int] = None
    issued = 0

    zeros_first = np.zeros(win, dtype=np.int16)
    zeros_hop = np.zeros(hop, dtype=np.int16)
    for i in range(len(windows)):
        chunk = zeros_first if i == 0 else zeros_hop
        cr = sess.push_chunk(chunk)
        while cr is None:
            cr = sess.push_chunk(zeros_hop)
        t = audio_cfg.window_seconds + i * audio_cfg.hop_seconds

        if (
            pending_grade is not None
            and challenge_responder is not None
            and i - pending_grade >= response_windows
        ):
            ch = sess._active_challenge  # noqa: SLF001
            if ch is not None:
                text_ok = challenge_responder(pending_grade)
                # Latency inside the human band; digits right or wrong per the responder.
                sess.grade_challenge(" ".join(ch.expected_digits) if text_ok else "", 900)
            pending_grade = None
        if cr.trigger_challenge:
            issued += 1
            if first_challenge is None:
                first_challenge = t
            pending_grade = i
        ov = cr.metrics.get("tier_override")
        if ov:
            overrides[str(ov)] = overrides.get(str(ov), 0) + 1
        if cr.interlock_active and first_block is None:
            first_block = t

    if sess.blocked and first_block is None:
        first_block = audio_cfg.window_seconds + (len(windows) - 1) * audio_cfg.hop_seconds
    return SessionOutcome(
        n_windows=len(windows),
        first_challenge_s=first_challenge,
        first_block_s=first_block,
        blocked=sess.blocked,
        challenges_issued=issued,
        overrides=overrides,
    )


def summarise(
    outcomes: Iterable[SessionOutcome],
    horizons_s: Sequence[float] = (30.0, 60.0, 120.0),
) -> Dict[str, object]:
    outs: List[SessionOutcome] = list(outcomes)
    n = max(1, len(outs))
    res: Dict[str, object] = {"sessions": len(outs)}
    for h in horizons_s:
        res[f"p_challenge_within_{int(h)}s"] = round(
            sum(1 for o in outs if o.first_challenge_s is not None and o.first_challenge_s <= h) / n, 4
        )
        res[f"p_block_within_{int(h)}s"] = round(
            sum(1 for o in outs if o.first_block_s is not None and o.first_block_s <= h) / n, 4
        )
    res["p_block_by_end"] = round(sum(1 for o in outs if o.blocked) / n, 4)
    res["mean_challenges_per_call"] = round(sum(o.challenges_issued for o in outs) / n, 4)
    ov: Dict[str, int] = {}
    for o in outs:
        for k in o.overrides:
            ov[k] = ov.get(k, 0) + 1
    res["sessions_with_override"] = ov
    return res


def bootstrap_sessions(
    llrs: np.ndarray,
    *,
    n_sessions: int,
    call_seconds: float,
    rng: np.random.Generator,
    asv: float,
    asv_enrolled: bool,
    channel_anomaly: float = 0.0,
    prosody_unnatural: float = 0.0,
    audio_cfg: Optional[AudioConfig] = None,
) -> List[List[WindowScore]]:
    """Build calls by drawing window scores i.i.d. from a pool.

    I.i.d. draws ignore within-call correlation (a genuine speaker who reads
    as synthetic tends to do so all call), which *overstates* how much
    multi-window averaging helps. Prefer per-speaker concatenation when the
    dump carries a speaker column; this is the fallback docs/13 Amendment 1
    anticipates for manifests without one.
    """
    audio_cfg = audio_cfg or settings.audio
    n_win = max(1, int(np.floor((call_seconds - audio_cfg.window_seconds) / audio_cfg.hop_seconds)) + 1)
    out: List[List[WindowScore]] = []
    for _ in range(n_sessions):
        draw = rng.choice(llrs, size=n_win, replace=True)
        out.append([
            WindowScore(float(v), asv, asv_enrolled, channel_anomaly, prosody_unnatural) for v in draw
        ])
    return out
