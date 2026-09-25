"""Calibrated Bayesian risk fusion (Layer 5).

Implements the formulation in ``docs/02-technical-strategy-and-architecture.md``
(equation for P(spoof|x)).

The fusion maps a 5-dim evidence vector into a calibrated posterior
probability of spoofing, applies an exponential moving average over
time, and emits a 0-100 risk score + ALLOW/CHALLENGE/BLOCK tier.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque, Dict, List, Optional

import numpy as np

from ..config import FusionConfig, RiskConfig, settings
from ..detectors.pipeline import WindowResult


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def calibrate(
    window: WindowResult,
    context_risk: float,
    fusion: FusionConfig,
    semantic_risk: float = 0.0,
) -> float:
    """Return P(spoof | features, context) in [0, 1].

    ``semantic_risk`` is the ASR/keyword scam-phrase signal from
    ``vanirakshak/semantic/`` (see docs/PROTOCOL.md's ``semantic`` verdict
    field) — an advisory context term, not an acoustic detector. It defaults
    to 0.0 so a caller that never supplies it reproduces the original
    acoustic-only fusion exactly.
    """
    cm_spoof = 1.0 - window.features.get("cm_genuine_prob", 0.5)
    asv_mismatch = 1.0 - window.asv_score  # high when not enrolled or not matching
    prosody_unn = window.prosody_unnatural
    chan_anom = window.channel_anomaly
    z = (
        fusion.intercept
        + fusion.w_cm * cm_spoof
        + fusion.w_asv * asv_mismatch
        + fusion.w_prosody * prosody_unn
        + fusion.w_channel * chan_anom
        + fusion.w_context * context_risk
        + fusion.w_semantic * max(0.0, min(1.0, semantic_risk))
    )
    return _sigmoid(z)


def to_tier(risk: float, rc: RiskConfig) -> str:
    if risk < rc.allow_threshold:
        return "ALLOW"
    if risk < rc.challenge_threshold:
        return "CHALLENGE"
    return "BLOCK"


@dataclass
class RiskDecision:
    risk: float             # 0-100
    p_spoof: float          # 0-1
    tier: str               # ALLOW | CHALLENGE | BLOCK
    raw_window: WindowResult
    ema_risk: float
    features: Dict[str, float]
    # Which deterministic rule forced BLOCK, if any: "impostor" | "clone".
    override: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["raw_window"] = self.raw_window.to_dict()
        return d


class RiskFusionEngine:
    """Stateful: holds the EMA across windows."""

    def __init__(
        self,
        risk_cfg: Optional[RiskConfig] = None,
        fusion_cfg: Optional[FusionConfig] = None,
    ) -> None:
        self.risk_cfg = risk_cfg or settings.risk
        self.fusion_cfg = fusion_cfg or settings.fusion
        self._ema: float | None = None
        self._clone_llrs: Deque[float] = deque(maxlen=max(1, int(getattr(self.risk_cfg, "clone_min_windows", 1))))

    def reset(self) -> None:
        self._ema = None
        self._clone_llrs.clear()

    def _clone_rule_fires(self, window: WindowResult) -> bool:
        k = getattr(self.risk_cfg, "clone_cm_llr_max", None)
        if k is None:
            return False
        # Only a real comparison against an enrolled voice can show "matches the
        # voiceprint"; the unenrolled 0.5 placeholder says nothing either way.
        if not window.asv_enrolled or window.asv_score < getattr(self.risk_cfg, "clone_asv_min", 0.0):
            self._clone_llrs.clear()
            return False
        self._clone_llrs.append(float(window.cm_score))
        if len(self._clone_llrs) < self._clone_llrs.maxlen:
            return False
        return float(np.mean(self._clone_llrs)) <= k

    def _force_block(self, risk: float) -> float:
        # Raise the score to the BLOCK line as well, so the reported risk and
        # the reported tier agree. A tier of BLOCK at risk 67 would contradict
        # the published thresholds and make the dashboard unreadable.
        floor = float(self.risk_cfg.challenge_threshold)
        self._ema = max(self._ema if self._ema is not None else risk, floor)
        return max(risk, floor)

    def update(
        self,
        window: WindowResult,
        context_risk: float = 0.0,
        semantic_risk: float = 0.0,
    ) -> RiskDecision:
        p = calibrate(window, context_risk, self.fusion_cfg, semantic_risk)
        # fast-rise override: if the posterior is decisively spoof, ignore the EMA
        if p > self.risk_cfg.fast_rise_threshold:
            risk = 100.0 * p
            self._ema = risk
        else:
            risk = 100.0 * p
            if self._ema is None:
                self._ema = risk
            else:
                self._ema = self.risk_cfg.ema_alpha * risk + (1.0 - self.risk_cfg.ema_alpha) * self._ema
                risk = self._ema
        risk = float(max(0.0, min(100.0, risk)))
        tier = to_tier(risk, self.risk_cfg)

        # Deterministic impostor override — see RiskConfig.impostor_block_below.
        # A caller whose voice is decisively NOT the enrolled speaker must engage
        # the interlock even if the weighted fusion lands short of BLOCK. Only
        # applies when an anchor actually produced the score: an unenrolled ASV
        # returns the neutral 0.5, which is never below the threshold.
        override: Optional[str] = None
        cutoff = getattr(self.risk_cfg, "impostor_block_below", 0.0)
        if cutoff > 0.0 and window.asv_score < cutoff:
            risk = self._force_block(risk)
            tier = "BLOCK"
            override = "impostor"

        # Deterministic clone override — see RiskConfig.clone_cm_llr_max.
        if self._clone_rule_fires(window) and override is None:
            risk = self._force_block(risk)
            tier = "BLOCK"
            override = "clone"

        return RiskDecision(
            risk=round(risk, 2),
            p_spoof=round(p, 4),
            tier=tier,
            raw_window=window,
            ema_risk=round(self._ema if self._ema is not None else risk, 2),
            features={**window.features, "semantic_risk": round(float(max(0.0, min(1.0, semantic_risk))), 4)},
            override=override,
        )
