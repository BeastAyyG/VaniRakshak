"""Calibrated Bayesian risk fusion (Layer 5).

Implements the formulation in ``docs/02-technical-strategy-and-architecture.md``
(equation for P(spoof|x)).

The fusion maps a 5-dim evidence vector into a calibrated posterior
probability of spoofing, applies an exponential moving average over
time, and emits a 0-100 risk score + ALLOW/CHALLENGE/BLOCK tier.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from ..config import FusionConfig, RiskConfig, settings
from ..detectors.pipeline import WindowResult


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def calibrate(window: WindowResult, context_risk: float, fusion: FusionConfig) -> float:
    """Return P(spoof | features, context) in [0, 1]."""
    cm_spoof = 1.0 - window.features.get("cm_genuine_prob", 0.5)
    # The ASV term counts only when there is something to compare against.
    #
    # Three cases, and they are NOT the same:
    #   1. caller claims an enrolled identity -> real score, use it.
    #   2. caller claims an identity nobody enrolled -> unverifiable; the ASV
    #      returns the neutral 0.5 and the pre-existing policy of treating
    #      that as a half-mismatch is kept deliberately (an unverifiable claim
    #      is legitimately weaker evidence than a verified one).
    #   3. caller claims nothing at all -> the ASV never ran. Scoring the
    #      neutral 0.5 as a 0.5 mismatch added a flat ``w_asv * 0.5`` (0.8 at
    #      the default weight) to every anonymous caller, enough on its own to
    #      move genuine audio from ALLOW (risk ~27) to CHALLENGE (risk ~37).
    #      Absence of identity evidence is not evidence of spoofing.
    #
    # ``asv_claimed`` is what separates 2 from 3: both return (0.5, False).
    asv_mismatch = (
        (1.0 - window.asv_score)
        if (window.asv_enrolled or getattr(window, "asv_claimed", False))
        else 0.0
    )
    prosody_unn = window.prosody_unnatural
    chan_anom = window.channel_anomaly
    z = (
        fusion.intercept
        + fusion.w_cm * cm_spoof
        + fusion.w_asv * asv_mismatch
        + fusion.w_prosody * prosody_unn
        + fusion.w_channel * chan_anom
        + fusion.w_context * context_risk
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

    def reset(self) -> None:
        self._ema = None

    def update(self, window: WindowResult, context_risk: float = 0.0) -> RiskDecision:
        p = calibrate(window, context_risk, self.fusion_cfg)
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
        cutoff = getattr(self.risk_cfg, "impostor_block_below", 0.0)
        if cutoff > 0.0 and window.asv_score < cutoff:
            # Raise the score to the BLOCK line as well, so the reported risk and
            # the reported tier agree. A tier of BLOCK at risk 67 would contradict
            # the published thresholds and make the dashboard unreadable.
            floor = float(self.risk_cfg.challenge_threshold)
            risk = max(risk, floor)
            self._ema = max(self._ema if self._ema is not None else risk, floor)
            tier = "BLOCK"

        return RiskDecision(
            risk=round(risk, 2),
            p_spoof=round(p, 4),
            tier=tier,
            raw_window=window,
            ema_risk=round(self._ema if self._ema is not None else risk, 2),
            features=window.features,
        )
