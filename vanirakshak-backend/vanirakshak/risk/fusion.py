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
        return RiskDecision(
            risk=round(risk, 2),
            p_spoof=round(p, 4),
            tier=tier,
            raw_window=window,
            ema_risk=round(self._ema if self._ema is not None else risk, 2),
            features=window.features,
        )
