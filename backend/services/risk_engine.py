"""
Risk Engine — computes factual-risk contribution from claim verification.

This is a SIGNAL, not a fraud verdict.
- SUPPORTED    → no factual contradiction → low/no risk contribution
- CONTRADICTED → factual suspicion        → risk contribution
- UNVERIFIABLE → insufficient evidence    → minimal risk contribution

The larger VaniRakshak system will combine this with voice-analysis
and speaker-similarity signals.  This module never says "fraud".
"""

from __future__ import annotations

from backend.schemas.verification import VerificationStatus, VerificationResult
from backend.schemas.risk import RiskLevel, ClaimRiskResult
from backend.config import get_settings


# ---------------------------------------------------------------------------
# Risk contribution per verification status
# ---------------------------------------------------------------------------

def _status_risk(status: VerificationStatus, confidence: float) -> float:
    """
    Map a single claim's verification status to a risk contribution.

    SUPPORTED     → 0.0 (no factual contradiction)
    CONTRADICTED  → 0.3 – 0.9 (scaled by confidence)
    UNVERIFIABLE  → 0.05 – 0.15 (low but non-zero)
    """
    if status == VerificationStatus.SUPPORTED:
        return 0.0

    if status == VerificationStatus.CONTRADICTED:
        # Higher confidence in contradiction → higher risk
        return min(0.9, 0.3 + confidence * 0.6)

    if status == VerificationStatus.UNVERIFIABLE:
        # Small risk contribution — we don't know
        return 0.1

    return 0.0  # fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_claim_risk(results: list[VerificationResult]) -> ClaimRiskResult:
    """
    Aggregate risk from all verified claims.

    Uses weighted average of individual risk contributions.
    Claims with higher confidence have more influence.
    """
    if not results:
        return ClaimRiskResult(
            claim_risk_score=0.0,
            risk_level=RiskLevel.NONE,
        )

    total_weight = 0.0
    weighted_risk = 0.0

    for r in results:
        weight = r.verification_confidence
        risk = _status_risk(r.status, r.verification_confidence)

        weighted_risk += risk * weight
        total_weight += weight

    if total_weight == 0:
        score = 0.0
    else:
        score = round(weighted_risk / total_weight, 3)

    # Classify risk level
    settings = get_settings()
    if score < settings.risk_threshold_low:
        level = RiskLevel.NONE
    elif score < settings.risk_threshold_medium:
        level = RiskLevel.LOW
    elif score < settings.risk_threshold_high:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.HIGH

    return ClaimRiskResult(
        claim_risk_score=score,
        risk_level=level,
    )


def compute_single_risk(result: VerificationResult) -> float:
    """Compute risk contribution for a single claim."""
    return round(_status_risk(result.status, result.verification_confidence), 3)
