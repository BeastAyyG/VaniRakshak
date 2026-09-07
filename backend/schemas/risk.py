"""
Risk schema — factual risk contribution from claim verification.

This module's risk score is a *signal*, not a fraud verdict.
The larger VaniRakshak system will combine it with voice-analysis
and speaker-similarity signals.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Categorical risk level derived from the numeric score."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ClaimRiskResult(BaseModel):
    """Aggregated factual-risk contribution from all verified claims."""

    claim_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregated factual risk score (0.0–1.0).",
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorical risk level.",
    )
