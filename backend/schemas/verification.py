"""
Verification schema — the outcome of comparing a claim against evidence.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from backend.schemas.evidence import Evidence


class VerificationStatus(str, Enum):
    """
    Primary verdict for a factual claim.

    SUPPORTED    — available evidence agrees with the claim.
    CONTRADICTED — available evidence conflicts with the claim.
    UNVERIFIABLE — insufficient evidence to determine truth.
    """

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


class VerificationResult(BaseModel):
    """Result of verifying a single claim against web evidence."""

    status: VerificationStatus = Field(
        ...,
        description="Primary verification verdict.",
    )
    verification_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the verification result (0.0–1.0).",
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of the verdict.",
    )
    query: str = Field(
        ...,
        description="The search query that was sent to the evidence provider.",
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence items used to reach the verdict.",
    )
