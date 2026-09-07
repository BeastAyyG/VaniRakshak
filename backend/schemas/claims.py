"""
Claim schemas — types, structured representation, and detected claims.

ClaimType is an extensible enum.  StructuredClaim captures the semantic
fields needed for verification.  DetectedClaim ties a claim back to its
audio timestamps.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Claim Type Enum
# ---------------------------------------------------------------------------

class ClaimType(str, Enum):
    """Extensible classification of factual claim domains."""

    WEATHER = "WEATHER"
    NEWS = "NEWS"
    FINANCE = "FINANCE"
    GOVERNMENT = "GOVERNMENT"
    TRANSPORT = "TRANSPORT"
    LOCATION = "LOCATION"
    TIME = "TIME"
    COMPANY = "COMPANY"
    PERSON = "PERSON"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    LAW = "LAW"
    GENERAL_FACT = "GENERAL_FACT"


# ---------------------------------------------------------------------------
# Certainty
# ---------------------------------------------------------------------------

class ClaimCertainty(str, Enum):
    """How certain the speaker is about the claim."""

    ASSERTED = "ASSERTED"          # stated as fact
    SPECULATIVE = "SPECULATIVE"    # may, might, reportedly
    CONDITIONAL = "CONDITIONAL"    # if, when (hypothetical)
    REPORTED = "REPORTED"          # "they said that …"


# ---------------------------------------------------------------------------
# Structured Claim
# ---------------------------------------------------------------------------

class StructuredClaim(BaseModel):
    """
    Semantic decomposition of a factual claim.

    Fields are optional because not every claim type uses every field.
    """

    claim_type: ClaimType = Field(
        ...,
        description="Domain classification of the claim.",
    )

    # Who / What
    entity: str | None = Field(
        default=None,
        description="Primary entity (person, org, product, etc.).",
    )
    organization: str | None = Field(
        default=None,
        description="Organization referenced in the claim.",
    )

    # Where
    location: str | None = Field(
        default=None,
        description="Geographic location referenced.",
    )

    # What happened / condition
    condition: str | None = Field(
        default=None,
        description="Condition or state described (e.g. 'heavy rain').",
    )
    event: str | None = Field(
        default=None,
        description="Event described (e.g. 'repo rate increase').",
    )

    # When
    time_reference: str | None = Field(
        default=None,
        description="Temporal reference (e.g. 'current', 'yesterday').",
    )

    # Certainty
    certainty: ClaimCertainty = Field(
        default=ClaimCertainty.ASSERTED,
        description="Speaker's expressed certainty.",
    )

    # Catch-all for domain-specific extras
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional domain-specific attributes.",
    )


# ---------------------------------------------------------------------------
# Detected Claim (tied to audio timestamps)
# ---------------------------------------------------------------------------

class DetectedClaim(BaseModel):
    """A factual claim detected from a transcript, anchored to audio time."""

    claim_id: str = Field(
        ...,
        description="Unique identifier for this claim within the analysis.",
    )
    start: float = Field(
        ...,
        ge=0.0,
        description="Start time (seconds) of the claim in the audio.",
    )
    end: float = Field(
        ...,
        ge=0.0,
        description="End time (seconds) of the claim in the audio.",
    )
    text: str = Field(
        ...,
        description="Original transcript text of the claim.",
    )
    claim_type: ClaimType = Field(
        ...,
        description="Domain classification.",
    )
    structured_claim: StructuredClaim = Field(
        ...,
        description="Parsed semantic structure of the claim.",
    )
