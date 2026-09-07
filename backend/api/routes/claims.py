"""
Claims API routes — factual claim verification endpoint.

POST /api/v1/claims/verify
  Accepts transcript segments, returns verified claims with evidence.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.schemas.transcript import TranscriptSegment
from backend.schemas.claims import ClaimType
from backend.schemas.verification import VerificationStatus
from backend.schemas.risk import RiskLevel
from backend.services.verification_pipeline import VerificationPipeline
from backend.services.evidence_provider import EvidenceProviderError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/claims", tags=["claims"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    """Request body for claim verification."""

    segments: list[TranscriptSegment] = Field(
        ...,
        min_length=1,
        description="Timestamped transcript segments to analyze.",
    )


class EvidenceResponse(BaseModel):
    title: str
    url: str
    content: str
    source_type: str
    retrieved_at: str | None = None
    relevance_score: float | None = None


class VerificationResponse(BaseModel):
    status: VerificationStatus
    verification_confidence: float
    reason: str
    query: str
    evidence: list[EvidenceResponse]


class StructuredClaimResponse(BaseModel):
    location: str | None = None
    condition: str | None = None
    organization: str | None = None
    entity: str | None = None
    event: str | None = None
    time_reference: str | None = None
    certainty: str | None = None


class ClaimResponse(BaseModel):
    claim_id: str
    start: float
    end: float
    text: str
    claim_type: ClaimType
    structured_claim: StructuredClaimResponse
    verification: VerificationResponse
    risk_contribution: float


class RiskResponse(BaseModel):
    claim_risk_score: float
    risk_level: RiskLevel


class VerifyResponse(BaseModel):
    """Response from claim verification endpoint."""

    transcript: list[TranscriptSegment]
    claims: list[ClaimResponse]
    risk: RiskResponse


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify factual claims in transcript",
    description=(
        "Accepts timestamped transcript segments, detects factual claims, "
        "verifies them against current web evidence using Tavily, "
        "and returns structured verification results."
    ),
)
async def verify_claims(request: VerifyRequest) -> VerifyResponse:
    """
    Full claim verification pipeline.

    1. Detect factual claims in transcript
    2. Structure and classify each claim
    3. Search for web evidence via Tavily
    4. Compare claim against evidence
    5. Return SUPPORTED / CONTRADICTED / UNVERIFIABLE with evidence
    """
    try:
        pipeline = VerificationPipeline()
        result = pipeline.run(request.segments)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EvidenceProviderError as exc:
        logger.error("Evidence provider error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Web evidence retrieval failed: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error in claim verification")
        raise HTTPException(
            status_code=500,
            detail="Internal error during claim verification.",
        )

    # Build response
    claims_response = []
    for vc in result.verified_claims:
        sc = vc.claim.structured_claim
        ev_responses = [
            EvidenceResponse(
                title=e.title,
                url=e.url,
                content=e.content,
                source_type=e.source_type.value,
                retrieved_at=e.retrieved_at.isoformat() if e.retrieved_at else None,
                relevance_score=e.relevance_score,
            )
            for e in vc.verification.evidence
        ]

        claims_response.append(
            ClaimResponse(
                claim_id=vc.claim.claim_id,
                start=vc.claim.start,
                end=vc.claim.end,
                text=vc.claim.text,
                claim_type=vc.claim.claim_type,
                structured_claim=StructuredClaimResponse(
                    location=sc.location,
                    condition=sc.condition,
                    organization=sc.organization,
                    entity=sc.entity,
                    event=sc.event,
                    time_reference=sc.time_reference,
                    certainty=sc.certainty.value if sc.certainty else None,
                ),
                verification=VerificationResponse(
                    status=vc.verification.status,
                    verification_confidence=vc.verification.verification_confidence,
                    reason=vc.verification.reason,
                    query=vc.verification.query,
                    evidence=ev_responses,
                ),
                risk_contribution=vc.risk_contribution,
            )
        )

    return VerifyResponse(
        transcript=request.segments,
        claims=claims_response,
        risk=RiskResponse(
            claim_risk_score=result.risk.claim_risk_score,
            risk_level=result.risk.risk_level,
        ),
    )
