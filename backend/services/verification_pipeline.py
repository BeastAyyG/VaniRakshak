"""
Verification Pipeline — orchestrates the full claim verification flow.

Transcript → Detect → Parse → Query → Search → Normalize → Verify → Risk

Handles:
- Multiple claims independently
- Deduplication of near-identical claims
- Graceful failure for individual claims
- Evidence provider errors → UNVERIFIABLE (never CONTRADICTED)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.schemas.transcript import TranscriptSegment
from backend.schemas.claims import DetectedClaim
from backend.schemas.evidence import Evidence
from backend.schemas.verification import VerificationStatus, VerificationResult
from backend.schemas.risk import ClaimRiskResult

from backend.services.claim_detector import detect_claims, split_into_sentences
from backend.services.claim_parser import parse_claim
from backend.services.query_builder import build_query, SearchQuery
from backend.services.evidence_provider import (
    WebEvidenceProvider,
    TavilyEvidenceProvider,
    EvidenceProviderError,
)
from backend.services.evidence_normalizer import normalize_evidence
from backend.services.claim_verifier import verify_claim
from backend.services.risk_engine import compute_claim_risk, compute_single_risk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class VerifiedClaim:
    """A claim that has been through the full verification pipeline."""

    claim: DetectedClaim
    query: str
    verification: VerificationResult
    risk_contribution: float


@dataclass
class PipelineResult:
    """Complete result of the verification pipeline."""

    segments: list[TranscriptSegment]
    verified_claims: list[VerifiedClaim]
    risk: ClaimRiskResult


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class VerificationPipeline:
    """
    Orchestrates claim verification from transcript segments to final result.
    """

    def __init__(
        self,
        evidence_provider: WebEvidenceProvider | None = None,
    ):
        """
        Initialize the pipeline.

        If no evidence_provider is given, creates a TavilyEvidenceProvider
        from environment settings.
        """
        if evidence_provider is not None:
            self._provider = evidence_provider
        else:
            self._provider = TavilyEvidenceProvider()

    def run(self, segments: list[TranscriptSegment]) -> PipelineResult:
        """
        Run the full verification pipeline.

        Steps:
        1. Convert segments to sentence dicts
        2. Detect factual claims
        3. Parse / structure each claim
        4. Build search queries
        5. Search via evidence provider
        6. Normalize evidence
        7. Verify each claim
        8. Compute risk
        """
        # Step 1: Prepare sentences
        sentence_dicts = self._segments_to_sentences(segments)

        # Step 2: Detect claims
        raw_claims = detect_claims(sentence_dicts)
        logger.info("Detected %d factual claim(s)", len(raw_claims))

        # Step 3: Parse / structure
        parsed_claims = [parse_claim(c) for c in raw_claims]

        # Step 4-7: Verify each claim independently
        verified: list[VerifiedClaim] = []
        seen_queries: dict[str, VerifiedClaim] = {}

        for claim in parsed_claims:
            # Build query
            sq = build_query(claim)

            # Deduplication: skip if we already searched a very similar query
            normalized_q = sq.query.strip().lower()
            if normalized_q in seen_queries:
                logger.info(
                    "Deduplicating query '%s' for claim '%s'",
                    sq.query,
                    claim.claim_id,
                )
                # Reuse the existing result
                existing = seen_queries[normalized_q]
                vc = VerifiedClaim(
                    claim=claim,
                    query=sq.query,
                    verification=existing.verification,
                    risk_contribution=existing.risk_contribution,
                )
                verified.append(vc)
                continue

            # Search + verify
            vc = self._verify_single(claim, sq)
            verified.append(vc)
            seen_queries[normalized_q] = vc

        # Step 8: Aggregate risk
        verification_results = [vc.verification for vc in verified]
        risk = compute_claim_risk(verification_results)

        return PipelineResult(
            segments=segments,
            verified_claims=verified,
            risk=risk,
        )

    def _verify_single(
        self,
        claim: DetectedClaim,
        sq: SearchQuery,
    ) -> VerifiedClaim:
        """Verify a single claim: search → normalize → compare."""

        # Search
        try:
            raw_results = self._provider.search(
                query=sq.query,
                topic=sq.topic,
                time_range=sq.time_range,
            )
        except EvidenceProviderError as exc:
            logger.warning(
                "Evidence provider failed for claim '%s': %s",
                claim.claim_id,
                exc,
            )
            # NEVER convert provider failure to CONTRADICTED
            result = VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verification_confidence=0.0,
                reason=f"Web search failed: {exc}",
                query=sq.query,
                evidence=[],
            )
            return VerifiedClaim(
                claim=claim,
                query=sq.query,
                verification=result,
                risk_contribution=compute_single_risk(result),
            )

        # Normalize
        evidence = normalize_evidence(raw_results)

        # Verify
        result = verify_claim(claim, evidence, sq.query)

        return VerifiedClaim(
            claim=claim,
            query=sq.query,
            verification=result,
            risk_contribution=compute_single_risk(result),
        )

    @staticmethod
    def _segments_to_sentences(
        segments: list[TranscriptSegment],
    ) -> list[dict]:
        """
        Convert TranscriptSegment list to the sentence dicts expected
        by the claim detector.

        If a segment contains multiple sentences, split them and
        distribute timestamps proportionally.
        """
        result: list[dict] = []

        for seg in segments:
            sentences = split_into_sentences(seg.text)

            if len(sentences) <= 1:
                result.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                })
            else:
                # Distribute timestamps proportionally by character length
                total_len = sum(len(s) for s in sentences)
                if total_len == 0:
                    continue

                duration = seg.end - seg.start
                current = seg.start

                for sent_text in sentences:
                    frac = len(sent_text) / total_len
                    sent_end = current + duration * frac
                    result.append({
                        "start": round(current, 2),
                        "end": round(sent_end, 2),
                        "text": sent_text,
                    })
                    current = sent_end

        return result
