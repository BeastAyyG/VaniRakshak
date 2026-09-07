"""
Test 4 — Tavily failure handling.

Simulate: API timeout / error.
Expected: graceful failure → UNVERIFIABLE. NEVER CONTRADICTED.
"""

from backend.services.evidence_provider import EvidenceProviderError
from backend.services.verification_pipeline import VerificationPipeline
from backend.schemas.transcript import TranscriptSegment
from backend.schemas.verification import VerificationStatus
from tests.conftest import MockEvidenceProvider


class TestEvidenceProviderFailure:

    def test_timeout_returns_unverifiable(self, mock_provider: MockEvidenceProvider):
        """
        API timeout should produce UNVERIFIABLE, never CONTRADICTED.
        """
        mock_provider.set_error(
            EvidenceProviderError("Connection timeout after 30s")
        )

        pipeline = VerificationPipeline(evidence_provider=mock_provider)
        result = pipeline.run([
            TranscriptSegment(
                start=4.0,
                end=8.0,
                text="It is raining heavily in Delhi right now.",
            ),
        ])

        # Should have detected the claim
        assert len(result.verified_claims) >= 1

        # Every claim should be UNVERIFIABLE due to provider failure
        for vc in result.verified_claims:
            assert vc.verification.status == VerificationStatus.UNVERIFIABLE
            assert vc.verification.status != VerificationStatus.CONTRADICTED

    def test_auth_error_returns_unverifiable(self, mock_provider: MockEvidenceProvider):
        """
        Authentication error should also be UNVERIFIABLE.
        """
        mock_provider.set_error(
            EvidenceProviderError("Invalid API key")
        )

        pipeline = VerificationPipeline(evidence_provider=mock_provider)
        result = pipeline.run([
            TranscriptSegment(
                start=0.0,
                end=4.0,
                text="RBI increased the repo rate yesterday.",
            ),
        ])

        for vc in result.verified_claims:
            assert vc.verification.status == VerificationStatus.UNVERIFIABLE

    def test_empty_results_not_contradicted(self, mock_provider: MockEvidenceProvider):
        """
        Empty search results (no evidence) → UNVERIFIABLE, not CONTRADICTED.
        """
        # Mock returns empty list (default behavior)
        pipeline = VerificationPipeline(evidence_provider=mock_provider)
        result = pipeline.run([
            TranscriptSegment(
                start=4.0,
                end=8.0,
                text="It is raining heavily in Delhi right now.",
            ),
        ])

        for vc in result.verified_claims:
            assert vc.verification.status != VerificationStatus.CONTRADICTED
