"""
Tests 6, 7 — Pipeline-level tests.

Test 6: Multiple claims → independent detection, queries, results.
Test 7: Timestamp preservation through the pipeline.
"""

from backend.schemas.transcript import TranscriptSegment
from backend.schemas.verification import VerificationStatus
from backend.services.verification_pipeline import VerificationPipeline
from backend.services.evidence_provider import WebResult
from tests.conftest import MockEvidenceProvider


class TestMultipleClaims:
    """Test 6: Multiple claims are detected and verified independently."""

    def test_two_claims_independent(self, mock_provider: MockEvidenceProvider):
        """
        Input: two factual claims in separate segments.
        Expected: two independent claims, two queries, two results.
        """
        mock_provider.set_response("delhi", [
            WebResult(
                title="Delhi Weather",
                url="https://example.com/weather",
                content="Heavy rain is currently reported in Delhi.",
                score=0.9,
            ),
        ])
        mock_provider.set_response("rbi", [
            WebResult(
                title="RBI Rate Decision",
                url="https://example.com/rbi",
                content="RBI has increased the repo rate by 25 basis points.",
                score=0.85,
            ),
        ])

        pipeline = VerificationPipeline(evidence_provider=mock_provider)
        result = pipeline.run([
            TranscriptSegment(
                start=4.0,
                end=8.0,
                text="It is raining heavily in Delhi right now.",
            ),
            TranscriptSegment(
                start=10.0,
                end=14.0,
                text="RBI increased the repo rate yesterday.",
            ),
        ])

        # Should detect at least 2 claims
        assert len(result.verified_claims) >= 2

        # Each should have its own claim_id
        ids = [vc.claim.claim_id for vc in result.verified_claims]
        assert len(set(ids)) == len(ids), "Claim IDs should be unique"

        # Each should have a query
        queries = [vc.query for vc in result.verified_claims]
        assert all(q for q in queries), "Each claim should have a query"


class TestTimestampPreservation:
    """Test 7: Timestamps survive the full pipeline."""

    def test_timestamps_preserved(self, mock_provider: MockEvidenceProvider):
        """
        Verify start/end remain attached to every claim after verification.
        """
        mock_provider.set_response("delhi", [
            WebResult(
                title="Weather",
                url="https://example.com",
                content="Rain in Delhi reported.",
                score=0.9,
            ),
        ])

        pipeline = VerificationPipeline(evidence_provider=mock_provider)
        result = pipeline.run([
            TranscriptSegment(
                start=14.2,
                end=18.7,
                text="It is raining heavily in Delhi right now.",
            ),
        ])

        assert len(result.verified_claims) >= 1

        for vc in result.verified_claims:
            assert vc.claim.start == 14.2, f"Start should be 14.2, got {vc.claim.start}"
            assert vc.claim.end == 18.7, f"End should be 18.7, got {vc.claim.end}"
            assert vc.claim.text, "Claim text should be preserved"

    def test_multiple_timestamps(self, mock_provider: MockEvidenceProvider):
        """Each claim keeps its own timestamp, not the other's."""
        pipeline = VerificationPipeline(evidence_provider=mock_provider)
        result = pipeline.run([
            TranscriptSegment(start=1.0, end=3.0, text="It is raining in Delhi now."),
            TranscriptSegment(start=5.0, end=9.0, text="RBI increased the repo rate yesterday."),
        ])

        if len(result.verified_claims) >= 2:
            starts = {vc.claim.start for vc in result.verified_claims}
            assert len(starts) > 1, "Different claims should have different timestamps"
