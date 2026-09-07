"""
Test risk engine — risk calculation correctness.
"""

from backend.schemas.verification import VerificationStatus, VerificationResult
from backend.schemas.risk import RiskLevel
from backend.services.risk_engine import compute_claim_risk, compute_single_risk


def _make_result(status: VerificationStatus, confidence: float = 0.8) -> VerificationResult:
    return VerificationResult(
        status=status,
        verification_confidence=confidence,
        reason="Test",
        query="test query",
        evidence=[],
    )


class TestRiskEngine:

    def test_supported_zero_risk(self):
        """SUPPORTED claim → 0.0 risk contribution."""
        r = _make_result(VerificationStatus.SUPPORTED, 0.9)
        risk = compute_single_risk(r)
        assert risk == 0.0

    def test_contradicted_has_risk(self):
        """CONTRADICTED claim → non-zero risk contribution."""
        r = _make_result(VerificationStatus.CONTRADICTED, 0.9)
        risk = compute_single_risk(r)
        assert risk > 0.3

    def test_unverifiable_low_risk(self):
        """UNVERIFIABLE → small but non-zero risk."""
        r = _make_result(VerificationStatus.UNVERIFIABLE, 0.5)
        risk = compute_single_risk(r)
        assert 0.0 < risk <= 0.2

    def test_aggregate_no_claims(self):
        """No claims → NONE risk level."""
        result = compute_claim_risk([])
        assert result.risk_level == RiskLevel.NONE
        assert result.claim_risk_score == 0.0

    def test_aggregate_all_supported(self):
        """All supported → NONE risk level."""
        results = [
            _make_result(VerificationStatus.SUPPORTED, 0.9),
            _make_result(VerificationStatus.SUPPORTED, 0.8),
        ]
        risk = compute_claim_risk(results)
        assert risk.risk_level == RiskLevel.NONE

    def test_aggregate_with_contradiction(self):
        """One contradicted claim → elevated risk."""
        results = [
            _make_result(VerificationStatus.SUPPORTED, 0.9),
            _make_result(VerificationStatus.CONTRADICTED, 0.85),
        ]
        risk = compute_claim_risk(results)
        assert risk.claim_risk_score > 0.0
        assert risk.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)
