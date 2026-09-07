"""
Tests 1, 2, 3 — Claim verification correctness.

Test 1: SUPPORTED (evidence agrees)
Test 2: CONTRADICTED (evidence conflicts)
Test 3: UNVERIFIABLE (private claim)
"""

from backend.schemas.claims import ClaimType, DetectedClaim, StructuredClaim, ClaimCertainty
from backend.schemas.evidence import Evidence, SourceType
from backend.schemas.verification import VerificationStatus
from backend.services.claim_verifier import verify_claim

from datetime import datetime, timezone


def _make_weather_claim(text: str = "It is raining heavily in Delhi right now.") -> DetectedClaim:
    return DetectedClaim(
        claim_id="test_v",
        start=4.0,
        end=8.0,
        text=text,
        claim_type=ClaimType.WEATHER,
        structured_claim=StructuredClaim(
            claim_type=ClaimType.WEATHER,
            location="Delhi",
            condition="heavy rain",
            time_reference="current",
            certainty=ClaimCertainty.ASSERTED,
        ),
    )


def _make_private_claim() -> DetectedClaim:
    return DetectedClaim(
        claim_id="test_private",
        start=10.0,
        end=14.0,
        text="Your personal bank account has been frozen.",
        claim_type=ClaimType.GENERAL_FACT,
        structured_claim=StructuredClaim(
            claim_type=ClaimType.GENERAL_FACT,
            certainty=ClaimCertainty.ASSERTED,
            extra={"is_private": True},
        ),
    )


def _make_evidence(content: str, title: str = "Test Source", url: str = "https://example.com") -> Evidence:
    return Evidence(
        title=title,
        url=url,
        content=content,
        source_type=SourceType.WEB,
        retrieved_at=datetime.now(timezone.utc),
    )


class TestClaimVerifier:

    # === Test 1 — SUPPORTED ===
    def test_supported_weather_claim(self):
        """
        Claim: heavy rain in Delhi.
        Evidence: confirms heavy rain in Delhi.
        Expected: SUPPORTED.
        """
        claim = _make_weather_claim()
        evidence = [
            _make_evidence("Heavy rain is currently reported in Delhi. IMD has issued a warning."),
            _make_evidence("Delhi is experiencing heavy rain and waterlogging."),
        ]
        result = verify_claim(claim, evidence, "current weather Delhi heavy rain")
        assert result.status == VerificationStatus.SUPPORTED
        assert result.verification_confidence > 0.5

    # === Test 2 — CONTRADICTED ===
    def test_contradicted_weather_claim(self):
        """
        Claim: heavy rain in Delhi.
        Evidence: Delhi is clear, no precipitation.
        Expected: CONTRADICTED.
        """
        claim = _make_weather_claim()
        evidence = [
            _make_evidence("Delhi is currently clear with no precipitation. The sky is sunny."),
            _make_evidence("No rain in Delhi today. Clear and dry conditions persist."),
        ]
        result = verify_claim(claim, evidence, "current weather Delhi heavy rain")
        assert result.status == VerificationStatus.CONTRADICTED
        assert result.verification_confidence > 0.5

    # === Test 3 — UNVERIFIABLE (private claim) ===
    def test_unverifiable_private_claim(self):
        """
        Claim: personal bank account frozen.
        Expected: UNVERIFIABLE (private info cannot be verified).
        """
        claim = _make_private_claim()
        evidence = []  # No evidence for private claims
        result = verify_claim(claim, evidence, "bank account frozen")
        assert result.status == VerificationStatus.UNVERIFIABLE

    # === Additional: no evidence → UNVERIFIABLE, NOT CONTRADICTED ===
    def test_no_evidence_is_unverifiable(self):
        """
        No search results should NEVER be CONTRADICTED.
        """
        claim = _make_weather_claim()
        result = verify_claim(claim, [], "current weather Delhi heavy rain")
        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.status != VerificationStatus.CONTRADICTED

    # === Additional: conflicting evidence → UNVERIFIABLE ===
    def test_conflicting_evidence(self):
        """
        Mixed evidence (some support, some contradict) → UNVERIFIABLE or lower confidence.
        """
        claim = _make_weather_claim()
        evidence = [
            _make_evidence("Heavy rain reported in Delhi NCR today."),  # supports
            _make_evidence("Delhi is clear and sunny today with no rain."),  # contradicts
        ]
        result = verify_claim(claim, evidence, "current weather Delhi heavy rain")
        # Should not force certainty — either UNVERIFIABLE or reduced confidence
        assert result.verification_confidence < 0.9
