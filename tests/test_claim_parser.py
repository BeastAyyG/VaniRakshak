"""
Test claim parser — structuring correctness.
"""

from backend.schemas.claims import ClaimType, ClaimCertainty, DetectedClaim, StructuredClaim
from backend.services.claim_parser import parse_claim


def _make_claim(text: str, claim_type: ClaimType = ClaimType.WEATHER) -> DetectedClaim:
    """Helper to create a minimal DetectedClaim for testing."""
    return DetectedClaim(
        claim_id="test_001",
        start=1.0,
        end=5.0,
        text=text,
        claim_type=claim_type,
        structured_claim=StructuredClaim(claim_type=claim_type),
    )


class TestClaimParser:

    def test_weather_location_extraction(self):
        claim = _make_claim("It is raining heavily in Delhi right now.", ClaimType.WEATHER)
        parsed = parse_claim(claim)
        assert parsed.structured_claim.location is not None
        assert "delhi" in parsed.structured_claim.location.lower()

    def test_weather_condition_extraction(self):
        claim = _make_claim("It is raining heavily in Delhi right now.", ClaimType.WEATHER)
        parsed = parse_claim(claim)
        assert parsed.structured_claim.condition is not None
        assert "rain" in parsed.structured_claim.condition.lower()

    def test_time_reference_current(self):
        claim = _make_claim("It is raining in Delhi right now.", ClaimType.WEATHER)
        parsed = parse_claim(claim)
        assert parsed.structured_claim.time_reference == "current"

    def test_time_reference_yesterday(self):
        claim = _make_claim("RBI increased the repo rate yesterday.", ClaimType.FINANCE)
        parsed = parse_claim(claim)
        assert parsed.structured_claim.time_reference == "yesterday"

    def test_org_extraction(self):
        claim = _make_claim("RBI increased the repo rate yesterday.", ClaimType.FINANCE)
        parsed = parse_claim(claim)
        assert parsed.structured_claim.organization is not None
        assert "rbi" in parsed.structured_claim.organization.lower()

    def test_event_extraction(self):
        claim = _make_claim("RBI increased the repo rate yesterday.", ClaimType.FINANCE)
        parsed = parse_claim(claim)
        assert parsed.structured_claim.event is not None
        assert "repo_rate" in parsed.structured_claim.event

    def test_timestamps_preserved(self):
        claim = _make_claim("It is raining in Delhi now.", ClaimType.WEATHER)
        parsed = parse_claim(claim)
        assert parsed.start == 1.0
        assert parsed.end == 5.0
        assert parsed.claim_id == "test_001"
