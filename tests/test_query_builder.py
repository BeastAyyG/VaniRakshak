"""
Test query builder — query generation correctness.
"""

from backend.schemas.claims import ClaimType, DetectedClaim, StructuredClaim
from backend.services.query_builder import build_query


def _make_parsed_claim(
    text: str,
    claim_type: ClaimType,
    location: str | None = None,
    condition: str | None = None,
    organization: str | None = None,
    event: str | None = None,
    time_reference: str | None = None,
) -> DetectedClaim:
    return DetectedClaim(
        claim_id="test_q",
        start=1.0,
        end=5.0,
        text=text,
        claim_type=claim_type,
        structured_claim=StructuredClaim(
            claim_type=claim_type,
            location=location,
            condition=condition,
            organization=organization,
            event=event,
            time_reference=time_reference,
        ),
    )


class TestQueryBuilder:

    def test_weather_query(self):
        claim = _make_parsed_claim(
            text="It is raining heavily in Delhi right now.",
            claim_type=ClaimType.WEATHER,
            location="Delhi",
            condition="heavy rain",
            time_reference="current",
        )
        sq = build_query(claim)
        assert "delhi" in sq.query.lower()
        assert "weather" in sq.query.lower()
        assert sq.topic == "general"  # weather uses general
        assert sq.time_range == "day"  # current → day

    def test_finance_query(self):
        claim = _make_parsed_claim(
            text="RBI increased the repo rate yesterday.",
            claim_type=ClaimType.FINANCE,
            organization="RBI",
            event="repo_rate_increase",
            time_reference="yesterday",
        )
        sq = build_query(claim)
        assert "rbi" in sq.query.lower()
        assert "repo" in sq.query.lower()
        assert sq.topic == "finance"
        assert sq.time_range == "day"

    def test_no_full_transcript_sent(self):
        """Query should NOT contain the full transcript text."""
        claim = _make_parsed_claim(
            text="This is a very long transcript segment that talks about many things including the weather in Delhi.",
            claim_type=ClaimType.WEATHER,
            location="Delhi",
            condition="rain",
            time_reference="current",
        )
        sq = build_query(claim)
        # Query should be focused, not the full text
        assert len(sq.query) < len(claim.text)
