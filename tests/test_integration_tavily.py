"""
Test 40 — Real Tavily integration test.

Proves: Python → Tavily SDK → Tavily API → real results → normalization.

SKIPPED unless TAVILY_API_KEY is set in environment.
Does NOT expose the API key in output.
"""

import os
import pytest

from backend.services.evidence_provider import TavilyEvidenceProvider
from backend.services.evidence_normalizer import normalize_evidence


# Skip if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("TAVILY_API_KEY") and not os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", ".env")
    ),
    reason="TAVILY_API_KEY not set — skipping real integration test",
)


class TestTavilyIntegration:

    def test_real_search_returns_results(self):
        """
        Real Tavily search for Delhi weather.
        Should return at least one result with title, url, content.
        """
        try:
            provider = TavilyEvidenceProvider()
        except ValueError:
            pytest.skip("TAVILY_API_KEY not configured")

        results = provider.search(
            query="current weather Delhi India",
            topic="general",
            time_range="day",
            max_results=3,
        )

        # Should get at least one result
        assert len(results) >= 1, "Tavily should return at least one result"

        # Each result should have content
        for r in results:
            assert r.title or r.content, "Result should have title or content"
            assert r.content, "Result should have content"

        # Normalization should work
        evidence = normalize_evidence(results)
        assert len(evidence) >= 1
        for ev in evidence:
            assert ev.title
            assert ev.content
            assert ev.retrieved_at is not None

    def test_real_search_finance(self):
        """
        Real search for RBI repo rate.
        """
        try:
            provider = TavilyEvidenceProvider()
        except ValueError:
            pytest.skip("TAVILY_API_KEY not configured")

        results = provider.search(
            query="RBI repo rate latest",
            topic="finance",
            time_range="month",
            max_results=3,
        )

        assert len(results) >= 1
        evidence = normalize_evidence(results)
        assert len(evidence) >= 1
