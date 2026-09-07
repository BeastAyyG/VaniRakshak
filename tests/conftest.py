"""
Shared test fixtures and mock providers.

All unit tests use mocked providers — no internet dependency.
"""

from __future__ import annotations

import pytest

from backend.services.evidence_provider import WebResult, EvidenceProviderError


class MockEvidenceProvider:
    """
    Mock web evidence provider for deterministic unit tests.

    Configure with predefined responses per query substring,
    or raise EvidenceProviderError to simulate failures.
    """

    def __init__(self):
        self._responses: dict[str, list[WebResult]] = {}
        self._error: Exception | None = None

    def set_response(self, query_substring: str, results: list[WebResult]):
        """Set response for queries containing the given substring."""
        self._responses[query_substring.lower()] = results

    def set_error(self, error: Exception):
        """Configure the provider to raise this error on search."""
        self._error = error

    def search(
        self,
        query: str,
        topic: str | None = None,
        time_range: str | None = None,
        max_results: int = 5,
    ) -> list[WebResult]:
        if self._error:
            raise self._error

        query_lower = query.lower()

        # Find the best matching response
        for key, results in self._responses.items():
            if key in query_lower:
                return results

        # Default: empty results (NOT an error — just no evidence)
        return []


@pytest.fixture
def mock_provider():
    """Provide a fresh MockEvidenceProvider instance."""
    return MockEvidenceProvider()


@pytest.fixture
def supported_weather_evidence():
    """Evidence that supports 'heavy rain in Delhi'."""
    return [
        WebResult(
            title="Delhi Weather Update",
            url="https://weather.example.com/delhi",
            content="Heavy rain is currently reported in Delhi. The Indian Meteorological Department has issued a warning for continuous heavy rainfall in Delhi NCR region.",
            score=0.95,
        ),
        WebResult(
            title="Delhi Rain Alert",
            url="https://news.example.com/delhi-rain",
            content="Delhi is experiencing heavy rain and waterlogging. Several areas have reported significant precipitation since morning.",
            score=0.88,
        ),
    ]


@pytest.fixture
def contradicted_weather_evidence():
    """Evidence that contradicts 'heavy rain in Delhi'."""
    return [
        WebResult(
            title="Delhi Weather Today",
            url="https://weather.example.com/delhi-today",
            content="Delhi is currently clear with no precipitation. The sky is sunny and temperatures are high. No rain is expected today.",
            score=0.92,
        ),
        WebResult(
            title="Delhi Forecast",
            url="https://forecast.example.com/delhi",
            content="Clear and dry conditions persist in Delhi. There is no rain reported and the weather remains sunny throughout the day.",
            score=0.85,
        ),
    ]
