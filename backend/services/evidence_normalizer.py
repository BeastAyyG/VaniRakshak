"""
Evidence Normalizer — converts raw WebResult into project Evidence schema.

Normalizes Tavily's internal format immediately so the rest of the system
never depends on Tavily's API structure.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from backend.schemas.evidence import Evidence, SourceType
from backend.services.evidence_provider import WebResult


# ---------------------------------------------------------------------------
# Source type classification
# ---------------------------------------------------------------------------

_OFFICIAL_DOMAINS = {
    "gov.in", ".gov", ".nic.in", "rbi.org.in", "sebi.gov.in",
    "pib.gov.in", "mha.gov.in", "meity.gov.in",
}

_NEWS_DOMAINS = {
    "ndtv.com", "thehindu.com", "hindustantimes.com",
    "indianexpress.com", "timesofindia.indiatimes.com",
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
    "economictimes.indiatimes.com", "livemint.com",
    "moneycontrol.com", "cnbc.com", "bloomberg.com",
    "theguardian.com", "washingtonpost.com", "nytimes.com",
    "aljazeera.com", "news18.com", "aajtak.in",
}

_WEATHER_DOMAINS = {
    "weather.com", "accuweather.com", "wunderground.com",
    "timeanddate.com", "imd.gov.in", "mausam.imd.gov.in",
}


def _classify_source(url: str) -> SourceType:
    """Classify a URL into a SourceType."""
    url_lower = url.lower()

    for domain in _OFFICIAL_DOMAINS:
        if domain in url_lower:
            return SourceType.OFFICIAL

    for domain in _NEWS_DOMAINS:
        if domain in url_lower:
            return SourceType.NEWS

    for domain in _WEATHER_DOMAINS:
        if domain in url_lower:
            return SourceType.OFFICIAL  # Weather services as official

    if url:
        return SourceType.WEB

    return SourceType.UNKNOWN


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_evidence(results: list[WebResult]) -> list[Evidence]:
    """
    Convert a list of raw WebResult objects into normalized Evidence.

    - Strips empty/irrelevant results
    - Classifies source type
    - Sets retrieval timestamp
    - Does NOT fabricate any content
    """
    now = datetime.now(timezone.utc)
    evidence_list: list[Evidence] = []

    for r in results:
        # Skip results with no useful content
        content = (r.content or "").strip()
        if not content:
            continue

        evidence_list.append(
            Evidence(
                title=(r.title or "Untitled").strip(),
                url=(r.url or "").strip(),
                content=content,
                source_type=_classify_source(r.url or ""),
                retrieved_at=now,
                relevance_score=r.score,
            )
        )

    return evidence_list
