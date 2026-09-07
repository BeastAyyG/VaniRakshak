"""
Query Builder — converts a structured claim into a focused search query.

Claim-type-aware: generates different query patterns for weather,
finance, government, etc.  Also maps to Tavily-specific parameters
(topic, time_range) so the evidence provider can use them.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.schemas.claims import ClaimType, StructuredClaim, DetectedClaim


@dataclass
class SearchQuery:
    """A search query ready for the evidence provider."""

    query: str
    topic: str | None = None       # Tavily topic: general, news, finance
    time_range: str | None = None   # Tavily time_range: day, week, month, year


# ---------------------------------------------------------------------------
# Claim type → Tavily topic mapping
# ---------------------------------------------------------------------------

_TYPE_TO_TOPIC: dict[ClaimType, str] = {
    ClaimType.WEATHER: "general",
    ClaimType.FINANCE: "finance",
    ClaimType.NEWS: "news",
    ClaimType.GOVERNMENT: "news",
    ClaimType.TRANSPORT: "news",
    ClaimType.COMPANY: "news",
    ClaimType.PERSON: "news",
    ClaimType.EVENT: "news",
    ClaimType.PRODUCT: "news",
    ClaimType.LAW: "news",
    ClaimType.LOCATION: "general",
    ClaimType.TIME: "general",
    ClaimType.GENERAL_FACT: "general",
}

# Time reference → Tavily time_range mapping
_TIME_TO_RANGE: dict[str, str] = {
    "current": "day",
    "today": "day",
    "yesterday": "day",
    "tonight": "day",
    "this_week": "week",
    "last_week": "week",
    "next_week": "week",
    "this_month": "month",
    "last_month": "month",
    "next_month": "month",
    "this_year": "year",
    "last_year": "year",
    "recent": "week",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_query(claim: DetectedClaim) -> SearchQuery:
    """
    Convert a DetectedClaim into a focused SearchQuery.

    Does NOT send the entire transcript — only the structured
    information needed to verify the specific claim.
    """
    sc = claim.structured_claim
    claim_type = sc.claim_type

    # Build query string based on claim type
    query = _build_query_string(sc)

    # Map to Tavily parameters
    topic = _TYPE_TO_TOPIC.get(claim_type, "general")
    time_range = None
    if sc.time_reference:
        time_range = _TIME_TO_RANGE.get(sc.time_reference)

    return SearchQuery(
        query=query,
        topic=topic,
        time_range=time_range,
    )


# ---------------------------------------------------------------------------
# Query string builders per claim type
# ---------------------------------------------------------------------------

def _build_query_string(sc: StructuredClaim) -> str:
    """Build an optimized query string from structured claim fields."""
    parts: list[str] = []

    if sc.claim_type == ClaimType.WEATHER:
        parts = _build_weather_query(sc)
    elif sc.claim_type == ClaimType.FINANCE:
        parts = _build_finance_query(sc)
    elif sc.claim_type in (ClaimType.GOVERNMENT, ClaimType.LAW):
        parts = _build_gov_query(sc)
    elif sc.claim_type in (ClaimType.COMPANY, ClaimType.PRODUCT):
        parts = _build_company_query(sc)
    elif sc.claim_type == ClaimType.TRANSPORT:
        parts = _build_transport_query(sc)
    else:
        parts = _build_general_query(sc)

    query = " ".join(p for p in parts if p)
    return query.strip() or sc.claim_type.value.lower()


def _build_weather_query(sc: StructuredClaim) -> list[str]:
    """Weather: 'current weather Delhi heavy rain'"""
    parts = []
    if sc.time_reference in ("current", "today", None):
        parts.append("current weather")
    else:
        parts.append(f"weather {sc.time_reference}")

    if sc.location:
        parts.append(sc.location)
    if sc.condition:
        parts.append(sc.condition)
    return parts


def _build_finance_query(sc: StructuredClaim) -> list[str]:
    """Finance: 'RBI repo rate increase yesterday'"""
    parts = []
    if sc.organization:
        parts.append(sc.organization)
    if sc.event:
        parts.append(sc.event.replace("_", " "))
    elif sc.condition:
        parts.append(sc.condition)
    if sc.time_reference and sc.time_reference != "current":
        parts.append(sc.time_reference.replace("_", " "))
    return parts


def _build_gov_query(sc: StructuredClaim) -> list[str]:
    """Government / Law: 'government scheme announcement'"""
    parts = []
    if sc.organization:
        parts.append(sc.organization)
    if sc.entity:
        parts.append(sc.entity)
    if sc.event:
        parts.append(sc.event.replace("_", " "))
    if sc.condition:
        parts.append(sc.condition)
    if sc.location:
        parts.append(sc.location)
    if sc.time_reference and sc.time_reference != "current":
        parts.append(sc.time_reference.replace("_", " "))
    return parts


def _build_company_query(sc: StructuredClaim) -> list[str]:
    """Company / Product: 'Apple launched product yesterday'"""
    parts = []
    if sc.organization:
        parts.append(sc.organization)
    if sc.entity:
        parts.append(sc.entity)
    if sc.event:
        parts.append(sc.event.replace("_", " "))
    if sc.condition:
        parts.append(sc.condition)
    if sc.time_reference and sc.time_reference != "current":
        parts.append(sc.time_reference.replace("_", " "))
    return parts


def _build_transport_query(sc: StructuredClaim) -> list[str]:
    """Transport: 'Delhi metro delay today'"""
    parts = []
    if sc.location:
        parts.append(sc.location)
    if sc.entity:
        parts.append(sc.entity)
    if sc.event:
        parts.append(sc.event.replace("_", " "))
    if sc.condition:
        parts.append(sc.condition)
    if sc.time_reference:
        parts.append(sc.time_reference.replace("_", " "))
    return parts


def _build_general_query(sc: StructuredClaim) -> list[str]:
    """General: combine all available fields."""
    parts = []
    if sc.organization:
        parts.append(sc.organization)
    if sc.entity:
        parts.append(sc.entity)
    if sc.location:
        parts.append(sc.location)
    if sc.event:
        parts.append(sc.event.replace("_", " "))
    if sc.condition:
        parts.append(sc.condition)
    if sc.time_reference:
        parts.append(sc.time_reference.replace("_", " "))
    return parts
