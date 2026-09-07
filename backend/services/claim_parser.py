"""
Claim Parser — extracts structured semantic fields from detected claims.

Takes a DetectedClaim and enriches its StructuredClaim with:
- location (city, country, region)
- organization
- condition / event
- time_reference (current, yesterday, last week, etc.)
- certainty (asserted vs speculative)

BASELINE IMPLEMENTATION: regex/pattern-based extraction.
Can be replaced by NER or an LLM extractor behind the same interface.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from backend.schemas.claims import (
    ClaimType,
    ClaimCertainty,
    DetectedClaim,
    StructuredClaim,
)


# ---------------------------------------------------------------------------
# Location extraction
# ---------------------------------------------------------------------------

# Major Indian cities + some international ones for demo coverage
_KNOWN_LOCATIONS: list[str] = [
    "Delhi", "New Delhi", "Mumbai", "Bangalore", "Bengaluru", "Chennai",
    "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Bhopal", "Patna", "Guwahati", "Srinagar", "Shimla",
    "Dehradun", "Thiruvananthapuram", "Kochi", "Goa", "Noida", "Gurugram",
    "Indore", "Nagpur", "Varanasi", "Agra", "Amritsar",
    # International
    "London", "New York", "Tokyo", "Beijing", "Paris", "Dubai",
    "Singapore", "Sydney", "Toronto", "Washington",
    # Countries
    "India", "China", "USA", "UK", "Japan", "Australia", "Pakistan",
    "Bangladesh", "Sri Lanka", "Nepal",
]

_LOCATION_PATTERNS = [
    re.compile(r"\bin\s+(" + "|".join(re.escape(loc) for loc in _KNOWN_LOCATIONS) + r")\b", re.IGNORECASE),
    re.compile(r"\b(" + "|".join(re.escape(loc) for loc in _KNOWN_LOCATIONS) + r")\s+(is|has|was|are|were|currently|today)\b", re.IGNORECASE),
    re.compile(r"\b(" + "|".join(re.escape(loc) for loc in _KNOWN_LOCATIONS) + r")\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Time reference extraction
# ---------------------------------------------------------------------------

_TIME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(right\s+now|currently|at\s+this\s+moment|at\s+present|presently)\b", re.IGNORECASE), "current"),
    (re.compile(r"\b(now)\b", re.IGNORECASE), "current"),
    (re.compile(r"\b(today)\b", re.IGNORECASE), "today"),
    (re.compile(r"\b(yesterday)\b", re.IGNORECASE), "yesterday"),
    (re.compile(r"\b(tomorrow)\b", re.IGNORECASE), "tomorrow"),
    (re.compile(r"\b(tonight)\b", re.IGNORECASE), "tonight"),
    (re.compile(r"\b(this\s+week)\b", re.IGNORECASE), "this_week"),
    (re.compile(r"\b(last\s+week)\b", re.IGNORECASE), "last_week"),
    (re.compile(r"\b(next\s+week)\b", re.IGNORECASE), "next_week"),
    (re.compile(r"\b(this\s+month)\b", re.IGNORECASE), "this_month"),
    (re.compile(r"\b(last\s+month)\b", re.IGNORECASE), "last_month"),
    (re.compile(r"\b(next\s+month)\b", re.IGNORECASE), "next_month"),
    (re.compile(r"\b(this\s+year)\b", re.IGNORECASE), "this_year"),
    (re.compile(r"\b(last\s+year)\b", re.IGNORECASE), "last_year"),
    (re.compile(r"\b(recently)\b", re.IGNORECASE), "recent"),
]


# ---------------------------------------------------------------------------
# Organization extraction
# ---------------------------------------------------------------------------

_KNOWN_ORGS: list[str] = [
    "RBI", "Reserve Bank of India", "SEBI", "IRCTC", "ISRO", "DRDO",
    "SBI", "HDFC", "ICICI", "PNB", "Reliance", "Tata", "Infosys",
    "Wipro", "TCS", "Adani", "Airtel", "Jio", "Vodafone", "Paytm",
    "Zomato", "Swiggy", "Flipkart", "Amazon", "Google", "Microsoft",
    "Apple", "Meta", "Tesla", "OpenAI",
    # Government bodies
    "Supreme Court", "High Court", "Parliament", "Lok Sabha", "Rajya Sabha",
    "Election Commission", "NITI Aayog", "NCERT", "UGC", "CBSE",
]

_ORG_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(org) for org in _KNOWN_ORGS) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Weather condition extraction
# ---------------------------------------------------------------------------

_WEATHER_CONDITIONS: dict[str, str] = {
    "heavy rain": "heavy rain",
    "raining heavily": "heavy rain",
    "raining": "rain",
    "rain": "rain",
    "light rain": "light rain",
    "drizzle": "light rain",
    "snowing": "snow",
    "snow": "snow",
    "heavy snow": "heavy snow",
    "storm": "storm",
    "thunderstorm": "thunderstorm",
    "hurricane": "hurricane",
    "cyclone": "cyclone",
    "tornado": "tornado",
    "flooding": "flood",
    "flood": "flood",
    "drought": "drought",
    "sunny": "sunny",
    "clear": "clear",
    "cloudy": "cloudy",
    "overcast": "overcast",
    "fog": "fog",
    "foggy": "fog",
    "hail": "hail",
    "heatwave": "heatwave",
    "heat wave": "heatwave",
    "cold wave": "cold wave",
    "cold": "cold",
    "hot": "hot",
    "warm": "warm",
    "humid": "humid",
    "windy": "windy",
    "wind": "wind",
}

_WEATHER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_WEATHER_CONDITIONS.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Event extraction (finance, government, etc.)
# ---------------------------------------------------------------------------

_EVENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(increased|hiked|raised)\s+(the\s+)?(repo\s*rate)", re.IGNORECASE), "repo_rate_increase"),
    (re.compile(r"\b(decreased|cut|lowered|reduced)\s+(the\s+)?(repo\s*rate)", re.IGNORECASE), "repo_rate_decrease"),
    (re.compile(r"\b(repo\s*rate)\b.*\b(increased|hiked|raised)\b", re.IGNORECASE), "repo_rate_increase"),
    (re.compile(r"\b(repo\s*rate)\b.*\b(decreased|cut|lowered|reduced)\b", re.IGNORECASE), "repo_rate_decrease"),
    (re.compile(r"\b(announced|launched|introduced|released|passed|approved)\b.*\b(scheme|policy|bill|act|product|service)\b", re.IGNORECASE), "announcement"),
    (re.compile(r"\b(scheme|policy|bill|act|product|service)\b.*\b(announced|launched|introduced|released|passed|approved)\b", re.IGNORECASE), "announcement"),
    (re.compile(r"\b(launched|released|introduced)\b.*\b(product|app|phone|device|software|version|update)\b", re.IGNORECASE), "product_launch"),
    (re.compile(r"\b(arrested|detained|charged)\b", re.IGNORECASE), "arrest"),
    (re.compile(r"\b(resigned|stepped\s+down)\b", re.IGNORECASE), "resignation"),
    (re.compile(r"\b(appointed|nominated|elected)\b", re.IGNORECASE), "appointment"),
    (re.compile(r"\b(earthquake|eruption|tsunami|landslide)\b", re.IGNORECASE), "natural_disaster"),
    (re.compile(r"\b(accident|crash|collision)\b", re.IGNORECASE), "accident"),
    (re.compile(r"\b(strike|protest|rally|demonstration)\b", re.IGNORECASE), "protest"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_claim(claim: DetectedClaim) -> DetectedClaim:
    """
    Enrich a DetectedClaim's StructuredClaim with extracted semantic fields.

    Returns a new DetectedClaim with the structured_claim updated.
    The original object is not mutated.
    """
    text = claim.text
    sc = claim.structured_claim

    # Extract fields
    location = _extract_location(text)
    organization = _extract_organization(text)
    time_ref = _extract_time_reference(text)
    condition = _extract_weather_condition(text) if sc.claim_type == ClaimType.WEATHER else None
    event = _extract_event(text)
    certainty = sc.certainty  # already set by detector

    # Build updated structured claim
    updated = StructuredClaim(
        claim_type=sc.claim_type,
        entity=sc.entity,
        organization=organization or sc.organization,
        location=location or sc.location,
        condition=condition or sc.condition,
        event=event or sc.event,
        time_reference=time_ref or sc.time_reference,
        certainty=certainty,
        extra=sc.extra,
    )

    # Return new DetectedClaim with updated structure
    return DetectedClaim(
        claim_id=claim.claim_id,
        start=claim.start,
        end=claim.end,
        text=claim.text,
        claim_type=claim.claim_type,
        structured_claim=updated,
    )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_location(text: str) -> str | None:
    """Extract the most likely geographic location from text."""
    for pattern in _LOCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _extract_organization(text: str) -> str | None:
    """Extract a known organization from text."""
    match = _ORG_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def _extract_time_reference(text: str) -> str | None:
    """Extract temporal reference from text."""
    for pattern, label in _TIME_PATTERNS:
        if pattern.search(text):
            return label
    # Default: if present tense verbs suggest current time
    if re.search(r"\b(is|are)\s+\w+ing\b", text, re.IGNORECASE):
        return "current"
    return None


def _extract_weather_condition(text: str) -> str | None:
    """Extract weather condition from text."""
    match = _WEATHER_PATTERN.search(text)
    if match:
        raw = match.group(1).lower()
        return _WEATHER_CONDITIONS.get(raw, raw)
    return None


def _extract_event(text: str) -> str | None:
    """Extract the type of event described."""
    for pattern, label in _EVENT_PATTERNS:
        if pattern.search(text):
            return label
    return None
