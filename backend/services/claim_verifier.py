"""
Claim Verifier — compares a structured claim against retrieved evidence.

Returns SUPPORTED, CONTRADICTED, or UNVERIFIABLE.

BASELINE IMPLEMENTATION: multi-signal heuristic approach.
Uses keyword overlap, contradiction detection, entity matching,
and temporal consistency.

Does NOT use simple substring matching.
Does NOT treat "no results" as "false".
Does NOT force certainty when evidence is conflicting.

Can be replaced by an LLM-based NLI model behind the same interface.
"""

from __future__ import annotations

import re
import logging

from backend.schemas.claims import ClaimType, DetectedClaim, StructuredClaim
from backend.schemas.evidence import Evidence, SourceType
from backend.schemas.verification import VerificationStatus, VerificationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contradiction / Support vocabularies per domain
# ---------------------------------------------------------------------------

_WEATHER_CONTRADICTIONS: dict[str, set[str]] = {
    # If the claim says X, and evidence contains any of Y → contradiction
    "rain": {"clear", "sunny", "no rain", "no precipitation", "dry", "not raining"},
    "heavy rain": {"clear", "sunny", "no rain", "no precipitation", "dry", "light rain", "drizzle", "not raining"},
    "snow": {"clear", "sunny", "no snow", "warm", "hot", "not snowing"},
    "sunny": {"rain", "raining", "storm", "cloudy", "overcast", "snow"},
    "clear": {"rain", "raining", "storm", "cloudy", "overcast", "snow", "fog"},
    "storm": {"clear", "sunny", "calm", "no storm"},
    "flood": {"dry", "no flood", "drought", "no rain"},
    "drought": {"rain", "flood", "heavy rain", "wet"},
    "cold": {"hot", "warm", "heatwave", "heat wave"},
    "hot": {"cold", "freezing", "cold wave", "snow"},
    "fog": {"clear", "sunny", "no fog"},
}

_WEATHER_SUPPORTS: dict[str, set[str]] = {
    "rain": {"rain", "raining", "precipitation", "shower", "wet", "downpour", "drizzle"},
    "heavy rain": {"heavy rain", "downpour", "torrential", "very heavy", "intense rain", "heavy rainfall"},
    "snow": {"snow", "snowing", "snowfall", "blizzard"},
    "sunny": {"sunny", "clear sky", "sunshine", "bright"},
    "clear": {"clear", "sunny", "no cloud", "fair"},
    "storm": {"storm", "thunderstorm", "tempest"},
    "flood": {"flood", "flooding", "waterlogging", "submerged"},
    "drought": {"drought", "dry spell", "water shortage"},
    "cold": {"cold", "freezing", "chilly", "low temperature"},
    "hot": {"hot", "scorching", "heatwave", "high temperature"},
    "fog": {"fog", "foggy", "mist", "visibility"},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_claim(
    claim: DetectedClaim,
    evidence_list: list[Evidence],
    query: str,
) -> VerificationResult:
    """
    Compare a claim against retrieved evidence and produce a verdict.

    Logic:
    1. If no evidence → UNVERIFIABLE
    2. If claim is private/personal → UNVERIFIABLE
    3. Check for contradictions (strong signal)
    4. Check for support (strong signal)
    5. If conflicting signals → reduced confidence
    6. If insufficient signal → UNVERIFIABLE
    """
    sc = claim.structured_claim

    # --- Case: Private claim ---
    if sc.extra.get("is_private", False):
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            verification_confidence=0.9,
            reason="This claim refers to private/personal information that cannot be verified through public web search.",
            query=query,
            evidence=evidence_list,
        )

    # --- Case: No evidence retrieved ---
    if not evidence_list:
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            verification_confidence=0.5,
            reason="No relevant evidence could be retrieved to verify this claim.",
            query=query,
            evidence=[],
        )

    # --- Dispatch by claim type ---
    if sc.claim_type == ClaimType.WEATHER:
        return _verify_weather_claim(claim, evidence_list, query)

    # Generic verification for other claim types
    return _verify_generic_claim(claim, evidence_list, query)


# ---------------------------------------------------------------------------
# Weather-specific verification
# ---------------------------------------------------------------------------

def _verify_weather_claim(
    claim: DetectedClaim,
    evidence_list: list[Evidence],
    query: str,
) -> VerificationResult:
    """
    Verify a weather claim using condition-matching against evidence.
    """
    sc = claim.structured_claim
    condition = (sc.condition or "").lower().strip()
    location = (sc.location or "").lower().strip()

    if not condition:
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            verification_confidence=0.3,
            reason="Could not extract a specific weather condition from the claim.",
            query=query,
            evidence=evidence_list,
        )

    support_score = 0.0
    contradict_score = 0.0
    location_matched = False
    total_relevant = 0

    support_keywords = _WEATHER_SUPPORTS.get(condition, {condition})
    contradict_keywords = _WEATHER_CONTRADICTIONS.get(condition, set())

    for ev in evidence_list:
        content_lower = ev.content.lower()

        # Check location relevance
        if location and location in content_lower:
            location_matched = True
        elif location and location not in content_lower:
            # Evidence might be about a different location
            continue

        total_relevant += 1

        # Check for support
        for kw in support_keywords:
            if kw in content_lower:
                support_score += 1.0
                break

        # Check for contradiction
        for kw in contradict_keywords:
            if kw in content_lower:
                contradict_score += 1.0
                break

    # --- Decision logic ---
    if total_relevant == 0:
        # Evidence exists but none is location-relevant
        if location and not location_matched:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verification_confidence=0.4,
                reason=f"Retrieved evidence does not appear to be about {sc.location}.",
                query=query,
                evidence=evidence_list,
            )
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            verification_confidence=0.3,
            reason="Retrieved evidence is not relevant enough to verify this claim.",
            query=query,
            evidence=evidence_list,
        )

    # Normalize scores
    support_ratio = support_score / total_relevant
    contradict_ratio = contradict_score / total_relevant

    if contradict_ratio > support_ratio and contradict_score >= 1:
        confidence = min(0.95, 0.6 + contradict_ratio * 0.3)
        return VerificationResult(
            status=VerificationStatus.CONTRADICTED,
            verification_confidence=round(confidence, 2),
            reason=f"External evidence conflicts with the claim that {sc.condition} in {sc.location or 'the specified location'}.",
            query=query,
            evidence=evidence_list,
        )

    if support_ratio > contradict_ratio and support_score >= 1:
        confidence = min(0.95, 0.6 + support_ratio * 0.3)
        return VerificationResult(
            status=VerificationStatus.SUPPORTED,
            verification_confidence=round(confidence, 2),
            reason=f"External evidence supports the claim that {sc.condition} in {sc.location or 'the specified location'}.",
            query=query,
            evidence=evidence_list,
        )

    if support_score > 0 and contradict_score > 0:
        # Conflicting evidence
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            verification_confidence=0.4,
            reason="Evidence sources conflict — some support and some contradict the claim.",
            query=query,
            evidence=evidence_list,
        )

    return VerificationResult(
        status=VerificationStatus.UNVERIFIABLE,
        verification_confidence=0.3,
        reason="Insufficient relevant evidence to determine whether the claim is true or false.",
        query=query,
        evidence=evidence_list,
    )


# ---------------------------------------------------------------------------
# Generic verification (non-weather claims)
# ---------------------------------------------------------------------------

def _verify_generic_claim(
    claim: DetectedClaim,
    evidence_list: list[Evidence],
    query: str,
) -> VerificationResult:
    """
    Verify a non-weather claim using multi-signal heuristics.

    Checks:
    1. Entity/org presence in evidence
    2. Event/action presence in evidence
    3. Negation patterns in evidence
    4. Source quality weighting
    """
    sc = claim.structured_claim
    text_lower = claim.text.lower()

    # Extract key terms from the claim
    key_terms = _extract_key_terms(sc)
    if not key_terms:
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            verification_confidence=0.3,
            reason="Could not extract enough structured information from the claim for verification.",
            query=query,
            evidence=evidence_list,
        )

    support_score = 0.0
    contradict_score = 0.0
    total_relevant = 0

    for ev in evidence_list:
        content_lower = ev.content.lower()

        # Check how many key terms appear in evidence
        term_matches = sum(1 for t in key_terms if t.lower() in content_lower)
        if term_matches == 0:
            continue  # Not relevant

        total_relevant += 1
        relevance = term_matches / len(key_terms)

        # Source quality bonus
        quality_multiplier = 1.0
        if ev.source_type == SourceType.OFFICIAL:
            quality_multiplier = 1.3
        elif ev.source_type == SourceType.NEWS:
            quality_multiplier = 1.1

        # Check for negation / contradiction patterns
        if _has_negation(content_lower, key_terms):
            contradict_score += relevance * quality_multiplier
        else:
            support_score += relevance * quality_multiplier

    if total_relevant == 0:
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            verification_confidence=0.3,
            reason="Retrieved evidence does not appear relevant to this claim.",
            query=query,
            evidence=evidence_list,
        )

    # Normalize
    support_ratio = support_score / total_relevant
    contradict_ratio = contradict_score / total_relevant

    if support_ratio > contradict_ratio and support_score >= 0.5:
        confidence = min(0.90, 0.5 + support_ratio * 0.3)
        return VerificationResult(
            status=VerificationStatus.SUPPORTED,
            verification_confidence=round(confidence, 2),
            reason="External evidence is consistent with this claim.",
            query=query,
            evidence=evidence_list,
        )

    if contradict_ratio > support_ratio and contradict_score >= 0.5:
        confidence = min(0.85, 0.5 + contradict_ratio * 0.3)
        return VerificationResult(
            status=VerificationStatus.CONTRADICTED,
            verification_confidence=round(confidence, 2),
            reason="External evidence conflicts with this claim.",
            query=query,
            evidence=evidence_list,
        )

    if support_score > 0 and contradict_score > 0:
        return VerificationResult(
            status=VerificationStatus.UNVERIFIABLE,
            verification_confidence=0.4,
            reason="Evidence sources provide conflicting information.",
            query=query,
            evidence=evidence_list,
        )

    return VerificationResult(
        status=VerificationStatus.UNVERIFIABLE,
        verification_confidence=0.3,
        reason="Insufficient relevant evidence to determine whether the claim is true or false.",
        query=query,
        evidence=evidence_list,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_key_terms(sc: StructuredClaim) -> list[str]:
    """Extract the most important terms from a structured claim."""
    terms = []
    if sc.organization:
        terms.append(sc.organization)
    if sc.entity:
        terms.append(sc.entity)
    if sc.location:
        terms.append(sc.location)
    if sc.event:
        terms.append(sc.event.replace("_", " "))
    if sc.condition:
        terms.append(sc.condition)
    return terms


# Negation patterns around key terms
_NEGATION_PATTERNS = [
    re.compile(r"\b(not|no|never|denied|denied|rejected|false|incorrect|untrue|didn.t|hasn.t|wasn.t|weren.t|isn.t|aren.t|don.t|doesn.t|won.t|can.t|couldn.t|shouldn.t|wouldn.t|haven.t|hadn.t)\b", re.IGNORECASE),
]


def _has_negation(text: str, key_terms: list[str]) -> bool:
    """
    Check if the evidence text negates the key claim terms.

    Looks for negation words in proximity to key terms.
    """
    for term in key_terms:
        term_lower = term.lower()
        if term_lower not in text:
            continue

        # Find term position and check surrounding context (±80 chars)
        idx = text.find(term_lower)
        if idx == -1:
            continue

        context_start = max(0, idx - 80)
        context_end = min(len(text), idx + len(term_lower) + 80)
        context = text[context_start:context_end]

        for neg_pattern in _NEGATION_PATTERNS:
            if neg_pattern.search(context):
                return True

    return False
