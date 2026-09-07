"""
Claim Detector — identifies factual claims in transcript text.

BASELINE IMPLEMENTATION: rule-based + heuristic.
This is explicitly a deterministic baseline for development/testing.
It can be replaced by a fine-tuned classifier without changing
any downstream interfaces.

Strategy:
1. Split text into sentences.
2. Filter out non-factual sentences (greetings, questions, acks).
3. Identify sentences containing factual-assertion patterns.
4. Classify each claim by domain (WEATHER, FINANCE, etc.).
5. Return DetectedClaim objects with original timestamps preserved.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from backend.schemas.claims import ClaimType, DetectedClaim, StructuredClaim, ClaimCertainty


# ---------------------------------------------------------------------------
# Sentence-level filters
# ---------------------------------------------------------------------------

# Patterns that indicate NON-factual content (should NOT trigger verification)
_NON_FACTUAL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Greetings / pleasantries
        r"^\s*(hello|hi|hey|good\s+(morning|afternoon|evening)|namaste)\b",
        r"^\s*(how\s+are\s+you|how\s+do\s+you\s+do)",
        r"^\s*(thank\s*(s|you)|thanks\s+a\s+lot|dhanyavaad)",
        r"^\s*(bye|goodbye|see\s+you|take\s+care)",
        # Acknowledgments
        r"^\s*(okay|ok|yes|no|right|sure|hmm|haan|achha|theek\s+hai)\s*[.!]?\s*$",
        # Pure questions (without embedded factual assertion)
        r"^\s*(can\s+you|could\s+you|will\s+you|would\s+you|do\s+you|did\s+you)\b",
        r"^\s*(what|who|where|when|why|how)\b.*\?\s*$",
        # Commands / requests
        r"^\s*(please|kindly)\b",
        # Very short utterances
        r"^\s*\S{1,3}\s*$",
    ]
]

# Patterns that indicate FACTUAL assertions (should trigger verification)
_FACTUAL_INDICATORS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # State / condition assertions
        r"\b(is|are|was|were)\s+(currently|now|presently|raining|snowing|clear|sunny|cloudy|flooding)",
        r"\b(is|are|was|were)\s+\w+ing\b",  # "is raining", "was flooding"
        r"\b(has|have|had)\s+(been|increased|decreased|launched|announced|released|blocked|frozen|expired)",
        # Event assertions
        r"\b(increased|decreased|raised|lowered|cut|hiked|dropped|fell|rose|surged|crashed)\b",
        r"\b(announced|launched|released|introduced|passed|approved|rejected|signed)\b",
        r"\b(happened|occurred|started|ended|opened|closed|banned|suspended|cancelled)\b",
        # Explicit factual framing
        r"\b(according\s+to|reports?\s+(say|indicate|suggest))",
        r"\b(officially|confirmed|reportedly)\b",
        # Weather patterns
        r"\b(rain|raining|snow|snowing|storm|hurricane|cyclone|flood|drought|temperature|weather|forecast)\b",
        # Financial patterns
        r"\b(repo\s+rate|interest\s+rate|stock|share|market|sensex|nifty|inflation|gdp)\b",
        # Institutional actions
        r"\b(rbi|sebi|government|ministry|court|supreme\s+court|parliament)\b.*\b(has|have|had|will|did|announced|increased|decreased)\b",
    ]
]

# Speculative modifiers → change certainty from ASSERTED to SPECULATIVE
_SPECULATIVE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(may|might|could|possibly|perhaps|probably|reportedly|allegedly)\b",
        r"\b(expected\s+to|likely\s+to|rumou?red)\b",
    ]
]

# ---------------------------------------------------------------------------
# Claim Type classification keywords
# ---------------------------------------------------------------------------

_CLAIM_TYPE_KEYWORDS: dict[ClaimType, list[re.Pattern]] = {
    ClaimType.WEATHER: [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(rain|raining|snow|snowing|storm|hurricane|cyclone|tornado|flood|drought|sunny|cloudy|fog|hail|temperature|weather|forecast|precipitation|wind|humidity|heatwave|cold\s*wave)\b",
        ]
    ],
    ClaimType.FINANCE: [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(repo\s*rate|interest\s*rate|stock|share|sensex|nifty|inflation|gdp|bond|market|trading|investment|bank\s*rate|monetary\s*policy|fiscal|budget|tax|revenue|profit|loss|dividend)\b",
            r"\b(rbi|sebi|reserve\s*bank)\b",
        ]
    ],
    ClaimType.GOVERNMENT: [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(government|minister|ministry|parliament|lok\s*sabha|rajya\s*sabha|cabinet|policy|scheme|bill|act|ordinance|election|vote|pm|chief\s*minister|president)\b",
        ]
    ],
    ClaimType.NEWS: [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(breaking|news|report|headlines?|press|media|journalist|coverage)\b",
        ]
    ],
    ClaimType.TRANSPORT: [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(flight|train|metro|bus|road|highway|airport|railway|traffic|accident|crash|delay|cancel|divert)\b",
        ]
    ],
    ClaimType.COMPANY: [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(company|startup|corporation|firm|limited|ltd|inc|pvt|enterprise|ceo|founder|acquisition|merger|ipo)\b",
        ]
    ],
    ClaimType.PRODUCT: [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(product|launched|release|version|update|feature|app|software|device|phone|model)\b",
        ]
    ],
    ClaimType.LAW: [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(court|judge|verdict|ruling|lawsuit|legal|law|section|ipc|crpc|constitution|supreme\s*court|high\s*court)\b",
        ]
    ],
    ClaimType.PERSON: [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(arrested|appointed|resigned|died|born|awarded|elected|nominated)\b",
        ]
    ],
    ClaimType.EVENT: [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(event|conference|summit|ceremony|festival|match|game|tournament|concert|rally|protest|strike)\b",
        ]
    ],
    ClaimType.LOCATION: [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(earthquake|eruption|tsunami|landslide|fire|blast|explosion|collapse)\b",
        ]
    ],
}


# ---------------------------------------------------------------------------
# Unverifiable-by-nature patterns (private / personal claims)
# ---------------------------------------------------------------------------

_PRIVATE_CLAIM_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(your\s+(account|bank|card|loan|kyc|aadhaar|pan|otp|password|balance|transaction))\b",
        r"\b(my\s+(account|bank|card|loan|kyc|aadhaar|pan|otp|password|balance|transaction))\b",
        r"\b(blocked|frozen|suspended|deactivated|expired)\b.*\b(your|my)\b",
    ]
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_claims(
    sentences: list[dict],
) -> list[DetectedClaim]:
    """
    Detect factual claims from a list of sentence dicts.

    Each dict must have: {"start": float, "end": float, "text": str}

    Returns a list of DetectedClaim objects for sentences identified
    as containing verifiable factual propositions.
    """
    claims: list[DetectedClaim] = []

    for sent in sentences:
        text = sent["text"].strip()
        start = sent["start"]
        end = sent["end"]

        if not text:
            continue

        # Step 1: Filter out non-factual content
        if _is_non_factual(text):
            continue

        # Step 2: Check for factual indicators
        if not _has_factual_indicator(text):
            continue

        # Step 3: Classify claim type
        claim_type = _classify_claim_type(text)

        # Step 4: Determine certainty
        certainty = _detect_certainty(text)

        # Step 5: Check if claim is private/unverifiable by nature
        is_private = _is_private_claim(text)

        # Step 6: Build DetectedClaim
        claim_id = f"claim_{uuid.uuid4().hex[:8]}"

        claim = DetectedClaim(
            claim_id=claim_id,
            start=start,
            end=end,
            text=text,
            claim_type=claim_type,
            structured_claim=StructuredClaim(
                claim_type=claim_type,
                certainty=certainty,
                extra={"is_private": is_private},
            ),
        )
        claims.append(claim)

    return claims


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_non_factual(text: str) -> bool:
    """Return True if the sentence is clearly not a factual assertion."""
    for pattern in _NON_FACTUAL_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _has_factual_indicator(text: str) -> bool:
    """Return True if the sentence contains a factual-assertion pattern."""
    for pattern in _FACTUAL_INDICATORS:
        if pattern.search(text):
            return True
    return False


def _classify_claim_type(text: str) -> ClaimType:
    """Classify the claim into a domain based on keyword matching."""
    # Score each type by number of matching patterns
    best_type = ClaimType.GENERAL_FACT
    best_score = 0

    for ctype, patterns in _CLAIM_TYPE_KEYWORDS.items():
        score = sum(1 for p in patterns if p.search(text))
        if score > best_score:
            best_score = score
            best_type = ctype

    return best_type


def _detect_certainty(text: str) -> ClaimCertainty:
    """Detect whether the speaker expresses certainty or speculation."""
    for pattern in _SPECULATIVE_PATTERNS:
        if pattern.search(text):
            return ClaimCertainty.SPECULATIVE
    return ClaimCertainty.ASSERTED


def _is_private_claim(text: str) -> bool:
    """Return True if the claim refers to private/personal information."""
    for pattern in _PRIVATE_CLAIM_PATTERNS:
        if pattern.search(text):
            return True
    return False


def split_into_sentences(text: str) -> list[str]:
    """
    Basic sentence splitter.

    Splits on sentence-ending punctuation and newlines.
    Used when the transcript is a single block of text
    rather than pre-segmented sentences.
    """
    # Split on .!? followed by space/newline, or on newlines
    raw = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in raw if s.strip()]
