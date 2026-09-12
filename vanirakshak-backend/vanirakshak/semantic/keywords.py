"""Scam-phrase keyword lexicon and scorer.

Matches against the *Hinglish-normalized* transcript produced by
``backend/services/asr/transliteration.py`` — that service already maps
Devanagari fraud vocabulary (KYC, OTP, bank, suspend, police, arrest, ...) to
their English tokens, so a single English keyword list catches Hindi,
Hinglish and English utterances alike. The raw (native-script / English)
transcript is matched too, as a fallback for terms the transliterator leaves
untouched.

Categories mirror the social-engineering script structure documented in
``docs/FLOWCHARTS_CORRECTED.md`` (impersonated authority, urgency, OTP/PIN
exfiltration, payment pressure) — each contributes independently so a call
that stacks multiple categories (the hallmark of a scripted scam) scores
higher than one hitting a single word in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

# category -> (weight, keywords). Weights are additive and capped at 1.0 by
# the caller — a single category maxing out is meaningful evidence, but not
# on its own enough to dominate the acoustic fusion (see VR_FUSION_W_SEMANTIC).
KEYWORD_CATEGORIES: Dict[str, Dict[str, object]] = {
    "otp_pin_exfiltration": {
        "weight": 0.45,
        "terms": ["otp", "pin", "cvv", "password", "one time password", "verification code"],
    },
    "kyc_account_threat": {
        "weight": 0.3,
        "terms": [
            "kyc", "account", "bank", "upi", "suspend", "suspended", "block", "blocked",
            "expire", "expired", "deactivate", "inactive", "verify", "verification", "update kyc",
        ],
    },
    "impersonated_authority": {
        "weight": 0.35,
        "terms": [
            "police", "arrest", "warrant", "cbi", "ed ", "income tax", "customs", "trai",
            "digital arrest", "court", "legal action", "fir", "cybercrime",
        ],
    },
    "urgency_pressure": {
        "weight": 0.2,
        "terms": [
            "urgent", "immediately", "right now", "turant", "abhi", "jaldi",
            "last warning", "final notice", "within 24 hours", "within 1 hour",
        ],
    },
    "payment_request": {
        "weight": 0.3,
        "terms": [
            "transfer", "send money", "gift card", "refund", "cashback", "pay now",
            "processing fee", "click this link", "click the link", "share the link",
        ],
    },
}

_WORD_RE_CACHE: Dict[str, re.Pattern] = {}


def _term_pattern(term: str) -> re.Pattern:
    pat = _WORD_RE_CACHE.get(term)
    if pat is None:
        # word-boundary match so "pin" doesn't match inside "spin"; terms with
        # internal spaces match as a literal phrase.
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(term.strip()) + r"(?![a-z0-9])")
        _WORD_RE_CACHE[term] = pat
    return pat


@dataclass
class KeywordMatch:
    category: str
    term: str


def score_text(*texts: str) -> Dict[str, object]:
    """Score one or more transcript variants (e.g. native + hinglish) for scam vocabulary.

    Returns ``{"semantic_risk": float in [0,1], "matches": [{"category", "term"}, ...],
    "categories_hit": [...]}``.
    """
    # Shared with claims.py: repairs ASR-split acronyms ("ke YC" -> "kyc") that
    # would otherwise slip past exact-match terms. See its docstring for why.
    from .claims import normalize_transcript

    haystack = normalize_transcript(" ".join(t for t in texts if t)).strip()
    matches: List[KeywordMatch] = []
    categories_hit: Dict[str, float] = {}

    if haystack:
        for category, spec in KEYWORD_CATEGORIES.items():
            terms = spec["terms"]  # type: ignore[index]
            weight = float(spec["weight"])  # type: ignore[arg-type]
            hit = False
            for term in terms:
                if _term_pattern(term).search(haystack):
                    matches.append(KeywordMatch(category=category, term=term))
                    hit = True
            if hit:
                categories_hit[category] = weight

    semantic_risk = min(1.0, sum(categories_hit.values()))
    return {
        "semantic_risk": round(semantic_risk, 4),
        "matches": [{"category": m.category, "term": m.term} for m in matches],
        "categories_hit": sorted(categories_hit.keys()),
    }
