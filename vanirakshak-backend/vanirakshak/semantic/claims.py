"""Claim verification against known-true institutional policy.

This answers a different question from ``keywords.py``. That module asks
"does this call use scam *vocabulary*?" — a bag-of-words signal. This module
asks "did the caller *assert or demand something that contradicts a published
policy*?", and when it fires it can name the specific rule broken and the
authority that published it.

Scope, stated honestly: this is **rule-based verification against a curated
knowledge base**, not open-ended fact-checking. It cannot verify arbitrary
claims ("your account balance is X", "I am calling from branch Y") — those
need live data this system does not have. What it *can* verify is the class
of claim that scam scripts structurally depend on, where the contradiction is
categorical rather than case-by-case: no legitimate bank ever asks for an
OTP, so "share your OTP" is false regardless of who is calling or what
account is involved. That categorical property is what makes these checkable
offline, deterministically, with a citation.

Every rule therefore records:
  * ``requires`` / ``optional`` — the lexical evidence that the claim was made
  * ``claim``     — what the caller is asserting or demanding, in plain words
  * ``finding``   — why that is false or impossible
  * ``authority`` — who published the rule being contradicted
  * ``severity``  — 0..1, how diagnostic of fraud a match is

Matching is intentionally conservative: a rule fires only when *all* of its
``requires`` groups are present in the same analysis window, so "the word OTP
appeared" alone is not a contradiction — "OTP" plus a request verb is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Each entry in `requires` is a group of synonyms; the group is satisfied if
# ANY of its terms appear. A rule fires only when EVERY group is satisfied.
# Hindi/Hinglish terms are included directly because the transliterator leaves
# common colloquial words romanized rather than translated.
CLAIM_RULES: List[Dict[str, object]] = [
    {
        "id": "otp_request",
        "requires": [
            ["otp", "one time password", "pin", "cvv", "verification code", "password"],
            ["share", "send", "tell", "give", "provide", "bhejo", "bhejiye", "batao", "bataiye", "dijiye", "de do", "read out"],
        ],
        "claim": "Caller is asking you to share an OTP / PIN / CVV.",
        "finding": "No bank, wallet, or government body ever asks for an OTP, PIN, or CVV — on any channel, for any reason. A request for one is fraudulent by definition, regardless of who the caller claims to be.",
        "authority": "RBI customer-protection guidance; every major Indian bank's published security policy",
        "severity": 0.95,
    },
    {
        "id": "remote_access_app",
        "requires": [
            ["anydesk", "teamviewer", "quicksupport", "screen share", "screen sharing", "remote access", "airdroid", "rustdesk"],
        ],
        "claim": "Caller wants you to install a remote-access or screen-sharing app.",
        "finding": "No bank or government agency resolves an account issue by having you install a screen-sharing app. These apps hand the caller live control of your device and are a standard step in remote-access fraud.",
        "authority": "RBI / CERT-In advisories on remote-access fraud",
        "severity": 0.9,
    },
    {
        "id": "digital_arrest",
        "requires": [
            ["digital arrest", "digital custody", "virtual arrest", "online arrest"],
        ],
        "claim": "Caller claims you are under 'digital arrest'.",
        "finding": "'Digital arrest' does not exist in Indian law. No police force, CBI, ED, or court can arrest, detain, or hold anyone over a phone or video call. The concept is an invention used only in this scam script.",
        "authority": "Ministry of Home Affairs / I4C public advisories",
        "severity": 1.0,
    },
    {
        "id": "rbi_calls_individual",
        "requires": [
            ["rbi", "reserve bank"],
            ["your account", "aapka account", "aapka khata", "kyc", "block", "blocked", "suspend", "suspended", "verify", "verification"],
        ],
        "claim": "Caller claims to be RBI contacting you about your personal account.",
        "finding": "The RBI does not hold retail customer accounts and does not call individuals about KYC, blocking, or verification. It regulates banks, not their customers — it has no reason or mechanism to call you.",
        "authority": "Reserve Bank of India public notices",
        "severity": 0.9,
    },
    {
        "id": "law_enforcement_payment",
        "requires": [
            ["police", "cbi", "ed", "enforcement directorate", "customs", "court", "narcotics", "cyber cell", "income tax"],
            ["pay", "payment", "transfer", "deposit", "fine", "settle", "settlement", "bail", "paise", "bhejo", "rupees", "amount"],
        ],
        "claim": "Caller claims to be law enforcement and is demanding payment.",
        "finding": "No Indian law-enforcement or tax agency collects fines, bail, or settlements by phone, UPI, bank transfer, gift card, or crypto. Legal proceedings happen through courts and written notices, never a call demanding immediate transfer.",
        "authority": "I4C / state cyber-crime advisories",
        "severity": 0.95,
    },
    {
        "id": "trai_disconnection",
        "requires": [
            ["trai", "telecom regulatory"],
            ["disconnect", "disconnection", "block", "blocked", "suspend", "suspended", "band ho", "number band"],
        ],
        "claim": "Caller claims TRAI will disconnect your phone number.",
        "finding": "TRAI does not disconnect individual phone numbers and does not call subscribers. Disconnection is handled by your operator, never by the regulator, and never via an unsolicited call demanding action.",
        "authority": "TRAI public advisory on impersonation calls",
        "severity": 0.9,
    },
    {
        "id": "parcel_narcotics",
        "requires": [
            ["parcel", "courier", "package", "fedex", "dhl", "consignment"],
            ["drugs", "narcotics", "illegal", "customs", "seized", "mdma", "passport", "arrest"],
        ],
        "claim": "Caller claims a parcel in your name contains illegal items.",
        "finding": "Courier companies and customs do not phone individuals to report seized contraband, and never route such a call to 'police' who then demand payment. This is a scripted pretext, not a real enforcement process.",
        "authority": "I4C / state cyber-crime advisories on courier fraud",
        "severity": 0.9,
    },
    {
        "id": "gift_card_or_crypto_payment",
        "requires": [
            ["gift card", "google play", "amazon card", "itunes", "bitcoin", "crypto", "usdt", "binance"],
            ["pay", "payment", "buy", "purchase", "send", "transfer", "settle"],
        ],
        "claim": "Caller wants payment in gift cards or cryptocurrency.",
        "finding": "No legitimate institution — bank, utility, court, or agency — accepts payment in gift cards or cryptocurrency. These are chosen purely because they are irreversible and untraceable.",
        "authority": "Standard fraud-prevention guidance (RBI, CERT-In)",
        "severity": 0.95,
    },
    {
        "id": "kyc_via_link_or_app",
        "requires": [
            ["kyc"],
            ["link", "click", "download", "install", "apk", "app", "form", "update now"],
        ],
        "claim": "Caller wants KYC completed via a link or downloaded app.",
        "finding": "KYC is never completed through a link sent in an SMS/WhatsApp message or an app side-loaded from a call. Banks complete KYC in-branch or in their own official app, and never under time pressure from an inbound call.",
        "authority": "RBI KYC master direction; bank security policies",
        "severity": 0.8,
    },
    {
        # Added after running test_audio/sample_60s_scam.wav through the real
        # ASR: that clip is the canonical Indian scam robocall ("your KYC has
        # expired, your account will be blocked today") and scored 0.0 because
        # it makes a threat without yet making a demand. The threat itself is
        # the checkable contradiction.
        "id": "kyc_same_day_blocking_threat",
        "requires": [
            ["kyc"],
            ["blocked", "block", "suspend", "suspended", "deactivate", "closed", "freeze", "frozen", "band"],
            ["today", "aaj", "immediately", "turant", "abhi", "within 24 hours", "right now", "tonight", "expired", "expire"],
        ],
        "claim": "Caller threatens your account will be blocked today over KYC.",
        "finding": "Banks do not cold-call or robocall customers threatening same-day account blocking over KYC. Real KYC updates come with written notice and a reasonable window, through your bank's own app or branch — never an unsolicited call creating same-day panic.",
        "authority": "RBI advisory on KYC-update fraud; RBI KYC master direction",
        "severity": 0.75,
    },
    {
        "id": "electricity_disconnection_tonight",
        "requires": [
            ["electricity", "bijli", "power", "meter"],
            ["disconnect", "disconnection", "cut", "band", "tonight", "aaj raat", "immediately", "turant"],
        ],
        "claim": "Caller threatens immediate electricity disconnection.",
        "finding": "Electricity boards issue written notices and statutory grace periods before disconnection. They do not call at short notice demanding instant payment to a personal number or UPI ID.",
        "authority": "State electricity board billing/disconnection rules",
        "severity": 0.8,
    },
]

# ASR routinely splits spoken acronyms into separate tokens — real Groq Whisper
# output on a Hindi scam sample transcribed "KYC" as "के YC" ("ke YC"), which an
# exact match on "kyc" misses entirely. These re-join the spelled-out forms
# before matching. Keep them narrow: each maps a plausible ASR rendering of a
# specific acronym, not a general fuzzy match.
_ACRONYM_REPAIRS: List[tuple] = [
    (re.compile(r"\bk\s*[ae]?y?\s*c\b|\bke\s+yc\b|\bkay\s*c\b"), "kyc"),
    (re.compile(r"\bo\s*t\s*p\b|\bot\s+p\b|\bo\s+tp\b"), "otp"),
    (re.compile(r"\bc\s*v\s*v\b"), "cvv"),
    (re.compile(r"\bu\s*p\s*i\b"), "upi"),
    (re.compile(r"\bc\s*b\s*i\b"), "cbi"),
    (re.compile(r"\br\s*b\s*i\b"), "rbi"),
    (re.compile(r"\bs\s*b\s*i\b"), "sbi"),
]


def normalize_transcript(text: str) -> str:
    """Lowercase and repair ASR-split acronyms so exact-match rules survive them."""
    out = (text or "").lower()
    for pattern, replacement in _ACRONYM_REPAIRS:
        out = pattern.sub(replacement, out)
    return out


_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def _pattern(term: str) -> re.Pattern:
    pat = _PATTERN_CACHE.get(term)
    if pat is None:
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(term.strip()) + r"(?![a-z0-9])")
        _PATTERN_CACHE[term] = pat
    return pat


def _group_hit(haystack: str, group: Sequence[str]) -> Optional[str]:
    """Return the first term in `group` present in `haystack`, else None."""
    for term in group:
        if _pattern(term).search(haystack):
            return term
    return None


@dataclass
class ClaimFinding:
    rule_id: str
    claim: str
    finding: str
    authority: str
    severity: float
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "claim": self.claim,
            "finding": self.finding,
            "authority": self.authority,
            "severity": round(float(self.severity), 3),
            "matched_terms": self.matched_terms,
        }


def verify_claims(*texts: str) -> Dict[str, object]:
    """Check transcript text for assertions that contradict known policy.

    Returns ``{"claim_risk": float in [0,1], "findings": [...], "rules_fired": [...]}``.

    ``claim_risk`` is the maximum severity among fired rules, not a sum: one
    confirmed policy contradiction ("share your OTP") is already decisive, and
    summing would let several weak matches outweigh a single categorical one.
    """
    haystack = normalize_transcript(" ".join(t for t in texts if t)).strip()
    findings: List[ClaimFinding] = []

    if haystack:
        for rule in CLAIM_RULES:
            matched: List[str] = []
            satisfied = True
            for group in rule["requires"]:  # type: ignore[index]
                hit = _group_hit(haystack, group)  # type: ignore[arg-type]
                if hit is None:
                    satisfied = False
                    break
                matched.append(hit)
            if satisfied:
                findings.append(
                    ClaimFinding(
                        rule_id=str(rule["id"]),
                        claim=str(rule["claim"]),
                        finding=str(rule["finding"]),
                        authority=str(rule["authority"]),
                        severity=float(rule["severity"]),  # type: ignore[arg-type]
                        matched_terms=matched,
                    )
                )

    claim_risk = max((f.severity for f in findings), default=0.0)
    return {
        "claim_risk": round(float(claim_risk), 4),
        "findings": [f.to_dict() for f in findings],
        "rules_fired": [f.rule_id for f in findings],
    }
