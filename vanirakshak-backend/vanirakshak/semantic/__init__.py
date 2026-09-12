"""Parallel semantic context signal: ASR transcript + scam-keyword scoring.

This package is deliberately separate from ``vanirakshak/detectors/`` — per
``docs/PROTOCOL.md`` and ``docs/FLOWCHARTS_CORRECTED.md`` the acoustic/
biometric pipeline (CM, ASV, channel, prosody -> Bayesian fusion) is and
remains the primary, authoritative forensic engine. What lives here is an
additional, advisory evidence term: what is the caller actually saying, and
does it contain the vocabulary of a live social-engineering script (OTP/KYC
requests, urgency, impersonated authority, payment pressure)?

It degrades to unavailable (never raises) exactly like ``ml_bridge.py``'s
``TargetSpeakerVerifier`` when the optional ASR stack (torch, groq,
GROQ_API_KEY) is not present, so the acoustic pipeline never depends on it.
"""

from .signal import SEMANTIC_SIGNAL, SemanticSignal

__all__ = ["SEMANTIC_SIGNAL", "SemanticSignal"]
