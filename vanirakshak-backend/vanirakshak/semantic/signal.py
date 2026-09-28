"""Bridge the merged ``backend/services/asr`` stack into the serving backend.

Mirrors the graceful-degradation pattern in ``vanirakshak/ml_bridge.py``:
this must NEVER raise into the caller. On a minimal install (no torch, no
``groq`` package, no ``GROQ_API_KEY``) it reports ``available: False`` with a
reason and the acoustic pipeline runs exactly as before — ``semantic_risk``
defaults to 0.0, which is a no-op in ``risk/fusion.py`` (``w_semantic * 0``).

The engine is ``settings.semantic.provider`` (``VR_SEMANTIC_PROVIDER``): a
cloud provider such as Groq uploads each window, a local one (``local``,
faster-whisper, IndicConformer) transcribes on the box.

Call sites must invoke :meth:`SemanticSignal.analyze_window` off the asyncio
event loop thread (e.g. via ``asyncio.to_thread``) — it blocks on a network
call (cloud) or on local inference.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .claims import verify_claims
from .keywords import score_text

# The merged ASR stack lives in the sibling ``backend/`` package at the repo
# root (D:\VaniRakshak\backend\services\asr), not inside vanirakshak-backend.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_UNAVAILABLE_BASE = {
    "available": False,
    "transcript": "",
    "hinglish_text": "",
    "language": None,
    "matches": [],
    "categories_hit": [],
    "semantic_risk": 0.0,
    "keyword_risk": 0.0,
    "claim_risk": 0.0,
    "claim_findings": [],
    "provider": None,
}


class SemanticSignal:
    """Lazily loads the ASR service once and scores audio windows for scam keywords."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._service = None
        self._provider: Optional[str] = None
        self.load_error: Optional[str] = None
        self.available = False

    def _ensure_loaded(self) -> bool:
        if self._service is not None:
            return True
        with self._lock:
            if self._service is not None:
                return True
            # G6: off unless explicitly enabled. Checked before the import so a
            # disabled layer never even constructs a client that could send
            # caller audio to a third party.
            from ..config import settings

            semantic = settings.semantic
            if not semantic.enabled:
                if semantic.provider_is_local:
                    consequence = (
                        f"enabling transcribes each audio window on this box with "
                        f"{semantic.provider}; zero_retention stays on"
                    )
                else:
                    consequence = (
                        f"enabling uploads each audio window to {semantic.provider} "
                        "(third-party) and turns off zero_retention"
                    )
                self.load_error = (
                    "disabled by default (set VR_SEMANTIC_ENABLED=1 to enable); " + consequence
                )
                return False

            if not semantic.provider_known:
                self.load_error = (
                    f"unknown VR_SEMANTIC_PROVIDER '{semantic.provider}' "
                    "(expected groq, local, whisper or indicconformer)"
                )
                return False

            # Every failure below is expected on a minimal install: torch,
            # the `groq` package, GROQ_API_KEY and faster-whisper are all
            # optional. Degrade
            # to available=False instead of raising — a missing dependency
            # or unset key must never break the live stream.
            try:
                from backend.services.asr.service import ASRService
                from backend.services.asr.types import ASRConfig
            except Exception as e:  # ImportError / ModuleNotFoundError
                self.load_error = f"ASR stack unavailable ({type(e).__name__}: {e})"
                return False

            try:
                service = ASRService(ASRConfig(provider=semantic.provider))
                service.warmup()
            except Exception as e:
                # Most commonly: GROQ_API_KEY not set (ValueError from
                # groq_provider._get_api_key), or faster-whisper not
                # installed for a local provider. Still a clean degrade.
                self.load_error = f"ASR warmup failed ({type(e).__name__}: {e})"
                return False

            self._service = service
            self._provider = semantic.provider
            self.available = True
            self.load_error = None
        return True

    def analyze_window(self, pcm_int16: np.ndarray, sample_rate: int = 16000) -> Dict[str, object]:
        """Transcribe one forensic window and score it for scam vocabulary.

        ``pcm_int16`` is the same raw window the acoustic pipeline analyses
        (see ``session.py``). Returns a dict matching the ``semantic`` field
        documented in ``docs/PROTOCOL.md``; never raises.
        """
        if pcm_int16 is None or pcm_int16.size == 0:
            return {**_UNAVAILABLE_BASE, "reason": "empty audio window"}

        if not self._ensure_loaded():
            return {**_UNAVAILABLE_BASE, "reason": self.load_error or "ASR service not loaded"}

        try:
            import torch

            waveform = torch.from_numpy(
                (pcm_int16.astype(np.float32) / 32768.0)
            ).unsqueeze(0)
            result = self._service.transcribe_chunk(
                waveform,
                start_sec=0.0,
                end_sec=float(pcm_int16.size) / float(sample_rate),
                language="auto",
            )
        except Exception as e:
            # Network hiccup, rate limit, local inference error, etc. — the
            # acoustic verdict must never depend on this succeeding.
            return {**_UNAVAILABLE_BASE, "reason": f"transcription failed ({type(e).__name__}: {e})"}

        text = (result.text or "").strip()
        hinglish = (result.hinglish_text or "").strip()
        scored = score_text(text, hinglish)
        claims = verify_claims(text, hinglish)

        # Two independent readings of the same transcript: `semantic_risk` is
        # "how much scam vocabulary is present" (bag of words), `claim_risk` is
        # "did the caller demand something that contradicts published policy"
        # (a citable contradiction). The fused term takes the max rather than
        # summing — a confirmed OTP request is already decisive on its own, and
        # summing would let stacked weak vocabulary outweigh it.
        fused_risk = max(float(scored["semantic_risk"]), float(claims["claim_risk"]))

        return {
            "available": True,
            "transcript": text,
            "hinglish_text": hinglish,
            "language": result.language,
            "matches": scored["matches"],
            "categories_hit": scored["categories_hit"],
            "semantic_risk": round(fused_risk, 4),
            "keyword_risk": scored["semantic_risk"],
            "claim_risk": claims["claim_risk"],
            "claim_findings": claims["findings"],
            "provider": result.provider or self._provider,
            "reason": None,
        }


# Module-level singleton, lazily loaded on first use.
SEMANTIC_SIGNAL = SemanticSignal()
