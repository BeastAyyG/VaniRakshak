"""Server-side transcription of a challenge response (docs/15 N2).

The challenge used to grade digits from ``response_text``, which the client
sends. A scripted client can always send the right digits, so the text check
proved nothing. The session now buffers the response audio while a challenge
is outstanding and this module transcribes it on the box.

Local engines only (``LOCAL_ASR_PROVIDERS``): the challenge must never upload
caller audio, whatever the semantic layer is configured to do. A cloud or
unknown provider is refused, and grading then fails closed.

Like ``semantic/signal.py`` this never raises: a missing engine reports
``(None, reason)`` and the caller decides what that means.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np

# Importing the semantic package puts the repo root (the sibling ``backend/``
# ASR stack) on sys.path, exactly as the semantic layer needs.
from ..semantic import signal as _semantic_signal  # noqa: F401


class ChallengeTranscriber:
    """Lazily loads one local ASR engine and transcribes response audio."""

    def __init__(self, provider: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._provider = provider
        self._service = None
        self.load_error: Optional[str] = None

    @property
    def provider(self) -> str:
        from ..config import settings

        return (self._provider or settings.challenge.asr_provider).strip().lower()

    def _ensure_loaded(self) -> bool:
        if self._service is not None:
            return True
        with self._lock:
            if self._service is not None:
                return True
            from ..config import LOCAL_ASR_PROVIDERS

            provider = self.provider
            if provider not in LOCAL_ASR_PROVIDERS:
                self.load_error = (
                    f"VR_CHALLENGE_ASR_PROVIDER '{provider}' is not a local engine; "
                    "the challenge never uploads caller audio"
                )
                return False
            try:
                from backend.services.asr.service import ASRService
                from backend.services.asr.types import ASRConfig
            except Exception as e:
                self.load_error = f"ASR stack unavailable ({type(e).__name__}: {e})"
                return False
            try:
                service = ASRService(ASRConfig(provider=provider))
                service.warmup()
            except Exception as e:
                self.load_error = f"ASR warmup failed ({type(e).__name__}: {e})"
                return False
            self._service = service
            self.load_error = None
        return True

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    def transcribe(self, pcm_int16: np.ndarray, sample_rate: int) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(transcript, None)`` or ``(None, reason)``. Blocking."""
        if not self._ensure_loaded():
            return None, self.load_error or "challenge transcriber not loaded"
        try:
            import torch

            waveform = torch.from_numpy(pcm_int16.astype(np.float32) / 32768.0).unsqueeze(0)
            result = self._service.transcribe_chunk(
                waveform,
                start_sec=0.0,
                end_sec=float(pcm_int16.size) / float(sample_rate),
                language="auto",
            )
        except Exception as e:
            return None, f"transcription failed ({type(e).__name__}: {e})"
        # Grade one reading, never both: joining the native transcript and
        # its romanisation would count every digit twice. Fall back to the
        # romanisation only when the native text holds no digits.
        from .engine import extract_digits

        native = (result.text or "").strip()
        roman = (result.hinglish_text or "").strip()
        return (native if extract_digits(native) or not roman else roman), None


# Module-level singleton, lazily loaded on first grade.
CHALLENGE_TRANSCRIBER = ChallengeTranscriber()
