"""Server-side transcription of a challenge response (docs/15 N2).

The challenge used to grade digits from ``response_text``, which the client
sends. A scripted client can always send the right digits, so the text check
proved nothing. The session now buffers the response audio while a challenge
is outstanding and this module transcribes it on the box.

Local engines only (``LOCAL_ASR_PROVIDERS``): the challenge must never upload
caller audio, whatever the semantic layer is configured to do. A cloud or
unknown provider is refused, and grading then fails closed.

Whisper-family providers (``local``, ``whisper``, ...) run faster-whisper
directly so the challenge can choose its own model size and languages
(docs/14 W2.c): whisper-tiny with free language detection heard Hindi digit
words as "Char punch", "Now, do." or Chinese, and even detection limited to
Hindi/English picks English on a two-word Hindi answer. So the answer is
decoded once per language in ``VR_CHALLENGE_ASR_LANGUAGES`` (default
``hi,en``) with that language forced, and graded on every reading. The model
size is ``VR_CHALLENGE_ASR_MODEL``. IndicConformer still goes through the shared ASR
stack.

Like ``semantic/signal.py`` this never raises: a missing engine reports
``(None, reason)`` and the caller decides what that means.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Importing the semantic package puts the repo root (the sibling ``backend/``
# ASR stack) on sys.path, exactly as the semantic layer needs.
from ..semantic import signal as _semantic_signal  # noqa: F401

_WHISPER_PROVIDERS = {"local", "local_whisper", "whisper", "faster_whisper", "existing_whisper"}


class _FasterWhisperDigits:
    """faster-whisper with detection limited to the expected languages."""

    def __init__(self, model_size: str, languages: List[str], device: str) -> None:
        try:
            import torch

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda" and hasattr(os, "add_dll_directory"):
                # CTranslate2 needs cuBLAS 12 on CUDA; on Windows the only copy
                # is often the one torch's CUDA build ships in torch/lib.
                lib = Path(torch.__file__).parent / "lib"
                if lib.is_dir():
                    os.add_dll_directory(str(lib))
        except ImportError:
            if device == "auto":
                device = "cpu"
        from faster_whisper import WhisperModel

        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute)
        self.languages = languages
        self.label = f"faster-whisper/{model_size}/{','.join(languages) or 'auto'}/{device}"

    def candidates(self, audio: np.ndarray) -> List[str]:
        """One reading per expected language, each decoded with that language
        forced. Detection on a two-word answer picks English for Hindi digits
        and spells "chhe teen" as "Chae deen" (docs/14 W2.c), so the engine is
        told the language instead of guessing it. With no languages set, one
        free-detection reading."""
        out = []
        for language in (self.languages or [None]):
            segments, _ = self.model.transcribe(
                audio, language=language, beam_size=5, temperature=0.0,
                condition_on_previous_text=False, without_timestamps=True,
            )
            out.append(" ".join(s.text.strip() for s in segments).strip())
        return out

    def transcribe(self, audio: np.ndarray) -> str:
        return " | ".join(self.candidates(audio))


class ChallengeTranscriber:
    """Lazily loads one local ASR engine and transcribes response audio."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_size: Optional[str] = None,
        languages: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._provider = provider
        self._model_size = model_size
        self._languages = languages
        self._device = device
        self._whisper: Optional[_FasterWhisperDigits] = None
        self._service = None
        self.load_error: Optional[str] = None

    @property
    def provider(self) -> str:
        from ..config import settings

        return (self._provider or settings.challenge.asr_provider).strip().lower()

    @property
    def label(self) -> str:
        if self._whisper is not None:
            return self._whisper.label
        return self.provider

    def _ensure_loaded(self) -> bool:
        if self._whisper is not None or self._service is not None:
            return True
        with self._lock:
            if self._whisper is not None or self._service is not None:
                return True
            from ..config import LOCAL_ASR_PROVIDERS, settings

            provider = self.provider
            if provider not in LOCAL_ASR_PROVIDERS:
                self.load_error = (
                    f"VR_CHALLENGE_ASR_PROVIDER '{provider}' is not a local engine; "
                    "the challenge never uploads caller audio"
                )
                return False
            if provider in _WHISPER_PROVIDERS:
                cfg = settings.challenge
                langs = self._languages if self._languages is not None else [
                    x.strip() for x in cfg.asr_languages.split(",") if x.strip()]
                try:
                    self._whisper = _FasterWhisperDigits(
                        self._model_size or cfg.asr_model, langs, self._device or cfg.asr_device)
                except Exception as e:
                    self.load_error = f"faster-whisper unavailable ({type(e).__name__}: {e})"
                    return False
                self.load_error = None
                return True
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

    def transcribe_candidates(
        self, pcm_int16: np.ndarray, sample_rate: int
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """Every reading the engine produced (one per expected language), or
        ``(None, reason)``. The grader passes the answer if any reading holds
        the digits in order; the audio still has to say them."""
        if not self._ensure_loaded():
            return None, self.load_error or "challenge transcriber not loaded"
        if self._whisper is None:
            text, err = self.transcribe(pcm_int16, sample_rate)
            return (None, err) if text is None else ([text], None)
        audio = pcm_int16.astype(np.float32) / 32768.0
        if sample_rate != 16000:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, 16000, sample_rate).astype(np.float32)
        try:
            return self._whisper.candidates(audio), None
        except Exception as e:
            return None, f"transcription failed ({type(e).__name__}: {e})"

    def transcribe(self, pcm_int16: np.ndarray, sample_rate: int) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(transcript, None)`` or ``(None, reason)``. Blocking."""
        if not self._ensure_loaded():
            return None, self.load_error or "challenge transcriber not loaded"
        if self._whisper is not None:
            cands, err = self.transcribe_candidates(pcm_int16, sample_rate)
            return (None, err) if cands is None else (" | ".join(cands), None)
        audio = pcm_int16.astype(np.float32) / 32768.0
        try:
            import torch

            result = self._service.transcribe_chunk(
                torch.from_numpy(audio).unsqueeze(0),
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
