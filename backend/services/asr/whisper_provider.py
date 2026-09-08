"""
Existing Faster-Whisper ASR Provider for VaniRakshak

Preserves the existing Whisper implementation behind the common ASRProvider interface.
Does NOT modify or replace the legacy whisper scripts or files.
"""

import time
import logging
from typing import Optional
import numpy as np
import torch

from .provider import ASRProvider
from .types import TranscriptionResult, ASRConfig
from .transliteration import HinglishTransliterationService

logger = logging.getLogger("vanirakshak.asr.whisper")


class ExistingWhisperASR(ASRProvider):
    """
    Adapter for faster-whisper ensuring backward compatibility.
    Singleton pattern ensures the model is loaded once and kept in memory.
    """
    _instance: Optional["ExistingWhisperASR"] = None

    @classmethod
    def get_instance(cls, config: Optional[ASRConfig] = None) -> "ExistingWhisperASR":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def __init__(self, config: Optional[ASRConfig] = None):
        super().__init__(config)
        self.model = None
        self._transliteration = HinglishTransliterationService()

    def load_model(self) -> None:
        if self._is_loaded and self.model is not None:
            return

        from faster_whisper import WhisperModel
        logger.info(f"[ASR] Loading Existing Whisper model on {self.device}...")
        t0 = time.time()
        
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        compute_type = "float16" if device_str == "cuda" else "int8"

        try:
            self.model = WhisperModel("tiny", device=device_str, compute_type=compute_type)
            self._is_loaded = True
            logger.info(f"[ASR] Existing Whisper model loaded successfully in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.error(f"[ASR] Failed to load Whisper model: {e}")
            raise RuntimeError(f"Whisper model loading failure: {str(e)}") from e

    def transcribe_tensor(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        language: str = "hi",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> TranscriptionResult:
        if not self._is_loaded:
            self.load_model()

        # Convert tensor to 1D float32 numpy array expected by faster-whisper
        audio_np = waveform.squeeze().cpu().numpy()
        duration = float(len(audio_np)) / float(sample_rate)

        t0 = time.time()
        try:
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                language=language if language != "auto" else None,
                vad_filter=True
            )
            text_parts = [s.text.strip() for s in segments if s.text.strip()]
            transcript = " ".join(text_parts).strip()
        except Exception as e:
            logger.error(f"[ASR] Whisper inference failed: {e}")
            raise RuntimeError(f"Whisper inference failure: {str(e)}") from e

        inference_time = time.time() - t0
        rtf = inference_time / max(duration, 0.001)

        # Hinglish representation
        hinglish = self._transliteration.to_hinglish(transcript)

        return TranscriptionResult(
            text=transcript,
            language=language,
            start_time=start_time,
            end_time=end_time,
            confidence=getattr(info, "language_probability", None),
            hinglish_text=hinglish,
            rtf=rtf,
            inference_time=inference_time,
            audio_duration=duration,
            device_used=str(self.device),
            provider="whisper"
        )
