"""
Groq Whisper ASR Provider for VaniRakshak Kavach
Powered by Groq Cloud LPU Inference (model: whisper-large-v3-turbo)

Delivers state-of-the-art multi-lingual speech-to-text with sub-second latency
for English, Hindi, Hinglish, and regional Indian languages.
"""

import os
import io
import time
import wave
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
import torch
from dotenv import load_dotenv

# Automatically load environment variables from .env
for p in [Path(".env"), Path("../.env"), Path("../../.env")]:
    if p.exists():
        load_dotenv(p)
        break


from .provider import ASRProvider
from .types import TranscriptionResult, ASRConfig
from .transliteration import HinglishTransliterationService

logger = logging.getLogger("vanirakshak.asr.groq")

# Default Groq Whisper model
DEFAULT_GROQ_MODEL = "whisper-large-v3-turbo"


class GroqWhisperASR(ASRProvider):
    """
    Adapter for Groq Cloud Whisper API.
    Replaces heavy local model execution with high-throughput LPU inference.
    """
    _instance: Optional["GroqWhisperASR"] = None

    @classmethod
    def get_instance(cls, config: Optional[ASRConfig] = None) -> "GroqWhisperASR":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def __init__(self, config: Optional[ASRConfig] = None):
        super().__init__(config)
        self.client = None
        self.model_name = getattr(self.config, "model_name", None) or DEFAULT_GROQ_MODEL
        if not self.model_name or "indic-conformer" in self.model_name:
            self.model_name = DEFAULT_GROQ_MODEL
        self._transliteration = HinglishTransliterationService()

    def _get_api_key(self) -> str:
        """Resolves Groq API key from config or environment variable."""
        key = getattr(self.config, "groq_api_key", None) or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please configure GROQ_API_KEY in your .env file."
            )
        return key.strip()

    def load_model(self) -> None:
        """Initializes Groq client."""
        if self._is_loaded and self.client is not None:
            return

        from groq import Groq
        api_key = self._get_api_key()
        logger.info(f"[ASR] Initializing Groq Whisper client (model='{self.model_name}')...")
        t0 = time.time()
        self.client = Groq(api_key=api_key)
        self._is_loaded = True
        logger.info(f"[ASR] Groq Whisper client ready in {time.time() - t0:.3f}s")

    def _tensor_to_wav_bytes(self, waveform: torch.Tensor, sample_rate: int = 16000) -> bytes:
        """Converts PyTorch tensor (1, N) into standard 16-bit PCM WAV bytes."""
        audio_np = waveform.detach().cpu().squeeze().numpy()
        
        # Clamp between -1.0 and 1.0, scale to int16
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767.0).astype(np.int16)
        
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    def transcribe_tensor(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        language: str = "hi",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> TranscriptionResult:
        """
        Transcribes a (1, N) audio tensor using Groq whisper-large-v3-turbo.
        """
        if waveform is None or waveform.numel() == 0:
            raise ValueError("Audio tensor cannot be empty")

        if not self._is_loaded:
            self.load_model()

        wav_bytes = self._tensor_to_wav_bytes(waveform, sample_rate)
        duration = float(waveform.shape[-1]) / float(sample_rate)

        t0 = time.time()
        try:
            call_kwargs = {
                "file": ("chunk.wav", wav_bytes),
                "model": self.model_name,
                "response_format": "verbose_json"
            }
            if language and language != "auto":
                call_kwargs["language"] = language

            resp = self.client.audio.transcriptions.create(**call_kwargs)
            transcript = (getattr(resp, "text", "") or "").strip()
            detected_lang = getattr(resp, "language", language) or language
            audio_duration = getattr(resp, "duration", None) or duration
        except Exception as e:
            logger.error(f"[ASR] Groq Whisper inference failed: {e}")
            raise RuntimeError(f"Groq Whisper transcription failed: {e}") from e

        inference_time = time.time() - t0
        rtf = inference_time / max(duration, 0.001)

        # Transliterate to Hinglish
        hinglish = self._transliteration.to_hinglish(transcript) if transcript else ""

        return TranscriptionResult(
            text=transcript,
            language=detected_lang,
            start_time=start_time,
            end_time=end_time,
            confidence=1.0,
            hinglish_text=hinglish,
            rtf=rtf,
            inference_time=inference_time,
            audio_duration=audio_duration,
            device_used="Groq Cloud LPU",
            provider="groq"
        )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.mp3",
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Direct transcription of audio bytes (MP3, WAV, M4A, FLAC).
        Returns a structured dictionary with segments, detected language, and full transcript.
        """
        if not audio_bytes:
            raise ValueError("Empty audio payload received")

        if not self._is_loaded:
            self.load_model()

        t0 = time.time()
        call_kwargs = {
            "file": (filename, audio_bytes),
            "model": self.model_name,
            "response_format": "verbose_json"
        }
        if language and language != "auto":
            call_kwargs["language"] = language

        try:
            resp = self.client.audio.transcriptions.create(**call_kwargs)
        except Exception as e:
            logger.error(f"[ASR] Groq audio transcription failed: {e}")
            raise RuntimeError(f"Groq audio transcription failed: {e}") from e

        inference_sec = time.time() - t0
        transcript = (getattr(resp, "text", "") or "").strip()
        detected_lang = getattr(resp, "language", "") or "unknown"
        duration = float(getattr(resp, "duration", 0.0) or 0.0)

        # Build segments
        segments = []
        raw_segments = getattr(resp, "segments", []) or []
        for s in raw_segments:
            s_text = getattr(s, "text", "") if hasattr(s, "text") else s.get("text", "")
            s_clean = s_text.strip() if s_text else ""
            if s_clean:
                s_start = getattr(s, "start", 0.0) if hasattr(s, "start") else s.get("start", 0.0)
                s_end = getattr(s, "end", 0.0) if hasattr(s, "end") else s.get("end", 0.0)
                segments.append({
                    "start": round(float(s_start), 1),
                    "end": round(float(s_end), 1),
                    "text": s_clean
                })

        lang_map = {
            "en": "English",
            "english": "English",
            "hi": "Hindi (हिंदी)",
            "hindi": "Hindi (हिंदी)",
            "ur": "Hinglish / Urdu",
            "mr": "Marathi",
            "bn": "Bengali",
            "ta": "Tamil",
            "te": "Telugu",
            "es": "Spanish"
        }
        lang_display = lang_map.get(detected_lang.lower(), detected_lang.capitalize() or "Hindi / English")
        hinglish = self._transliteration.to_hinglish(transcript) if transcript else ""

        return {
            "engine": f"Groq Whisper ({self.model_name})",
            "transcript": transcript,
            "hindi_text": transcript,
            "hinglish_text": hinglish,
            "segments": segments,
            "language_code": detected_lang.lower(),
            "language": lang_display,
            "language_probability": 0.99,
            "duration": round(duration, 2),
            "inference_sec": round(inference_sec, 3),
            "device": "Groq Cloud LPU"
        }

    def get_device_info(self) -> str:
        return "Groq Cloud LPU (whisper-large-v3-turbo)"
