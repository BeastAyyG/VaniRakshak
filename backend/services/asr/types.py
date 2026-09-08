"""
VaniRakshak ASR - Data Models and Configuration Types
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os
import torch


@dataclass
class TranscriptionResult:
    """
    Standardized result for an audio transcription slice or chunk.
    Canonical transcript is native-script Hindi (text),
    while hinglish_text is the normalized/transliterated Romanized form.
    """
    text: str
    language: str = "hi"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: Optional[float] = None
    hinglish_text: Optional[str] = None
    rtf: Optional[float] = None  # Real-Time Factor: inference_time / audio_duration
    inference_time: Optional[float] = None
    audio_duration: Optional[float] = None
    device_used: Optional[str] = None
    provider: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "hindi_text": self.text,
            "hinglish_text": self.hinglish_text or "",
            "language": self.language,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "rtf": round(self.rtf, 4) if self.rtf is not None else None,
            "inference_time": round(self.inference_time, 4) if self.inference_time is not None else None,
            "audio_duration": round(self.audio_duration, 4) if self.audio_duration is not None else None,
            "device_used": self.device_used,
            "provider": self.provider
        }


@dataclass
class AudioChunk:
    """
    Represents a discrete audio window (e.g. 4-second chunk) in memory.
    """
    tensor: torch.Tensor
    start_sec: float
    end_sec: float
    sample_rate: int = 16000

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class ASRConfig:
    """
    Configuration parameters for ASR service.
    Loads defaults from environment variables where available.
    """
    model_name: str = field(
        default_factory=lambda: os.getenv("ASR_MODEL", "whisper-large-v3-turbo")
    )
    language: str = field(
        default_factory=lambda: os.getenv("ASR_LANGUAGE", "auto")
    )

    target_sample_rate: int = field(
        default_factory=lambda: int(os.getenv("ASR_SAMPLE_RATE", "16000"))
    )
    device: str = field(
        default_factory=lambda: os.getenv("ASR_DEVICE", "auto")
    )
    chunk_seconds: float = field(
        default_factory=lambda: float(os.getenv("ASR_CHUNK_SECONDS", "4.0"))
    )
    provider: str = field(
        default_factory=lambda: os.getenv("ASR_PROVIDER", "groq")
    )
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )

