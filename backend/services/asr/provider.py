"""
ASR Provider Base Class and Factory for VaniRakshak

Implements the ASRProvider interface to ensure:
- IndicConformer is the primary Hindi/Hinglish ASR provider
- Existing Faster-Whisper is preserved behind the same interface
- Zero disruption to legacy implementations
"""

from abc import ABC, abstractmethod
from typing import Optional
import os
import logging
import torch

from .types import TranscriptionResult, ASRConfig

logger = logging.getLogger("vanirakshak.asr.provider")


class ASRProvider(ABC):
    """Abstract base class for all ASR engines in VaniRakshak."""

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.device = self._detect_device(self.config.device)
        self._is_loaded = False

    def _detect_device(self, preferred_device: str = "auto") -> torch.device:
        """
        Determines execution device:
        - If CUDA is available and requested/auto -> 'cuda'
        - Otherwise gracefully falls back to 'cpu'
        """
        if preferred_device.lower() in ("cuda", "gpu"):
            if torch.cuda.is_available():
                logger.info("[ASR] Using device: cuda")
                return torch.device("cuda")
            else:
                logger.warning("[ASR] CUDA requested but not available. Falling back to CPU.")
                logger.info("[ASR] Using device: cpu")
                return torch.device("cpu")

        if preferred_device.lower() == "cpu":
            logger.info("[ASR] Using device: cpu")
            return torch.device("cpu")

        # Auto detection
        if torch.cuda.is_available():
            logger.info("[ASR] Using device: cuda")
            return torch.device("cuda")
        else:
            logger.info("[ASR] Using device: cpu")
            return torch.device("cpu")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load_model(self) -> None:
        """Loads model weights into memory ONCE during application startup."""
        pass

    @abstractmethod
    def transcribe_tensor(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        language: str = "hi",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> TranscriptionResult:
        """
        Transcribes a (1, N) 16kHz mono audio tensor into a TranscriptionResult.
        """
        pass

    def get_device_info(self) -> str:
        return str(self.device)


def get_asr_provider(provider_name: Optional[str] = None, config: Optional[ASRConfig] = None) -> ASRProvider:
    """
    Factory function to retrieve configured ASR provider.
    Defaults to 'groq' if GROQ_API_KEY is present, or configured provider.
    """
    cfg = config or ASRConfig()
    default_provider = "groq" if os.getenv("GROQ_API_KEY") else "indicconformer"
    selected = (provider_name or cfg.provider or default_provider).lower()

    if selected in ("groq", "groq_whisper", "groqcloud", "whisper_turbo"):
        from .groq_provider import GroqWhisperASR
        return GroqWhisperASR.get_instance(cfg)
    elif selected in ("indicconformer", "indic_conformer", "indic"):
        from .indicconformer import IndicConformerASR
        return IndicConformerASR.get_instance(cfg)
    elif selected in ("whisper", "faster_whisper", "existing_whisper", "local", "local_whisper"):
        from .whisper_provider import ExistingWhisperASR
        return ExistingWhisperASR.get_instance(cfg)
    else:
        raise ValueError(f"Unknown ASR provider '{selected}'. Supported: 'groq', 'indicconformer', 'whisper', 'local'")

