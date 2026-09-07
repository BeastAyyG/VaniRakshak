"""
AI4Bharat IndicConformer ASR Provider for VaniRakshak

Official Reference:
- AI4Bharat IndicConformerASR: https://github.com/AI4Bharat/IndicConformerASR
- Hugging Face IndicConformer 600M Multilingual: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual

Features:
- Singleton lifecycle: Loaded ONCE during application startup and kept in memory.
- Dynamic hardware acceleration: Uses CUDA GPU if available, otherwise falls back gracefully to CPU.
- Explicit language identifier: 'hi' (Hindi / Hinglish code-switching).
- Decoding: Native CTC / RNNT decoding with torch.inference_mode().
- Error reporting: Clear, actionable diagnostic exceptions.
"""

import os
import time
import logging
from typing import Optional, Union, Dict, Any
import torch

from .provider import ASRProvider
from .types import TranscriptionResult, ASRConfig
from .transliteration import HinglishTransliterationService

logger = logging.getLogger("vanirakshak.asr.indicconformer")


class IndicConformerASR(ASRProvider):
    """
    Primary Hindi / Hinglish Speech-to-Text implementation using
    AI4Bharat's IndicConformer 600M Multilingual model.
    """
    _instance: Optional["IndicConformerASR"] = None

    @classmethod
    def get_instance(cls, config: Optional[ASRConfig] = None) -> "IndicConformerASR":
        """Singleton accessor to ensure model is loaded only once across the process."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resets singleton instance (useful for unit testing)."""
        cls._instance = None

    def __init__(self, config: Optional[ASRConfig] = None):
        super().__init__(config)
        self.model = None
        self.model_name = self.config.model_name
        self.language = self.config.language or "hi"
        self._transliteration = HinglishTransliterationService()

    def load_model(self) -> None:
        """
        Loads the official AI4Bharat IndicConformer model.
        Loaded once at startup and maintained in memory.
        """
        if self._is_loaded and self.model is not None:
            return

        logger.info(f"[ASR] Loading IndicConformer: {self.model_name}...")
        logger.info(f"[ASR] Target Language: {self.language}")
        logger.info(f"[ASR] Device: {self.device}")

        t0 = time.time()
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        try:
            from transformers import AutoModel

            kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
            }
            if hf_token:
                kwargs["token"] = hf_token

            # Load model
            self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
            self.model.to(self.device)
            self.model.eval()

            load_time = time.time() - t0
            self._is_loaded = True
            logger.info(f"[ASR] Model loaded successfully in {load_time:.2f}s on {self.device}")

        except Exception as e:
            err_msg = str(e)
            logger.error(f"[ASR] Model loading failure: {err_msg}")
            
            allow_fallback = os.getenv("ASR_FALLBACK_TO_WHISPER", "true").lower() in ("true", "1", "yes")
            if allow_fallback and ("gated repo" in err_msg.lower() or "401" in err_msg or "restricted" in err_msg.lower() or "not found" in err_msg.lower()):
                logger.warning(
                    f"[ASR] Hugging Face gated access restricted for '{self.model_name}'. "
                    f"Engaging ExistingWhisperASR fallback provider to ensure uninterrupted service."
                )
                from .whisper_provider import ExistingWhisperASR
                self._fallback_provider = ExistingWhisperASR.get_instance(self.config)
                self._fallback_provider.load_model()
                self._is_loaded = True
                return

            if "gated repo" in err_msg.lower() or "401" in err_msg or "restricted" in err_msg.lower():
                raise RuntimeError(
                    f"ASR model failed to load: Hugging Face gated access required for '{self.model_name}'. "
                    f"Please request access at https://huggingface.co/{self.model_name} and set your HF_TOKEN "
                    f"in .env or run `huggingface-cli login`. Original error: {err_msg}"
                ) from e
            elif "out of memory" in err_msg.lower() or "cuda oom" in err_msg.lower():
                raise RuntimeError(
                    f"ASR model failed to load: CUDA Out of Memory on {self.device}. "
                    f"Consider setting ASR_DEVICE=cpu. Original error: {err_msg}"
                ) from e
            else:
                raise RuntimeError(f"ASR model failed to load: {err_msg}") from e

    def transcribe_tensor(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        language: str = "hi",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        decoding_strategy: str = "ctc"
    ) -> TranscriptionResult:
        """
        Runs inference on 16kHz mono audio tensor:
        Audio tensor -> IndicConformer -> Hindi text -> Hinglish transliteration -> TranscriptionResult
        """
        if not self._is_loaded:
            self.load_model()

        if getattr(self, "_fallback_provider", None) is not None:
            res = self._fallback_provider.transcribe_tensor(
                waveform=waveform,
                sample_rate=sample_rate,
                language=language or self.language or "hi",
                start_time=start_time,
                end_time=end_time
            )
            res.provider = "indicconformer (fallback: whisper)"
            return res

        if self.model is None:
            raise RuntimeError("ASR model is not loaded.")

        if waveform is None or waveform.numel() == 0:
            raise ValueError("Inference failure: Input audio waveform is empty.")

        # Ensure (1, N) or (B, N) shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        duration = waveform.shape[-1] / float(sample_rate)
        lang = language or self.language or "hi"

        t0 = time.time()
        logger.info(f"[ASR] Processing chunk: {start_time or 0.0:.1f}-{end_time or duration:.1f}s (duration: {duration:.2f}s)")

        # Move to model device
        audio_tensor = waveform.to(self.device)

        try:
            with torch.inference_mode():
                # Official AI4Bharat IndicConformer inference call: model(wav, lang, strategy)
                output = self.model(audio_tensor, lang, decoding_strategy)

            # Format raw output into string transcript
            if isinstance(output, (list, tuple)):
                raw_text = " ".join(str(item) for item in output if item).strip()
            elif isinstance(output, dict):
                raw_text = output.get("text", output.get("transcription", "")).strip()
            else:
                raw_text = str(output).strip()

        except Exception as e:
            logger.error(f"[ASR] Inference failure on {self.device}: {e}")
            raise RuntimeError(f"ASR inference failure: {str(e)}") from e

        inference_time = time.time() - t0
        rtf = inference_time / max(duration, 0.001)

        # Generate secondary Hinglish representation while preserving technical/fraud English terms
        hinglish = self._transliteration.to_hinglish(raw_text)

        logger.info(f"[ASR] Result (Hindi): {raw_text}")
        logger.info(f"[ASR] Result (Hinglish): {hinglish}")
        logger.info(f"[ASR] Latency: {inference_time:.2f}s | RTF: {rtf:.2f}")

        return TranscriptionResult(
            text=raw_text,
            language=lang,
            start_time=start_time,
            end_time=end_time,
            confidence=None,  # IndicConformer CTC doesn't yield a direct calibrated confidence score; do not fake it
            hinglish_text=hinglish,
            rtf=rtf,
            inference_time=inference_time,
            audio_duration=duration,
            device_used=str(self.device),
            provider="indicconformer"
        )
