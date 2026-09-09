"""
Unified ASR Service for VaniRakshak Kavach

Orchestrates:
Audio Preprocessing -> 4-second Chunks -> ASRProvider (IndicConformer / Whisper) -> Hinglish Transliteration -> TranscriptBuffer
"""

import io
import time
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import numpy as np
import torch

from .types import TranscriptionResult, AudioChunk, ASRConfig
from .preprocessing import AudioPreprocessor, AudioPreprocessingError
from .provider import get_asr_provider, ASRProvider

from .buffer import TranscriptBuffer
from .transliteration import HinglishTransliterationService

logger = logging.getLogger("vanirakshak.asr.service")


class ASRService:
    """
    High-level ASR Service consumed by the rest of VaniRakshak Kavach.
    Hides internal provider and decoding details.
    """
    _instance: Optional["ASRService"] = None

    @classmethod
    def get_instance(cls, config: Optional[ASRConfig] = None) -> "ASRService":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.preprocessor = AudioPreprocessor(target_sample_rate=self.config.target_sample_rate)
        self.provider: ASRProvider = get_asr_provider(self.config.provider, self.config)
        self.transliteration = HinglishTransliterationService()

    def warmup(self) -> None:
        """Loads model into memory ONCE during service startup."""
        if not self.provider.is_loaded:
            self.provider.load_model()

    def transcribe_chunk(
        self,
        audio_chunk: Union[AudioChunk, torch.Tensor],
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribes a single 4-second audio window.
        Returns TranscriptionResult containing both native Hindi and Romanized Hinglish.
        """
        if isinstance(audio_chunk, AudioChunk):
            waveform = audio_chunk.tensor
            s_sec = audio_chunk.start_sec
            e_sec = audio_chunk.end_sec
            sr = audio_chunk.sample_rate
        else:
            waveform = audio_chunk
            s_sec = start_sec or 0.0
            sr = self.config.target_sample_rate
            e_sec = end_sec or (s_sec + waveform.shape[-1] / float(sr))

        lang = language or self.config.language or "auto"
        return self.provider.transcribe_tensor(
            waveform=waveform,
            sample_rate=sr,
            language=lang,
            start_time=s_sec,
            end_time=e_sec
        )

    def transcribe(
        self,
        audio: Union[str, Path, bytes, io.BytesIO, torch.Tensor, np.ndarray],
        language: Optional[str] = None,
        chunk_seconds: Optional[float] = None,
        overlap_seconds: float = 0.0
    ) -> Dict[str, Any]:
        """
        Processes complete call audio:
        1. Decodes and normalizes to 16kHz mono.
        2. Windows audio into 4-second chunks.
        3. Transcribes each chunk via Groq / IndicConformer ASR.
        4. Buffers chunks with start/end timestamps.
        5. Returns structured payload with both Hindi and Hinglish transcripts.
        """
        t_start = time.time()
        c_sec = chunk_seconds or self.config.chunk_seconds or 4.0
        lang = language or self.config.language or "auto"


        # 1. Preprocess
        try:
            waveform, sr, total_duration = self.preprocessor.load_and_preprocess(audio)
        except AudioPreprocessingError as preproc_err:
            # If local waveform decoding fails (e.g. mp3/m4a without ffmpeg/torchaudio),
            # but provider is Groq / cloud ASR which natively supports all audio formats:
            if hasattr(self.provider, "transcribe_bytes"):
                direct_res = self.transcribe_direct(audio, language=lang)
                chunks = [
                    {
                        "chunk_index": i,
                        "start": s.get("start", 0.0),
                        "end": s.get("end", 0.0),
                        "hindi_text": s.get("text", ""),
                        "hinglish_text": self.transliteration.to_hinglish(s.get("text", "")),
                        "confidence": 1.0,
                        "rtf": round(direct_res.get("inference_sec", 0.0) / max(direct_res.get("duration", 1.0), 0.001), 3)
                    }
                    for i, s in enumerate(direct_res.get("segments", []))
                ]
                return {
                    "hindi_text": direct_res.get("transcript", ""),
                    "hinglish_text": direct_res.get("hinglish_text", ""),
                    "full_hindi_text": direct_res.get("transcript", ""),
                    "full_hinglish_text": direct_res.get("hinglish_text", ""),
                    "total_duration": direct_res.get("duration", 0.0),
                    "chunk_count": len(chunks),
                    "chunks": chunks,
                    "language": direct_res.get("language", lang),
                    "provider": "groq",
                    "device": self.provider.get_device_info(),
                    "model": getattr(self.provider, "model_name", "whisper-large-v3-turbo"),
                    "audio_duration_seconds": direct_res.get("duration", 0.0),
                    "inference_time_seconds": direct_res.get("inference_sec", 0.0),
                    "total_processing_time_seconds": direct_res.get("inference_sec", 0.0),
                    "real_time_factor": round(direct_res.get("inference_sec", 0.0) / max(direct_res.get("duration", 1.0), 0.001), 3)
                }
            raise preproc_err

        # 2. Window into 4-second chunks
        chunks = self.preprocessor.split_into_chunks(
            waveform,
            sample_rate=sr,
            chunk_seconds=c_sec,
            overlap_seconds=overlap_seconds
        )

        buffer = TranscriptBuffer()
        total_inference_time = 0.0

        # 3. Transcribe each chunk
        for chunk in chunks:
            chunk_res = self.transcribe_chunk(chunk, language=lang)
            buffer.add_chunk_result(chunk_res)
            if chunk_res.inference_time:
                total_inference_time += chunk_res.inference_time

        overall_time = time.time() - t_start
        overall_rtf = total_inference_time / max(total_duration, 0.001)

        result_payload = buffer.to_dict()
        result_payload.update({
            "full_hindi_text": buffer.full_hindi_text,
            "full_hinglish_text": buffer.full_hinglish_text,
            "language": lang,
            "provider": self.config.provider,
            "device": self.provider.get_device_info(),
            "model": self.config.model_name,
            "audio_duration_seconds": round(total_duration, 2),
            "inference_time_seconds": round(total_inference_time, 3),
            "total_processing_time_seconds": round(overall_time, 3),
            "real_time_factor": round(overall_rtf, 3)
        })


        return result_payload

    def transcribe_direct(
        self,
        audio: Union[str, Path, bytes, io.BytesIO],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Direct audio file or bytes transcription via Groq Whisper Cloud
        without tensor conversion for ultra-low latency.
        """
        if hasattr(self.provider, "transcribe_bytes"):
            if isinstance(audio, (str, Path)):
                p = Path(audio)
                with open(p, "rb") as f:
                    audio_bytes = f.read()
                filename = p.name
            elif isinstance(audio, io.BytesIO):
                audio.seek(0)
                audio_bytes = audio.read()
                filename = "audio.wav"
            else:
                audio_bytes = audio
                filename = "audio.mp3"
            return self.provider.transcribe_bytes(audio_bytes, filename=filename, language=language)
        return self.transcribe(audio, language=language)

