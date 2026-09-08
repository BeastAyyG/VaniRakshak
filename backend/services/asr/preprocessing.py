"""
Audio Preprocessing Pipeline for VaniRakshak ASR

Converts incoming call audio (WAV, MP3, M4A, FLAC) into:
- Sample rate: 16000 Hz
- Channels: 1 (Mono)
- Normalized PyTorch FloatTensor [-1.0, 1.0]
- Discrete 4-second audio chunks with optional overlap support
"""

import io
import os
import wave
import logging
from pathlib import Path
from typing import Union, Tuple, List, Optional
import numpy as np
import torch

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    import soundfile as sf
except ImportError:
    sf = None

from .types import AudioChunk

logger = logging.getLogger("vanirakshak.asr.preprocessing")


TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_SEC = 0.1  # Audios shorter than 100ms are treated as invalid/empty


class AudioPreprocessingError(Exception):
    """Raised when audio decoding, conversion, or validation fails."""
    pass


class AudioPreprocessor:
    """
    Handles robust loading, channel collapsing, resampling, and windowing
    for incoming voice call recordings.
    """

    def __init__(self, target_sample_rate: int = TARGET_SAMPLE_RATE):
        self.target_sample_rate = target_sample_rate

    def load_and_preprocess(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO, torch.Tensor, np.ndarray],
        source_sample_rate: Optional[int] = None
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Loads and standardizes audio to (1, N) 16kHz mono PyTorch tensor.

        Returns:
            Tuple of (audio_tensor, sample_rate, duration_seconds)
        """
        waveform, sr = self._decode_input(audio_input, source_sample_rate)

        # 1. Validate non-empty
        if waveform is None or waveform.numel() == 0:
            raise AudioPreprocessingError("Audio waveform is empty or has 0 samples.")

        # 2. Convert Stereo -> Mono (average across channel dimension)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.dim() > 2:
            waveform = waveform.squeeze()
            if waveform.dim() > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            else:
                waveform = waveform.unsqueeze(0)

        # 3. Resample to target sample rate (16000 Hz) if needed
        if sr != self.target_sample_rate:
            if sr <= 0:
                raise AudioPreprocessingError(f"Invalid source sample rate: {sr}")
            if torchaudio is not None:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)
            else:
                new_len = int(round(waveform.shape[-1] * (self.target_sample_rate / float(sr))))
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
                ).squeeze(0)
            sr = self.target_sample_rate

        # Ensure float32 format
        if waveform.dtype != torch.float32:
            waveform = waveform.to(torch.float32)

        # Normalize amplitude if outside [-1.0, 1.0]
        max_val = torch.max(torch.abs(waveform)).item()
        if max_val > 1.0:
            waveform = waveform / max_val

        duration = waveform.shape[-1] / float(sr)

        # 4. Check minimum duration
        if duration < MIN_AUDIO_DURATION_SEC:
            raise AudioPreprocessingError(
                f"Audio is too short ({duration:.3f}s). Minimum required is {MIN_AUDIO_DURATION_SEC}s."
            )

        return waveform, sr, duration

    @staticmethod
    def _read_with_wave(source: Union[str, Path, io.BytesIO]) -> Tuple[torch.Tensor, int]:
        """Fallback WAV decoder using Python standard library 'wave'."""
        with wave.open(source, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            
            if sampwidth == 2:
                audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                audio_np = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            elif sampwidth == 1:
                audio_np = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                audio_np = np.frombuffer(frames, dtype=np.float32)
                
            if n_channels > 1:
                audio_np = audio_np.reshape(-1, n_channels).T
            else:
                audio_np = audio_np.reshape(1, -1)
            return torch.from_numpy(audio_np.copy()), sr

    def _decode_input(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO, torch.Tensor, np.ndarray],
        source_sample_rate: Optional[int]
    ) -> Tuple[torch.Tensor, int]:
        """Loads audio from various sources into a torch tensor."""
        if isinstance(audio_input, torch.Tensor):
            sr = source_sample_rate or self.target_sample_rate
            if audio_input.dim() == 1:
                audio_input = audio_input.unsqueeze(0)
            return audio_input, sr

        if isinstance(audio_input, np.ndarray):
            tensor = torch.from_numpy(audio_input)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            sr = source_sample_rate or self.target_sample_rate
            return tensor, sr

        # File path
        if isinstance(audio_input, (str, Path)):
            path_obj = Path(audio_input)
            if not path_obj.exists():
                raise AudioPreprocessingError(f"Audio file not found: {path_obj}")
            if path_obj.stat().st_size == 0:
                raise AudioPreprocessingError(f"Audio file is empty (0 bytes): {path_obj}")

            if sf is not None:
                try:
                    data, sr = sf.read(str(path_obj), dtype="float32")
                    tensor = torch.from_numpy(data)
                    if tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.dim() == 2:
                        tensor = tensor.t()
                    return tensor, sr
                except Exception:
                    pass

            if torchaudio is not None:
                try:
                    waveform, sr = torchaudio.load(str(path_obj))
                    return waveform, sr
                except Exception:
                    pass

            try:
                return self._read_with_wave(str(path_obj))
            except Exception as e:
                raise AudioPreprocessingError(f"Failed to decode audio file '{path_obj.name}': {e}") from e

        # In-memory bytes
        if isinstance(audio_input, (bytes, io.BytesIO)):
            bio = io.BytesIO(audio_input) if isinstance(audio_input, bytes) else audio_input
            if bio.getbuffer().nbytes == 0:
                raise AudioPreprocessingError("In-memory audio bytes buffer is empty.")
            bio.seek(0)

            if sf is not None:
                try:
                    data, sr = sf.read(bio, dtype="float32")
                    tensor = torch.from_numpy(data)
                    if tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.dim() == 2:
                        tensor = tensor.t()
                    return tensor, sr
                except Exception:
                    bio.seek(0)

            if torchaudio is not None:
                try:
                    waveform, sr = torchaudio.load(bio)
                    return waveform, sr
                except Exception:
                    bio.seek(0)

            try:
                return self._read_with_wave(bio)
            except Exception as e:
                raise AudioPreprocessingError(f"Failed to decode audio bytes: {e}") from e

        raise AudioPreprocessingError(f"Unsupported audio input type: {type(audio_input)}")


    def split_into_chunks(
        self,
        waveform: torch.Tensor,
        sample_rate: int = TARGET_SAMPLE_RATE,
        chunk_seconds: float = 4.0,
        overlap_seconds: float = 0.0
    ) -> List[AudioChunk]:
        """
        Splits full waveform into consecutive windows (e.g. 4.0-second chunks).
        Supports optional overlap.
        """
        if chunk_seconds <= 0:
            raise ValueError(f"chunk_seconds must be positive, got {chunk_seconds}")
        if overlap_seconds >= chunk_seconds:
            raise ValueError("overlap_seconds cannot be greater than or equal to chunk_seconds")

        total_samples = waveform.shape[-1]
        samples_per_chunk = int(chunk_seconds * sample_rate)
        step_samples = int((chunk_seconds - overlap_seconds) * sample_rate)

        chunks: List[AudioChunk] = []
        start_idx = 0

        while start_idx < total_samples:
            end_idx = min(start_idx + samples_per_chunk, total_samples)
            chunk_tensor = waveform[:, start_idx:end_idx]

            chunk_duration = chunk_tensor.shape[-1] / float(sample_rate)
            if chunk_duration >= MIN_AUDIO_DURATION_SEC:
                start_sec = start_idx / float(sample_rate)
                end_sec = end_idx / float(sample_rate)
                chunks.append(
                    AudioChunk(
                        tensor=chunk_tensor,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        sample_rate=sample_rate
                    )
                )

            if end_idx >= total_samples:
                break

            start_idx += step_samples

        return chunks
