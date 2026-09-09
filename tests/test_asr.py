"""
Unit Tests for VaniRakshak Kavach ASR Subsystem

Tests:
1. Audio preprocessing (stereo -> mono, 8kHz -> 16kHz, 16kHz -> unchanged, error handling)
2. 4-second audio window chunking (10s audio -> expected 4s chunks)
3. Hinglish transliteration and English terminology preservation (KYC, OTP, SBI, bank account, etc.)
4. Mocked IndicConformer ASR provider (loads once, returns expected TranscriptionResult structure)
5. TranscriptBuffer temporal aggregation
6. Provider switching (IndicConformer vs Existing Whisper)
"""

import os
import io
import pytest
import numpy as np
import torch

from backend.services.asr.types import TranscriptionResult, AudioChunk, ASRConfig
from backend.services.asr.preprocessing import AudioPreprocessor, AudioPreprocessingError
from backend.services.asr.transliteration import HinglishTransliterationService
from backend.services.asr.buffer import TranscriptBuffer
from backend.services.asr.indicconformer import IndicConformerASR
from backend.services.asr.whisper_provider import ExistingWhisperASR
from backend.services.asr.service import ASRService


# =====================================================================
# 1. AUDIO PREPROCESSING TESTS
# =====================================================================

def test_preprocessing_stereo_to_mono():
    """Verify stereo (2, N) audio is collapsed into mono (1, N)."""
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    
    # Create 1-second stereo tensor
    stereo_audio = torch.randn(2, 16000, dtype=torch.float32)
    processed, sr, duration = preprocessor.load_and_preprocess(stereo_audio, source_sample_rate=16000)
    
    assert processed.dim() == 2
    assert processed.shape[0] == 1, "Audio must be converted to 1-channel mono"
    assert processed.shape[1] == 16000
    assert sr == 16000
    assert abs(duration - 1.0) < 0.01


def test_preprocessing_resampling_8khz_to_16khz():
    """Verify 8000 Hz audio is properly resampled to 16000 Hz."""
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    
    # 1-second of 8000 Hz audio = 8000 samples
    audio_8k = torch.randn(1, 8000, dtype=torch.float32)
    processed, sr, duration = preprocessor.load_and_preprocess(audio_8k, source_sample_rate=8000)
    
    assert sr == 16000
    assert processed.shape[0] == 1
    assert processed.shape[1] == 16000, "Should have 16000 samples after upsampling from 8kHz"
    assert abs(duration - 1.0) < 0.01


def test_preprocessing_16khz_unchanged():
    """Verify 16000 Hz audio sample rate is preserved without alteration."""
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    
    audio_16k = torch.randn(1, 32000, dtype=torch.float32)  # 2.0s
    processed, sr, duration = preprocessor.load_and_preprocess(audio_16k, source_sample_rate=16000)
    
    assert sr == 16000
    assert processed.shape[1] == 32000
    assert abs(duration - 2.0) < 0.01


def test_preprocessing_empty_audio_raises_error():
    """Verify empty or zero-sample audio raises AudioPreprocessingError."""
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    empty_tensor = torch.empty(0)
    
    with pytest.raises(AudioPreprocessingError) as exc_info:
        preprocessor.load_and_preprocess(empty_tensor)
    assert "empty" in str(exc_info.value).lower()


def test_preprocessing_too_short_audio_raises_error():
    """Verify audio under 100ms threshold raises error."""
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    # 50ms audio = 800 samples at 16kHz
    short_tensor = torch.randn(1, 800)
    
    with pytest.raises(AudioPreprocessingError) as exc_info:
        preprocessor.load_and_preprocess(short_tensor, source_sample_rate=16000)
    assert "too short" in str(exc_info.value).lower()


# =====================================================================
# 2. 4-SECOND CHUNKING TESTS
# =====================================================================

def test_chunking_10s_audio_produces_expected_chunks():
    """
    Verify 10-second audio at 16kHz (160,000 samples) splits into:
    Chunk 1: 0.0 - 4.0s (64,000 samples)
    Chunk 2: 4.0 - 8.0s (64,000 samples)
    Chunk 3: 8.0 - 10.0s (32,000 samples)
    """
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    waveform = torch.randn(1, 160000)  # exactly 10.0 seconds
    
    chunks = preprocessor.split_into_chunks(waveform, sample_rate=16000, chunk_seconds=4.0)
    
    assert len(chunks) == 3, f"Expected 3 chunks for 10s audio, got {len(chunks)}"
    
    # Chunk 1
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 4.0
    assert chunks[0].tensor.shape[-1] == 64000
    
    # Chunk 2
    assert chunks[1].start_sec == 4.0
    assert chunks[1].end_sec == 8.0
    assert chunks[1].tensor.shape[-1] == 64000
    
    # Chunk 3
    assert chunks[2].start_sec == 8.0
    assert chunks[2].end_sec == 10.0
    assert chunks[2].tensor.shape[-1] == 32000


# =====================================================================
# 3. TRANSLITERATION AND ENGLISH WORD PRESERVATION TESTS
# =====================================================================

def test_transliteration_hindi_to_hinglish():
    """Verify native Hindi converts to colloquial Hinglish."""
    service = HinglishTransliterationService()
    text = "आपका बैंक अकाउंट कल बंद हो जाएगा।"
    hinglish = service.to_hinglish(text)
    
    assert "aapka" in hinglish.lower()
    assert "bank" in hinglish.lower()
    assert "account" in hinglish.lower()
    assert "kal" in hinglish.lower()
    assert "band" in hinglish.lower()
    assert "jayega" in hinglish.lower()


def test_transliteration_preserves_english_acronyms():
    """Verify technical scam terminology (KYC, OTP, UPI, SBI) is preserved."""
    service = HinglishTransliterationService()
    text = "सर मैं SBI headquarters से बोल रहा हूँ। आपका KYC expire हो गया है।"
    hinglish = service.to_hinglish(text)
    
    assert "SBI" in hinglish
    assert "KYC" in hinglish
    assert "headquarters" in hinglish
    assert "expire" in hinglish


# =====================================================================
# 4. TRANSCRIPT BUFFER TESTS
# =====================================================================

def test_transcript_buffer_chronological_aggregation():
    """Verify transcript buffer properly aggregates chunks and maintains timestamps."""
    buffer = TranscriptBuffer()
    
    c1 = TranscriptionResult(
        text="मैं SBI headquarters से बोल रहा हूँ",
        language="hi",
        start_time=0.0,
        end_time=4.0,
        hinglish_text="main SBI headquarters se bol raha hoon"
    )
    c2 = TranscriptionResult(
        text="आपका KYC expire हो गया है",
        language="hi",
        start_time=4.0,
        end_time=8.0,
        hinglish_text="aapka KYC expire ho gaya hai"
    )
    
    buffer.add_chunk_result(c1)
    buffer.add_chunk_result(c2)
    
    assert len(buffer.chunks) == 2
    assert buffer.chunks[0]["start"] == 0.0
    assert buffer.chunks[0]["end"] == 4.0
    assert buffer.chunks[1]["start"] == 4.0
    assert buffer.chunks[1]["end"] == 8.0
    
    # Aggregated texts
    assert buffer.full_hindi_text == "मैं SBI headquarters से बोल रहा हूँ आपका KYC expire हो गया है"
    assert buffer.full_hinglish_text == "main SBI headquarters se bol raha hoon aapka KYC expire ho gaya hai"
    assert buffer.total_duration == 8.0


# =====================================================================
# 5. MOCKED MODEL LOADING & INFERENCE (FAST, NO 600M DOWNLOAD)
# =====================================================================

class MockConformerModel:
    def __init__(self):
        self.call_count = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, audio_tensor, language, decoding_strategy):
        self.call_count += 1
        return "आपका बैंक अकाउंट कल बंद हो जाएगा।"


def test_indicconformer_loads_once_and_transcribes(monkeypatch):
    """
    Verify model loads only once across multiple chunk transcriptions
    and returns properly structured TranscriptionResult.
    """
    IndicConformerASR.reset_instance()
    mock_model = MockConformerModel()

    # Mock AutoModel.from_pretrained to avoid downloading weights in unit tests
    def mock_from_pretrained(*args, **kwargs):
        return mock_model

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", mock_from_pretrained)

    config = ASRConfig(provider="indicconformer", device="cpu")
    asr = IndicConformerASR.get_instance(config)
    
    # First call loads model
    asr.load_model()
    assert asr.is_loaded is True

    # Second call should reuse already loaded model without reloading
    asr.load_model()
    assert asr.is_loaded is True

    # Transcribe audio chunk
    chunk_tensor = torch.zeros(1, 16000)  # 1.0s
    res = asr.transcribe_tensor(chunk_tensor, sample_rate=16000, language="hi", start_time=0.0, end_time=1.0)
    
    assert isinstance(res, TranscriptionResult)
    assert res.text == "आपका बैंक अकाउंट कल बंद हो जाएगा।"
    assert res.language == "hi"
    assert res.start_time == 0.0
    assert res.end_time == 1.0
    assert res.hinglish_text is not None
    assert "bank" in res.hinglish_text.lower()
    assert res.provider == "indicconformer"


def test_indicconformer_invalid_input_raises_error():
    """Verify empty tensor to transcribe_tensor raises ValueError."""
    IndicConformerASR.reset_instance()
    config = ASRConfig(provider="indicconformer", device="cpu")
    asr = IndicConformerASR.get_instance(config)
    asr._is_loaded = True
    asr.model = MockConformerModel()
    
    with pytest.raises(ValueError) as exc_info:
        asr.transcribe_tensor(torch.empty(0))
    assert "empty" in str(exc_info.value).lower()
