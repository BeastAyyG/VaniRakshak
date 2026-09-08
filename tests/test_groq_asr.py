"""
Automated Integration and Unit Tests for Groq Whisper ASR Subsystem in VaniRakshak
Powered by Groq Cloud LPU (whisper-large-v3-turbo)
"""

import os
import pytest
import torch
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment is loaded
for p in [Path(".env"), Path("../.env")]:
    if p.exists():
        load_dotenv(p)
        break

from backend.services.asr.types import TranscriptionResult, ASRConfig
from backend.services.asr.groq_provider import GroqWhisperASR
from backend.services.asr.service import ASRService
from backend.services.asr.provider import get_asr_provider


# =====================================================================
# 1. LIVE GROQ CLOUD TRANSCRIPTION TESTS
# =====================================================================

@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required for live test")
def test_groq_transcribe_bytes_english_kyc():
    """Verify live Groq whisper-large-v3-turbo transcription on English bank scam audio."""
    provider = GroqWhisperASR.get_instance()
    
    audio_path = Path("test_audio/sample_kyc_en.mp3")
    assert audio_path.exists(), "Sample KYC audio file must exist"
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    result = provider.transcribe_bytes(audio_bytes, filename="sample_kyc_en.mp3")

    assert "transcript" in result
    transcript = result["transcript"].lower()
    assert "sbi" in transcript or "bank" in transcript or "kyc" in transcript
    assert "expire" in transcript or "blocked" in transcript
    assert result["language_code"] == "en" or "english" in result["language"].lower()
    assert result["inference_sec"] < 4.0, f"Inference took {result['inference_sec']}s, expected < 4s"
    assert result["device"] == "Groq Cloud LPU"


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required for live test")
def test_groq_transcribe_bytes_hindi_bijli():
    """Verify live Groq whisper-large-v3-turbo transcription on native Hindi electricity scam audio."""
    provider = GroqWhisperASR.get_instance()
    
    audio_path = Path("test_audio/sample_bijli_hi.mp3")
    assert audio_path.exists(), "Sample Bijli audio file must exist"
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    result = provider.transcribe_bytes(audio_bytes, filename="sample_bijli_hi.mp3")

    assert "transcript" in result
    assert "बिजली" in result["transcript"] or "उपभोक्ता" in result["transcript"] or "उपभोकता" in result["transcript"]
    assert "hinglish_text" in result
    hinglish = result["hinglish_text"].lower()
    assert "electricity" in hinglish or "bijli" in hinglish or "bill" in hinglish
    assert "kat" in hinglish or "kaat" in hinglish or "jayegi" in hinglish



@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required for live test")
def test_groq_transcribe_tensor():
    """Verify transcribe_tensor converts tensor to in-memory WAV and transcribes via Groq."""
    provider = GroqWhisperASR.get_instance()
    
    # 2 seconds of silence/low noise
    tensor = torch.randn(1, 32000) * 0.01
    res = provider.transcribe_tensor(tensor, sample_rate=16000, language="en", start_time=0.0, end_time=2.0)

    assert isinstance(res, TranscriptionResult)
    assert res.provider == "groq"
    assert res.start_time == 0.0
    assert res.end_time == 2.0
    assert res.inference_time is not None
    assert res.inference_time < 5.0


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required for live test")
def test_asr_service_full_pipeline_with_groq():
    """Verify unified ASRService orchestrates Groq provider cleanly."""
    ASRService.reset_instance()
    config = ASRConfig(provider="groq", model_name="whisper-large-v3-turbo")
    service = ASRService.get_instance(config)

    audio_path = Path("test_audio/sample_kyc_en.mp3")
    res = service.transcribe(audio_path)

    assert "hindi_text" in res
    assert "hinglish_text" in res
    assert "chunks" in res
    assert len(res["chunks"]) >= 1
    assert "sbi" in res["hindi_text"].lower() or "bank" in res["hindi_text"].lower()


# =====================================================================
# 2. ERROR HANDLING AND INPUT VALIDATION TESTS
# =====================================================================

def test_groq_empty_tensor_raises_error():
    """Verify passing empty tensor to transcribe_tensor raises ValueError."""
    provider = GroqWhisperASR.get_instance()
    with pytest.raises(ValueError) as exc:
        provider.transcribe_tensor(torch.empty(0))
    assert "empty" in str(exc.value).lower()


def test_groq_empty_bytes_raises_error():
    """Verify passing empty bytes to transcribe_bytes raises ValueError."""
    provider = GroqWhisperASR.get_instance()
    with pytest.raises(ValueError) as exc:
        provider.transcribe_bytes(b"")
    assert "empty" in str(exc.value).lower()


def test_provider_factory_selects_groq():
    """Verify get_asr_provider resolves to GroqWhisperASR."""
    provider = get_asr_provider("groq")
    assert isinstance(provider, GroqWhisperASR)
