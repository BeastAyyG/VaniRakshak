"""
AI4Bharat IndicConformer / Groq Whisper ASR Engine — VaniRakshak Indic Edition
Powered by Groq Cloud Whisper Large v3 Turbo
Supports Hindi, Hinglish, English (Indian Accent), and 22+ Indian Languages with sub-second latency.
"""

import os
import io
import sys
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load environment
for p in [Path(".env"), Path("../.env"), Path("../../.env")]:
    if p.exists():
        load_dotenv(p)
        break

logger = logging.getLogger("indic_conformer_engine")
logging.basicConfig(level=logging.INFO)

# Model Instance Wrapper
_groq_client = None
_model_name = "whisper-large-v3-turbo"

def get_groq_client():
    """Initializes Groq Cloud Whisper Client."""
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY is not set in environment!")
        logger.info(f"Initializing Groq Cloud Whisper Engine ({_model_name})...")
        t0 = time.time()
        _groq_client = Groq(api_key=api_key)
        logger.info(f"Groq Cloud Whisper Engine ready in {time.time() - t0:.2f}s")
    return _groq_client

def get_indic_conformer_model():
    """Compatibility alias for server lifespan."""
    return get_groq_client()

def transcribe_with_indic_conformer(audio_bytes: bytes) -> Dict:
    """
    Transcribes audio bytes (up to 1 minute / 60s max) using Groq Whisper Large v3 Turbo.
    Decodes Hindi, Hinglish, English (Indian Accent), and regional Indian speech.
    """
    t0 = time.time()
    client = get_groq_client()
    
    try:
        resp = client.audio.transcriptions.create(
            file=("audio.mp3", audio_bytes),
            model=_model_name,
            response_format="verbose_json"
        )
    except Exception as exc:
        logger.error(f"Groq transcription error: {exc}")
        raise RuntimeError(f"Groq transcription failure: {exc}") from exc

    transcript = (getattr(resp, "text", "") or "").strip()
    inference_sec = time.time() - t0
    detected_lang = getattr(resp, "language", "hi") or "hi"
    duration = float(getattr(resp, "duration", 0.0) or 0.0)

    segments_list = []
    raw_segments = getattr(resp, "segments", []) or []
    for s in raw_segments:
        s_text = getattr(s, "text", "") if hasattr(s, "text") else s.get("text", "")
        if s_text and s_text.strip():
            s_start = getattr(s, "start", 0.0) if hasattr(s, "start") else s.get("start", 0.0)
            s_end = getattr(s, "end", 0.0) if hasattr(s, "end") else s.get("end", 0.0)
            segments_list.append({
                "start": round(float(s_start), 1),
                "end": round(float(s_end), 1),
                "text": s_text.strip()
            })

    lang_map = {
        "hi": "Hindi (हिंदी) / Hinglish",
        "hindi": "Hindi (हिंदी) / Hinglish",
        "en": "English (Indian Accent)",
        "english": "English (Indian Accent)",
        "ur": "Hinglish / Urdu",
        "mr": "Marathi (मराठी)",
        "bn": "Bengali (বাংলা)",
        "gu": "Gujarati (ગુજરાતી)",
        "ta": "Tamil (தமிழ்)",
        "te": "Telugu (తెలుగు)",
        "kn": "Kannada (ಕನ್ನಡ)",
        "pa": "Punjabi (ਪੰਜਾਬੀ)"
    }
    lang_display = lang_map.get(detected_lang.lower(), f"Indic ({detected_lang.upper()})")

    # Generate dual representation: Hindi (native script) + Romanized Hinglish.
    # The transliteration service lives in a different project layout and is
    # not vendored here — import it lazily and fall back to the raw
    # transcript so a successful Groq call never ends in ModuleNotFoundError.
    try:
        from backend.services.asr.transliteration import HinglishTransliterationService
        hinglish_text = HinglishTransliterationService().to_hinglish(transcript)
    except Exception as exc:
        logger.warning(f"Hinglish transliteration unavailable ({exc}); returning raw transcript")
        hinglish_text = transcript

    return {
        "engine": "Groq Cloud Whisper (whisper-large-v3-turbo)",
        "model": "whisper-large-v3-turbo",
        "transcript": transcript,
        "hindi_text": transcript,
        "hinglish_text": hinglish_text,
        "segments": segments_list,
        "language_code": detected_lang.lower(),
        "language": lang_display,
        # Groq verbose_json returns no language confidence — None, not a
        # fabricated 0.99. No consumer reads this key (checked server.py + UI).
        "language_probability": None,
        "duration": round(duration, 2),
        "inference_sec": round(inference_sec, 3)
    }

if __name__ == "__main__":
    print("Testing Groq Whisper Engine...")
    model = get_indic_conformer_model()
    print("Groq Whisper Engine loaded successfully!")
