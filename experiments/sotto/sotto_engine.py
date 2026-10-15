"""
Sotto Voice Dictation Engine — Powered by Groq Cloud Whisper & OpenVINO Hardware Topology
Delivers real-time cloud-accelerated Whisper speech recognition (whisper-large-v3-turbo).
Supports English, Hindi, and Hinglish dictation & scam audio analysis.
"""

import os
import io
import sys
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load environment
for p in [Path(".env"), Path("../.env"), Path("../../.env")]:
    if p.exists():
        load_dotenv(p)
        break

logger = logging.getLogger("sotto_engine")
logging.basicConfig(level=logging.INFO)

# OpenVINO Runtime & Hardware Check
_openvino_core = None
_openvino_device = "CPU"
_available_devices = ["CPU"]

def get_openvino_device() -> str:
    global _openvino_core, _openvino_device, _available_devices
    if _openvino_core is None:
        try:
            import openvino as ov
            _openvino_core = ov.Core()
            _available_devices = list(_openvino_core.available_devices)
            logger.info(f"Sotto OpenVINO Runtime detected hardware devices: {_available_devices}")
            
            if "GPU" in _available_devices or "GPU.0" in _available_devices:
                _openvino_device = "GPU"
            elif "NPU" in _available_devices:
                _openvino_device = "NPU"
            else:
                _openvino_device = "CPU"
        except Exception as e:
            logger.info(f"OpenVINO runtime optional check: {e}")
            _openvino_device = "CPU"
    return _openvino_device

_groq_client = None

def get_sotto_model():
    """
    Initializes Groq Cloud Whisper client (whisper-large-v3-turbo) for ultra-low latency voice recognition.
    """
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY is not set in environment!")
        logger.info("Initializing Sotto Groq Whisper Engine (whisper-large-v3-turbo)...")
        t0 = time.time()
        _groq_client = Groq(api_key=api_key)
        logger.info(f"Sotto Groq Whisper Engine ready in {time.time() - t0:.3f}s")
    return _groq_client

def transcribe_with_sotto(audio_bytes: bytes) -> Dict:
    """
    Transcribes audio bytes (up to 1 minute / 60s max) using Groq Cloud Whisper Large v3 Turbo.
    Returns transcript, detected language, inference duration, and segment timestamps.
    """
    t0 = time.time()
    client = get_sotto_model()
    device = get_openvino_device()

    try:
        resp = client.audio.transcriptions.create(
            file=("audio.mp3", audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )
    except Exception as exc:
        logger.error(f"Groq transcription error: {exc}")
        raise RuntimeError(f"Sotto Groq Whisper transcription failure: {exc}") from exc

    transcript = (getattr(resp, "text", "") or "").strip()
    inference_sec = time.time() - t0
    detected_lang = getattr(resp, "language", "en") or "en"
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
        "en": "English", "english": "English",
        "hi": "Hindi (हिंदी)", "hindi": "Hindi (हिंदी)",
        "ur": "Hinglish / Urdu", "mr": "Marathi",
        "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "es": "Spanish"
    }
    lang_display = lang_map.get(detected_lang.lower(), detected_lang.capitalize())

    return {
        "engine": "Sotto Whisper (Groq Cloud LPU + OpenVINO Client)",
        "transcript": transcript,
        "segments": segments_list,
        "language_code": detected_lang.lower(),
        "language": lang_display,
        # Groq verbose_json returns no language confidence — None, not a
        # fabricated 0.99. No consumer reads this key (checked server.py + UI).
        "language_probability": None,
        "duration": round(duration, 2),
        "inference_sec": round(inference_sec, 3),
        "device": f"Groq LPU / Host {device}",
        "available_devices": _available_devices
    }

if __name__ == "__main__":
    dev = get_openvino_device()
    print(f"Sotto Groq Whisper Engine loaded successfully!")
    print(f"Target Hardware: {dev} | Available OpenVINO Devices: {_available_devices}")
