"""
VaniRakshak - Standalone Benchmark & Verification Script for AI4Bharat IndicConformer ASR

Usage:
    python scripts/test_indicconformer.py [path_to_audio_file]

Example:
    python scripts/test_indicconformer.py test_audio/hindi_sample.wav
"""

import sys
import os
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from backend.services.asr.service import ASRService
from backend.services.asr.types import ASRConfig


def run_indicconformer_test(audio_path: str = "test_audio/hindi_sample.wav") -> bool:
    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"[-] Error: Audio file not found: {audio_file}")
        return False

    config = ASRConfig(
        provider="indicconformer",
        model_name=os.getenv("ASR_MODEL", "ai4bharat/indic-conformer-600m-multilingual"),
        language=os.getenv("ASR_LANGUAGE", "hi"),
        device=os.getenv("ASR_DEVICE", "auto")
    )

    print("=" * 65)
    print("           IndicConformer ASR Test")
    print("=" * 65)
    print(f"Model: {config.model_name}")
    print(f"Language: {config.language}")
    print(f"Audio: {audio_file}")

    try:
        service = ASRService.get_instance(config)
        print(f"Device: {service.provider.get_device_info()}")
        print("[*] Warming up ASR model...")
        service.warmup()
    except Exception as e:
        print(f"\n[-] Model loading failure: {e}")
        print("=" * 65)
        print("FAILED")
        print("=" * 65)
        return False

    print(f"[*] Transcribing audio: '{audio_file}'...")
    try:
        t0 = time.time()
        result = service.transcribe(audio_file)
        elapsed = time.time() - t0
    except Exception as e:
        print(f"\n[-] Transcription failure: {e}")
        print("=" * 65)
        print("FAILED")
        print("=" * 65)
        return False

    hindi_transcript = result.get("hindi_text", "")
    hinglish_transcript = result.get("hinglish_text", "")
    audio_dur = result.get("audio_duration_seconds", 0.0)
    infer_time = result.get("inference_time_seconds", 0.0)
    rtf = result.get("real_time_factor", 0.0)
    chunks = result.get("chunks", [])

    print(f"Audio Duration: {audio_dur:.2f}s")
    print(f"Inference Time: {infer_time:.2f}s (Real-Time Factor RTF: {rtf:.2f}x)")
    print(f"4-Second Chunks Processed: {len(chunks)}")
    print("-" * 65)
    print(f"Hindi transcript: {hindi_transcript}")
    print(f"Hinglish: {hinglish_transcript}")
    print("-" * 65)

    # Check for fraud keywords
    fraud_keywords = ["kyc", "otp", "bank", "account", "suspend", "expire", "sbi", "rbi", "police", "arrest"]
    detected = [kw for kw in fraud_keywords if kw in hinglish_transcript.lower() or kw in hindi_transcript.lower()]
    if detected:
        print(f"[!] Critical Scam / Fraud Indicators Detected: {detected}")
    else:
        print("[+] Normal conversational transcript.")

    print("=" * 65)
    print("SUCCESS")
    print("=" * 65)
    return True


if __name__ == "__main__":
    target_audio = sys.argv[1] if len(sys.argv) > 1 else "test_audio/hindi_sample.wav"
    success = run_indicconformer_test(target_audio)
    sys.exit(0 if success else 1)
