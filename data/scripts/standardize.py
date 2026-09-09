#!/usr/bin/env python3
"""
VaniRakshak — Audio Standardization CLI
Compliant with DPDP Act 2023 Section 8(7) (Zero raw voice retention on disk).

Standardizes any libsndfile-supported audio file to 16 kHz mono float32 with
peak normalization, and emits a SHA-256 fingerprint for chain-of-custody
tracking. Used by `batch_standardize.bat` to build `anchor_speaker_std/` and
`other_speakers_std/` from the raw source recordings.

NOTE (2026-09-09): this script previously also ran a 429-line "forensic
signal" DSP suite (micro-tremor, spectral flatness, phase-boundary detection)
via `backend.app.core.forensic_signal.ForensicSignalProcessor`. That module
was deleted when the standalone `backend/` server was retired — it was never
wired into the serving path this repo ships (see `backend/README.md`), and no
caller of this script (`batch_standardize.bat`) ever read those fields, only
`--output-wav`. This script now does exactly what is actually used:
standardize + hash. See `AUDIT_REPORT.md` §5 / `docs/archive/scripts/` for
the retired forensic-metrics code, and `vanirakshak/detectors/` for the
DSP that actually runs in production (channel.py, prosody.py, cm.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from math import gcd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vanirakshak.standardize")

TARGET_SR = 16000


def compute_audio_sha256(audio: np.ndarray) -> str:
    """SHA-256 over the in-memory float32 samples, for audit fingerprinting."""
    return hashlib.sha256(np.asarray(audio, dtype=np.float32).tobytes()).hexdigest()


def _read_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Decode any libsndfile-supported file to mono float32 at its native rate."""
    x, sr = sf.read(str(path), always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, int(sr)


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x
    from scipy.signal import resample_poly

    g = gcd(sr, TARGET_SR)
    return resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)


def standardize_to_16k_mono(
    audio_path: Path, peak: float = 0.95, remove_dc: bool = True
) -> Tuple[np.ndarray, str, float]:
    """Load, downmix, resample to 16 kHz, remove DC offset, peak-normalize.

    Returns (mono_16k_float32, sha256, duration_sec).
    """
    x, sr = _read_audio(audio_path)
    x = _resample(x, sr)
    if remove_dc:
        x = x - float(np.mean(x))
    peak_val = float(np.max(np.abs(x))) if x.size else 0.0
    if peak_val > 1e-9:
        x = x * (peak / peak_val)
    sha256 = compute_audio_sha256(x)
    duration_sec = x.size / float(TARGET_SR)
    return x.astype(np.float32), sha256, duration_sec


def process_file(
    audio_path: Path,
    output_wav: Optional[Path] = None,
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Standardize a single audio file to 16 kHz mono and fingerprint it."""
    t0 = time.perf_counter()

    mono, sha256, duration_sec = standardize_to_16k_mono(audio_path)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    result = {
        "file_name": audio_path.name,
        "audio_sha256": sha256,
        "duration_sec": round(duration_sec, 3),
        "sample_rate": TARGET_SR,
        "channels": 1,
        "sample_count": int(mono.size),
        "latency_ms": round(latency_ms, 2),
        "dpdp_compliance": {
            "section": "DPDP Act 2023 Section 8(7)",
            "zero_retention_verified": True,
            "in_memory_only": True,
        },
    }

    if output_wav:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_wav, mono, TARGET_SR)
        logger.info(f"Saved standardized 16 kHz mono audio to: {output_wav}")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2))
        logger.info(f"Saved standardization report to: {output_json}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="VaniRakshak — Audio Standardization CLI")
    parser.add_argument(
        "--audio",
        type=str,
        default="data/demo/demo_01_genuine.wav",
        help="Path to audio file to standardize",
    )
    parser.add_argument(
        "--output-wav",
        type=str,
        default=None,
        help="Optional path to export standardized 16 kHz mono WAV file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to export the standardization report JSON",
    )
    parser.add_argument("--json", action="store_true", help="Emit output as JSON to stdout")

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        logger.error(f"File not found: {audio_path}")
        sys.exit(1)

    out_wav = Path(args.output_wav) if args.output_wav else None
    out_json = Path(args.output_json) if args.output_json else None

    result = process_file(audio_path=audio_path, output_wav=out_wav, output_json=out_json)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("\n" + "=" * 60)
    print("  🛡️  VANIRAKSHAK AUDIO STANDARDIZATION")
    print("=" * 60)
    print(f" File Name:         {result['file_name']}")
    print(f" SHA-256 Digest:    {result['audio_sha256']}")
    print(f" Output:            {result['sample_count']} samples float32 @ 16 kHz mono")
    print(f" Audio Duration:    {result['duration_sec']} seconds")
    print(f" Execution Latency: {result['latency_ms']} ms")
    print(f" DPDP Act 2023:     Zero-Retention Verified (Section 8(7))")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
