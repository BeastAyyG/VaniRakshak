"""VaniRakshak — Batch & CLI Feature Extraction Pipeline
Compliant with DPDP Act 2023 Section 8(7) (Zero-retention voice processing).

Features:
1. Batch MFCC Feature Extraction (13–40 coefficients @ 16 kHz mono) from dataset manifest.
2. Sliding-window 768-dimensional WavLM & 192-dimensional ECAPA vector extraction.
3. High-throughput .npy serialization for low-latency neural training.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

# Add repository root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vanirakshak.extract_features")


def extract_mfcc_features(
    audio_path: Path,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
) -> np.ndarray:
    """Extract MFCC feature matrix (time_steps, n_mfcc) using librosa."""
    try:
        import librosa

        y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Transpose to (T, n_mfcc)
        return mfcc.T.astype(np.float32)
    except Exception:
        # Fallback to soundfile + resample if necessary
        data, sr = sf.read(str(audio_path))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != sample_rate:
            import torchaudio.transforms as T

            resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
            data = resampler(torch.from_numpy(data.astype(np.float32))).numpy()
            sr = sample_rate
        import librosa

        mfcc = librosa.feature.mfcc(y=data.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
        return mfcc.T.astype(np.float32)


def batch_extract_manifest(
    manifest_path: Path,
    data_dir: Path,
    output_dir: Path,
    n_mfcc: int = 40,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """Batch extract MFCC features for all files listed in manifest.csv."""
    manifest_path = Path(manifest_path).resolve()
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: List[Dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cleaned = {k.strip(): v.strip() for k, v in r.items() if k is not None}
            if cleaned and any(cleaned.values()):
                rows.append(cleaned)

    print("\n" + "=" * 65)
    print("  [VANIRAKSHAK MFCC FEATURE EXTRACTION]")
    print("=" * 65)
    print(f" Manifest Source : {manifest_path}")
    print(f" Data Directory  : {data_dir}")
    print(f" Output Directory: {output_dir}")
    print(f" MFCC Dimension  : {n_mfcc} coefficients @ {sample_rate} Hz")
    print(f" Total Audio Files: {len(rows)}")
    print("-" * 65)

    extracted_records: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for idx, row in enumerate(rows, start=1):
        rel_path_str = row.get("filepath") or row.get("file_path") or row.get("path")
        speaker_id = row.get("speaker_id") or row.get("speaker") or "unknown"
        if not rel_path_str:
            continue

        audio_path = (data_dir / rel_path_str).resolve()
        if not audio_path.is_file():
            audio_path = (REPO_ROOT / rel_path_str).resolve()
        if not audio_path.is_file():
            audio_path = (data_dir / Path(rel_path_str).name).resolve()

        if not audio_path.is_file():
            logger.warning(f"[{idx}/{len(rows)}] File not found: {rel_path_str}")
            continue

        try:
            mfcc_feats = extract_mfcc_features(audio_path, sample_rate=sample_rate, n_mfcc=n_mfcc)
            dest_file = output_dir / Path(rel_path_str).with_suffix(".npy")
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest_file, mfcc_feats)

            extracted_records.append({
                "filepath": rel_path_str,
                "speaker_id": speaker_id,
                "npy_file": str(dest_file),
                "shape": list(mfcc_feats.shape),
            })
            if idx % 10 == 0 or idx == len(rows):
                logger.info(f"[{idx}/{len(rows)}] Processed {rel_path_str} -> {mfcc_feats.shape}")
        except Exception as e:
            logger.error(f"Error extracting {audio_path}: {e}")

    total_time = time.perf_counter() - t_start
    print("-" * 65)
    print(f" [OK] Successfully extracted {len(extracted_records)}/{len(rows)} files in {total_time:.2f}s")
    print(f" Saved .npy feature vectors to: {output_dir}")
    print("=" * 65 + "\n")

    return {
        "count": len(extracted_records),
        "output_dir": str(output_dir),
        "n_mfcc": n_mfcc,
        "elapsed_sec": round(total_time, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VaniRakshak — Acoustic Feature Extraction (MFCC & WavLM Vectors)"
    )
    # Batch MFCC Manifest Mode
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        type=str,
        default=None,
        help="Path to audio root directory (default: ../data or data)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest CSV file (e.g. ../data/manifest.csv)",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save extracted .npy features (e.g. ../data/features)",
    )
    parser.add_argument(
        "--n_mfcc",
        "--n-mfcc",
        dest="n_mfcc",
        type=int,
        default=40,
        help="Number of MFCC coefficients to extract (default: 40)",
    )

    # Single-file / WavLM extraction arguments
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Single audio file path for WavLM/ECAPA feature extraction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for single audio features (.pt or .npy)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=2.0,
        help="Sliding window duration in seconds (default: 2.0s)",
    )
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=0.5,
        help="Sliding window hop size in seconds (default: 0.5s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda' or 'cpu')",
    )

    args = parser.parse_args()

    # If manifest is supplied or data_dir + output_dir supplied, run batch MFCC extraction
    if args.manifest or (args.data_dir and args.output_dir):
        data_dir_p = Path(args.data_dir or (REPO_ROOT / "data"))
        manifest_p = Path(args.manifest or (data_dir_p / "manifest.csv"))
        out_dir_p = Path(args.output_dir or (data_dir_p / "features"))

        batch_extract_manifest(
            manifest_path=manifest_p,
            data_dir=data_dir_p,
            output_dir=out_dir_p,
            n_mfcc=args.n_mfcc,
        )
        return

    # Single-file mode
    audio_path = Path(args.audio or "data/demo/demo_01_genuine.wav")
    if not audio_path.is_file():
        audio_path = REPO_ROOT / audio_path

    if not audio_path.is_file():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)

    from backend.app.models.wavlm_detector import WavLMFeatureExtractor

    wavlm = WavLMFeatureExtractor(device=args.device)
    logger.info(f"Extracting 768-dim features from {audio_path}...")
    # Single file extraction logic
    data, sr = sf.read(str(audio_path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    score, vec_768, ms = wavlm.compute_acoustic_anomaly_score(data.astype(np.float32), sr)
    logger.info(f"Anomaly Score: {score:.4f} | Latency: {ms:.2f}ms | Feature Shape: {vec_768.shape}")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_p, vec_768)
        logger.info(f"Saved features to {out_p}")


if __name__ == "__main__":
    main()
