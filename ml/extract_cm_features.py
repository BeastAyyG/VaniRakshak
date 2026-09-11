"""Feature extraction pipeline for frozen SSL (WavLM) anti-spoofing countermeasure.

Extracts intermediate representations from microsoft/wavlm-base-plus across
4.0s (64,000 samples) audio windows in fp16, saving compressed feature tensors
for fast AASIST-L head training (Recipe A in docs/08-cm-training-plan.md).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import socket
import sys
import time
from typing import Optional

# Force IPv4 resolution to bypass Windows DNS IPv6 hanging/timeout on CDN
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(*args, **kwargs):
    responses = _orig_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET] or responses


socket.getaddrinfo = _ipv4_getaddrinfo

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vanirakshak.extract_cm_features")

SAMPLE_RATE = 16000
WINDOW_S = 4.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_S)  # 64,000


def load_and_preprocess_audio(
    audio_path: Path,
    target_samples: int = WINDOW_SAMPLES,
    eval_mode: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Load audio, resample to 16kHz mono, normalize, and slice/pad to target_samples."""
    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

    n_samples = len(data)
    if n_samples < target_samples:
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[:n_samples] = data
        data = padded
    elif n_samples > target_samples:
        if eval_mode:
            # Centre crop for evaluation
            start = (n_samples - target_samples) // 2
        else:
            # Random crop for training
            if rng is None:
                rng = np.random.default_rng()
            start = rng.integers(0, n_samples - target_samples + 1)
        data = data[start : start + target_samples]

    # Per-utterance zero-mean unit-variance normalization
    std = np.std(data)
    if std > 1e-6:
        data = (data - np.mean(data)) / std

    return data.astype(np.float32)


def extract_manifest_features(
    manifest_path: Path,
    output_dir: Path,
    model_id: str = "microsoft/wavlm-base-plus",
    layer_index: int = -1,
    batch_size: int = 32,
    device: str = "cuda",
    eval_mode: bool = False,
    limit: Optional[int] = None,
) -> Path:
    """Extract features for all utterances listed in a TSV manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "tensors"
    features_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature extractor and model: %s on %s", model_id, device)
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
        model = WavLMModel.from_pretrained(model_id, local_files_only=True)
    except Exception:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = WavLMModel.from_pretrained(model_id)
    model.eval()

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dev = torch.device(device if use_cuda else "cpu")

    if use_cuda:
        model = model.half().to(dev)
    else:
        model = model.to(dev)

    # Read manifest
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for r in reader:
            if len(r) >= 2:
                rows.append(r)
            if limit and len(rows) >= limit:
                break

    logger.info("Processing %d utterances from %s", len(rows), manifest_path.name)
    out_manifest = output_dir / f"{manifest_path.stem}_features.tsv"
    rng = np.random.default_rng(42)

    written_count = 0
    with open(out_manifest, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["feature_path", "label", "attack_id"])

        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            waveforms = []
            valid_batch_rows = []

            for row in batch_rows:
                audio_path = Path(row[0])
                if not audio_path.is_file():
                    logger.warning("File not found: %s", audio_path)
                    continue

                utt_name = audio_path.stem
                sub_dir = features_dir / (utt_name[:4] if len(utt_name) >= 4 else "misc")
                save_path = sub_dir / f"{utt_name}.npy"
                if save_path.is_file() and save_path.stat().st_size > 0:
                    label = row[1]
                    attack_id = row[2] if len(row) > 2 else "-"
                    writer.writerow([str(save_path.resolve()), label, attack_id])
                    written_count += 1
                    continue

                try:
                    wav = load_and_preprocess_audio(
                        audio_path,
                        target_samples=WINDOW_SAMPLES,
                        eval_mode=eval_mode,
                        rng=rng,
                    )
                    waveforms.append(wav)
                    valid_batch_rows.append(row)
                except Exception as exc:
                    logger.warning("Error loading %s: %s", audio_path, exc)

            if waveforms:
                inputs = feature_extractor(
                    waveforms,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                )
                input_values = inputs.input_values.to(dev)
                if use_cuda:
                    input_values = input_values.half()

                with torch.inference_mode():
                    outputs = model(input_values, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[layer_index]  # (B, T, D)
                    # Keep in fp16 on CPU
                    feats = hidden_states.detach().cpu().to(torch.float16).numpy()

                for feat, r in zip(feats, valid_batch_rows):
                    utt_name = Path(r[0]).stem
                    sub_dir = features_dir / (utt_name[:4] if len(utt_name) >= 4 else "misc")
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    save_path = sub_dir / f"{utt_name}.npy"
                    np.save(save_path, feat)

                    label = r[1]
                    attack_id = r[2] if len(r) > 2 else "-"
                    writer.writerow([str(save_path.resolve()), label, attack_id])
                    written_count += 1

            if (i // batch_size) % 20 == 0:
                out_f.flush()
                logger.info("Processed %d / %d utterances", written_count, len(rows))

    logger.info("Extracted %d features successfully -> %s", written_count, out_manifest)
    return out_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WavLM features for AASIST training")
    parser.add_argument("--manifest", type=Path, required=True, help="Input TSV manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save features")
    parser.add_argument("--model-id", type=str, default="microsoft/wavlm-base-plus")
    parser.add_argument("--layer-index", type=int, default=-1, help="Layer hidden state (-1=last)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-mode", action="store_true", help="Centre crop instead of random crop")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items for test")
    args = parser.parse_args()

    extract_manifest_features(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_id=args.model_id,
        layer_index=args.layer_index,
        batch_size=args.batch_size,
        device=args.device,
        eval_mode=args.eval_mode,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
