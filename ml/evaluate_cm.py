"""Comprehensive evaluation suite for trained AASIST / WavLM countermeasures.

Computes ASVspoof 5 Track 1 primary minDCF, actDCF, CLLR, pooled EER,
and per-attack EER breakdown as required by docs/08-cm-training-plan.md §5.
Supports evaluation on pre-extracted feature tensors (fast) and raw audio waveforms
(streaming feature extraction, with optional sampling limit and Platt calibration).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compute_mindcf import (
    compute_act_dcf as compute_actdcf,
    compute_cllr,
    compute_eer,
    compute_mindcf,
    evaluate_system,
)
from ml.aasist_model import AASISTLHead, AttentiveMerging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vanirakshak.evaluate_cm")


class FeatureDataset(Dataset):
    """Dataset of pre-extracted numpy features."""

    def __init__(self, manifest_tsv: Path, limit: Optional[int] = None) -> None:
        self.samples = []
        with open(manifest_tsv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    feat_path = row[0]
                    label_str = row[1].strip().lower()
                    attack_id = row[2].strip() if len(row) > 2 else "-"
                    label = 1 if label_str in ("bonafide", "bona-fide", "genuine", "1") else 0
                    self.samples.append((feat_path, label, attack_id))
                    if limit and len(self.samples) >= limit:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        feat_path, label, attack_id = self.samples[idx]
        feat = np.load(feat_path)
        return torch.from_numpy(feat).float(), label, attack_id


def evaluate_scores(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    attack_scores: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, object]:
    """Compute pooled and per-attack metrics."""
    bon = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spf = np.asarray(spoof_scores, dtype=np.float64).ravel()

    metrics = evaluate_system(bon, spf)

    # Per-attack breakdown
    per_attack_eer = {}
    if attack_scores:
        for attack_id, scores in sorted(attack_scores.items()):
            atk_spf = np.asarray(scores, dtype=np.float64).ravel()
            if atk_spf.size > 0 and bon.size > 0:
                eer_val, _ = compute_eer(bon, atk_spf)
                per_attack_eer[attack_id] = round(eer_val * 100.0, 3)
            else:
                per_attack_eer[attack_id] = float("nan")

    metrics["per_attack_eer"] = per_attack_eer
    return metrics


def evaluate_checkpoint_on_features(
    checkpoint_path: Path,
    feature_manifest: Path,
    batch_size: int = 64,
    device: str = "cuda",
    limit: Optional[int] = None,
    apply_calibration: bool = False,
) -> Dict[str, object]:
    """Evaluate a saved AASIST-L checkpoint on extracted feature tensors."""
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    logger.info("Loading checkpoint from %s onto %s", checkpoint_path, dev)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt["state_dict"]

    in_dim = 768 if ckpt.get("frontend") == "microsoft/wavlm-base-plus" else 1024
    model = AASISTLHead(in_dim=in_dim)
    model.load_state_dict(state_dict, strict=False)
    model.to(dev)
    model.eval()

    cal_w = ckpt.get("calibration", {}).get("scale_w", 1.0) if apply_calibration else 1.0
    cal_b = ckpt.get("calibration", {}).get("bias_b", 0.0) if apply_calibration else 0.0
    if apply_calibration:
        logger.info("Applying stored Platt calibration: scale_w=%.6f, bias_b=%.6f", cal_w, cal_b)

    from ml.train_cm import FastPrefetchLoader

    dataset = FeatureDataset(feature_manifest, limit=limit)
    loader = FastPrefetchLoader(dataset, batch_size=batch_size, shuffle=False, n_threads=16)

    bonafide_scores: List[float] = []
    spoof_scores: List[float] = []
    attack_scores: Dict[str, List[float]] = {}
    use_amp = (dev.type == "cuda")

    logger.info("Scoring %d samples from %s", len(dataset), feature_manifest.name)
    with torch.no_grad():
        for feats, labels, attack_ids in loader:
            feats = feats.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                _, logits = model(feats)
            # LLR: bonafide logit - spoof logit
            raw_llr = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
            llr = cal_w * raw_llr + cal_b

            for s, y, atk in zip(llr, labels.numpy(), attack_ids):
                score = float(s)
                if y == 1:
                    bonafide_scores.append(score)
                else:
                    spoof_scores.append(score)
                    if atk not in attack_scores:
                        attack_scores[atk] = []
                    attack_scores[atk].append(score)

    results = evaluate_scores(
        np.array(bonafide_scores), np.array(spoof_scores), attack_scores
    )

    logger.info("=== Evaluation Results for %s ===", checkpoint_path.name)
    logger.info("minDCF    : %.4f", results.get("minDCF", float("nan")))
    logger.info("actDCF    : %.4f", results.get("actDCF", float("nan")))
    logger.info("CLLR      : %.4f", results.get("CLLR", float("nan")))
    logger.info("Pooled EER: %.3f %%", results.get("EER", float("nan")) * 100.0)
    logger.info("Per-attack EER breakdown:")
    for atk, val in results.get("per_attack_eer", {}).items():
        logger.info("  Attack %-10s: %.3f %%", atk, val)

    return results


def evaluate_checkpoint_on_audio(
    checkpoint_path: Path,
    audio_manifest: Path,
    batch_size: int = 64,
    device: str = "cuda",
    limit: Optional[int] = None,
    apply_calibration: bool = False,
    workers: int = 16,
) -> Dict[str, object]:
    """Evaluate an AASIST-L checkpoint directly on raw audio files via streaming feature extraction."""
    from concurrent.futures import ThreadPoolExecutor
    import soundfile as sf
    from transformers import WavLMModel
    from ml.augment import prepare_audio_window

    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    logger.info("Loading checkpoint from %s onto %s", checkpoint_path, dev)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt["state_dict"]

    in_dim = 768 if ckpt.get("frontend") == "microsoft/wavlm-base-plus" else 1024
    model = AASISTLHead(in_dim=in_dim).to(dev)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    logger.info("Loading WavLM feature extractor onto %s", dev)
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    if dev.type == "cuda":
        wavlm = wavlm.half().to(dev)
    else:
        wavlm = wavlm.to(dev)
    wavlm.eval()

    cal_w = ckpt.get("calibration", {}).get("scale_w", 1.0) if apply_calibration else 1.0
    cal_b = ckpt.get("calibration", {}).get("bias_b", 0.0) if apply_calibration else 0.0
    if apply_calibration:
        logger.info("Applying stored Platt calibration: scale_w=%.6f, bias_b=%.6f", cal_w, cal_b)

    records = []
    with open(audio_manifest, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                path = row[0].strip()
                label = 1 if row[1].strip().lower() in ("bonafide", "bona-fide", "genuine", "1") else 0
                atk = row[2].strip() if len(row) > 2 else "-"
                records.append((path, label, atk))
                if limit and len(records) >= limit:
                    break

    logger.info("Loaded %d audio records from %s to evaluate", len(records), audio_manifest.name)

    def load_audio(item: Tuple[str, int, str]) -> Tuple[np.ndarray, int, str]:
        p, label, atk = item
        try:
            x, sr = sf.read(p, dtype="float32")
            w = prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False)
            return w, label, atk
        except Exception as e:
            logger.warning("Error loading %s: %s; zeroing", p, e)
            return np.zeros(64000, dtype=np.float32), label, atk

    bonafide_scores: List[float] = []
    spoof_scores: List[float] = []
    attack_scores: Dict[str, List[float]] = {}
    use_amp = (dev.type == "cuda")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(0, len(records), batch_size):
            batch_items = records[i : i + batch_size]
            processed = list(executor.map(load_audio, batch_items))

            audios = np.stack([p[0] for p in processed])
            labels = [p[1] for p in processed]
            atks = [p[2] for p in processed]

            audio_t = torch.from_numpy(audios).to(dev)
            if dev.type == "cuda":
                audio_t = audio_t.half()

            with torch.no_grad():
                hidden = wavlm(audio_t, output_hidden_states=True).hidden_states[-1].float()
                _, logits = model(hidden)
                raw_llrs = (logits[:, 1] - logits[:, 0]).cpu().numpy()
                llrs = cal_w * raw_llrs + cal_b

            for s, y, atk in zip(llrs, labels, atks):
                score = float(s)
                if y == 1:
                    bonafide_scores.append(score)
                else:
                    spoof_scores.append(score)
                    if atk not in attack_scores:
                        attack_scores[atk] = []
                    attack_scores[atk].append(score)

            if (i // batch_size) % 25 == 0 and i > 0:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                logger.info("Scored %d / %d utts (%.1f utts/s)", i, len(records), rate)

    results = evaluate_scores(
        np.array(bonafide_scores), np.array(spoof_scores), attack_scores
    )

    logger.info("=== Audio Evaluation Results for %s ===", checkpoint_path.name)
    logger.info("minDCF    : %.4f", results.get("minDCF", float("nan")))
    logger.info("actDCF    : %.4f", results.get("actDCF", float("nan")))
    logger.info("actDCF(B) : %.4f", results.get("actDCF_bayes", float("nan")))
    logger.info("CLLR      : %.4f", results.get("CLLR", float("nan")))
    logger.info("Pooled EER: %.3f %%", results.get("EER", float("nan")) * 100.0)
    logger.info("Per-attack EER breakdown:")
    for atk, val in results.get("per_attack_eer", {}).items():
        logger.info("  Attack %-10s: %.3f %%", atk, val)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AASIST countermeasure")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest TSV (feature or audio)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-report", type=Path, default=None, help="Save JSON report")
    parser.add_argument(
        "--audio-eval",
        action="store_true",
        help="Evaluate directly on raw audio waveforms instead of pre-cached feature tensors",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of utterances to evaluate (e.g. 20000 for eval partition sample)",
    )
    parser.add_argument(
        "--apply-calibration",
        action="store_true",
        help="Apply Platt scaling parameters stored in checkpoint if available",
    )
    parser.add_argument("--workers", type=int, default=16, help="Audio loader thread pool workers")
    args = parser.parse_args()

    if args.audio_eval:
        results = evaluate_checkpoint_on_audio(
            checkpoint_path=args.checkpoint,
            audio_manifest=args.manifest,
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
            apply_calibration=args.apply_calibration,
            workers=args.workers,
        )
    else:
        results = evaluate_checkpoint_on_features(
            checkpoint_path=args.checkpoint,
            feature_manifest=args.manifest,
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
            apply_calibration=args.apply_calibration,
        )

    if args.out_report:
        Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
        # Format consistent with reports/cm_aasist_eval_partition.json
        n_bon = len([v for v in results.get("per_attack_eer", {})]) # or store structured report
        out_dict = {
            "checkpoint": str(args.checkpoint.as_posix()),
            "partition": args.manifest.stem,
            "metrics": results,
            "per_attack_eer": results.get("per_attack_eer", {}),
        }
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)
        logger.info("Saved report to %s", args.out_report)


if __name__ == "__main__":
    main()
