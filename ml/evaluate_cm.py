"""Comprehensive evaluation suite for trained AASIST / WavLM countermeasures.

Computes ASVspoof 5 Track 1 primary minDCF, actDCF, CLLR, pooled EER,
and per-attack EER breakdown as required by docs/08-cm-training-plan.md §5.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
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

    def __init__(self, manifest_tsv: Path) -> None:
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

    from ml.train_cm import FastPrefetchLoader

    dataset = FeatureDataset(feature_manifest)
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
            llr = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AASIST countermeasure")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--manifest", type=Path, required=True, help="Feature TSV manifest")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-report", type=Path, default=None, help="Save JSON report")
    args = parser.parse_args()

    results = evaluate_checkpoint_on_features(
        checkpoint_path=args.checkpoint,
        feature_manifest=args.manifest,
        batch_size=args.batch_size,
        device=args.device,
    )

    if args.out_report:
        Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved report to %s", args.out_report)


if __name__ == "__main__":
    main()
