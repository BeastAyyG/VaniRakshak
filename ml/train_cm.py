"""Training pipeline for AASIST anti-spoofing countermeasure.

Implements Recipe A (AASIST-L head on frozen WavLM features) and Recipe C
(WavLM fine-tuning with Attentive Merging) as specified in docs/08-cm-training-plan.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compute_mindcf import compute_eer, compute_mindcf, evaluate_system
from ml.aasist_model import AASISTLHead, AttentiveMerging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vanirakshak.train_cm")


class FeatureDataset(Dataset):
    """Memory-mapped or cached feature dataset."""

    def __init__(self, manifest_tsv: Path) -> None:
        self.samples = []
        n_bonafide = 0
        n_spoof = 0

        with open(manifest_tsv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    feat_path = row[0]
                    label_str = row[1].strip().lower()
                    attack_id = row[2].strip() if len(row) > 2 else "-"
                    label = 1 if label_str in ("bonafide", "bona-fide", "genuine", "1") else 0
                    if label == 1:
                        n_bonafide += 1
                    else:
                        n_spoof += 1
                    self.samples.append((feat_path, label, attack_id))

        self.n_bonafide = n_bonafide
        self.n_spoof = n_spoof
        logger.info(
            "Loaded %d samples from %s (bonafide=%d, spoof=%d, ratio=%.3f)",
            len(self.samples),
            manifest_tsv.name,
            n_bonafide,
            n_spoof,
            (n_bonafide / max(1, n_spoof)),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        feat_path, label, attack_id = self.samples[idx]
        feat = np.load(feat_path)
        return torch.from_numpy(feat).float(), label, attack_id


class FastPrefetchLoader:
    """Multi-threaded prefetching batch loader for cached feature tensors."""

    def __init__(
        self,
        dataset: FeatureDataset,
        batch_size: int = 128,
        shuffle: bool = False,
        drop_last: bool = False,
        n_threads: int = 16,
        queue_size: int = 8,
    ) -> None:
        self.samples = dataset.samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_threads = n_threads
        self.queue_size = queue_size

    def __len__(self) -> int:
        n = len(self.samples)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.samples)
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)

        if self.drop_last:
            n_batches = n // self.batch_size
            total_items = n_batches * self.batch_size
            indices = indices[:total_items]

        q = queue.Queue(maxsize=self.queue_size)
        sentinel = object()

        def loader_thread():
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                for start in range(0, len(indices), self.batch_size):
                    batch_idx = indices[start : start + self.batch_size]
                    batch_samples = [self.samples[i] for i in batch_idx]
                    feat_paths = [s[0] for s in batch_samples]
                    labels = torch.tensor([s[1] for s in batch_samples], dtype=torch.long)
                    atks = [s[2] for s in batch_samples]

                    feats = list(executor.map(np.load, feat_paths))
                    feats_tensor = torch.from_numpy(np.stack(feats)).float()
                    q.put((feats_tensor, labels, atks))
            q.put(sentinel)

        t = threading.Thread(target=loader_thread, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item


def train_recipe_a(
    train_manifest: Path,
    dev_manifest: Path,
    output_checkpoint: Path,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    patience: int = 10,
) -> Path:
    """Train Recipe A: AASIST-L head on pre-extracted WavLM features."""
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    logger.info("Starting Recipe A training on device: %s", dev)

    train_set = FeatureDataset(train_manifest)
    dev_set = FeatureDataset(dev_manifest)

    train_loader = FastPrefetchLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train_set) > batch_size),
        n_threads=16,
        queue_size=8,
    )
    dev_loader = FastPrefetchLoader(
        dev_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        n_threads=16,
        queue_size=8,
    )

    # Class-weighted Cross-Entropy: spoof_weight = n_bonafide / n_spoof
    # Class 0 = spoof, Class 1 = bonafide
    spoof_weight = float(train_set.n_bonafide / max(1, train_set.n_spoof))
    class_weights = torch.tensor([spoof_weight, 1.0], device=dev, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Class weights: spoof=%.4f, bonafide=1.0000", spoof_weight)

    # Instantiate AASIST-L head (768 in_dim for WavLM-base-plus)
    model = AASISTLHead(in_dim=768).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("AASIST-L head initialized with %d trainable parameters", n_params)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_dev_mindcf = float("inf")
    best_state_dict = None
    best_eer_threshold = 0.0
    best_dev_eer = 0.0
    best_epoch = 0
    patience_counter = 0

    use_amp = (dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for feats, labels, _ in train_loader:
            feats = feats.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                _, logits = model(feats)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1
            if n_batches % 200 == 0:
                logger.info(
                    "Epoch %3d: step %4d/%d (%.1f%%) - loss: %.4f",
                    epoch,
                    n_batches,
                    len(train_loader),
                    (n_batches / len(train_loader)) * 100.0,
                    total_loss / n_batches,
                )
                sys.stderr.flush()

        scheduler.step()
        train_loss = total_loss / max(1, n_batches)

        # Validation on dev set
        model.eval()
        bon_scores = []
        spf_scores = []

        with torch.no_grad():
            for feats, labels, _ in dev_loader:
                feats = feats.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    _, logits = model(feats)
                # LLR: bonafide logit - spoof logit
                llr = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
                labels_np = labels.numpy()
                mask = (labels_np == 1)
                bon_scores.extend(llr[mask].tolist())
                spf_scores.extend(llr[~mask].tolist())

        bon_arr = np.array(bon_scores)
        spf_arr = np.array(spf_scores)
        dev_metrics = evaluate_system(bon_arr, spf_arr)
        dev_mindcf = dev_metrics["minDCF"]
        dev_eer = dev_metrics["EER"]
        _, threshold_eer = compute_eer(bon_arr, spf_arr)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %3d/%d [%.1fs] - Loss: %.4f | Dev minDCF: %.4f, EER: %.3f%%, actDCF: %.4f",
            epoch,
            epochs,
            elapsed,
            train_loss,
            dev_mindcf,
            dev_eer * 100.0,
            dev_metrics.get("actDCF", float("nan")),
        )
        sys.stderr.flush()

        if dev_mindcf < best_dev_mindcf:
            best_dev_mindcf = dev_mindcf
            best_dev_eer = dev_eer
            best_eer_threshold = float(threshold_eer)
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    logger.info(
        "Training complete. Best epoch %d: dev minDCF=%.4f, dev EER=%.3f%%, threshold=%.4f",
        best_epoch,
        best_dev_mindcf,
        best_dev_eer * 100.0,
        best_eer_threshold,
    )

    # Save compliant checkpoint matching cm_aasist.py schema
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "architecture": "aasist-l",
        "frontend": "microsoft/wavlm-base-plus",
        "sample_rate": 16000,
        "window_s": 4.0,
        "state_dict": best_state_dict,
        "eer_threshold": float(best_eer_threshold),
        "dev_eer": float(best_dev_eer),  # fraction
        "dev_min_dcf": float(best_dev_mindcf),
    }

    torch.save(checkpoint_payload, str(output_checkpoint))
    logger.info("Saved validated checkpoint to %s", output_checkpoint)
    return output_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AASIST Countermeasure")
    parser.add_argument("--train-manifest", type=Path, required=True, help="Path to train features TSV")
    parser.add_argument("--dev-manifest", type=Path, required=True, help="Path to dev features TSV")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "cm_aasist.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_recipe_a(
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        output_checkpoint=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
