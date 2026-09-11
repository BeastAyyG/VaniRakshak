"""Training pipeline for AASIST anti-spoofing countermeasure (Recipe A).

Implements Recipe A with stochastic codec and telephony augmentation as specified in
docs/08-cm-training-plan.md §§4, 12.1 and docs/03-codec-augmentation-and-telephony.md:
  - Audio dataset reading canonical ASVspoof 5 manifests
  - Stochastic on-the-fly telephony/codec augmentation (G.711, AMR-WB, GSM-EFR, noise, RIR)
  - Frozen WavLM frontend (microsoft/wavlm-base-plus) on GPU
  - AASIST-L head architecture (~89.5k parameters)
  - Class-weighted cross-entropy (spoof_weight = n_bonafide / n_spoof)
  - AdamW optimizer with cosine annealing
  - Fast dev validation against pre-cached fp16 feature tensors
  - In-training evaluation on benchmark demo clips (verifying YouTube domain-shift resolution)
  - Best-model checkpointing to data/cm_aasist.pt matching detector schema
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
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from transformers import WavLMModel

# Add repository root and backend to path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "vanirakshak-backend"
for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmarks.compute_mindcf import compute_eer, compute_mindcf, evaluate_system
from ml.aasist_model import AASISTLHead
from ml.augment import CodecTelephonyAugmenter, prepare_audio_window
from vanirakshak.detectors.cm_aasist import _validate_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vanirakshak.train_cm")


# ---------------------------------------------------------------------------
# Audio Dataset & DataLoader for Training
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """Raw audio dataset from TSV manifest."""

    def __init__(self, manifest_tsv: Path, corpus_audio_dir: Optional[Path] = None) -> None:
        self.samples: List[Tuple[str, int, str]] = []
        n_bonafide = 0
        n_spoof = 0

        manifest_tsv = Path(manifest_tsv)
        if not manifest_tsv.is_file():
            raise FileNotFoundError(f"Manifest TSV not found at {manifest_tsv}")

        default_audio_dir = corpus_audio_dir or Path(r"A:\cm-corpus\asvspoof5\flac_T")

        with open(manifest_tsv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    raw_path = row[0].strip()
                    label_str = row[1].strip().lower()
                    attack_id = row[2].strip() if len(row) > 2 else "-"
                    label = 1 if label_str in ("bonafide", "bona-fide", "genuine", "1") else 0

                    p = Path(raw_path)
                    if not p.is_file() and default_audio_dir.is_dir():
                        candidate = default_audio_dir / p.name
                        if candidate.is_file():
                            raw_path = str(candidate)

                    if label == 1:
                        n_bonafide += 1
                    else:
                        n_spoof += 1
                    self.samples.append((raw_path, label, attack_id))

        self.n_bonafide = n_bonafide
        self.n_spoof = n_spoof
        logger.info(
            "Loaded %d audio records from %s (bonafide=%d, spoof=%d, ratio=%.4f)",
            len(self.samples),
            manifest_tsv.name,
            n_bonafide,
            n_spoof,
            (n_bonafide / max(1, n_spoof)),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int, str]:
        return self.samples[idx]


class FastAudioAugmentLoader:
    """Multi-threaded prefetching loader for raw audio with stochastic augmentation."""

    def __init__(
        self,
        dataset: AudioDataset,
        augmenter: CodecTelephonyAugmenter,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = True,
        n_threads: int = 24,
        queue_size: int = 8,
        max_samples: Optional[int] = None,
    ) -> None:
        self.samples = dataset.samples
        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
        self.augmenter = augmenter
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

    def _load_and_augment(self, item: Tuple[str, int, str]) -> Tuple[np.ndarray, int]:
        path, label, _ = item
        try:
            audio, sr = sf.read(path, dtype="float32")
            w = prepare_audio_window(audio, sr=sr, target_duration=4.0, random_crop=True)
            w_aug = self.augmenter(w)
            return w_aug, label
        except Exception as e:
            logger.warning("Error loading %s: %s; using zero window", path, e)
            return np.zeros(64000, dtype=np.float32), label

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

        def producer():
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                for start in range(0, len(indices), self.batch_size):
                    batch_idx = indices[start : start + self.batch_size]
                    batch_items = [self.samples[i] for i in batch_idx]

                    processed = list(executor.map(self._load_and_augment, batch_items))
                    audios = np.stack([p[0] for p in processed])
                    labels = np.array([p[1] for p in processed], dtype=np.int64)

                    audio_tensor = torch.from_numpy(audios).float()
                    label_tensor = torch.from_numpy(labels).long()
                    q.put((audio_tensor, label_tensor))
            q.put(sentinel)

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item


# ---------------------------------------------------------------------------
# Pre-cached Feature Dataset for Dev Validation
# ---------------------------------------------------------------------------

class FeatureDataset(Dataset):
    """Dataset for pre-extracted fp16 numpy feature tensors."""

    def __init__(self, manifest_tsv: Path) -> None:
        self.samples = []
        n_bonafide = 0
        n_spoof = 0

        with open(manifest_tsv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    feat_path = row[0].strip()
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
            "Loaded %d feature records from %s (bonafide=%d, spoof=%d)",
            len(self.samples),
            manifest_tsv.name,
            n_bonafide,
            n_spoof,
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
        batch_size: int = 256,
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


# ---------------------------------------------------------------------------
# Benchmark Demo Clip Evaluator
# ---------------------------------------------------------------------------

def evaluate_demo_clips(
    model: AASISTLHead,
    wavlm: WavLMModel,
    threshold: float,
    dev: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate model on the 4 demo benchmark clips via full pipeline and fusion."""
    from vanirakshak.detectors.pipeline import DetectionPipeline
    from vanirakshak.risk.fusion import RiskFusionEngine

    demo_dir = REPO_ROOT / "data" / "demo"
    demo_clips = [
        ("demo_01_genuine.wav", "narendra_modi"),
        ("demo_02_degraded_copy.wav", "narendra_modi"),
        ("demo_03_partial_splice.wav", "narendra_modi"),
        ("demo_04_impostor.wav", "narendra_modi"),
    ]
    results = {}

    model.eval()

    class _TempCM:
        def __init__(self, m: AASISTLHead, w: WavLMModel, t: float) -> None:
            self.m = m
            self.w = w
            self.t = t

        def score(self, x: np.ndarray, sr: int) -> float:
            audio = np.asarray(x, dtype=np.float32).ravel()
            target_samples = 64000
            if len(audio) < target_samples:
                padded = np.zeros(target_samples, dtype=np.float32)
                padded[: len(audio)] = audio
                audio = padded
            elif len(audio) > target_samples:
                start = (len(audio) - target_samples) // 2
                audio = audio[start : start + target_samples]

            std = float(np.std(audio))
            if std > 1e-6:
                audio = (audio - float(np.mean(audio))) / std
            inp = torch.from_numpy(audio).unsqueeze(0).half().to(dev)

            with torch.no_grad():
                w_out = self.w(inp, output_hidden_states=True)
                feat = w_out.hidden_states[-1]
                _, logits = self.m(feat.float())
                raw_llr = float((logits[0, 1] - logits[0, 0]).item())
                return raw_llr - self.t

    cm_temp = _TempCM(model, wavlm, threshold)
    pipe = DetectionPipeline(cm=cm_temp)

    # Enrol Modi using genuine demo audio
    modi_path = demo_dir / "demo_01_genuine.wav"
    if modi_path.is_file():
        modi_audio, sr = sf.read(str(modi_path), dtype="float32")
        pipe.enrol("narendra_modi", modi_audio, sr)

    for fname, claimed_user in demo_clips:
        p = demo_dir / fname
        if not p.is_file():
            continue
        audio, sr = sf.read(str(p), dtype="float32")
        res = pipe.analyse(audio, sr, claimed_user=claimed_user)
        fusion = RiskFusionEngine()
        dec = fusion.update(res)
        results[fname] = {
            "llr": res.cm_score,
            "risk": dec.risk,
            "tier": dec.tier,
            "cm_genuine_prob": res.features.get("cm_genuine_prob", 0.5),
            "asv_score": res.asv_score,
        }
    return results


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_recipe_a(
    train_manifest: Path,
    dev_manifest: Path,
    output_checkpoint: Path,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    patience: int = 5,
    init_weights: Optional[Path] = None,
    max_train_utts: Optional[int] = None,
    n_workers: int = 24,
) -> Path:
    """Train Recipe A with on-the-fly codec/telephony augmentation."""
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    logger.info("Initializing Recipe A Retraining on device: %s", dev)

    # 1. Setup Audio Training Set & DataLoader
    train_set = AudioDataset(train_manifest)
    augmenter = CodecTelephonyAugmenter(
        p_codec=0.5,
        p_noise=0.3,
        p_bandpass=0.3,
        p_rir=0.2,
        p_gain=0.3,
        seed=42,
    )
    train_loader = FastAudioAugmentLoader(
        dataset=train_set,
        augmenter=augmenter,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        n_threads=n_workers,
        queue_size=8,
        max_samples=max_train_utts,
    )

    # 2. Setup Dev Set & DataLoader
    dev_set = FeatureDataset(dev_manifest)
    dev_loader = FastPrefetchLoader(
        dataset=dev_set,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        n_threads=16,
        queue_size=8,
    )

    # 3. Class-weighted Cross-Entropy: spoof_weight = n_bonafide / n_spoof
    # Class 0 = spoof, Class 1 = bonafide
    spoof_weight = float(train_set.n_bonafide / max(1, train_set.n_spoof))
    class_weights = torch.tensor([spoof_weight, 1.0], device=dev, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Class weights: spoof=%.4f, bonafide=1.0000", spoof_weight)

    # 4. Load Frozen WavLM Frontend
    logger.info("Loading frozen WavLM frontend: microsoft/wavlm-base-plus...")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").half().to(dev).eval()
    for p in wavlm.parameters():
        p.requires_grad = False

    # 5. Initialize AASIST-L Head
    model = AASISTLHead(in_dim=768).to(dev)
    if init_weights and Path(init_weights).is_file():
        logger.info("Loading initial weights from %s...", init_weights)
        init_ckpt = torch.load(str(init_weights), map_location="cpu")
        if "state_dict" in init_ckpt:
            model.load_state_dict(init_ckpt["state_dict"], strict=False)
            logger.info("Successfully loaded state_dict from %s", init_weights)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("AASIST-L Head initialized with %d trainable parameters", n_params)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = (dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_dev_mindcf = float("inf")
    best_state_dict = None
    best_eer_threshold = 0.0
    best_dev_eer = 0.0
    best_epoch = 0
    patience_counter = 0

    logger.info(
        "Starting training loop: %d epochs, %d batches/epoch, lr=%.2e, patience=%d",
        epochs,
        len(train_loader),
        lr,
        patience,
    )

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for audio_batch, labels in train_loader:
            audio_batch = audio_batch.to(dev, dtype=torch.float16, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)

            # Frozen WavLM forward pass on GPU
            with torch.no_grad():
                w_out = wavlm(audio_batch, output_hidden_states=True)
                feats = w_out.hidden_states[-1]  # (B, 199, 768)

            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                _, logits = model(feats.float())
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

            if n_batches % 200 == 0 or n_batches == len(train_loader):
                elapsed_now = time.perf_counter() - t0
                speed = (n_batches * batch_size) / max(0.1, elapsed_now)
                logger.info(
                    "Epoch %2d: step %4d/%d (%.1f%%) - loss: %.4f - speed: %.1f utts/s",
                    epoch,
                    n_batches,
                    len(train_loader),
                    (n_batches / len(train_loader)) * 100.0,
                    total_loss / n_batches,
                    speed,
                )
                sys.stderr.flush()

        scheduler.step()
        train_loss = total_loss / max(1, n_batches)

        # -------------------------------------------------------------------
        # Validation on Dev Set
        # -------------------------------------------------------------------
        logger.info("Evaluating epoch %d on dev set (140,950 records)...", epoch)
        t_val0 = time.perf_counter()
        model.eval()
        bon_scores = []
        spf_scores = []

        with torch.no_grad():
            for feats, labels, _ in dev_loader:
                feats = feats.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    _, logits = model(feats)
                # Raw LLR: bonafide logit - spoof logit
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
        t_val1 = time.perf_counter()

        # Evaluate demo clips
        demo_res = evaluate_demo_clips(model, wavlm, threshold_eer, dev)
        demo_01 = demo_res.get("demo_01_genuine.wav", {})

        elapsed_total = time.perf_counter() - t0
        logger.info(
            "Epoch %2d/%d [%.1fs, val: %.1fs] - Loss: %.4f | Dev minDCF: %.4f, EER: %.3f%%, tau: %.4f | Demo01 (Genuine): LLR=%+.2f, Risk=%.1f (%s)",
            epoch,
            epochs,
            elapsed_total,
            t_val1 - t_val0,
            train_loss,
            dev_mindcf,
            dev_eer * 100.0,
            threshold_eer,
            demo_01.get("llr", float("nan")),
            demo_01.get("risk", float("nan")),
            demo_01.get("tier", "UNKNOWN"),
        )
        sys.stderr.flush()

        if dev_mindcf < best_dev_mindcf:
            best_dev_mindcf = dev_mindcf
            best_dev_eer = dev_eer
            best_eer_threshold = float(threshold_eer)
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint immediately upon improvement
            output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_payload = {
                "architecture": "aasist-l",
                "frontend": "microsoft/wavlm-base-plus",
                "sample_rate": 16000,
                "window_s": 4.0,
                "state_dict": best_state_dict,
                "eer_threshold": float(best_eer_threshold),
                "dev_eer": float(best_dev_eer),
                "dev_min_dcf": float(best_dev_mindcf),
            }
            torch.save(checkpoint_payload, str(output_checkpoint))
            _validate_checkpoint(checkpoint_payload, output_checkpoint)
            logger.info(
                "*** New best checkpoint at epoch %d saved to %s (minDCF=%.4f, EER=%.3f%%) ***",
                best_epoch,
                output_checkpoint,
                best_dev_mindcf,
                best_dev_eer * 100.0,
            )
        else:
            patience_counter += 1
            logger.info("Dev minDCF did not improve (patience: %d/%d)", patience_counter, patience)
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    logger.info(
        "Training finished. Best epoch %d: dev minDCF=%.4f, dev EER=%.3f%%, threshold=%.4f",
        best_epoch,
        best_dev_mindcf,
        best_dev_eer * 100.0,
        best_eer_threshold,
    )
    return output_checkpoint


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train AASIST Countermeasure with Codec Augmentation")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path(r"A:\cm-corpus\manifests\train.tsv"),
        help="Path to train manifest TSV",
    )
    parser.add_argument(
        "--dev-manifest",
        type=Path,
        default=Path(r"C:\Users\singu\cm-features\dev\dev_features.tsv"),
        help="Path to dev feature TSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "cm_aasist.pt",
        help="Output checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--init-weights", type=Path, default=REPO_ROOT / "data" / "cm_aasist.pt")
    parser.add_argument("--max-train-utts", type=int, default=None)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    init_w = args.init_weights if (args.init_weights and args.init_weights.is_file()) else None

    train_recipe_a(
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        output_checkpoint=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        patience=args.patience,
        init_weights=init_w,
        max_train_utts=args.max_train_utts,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
