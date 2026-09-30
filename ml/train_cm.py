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
from typing import Any, Dict, List, Optional, Tuple

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
from ml.real_codecs import REAL_CODECS, RealCodecAugmenter
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
        generator: Optional[torch.Generator] = None,
        seed: int = 42,
        real_codec: Optional["RealCodecAugmenter"] = None,
    ) -> None:
        self.samples = dataset.samples
        # docs/16: a real codec round trip on the RAW utterance, before
        # prepare_audio_window normalises it. None = off.
        self.real_codec = real_codec
        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
        self.augmenter = augmenter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_threads = n_threads
        self.queue_size = queue_size
        self.generator = generator
        # Per-item RNG streams keep the crop offset and augmentation choice
        # reproducible: the 24 producer threads would otherwise race on a
        # single shared generator, so which utterance gets which augmentation
        # would vary run to run even with a fixed seed.
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.samples)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _load_and_augment(
        self, task: Tuple[Tuple[str, int, str], int, int]
    ) -> Tuple[np.ndarray, int]:
        item, index, epoch = task
        path, label, _ = item
        # Seeded on (seed, epoch, sample index) -- independent of thread
        # scheduling, and still different every epoch. The epoch travels in
        # the task rather than being read off self, which the producer thread
        # would race against the next __iter__ call.
        rng = np.random.default_rng([self.seed, epoch, int(index)])
        try:
            audio, sr = sf.read(path, dtype="float32")
            if self.real_codec is not None and self.real_codec.p > 0:
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    raise ValueError(f"real-codec augmentation expects 16 kHz audio, got {sr}")
                audio = self.real_codec(audio, rng)
            w = prepare_audio_window(
                audio, sr=sr, target_duration=4.0, random_crop=True, rng=rng
            )
            w_aug = self.augmenter(w, rng=rng)
            return w_aug, label
        except Exception as e:
            logger.warning("Error loading %s: %s; using zero window", path, e)
            return np.zeros(64000, dtype=np.float32), label

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1
        n = len(self.samples)
        if self.shuffle:
            if self.generator is not None:
                indices = torch.randperm(n, generator=self.generator).numpy()
            else:
                indices = np.arange(n)
                np.random.shuffle(indices)
        else:
            indices = np.arange(n)

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
                    batch_items = [(self.samples[i], int(i), epoch) for i in batch_idx]

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
        seed: int = 42,
        n_threads: int = 16,
        queue_size: int = 8,
    ) -> None:
        self.samples = dataset.samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_threads = n_threads
        self.seed = seed
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
            # Dev ordering does not change any metric, but an unseeded global
            # shuffle here is the same defect as the one fixed above.
            np.random.default_rng(self.seed).shuffle(indices)

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
# Streaming Audio Dataset & Loader for Zero-Disk Dev Validation
# ---------------------------------------------------------------------------

def is_audio_manifest(manifest_tsv: Path) -> bool:
    """Check if the first data row in manifest_tsv points to an audio file (.flac/.wav)."""
    p = Path(manifest_tsv)
    if not p.is_file():
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    target = row[0].strip().lower()
                    if target.endswith(".flac") or target.endswith(".wav"):
                        return True
                    if target.endswith(".npy"):
                        return False
    except Exception:
        return False
    return False


class AudioDevDataset(Dataset):
    """Raw audio dataset for dev validation (no augmentation, center-crop)."""

    def __init__(
        self,
        manifest_tsv: Path,
        corpus_audio_dir: Optional[Path] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.samples: List[Tuple[str, int, str]] = []
        n_bonafide = 0
        n_spoof = 0

        manifest_tsv = Path(manifest_tsv)
        if not manifest_tsv.is_file():
            raise FileNotFoundError(f"Dev manifest TSV not found at {manifest_tsv}")

        default_audio_dir = corpus_audio_dir or Path(r"A:\cm-corpus\asvspoof5\flac_D")

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
                    if limit and len(self.samples) >= limit:
                        break

        self.n_bonafide = n_bonafide
        self.n_spoof = n_spoof
        logger.info(
            "Loaded %d dev audio records from %s (bonafide=%d, spoof=%d)",
            len(self.samples),
            manifest_tsv.name,
            n_bonafide,
            n_spoof,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int, str]:
        return self.samples[idx]


class FastAudioDevLoader:
    """Multi-threaded prefetching batch loader for streaming raw dev audio."""

    def __init__(
        self,
        dataset: AudioDevDataset,
        batch_size: int = 64,
        n_threads: int = 16,
        queue_size: int = 8,
    ) -> None:
        self.samples = dataset.samples
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.queue_size = queue_size

    def __len__(self) -> int:
        n = len(self.samples)
        return (n + self.batch_size - 1) // self.batch_size

    def _load_audio(self, item: Tuple[str, int, str]) -> Tuple[np.ndarray, int, str]:
        path, label, attack_id = item
        try:
            audio, sr = sf.read(path, dtype="float32")
            w = prepare_audio_window(audio, sr=sr, target_duration=4.0, random_crop=False)
            return w, label, attack_id
        except Exception as e:
            logger.warning("Error loading dev audio %s: %s; using zero window", path, e)
            return np.zeros(64000, dtype=np.float32), label, attack_id

    def __iter__(self):
        n = len(self.samples)
        indices = np.arange(n)
        q = queue.Queue(maxsize=self.queue_size)
        sentinel = object()

        def producer():
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                for start in range(0, len(indices), self.batch_size):
                    batch_idx = indices[start : start + self.batch_size]
                    batch_items = [self.samples[i] for i in batch_idx]

                    processed = list(executor.map(self._load_audio, batch_items))
                    audios = np.stack([p[0] for p in processed])
                    labels = np.array([p[1] for p in processed], dtype=np.int64)
                    attacks = [p[2] for p in processed]

                    audio_tensor = torch.from_numpy(audios).float()
                    label_tensor = torch.from_numpy(labels).long()
                    q.put((audio_tensor, label_tensor, attacks))
            q.put(sentinel)

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item


# ---------------------------------------------------------------------------
# Benchmark Demo Clip Evaluator
# ---------------------------------------------------------------------------

def select_wavlm_features(w_out, layer_index: int, merge_layers: Optional[int] = None):
    """Pick the WavLM representation(s) the AASIST head will consume.

    Returns a single (B, T, D) tensor for the single-layer recipe, or a list
    of them when ``merge_layers`` is set, for AASISTLHead's attentive merge
    (Pan et al., docs/08 [R10]). ``merge_layers=12`` selects hidden_states
    0..11 -- the embedding output plus the early-to-middle transformer
    layers that docs/08 §12.6 measured as carrying the spoofing signal.

    Every extraction site in this file routes through here, so train and
    inference cannot drift apart on which layers they read.

    Returned tensors are cast to float32 here (WavLM runs in half): doing it
    in one place means a caller cannot forget it on the merged path, where
    the value is a list and ``.float()`` would raise AttributeError.
    """
    if merge_layers:
        return [w_out.hidden_states[i].float() for i in range(merge_layers)]
    return w_out.hidden_states[layer_index].float()


def evaluate_demo_clips(
    model: AASISTLHead,
    wavlm: WavLMModel,
    threshold: float,
    dev: torch.device,
    layer_index: int = 5,
    merge_layers: Optional[int] = None,
    freeze_merge: bool = False,
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
        def __init__(self, m: AASISTLHead, w: WavLMModel, t: float, layer_idx: int = 5,
                     merge_layers: Optional[int] = None) -> None:
            self.m = m
            self.w = w
            self.t = t
            self.layer_idx = layer_idx
            self.merge_layers = merge_layers

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
            inp = torch.from_numpy(audio).unsqueeze(0).to(dev, dtype=self.w.dtype)

            with torch.no_grad():
                w_out = self.w(inp, output_hidden_states=True)
                feat = select_wavlm_features(w_out, self.layer_idx, self.merge_layers)
                _, logits = self.m(feat)
                raw_llr = float((logits[0, 1] - logits[0, 0]).item())
                return raw_llr - self.t

    cm_temp = _TempCM(model, wavlm, threshold, layer_idx=layer_index, merge_layers=merge_layers)
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
    max_dev_utts: Optional[int] = None,
    n_workers: int = 24,
    layer_index: int = 5,
    merge_layers: Optional[int] = None,
    freeze_merge: bool = False,
    streaming_dev: bool = False,
    seed: int = 42,
    unfreeze_top_n: int = 0,
    encoder_lr: float = 1e-5,
    aug_p_codec: float = 0.5,
    aug_p_noise: float = 0.3,
    aug_p_bandpass: float = 0.3,
    aug_p_rir: float = 0.2,
    aug_p_gain: float = 0.3,
    real_codec_p: float = 0.0,
    real_codecs: Optional[List[str]] = None,
    codec_aug_p: float = 0.0,
    codec_aug_codecs: Optional[List[str]] = None,
) -> Path:
    """Train Recipe A with on-the-fly codec/telephony augmentation.

    The five ``aug_p_*`` values are the augmenter's per-effect probabilities.
    They are CLI-exposed and recorded in the checkpoint because they are an
    experiment variable, not a constant: docs/10 attributes the A19/A20
    regression to the codec/telephony augmenter, so "same recipe with the
    band-limiting effects off" has to be expressible and auditable.
    """
    # Seed RNGs for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loader_generator = torch.Generator().manual_seed(seed)

    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    logger.info("Initializing Recipe A Retraining on device: %s (WavLM layer_index=%d, seed=%d)", dev, layer_index, seed)

    # 1. Setup Audio Training Set & DataLoader
    train_set = AudioDataset(train_manifest)
    augmenter = CodecTelephonyAugmenter(
        p_codec=aug_p_codec,
        p_noise=aug_p_noise,
        p_bandpass=aug_p_bandpass,
        p_rir=aug_p_rir,
        p_gain=aug_p_gain,
        seed=seed,
    )
    logger.info(
        "Augmenter probabilities: codec=%.2f noise=%.2f bandpass=%.2f rir=%.2f gain=%.2f",
        aug_p_codec, aug_p_noise, aug_p_bandpass, aug_p_rir, aug_p_gain,
    )
    if real_codec_p > 0 and codec_aug_p > 0:
        raise ValueError("use only one of --real-codec-p (docs/16) and --codec-aug-p (docs/17)")
    _aug_p = codec_aug_p if codec_aug_p > 0 else real_codec_p
    _aug_codecs = codec_aug_codecs or real_codecs or REAL_CODECS
    real_codec = RealCodecAugmenter(_aug_p, _aug_codecs) if _aug_p > 0 else None
    if real_codec is not None:
        tag = "docs/17" if codec_aug_p > 0 else "docs/16"
        logger.info("Real-codec augmentation (%s): %s", tag, real_codec.describe())
    train_loader = FastAudioAugmentLoader(
        dataset=train_set,
        augmenter=augmenter,
        real_codec=real_codec,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        n_threads=n_workers,
        queue_size=8,
        max_samples=max_train_utts,
        generator=loader_generator,
        seed=seed,
    )

    # 2. Setup Dev Set & DataLoader
    is_audio_dev = streaming_dev or is_audio_manifest(dev_manifest)
    if is_audio_dev:
        logger.info(
            "Dev validation configured in STREAMING mode directly from audio (%s)",
            dev_manifest,
        )
        dev_set = AudioDevDataset(dev_manifest, limit=max_dev_utts)
        dev_loader = FastAudioDevLoader(
            dataset=dev_set,
            batch_size=batch_size,
            n_threads=min(16, n_workers),
            queue_size=8,
        )
    else:
        logger.info(
            "Dev validation configured from pre-cached feature tensors (%s)",
            dev_manifest,
        )
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

    # 4. Load WavLM Frontend (frozen by default; top-N layers optionally tuned)
    #
    # dtype: fp16 when fully frozen -- that is the numerics every published
    # benchmark was produced under, so it must not move. bf16 when fine-tuning,
    # because fp16 encoder fine-tuning diverges. docs/10 G3 requires the same
    # dtype in training, the eval harnesses and runtime.
    frontend_dtype = torch.float32
    if dev.type == "cuda":
        frontend_dtype = torch.bfloat16 if unfreeze_top_n > 0 else torch.float16
    logger.info(
        "Loading WavLM frontend: microsoft/wavlm-base-plus (dtype=%s, unfreeze_top_n=%d)",
        frontend_dtype, unfreeze_top_n,
    )
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(frontend_dtype).to(dev)
    for p in wavlm.parameters():
        p.requires_grad = False

    n_encoder_layers = len(wavlm.encoder.layers)
    if unfreeze_top_n < 0 or unfreeze_top_n > n_encoder_layers:
        raise ValueError(
            f"--unfreeze-top-n must be in [0, {n_encoder_layers}], got {unfreeze_top_n}"
        )

    # Unfreeze the top N layers *whose outputs are actually consumed*, not the
    # top N overall. select_wavlm_features takes hidden_states[0:merge_layers],
    # i.e. the embedding output plus layers 0..merge_layers-2 -- so with
    # --merge-layers 12 layer 11's output is never read, and unfreezing it
    # trains nothing (measured: 57 of 76 tensors received gradients, exactly
    # three layers' worth). Anchor the window to the last layer the head sees.
    if merge_layers:
        last_used_layer = merge_layers - 2
    else:
        last_used_layer = (
            n_encoder_layers - 1 if layer_index in (-1, n_encoder_layers)
            else layer_index - 1
        )
    last_used_layer = max(0, min(last_used_layer, n_encoder_layers - 1))

    encoder_param_names: List[str] = []
    if unfreeze_top_n > 0:
        first_trainable = last_used_layer + 1 - unfreeze_top_n
        if first_trainable < 0:
            raise ValueError(
                f"--unfreeze-top-n {unfreeze_top_n} exceeds the {last_used_layer + 1} "
                f"encoder layers this configuration actually reads "
                f"(merge_layers={merge_layers}, layer_index={layer_index})"
            )
        for name, param in wavlm.named_parameters():
            if name.startswith("encoder.layers."):
                idx = int(name.split(".")[2])
                if first_trainable <= idx <= last_used_layer:
                    param.requires_grad = True
                    encoder_param_names.append(name)
        # LayerDrop MUST be off before train() mode. WavLM-base-plus ships
        # layerdrop=0.1, which in training mode randomly skips encoder layers
        # and returns a hidden_states tuple of *varying length* (measured: 10
        # to 13 over 12 batches). That either raises IndexError in
        # select_wavlm_features or -- far worse, when merge_layers is small
        # enough not to overrun -- silently merges a different random set of
        # layers on every batch. Every standard fine-tuning recipe disables it.
        wavlm.config.layerdrop = 0.0
        if hasattr(wavlm, "encoder") and hasattr(wavlm.encoder, "config"):
            wavlm.encoder.config.layerdrop = 0.0
        # Dropout in the tuned layers is wanted; the frozen stack below still
        # sees train-mode dropout too, which is a second variable moving
        # alongside the unfreeze (docs/10 G3 "watch item"). Recorded, not hidden.
        wavlm.train()
        wavlm.gradient_checkpointing_enable()
        n_enc_trainable = sum(
            p.numel() for p in wavlm.parameters() if p.requires_grad
        )
        logger.info(
            "Unfroze encoder layers %d-%d (%d tensors, %d parameters); "
            "gradient checkpointing on, frontend in train() mode",
            first_trainable, last_used_layer,
            len(encoder_param_names), n_enc_trainable,
        )
    else:
        wavlm.eval()

    # 5. Initialize AASIST-L Head
    model = AASISTLHead(in_dim=768, merge_layers=merge_layers,
                        freeze_merge=freeze_merge).to(dev)
    # Warm start is opt-in and must never be silent. It defaulted to
    # data/cm_aasist.pt, which made every run a fine-tune of one head while
    # looking like an independent experiment -- see docs/10 (2026-09-13).
    applied_init_weights: Optional[str] = None
    if init_weights and Path(init_weights).is_file():
        logger.info("Loading initial weights from %s...", init_weights)
        init_ckpt = torch.load(str(init_weights), map_location="cpu")
        if "state_dict" in init_ckpt:
            incompatible = model.load_state_dict(init_ckpt["state_dict"], strict=False)
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
            applied_init_weights = str(init_weights)
            logger.info(
                "Warm start from %s: %d/%d keys applied, %d missing, %d unexpected",
                init_weights,
                len(init_ckpt["state_dict"]) - len(unexpected),
                len(init_ckpt["state_dict"]),
                len(missing), len(unexpected),
            )
            if missing or unexpected:
                # strict=False is what let a head with a different merge
                # config inherit a parent silently. Say so, every time.
                logger.warning(
                    "Partial warm start from %s. Missing keys keep their random "
                    "init; unexpected keys are discarded. The two models are not "
                    "the same architecture. missing(%d)=%s unexpected(%d)=%s",
                    init_weights, len(missing), missing[:8],
                    len(unexpected), unexpected[:8],
                )
        else:
            logger.warning(
                "--init-weights %s has no 'state_dict' key: no warm start applied, "
                "training from random init.", init_weights,
            )
    elif init_weights:
        logger.warning(
            "--init-weights %s is not a file: no warm start applied, training "
            "from random init.", init_weights,
        )
    else:
        logger.info("No --init-weights given: training from random init.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("AASIST-L Head initialized with %d trainable parameters", n_params)

    param_groups = [{"params": list(model.parameters()), "lr": lr}]
    if unfreeze_top_n > 0:
        enc_params = [p for p in wavlm.parameters() if p.requires_grad]
        param_groups.append({"params": enc_params, "lr": encoder_lr})
        logger.info(
            "Discriminative LR: head=%.2e (%d tensors), encoder=%.2e (%d tensors)",
            lr, len(param_groups[0]["params"]), encoder_lr, len(enc_params),
        )
    optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = (dev.type == "cuda")
    # GradScaler is for fp16 only. bf16 has fp32 range and must not be scaled;
    # scaling it silently degrades the gradients it was meant to protect.
    amp_dtype = torch.bfloat16 if unfreeze_top_n > 0 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype is torch.float16))

    best_dev_mindcf = float("inf")
    best_state_dict = None
    best_eer_threshold = 0.0
    best_dev_eer = 0.0
    best_epoch = 0
    patience_counter = 0

    logger.info(
        "Starting training loop: %d epochs, %d batches/epoch, lr=%.2e, patience=%d, layer_index=%d",
        epochs,
        len(train_loader),
        lr,
        patience,
        layer_index,
    )

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        # Restored each epoch: the dev pass below puts the frontend back into
        # eval(), which would otherwise silently persist into the next epoch.
        if unfreeze_top_n > 0:
            wavlm.train()
        total_loss = 0.0
        n_batches = 0

        for audio_batch, labels in train_loader:
            # Must match the frontend's dtype -- hardcoding float16 fed fp16
            # audio into a bf16 encoder once --unfreeze-top-n was added.
            audio_batch = audio_batch.to(dev, dtype=frontend_dtype, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                if unfreeze_top_n > 0:
                    # Graph must reach the encoder, so no no_grad() here.
                    w_out = wavlm(audio_batch, output_hidden_states=True)
                    feats = select_wavlm_features(w_out, layer_index, merge_layers)
                else:
                    with torch.no_grad():
                        w_out = wavlm(audio_batch, output_hidden_states=True)
                        feats = select_wavlm_features(w_out, layer_index, merge_layers)
                _, logits = model(feats)
                loss = criterion(logits, labels)

            clip_targets = list(model.parameters())
            if unfreeze_top_n > 0:
                clip_targets += [p for p in wavlm.parameters() if p.requires_grad]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(clip_targets, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(clip_targets, max_norm=1.0)
                optimizer.step()

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
        logger.info("Evaluating epoch %d on dev set (%d records)...", epoch, len(dev_set))
        t_val0 = time.perf_counter()
        model.eval()
        # The frontend is in train() mode while fine-tuning, so dropout would
        # otherwise be live during validation -- corrupting the very dev metric
        # that drives early stopping and checkpoint selection.
        wavlm.eval()
        bon_scores = []
        spf_scores = []

        with torch.no_grad():
            if is_audio_dev:
                for audio_batch, labels, _ in dev_loader:
                    audio_batch = audio_batch.to(
                        dev, dtype=frontend_dtype, non_blocking=True,
                    )
                    w_out = wavlm(audio_batch, output_hidden_states=True)
                    feats = select_wavlm_features(w_out, layer_index, merge_layers)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        _, logits = model(feats)
                    llr = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
                    labels_np = labels.numpy()
                    mask = (labels_np == 1)
                    bon_scores.extend(llr[mask].tolist())
                    spf_scores.extend(llr[~mask].tolist())
            else:
                for feats, labels, _ in dev_loader:
                    feats = feats.to(dev, non_blocking=True)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        _, logits = model(feats)
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
        demo_res = evaluate_demo_clips(model, wavlm, threshold_eer, dev, layer_index=layer_index,
                                   merge_layers=merge_layers)
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
            output_checkpoint = Path(output_checkpoint)
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
                "layer_index": int(layer_index),
                "merge_layers": int(merge_layers) if merge_layers else None,
                "freeze_merge": bool(freeze_merge),
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "unfreeze_top_n": int(unfreeze_top_n),
                "encoder_lr": float(encoder_lr) if unfreeze_top_n > 0 else None,
                "frontend_dtype": str(frontend_dtype).replace("torch.", ""),
                # Provenance. A checkpoint that does not say where its weights
                # came from cannot be compared to another one honestly.
                "init_weights": applied_init_weights,
                "augmentation": {
                    "p_codec": float(aug_p_codec),
                    "p_noise": float(aug_p_noise),
                    "p_bandpass": float(aug_p_bandpass),
                    "p_rir": float(aug_p_rir),
                    "p_gain": float(aug_p_gain),
                },
                "real_codec": real_codec.describe() if real_codec is not None else None,
            }
            logger.info(
                "Checkpoint provenance: init_weights=%s augmentation=%s",
                applied_init_weights, checkpoint_payload["augmentation"],
            )
            if unfreeze_top_n > 0:
                # Only the tuned layers, never the whole frontend: WavLM-base-plus
                # is ~94.7M params (379 MB fp32 / 189 MB bf16) and these
                # checkpoints have been force-added to a public repo before.
                checkpoint_payload["frontend_state_dict"] = {
                    k: v.detach().to("cpu")
                    for k, v in wavlm.state_dict().items()
                    if k in set(encoder_param_names)
                }
                checkpoint_payload["frontend_layer_indices"] = sorted(
                    {int(k.split(".")[2]) for k in encoder_param_names}
                )
            torch.save(checkpoint_payload, str(output_checkpoint))
            _validate_checkpoint(checkpoint_payload, output_checkpoint)
            logger.info(
                "*** New best checkpoint at epoch %d saved to %s (minDCF=%.4f, EER=%.3f%%, layer=%d) ***",
                best_epoch,
                output_checkpoint,
                best_dev_mindcf,
                best_dev_eer * 100.0,
                layer_index,
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
        default=Path(r"A:\cm-corpus\manifests\dev.tsv")
        if Path(r"A:\cm-corpus\manifests\dev.tsv").is_file()
        else Path(r"C:\Users\singu\cm-features\dev\dev_features.tsv"),
        help="Path to dev manifest TSV (raw audio or pre-cached features)",
    )
    parser.add_argument(
        "--output-checkpoint",
        "--output",
        dest="output",
        type=Path,
        default=REPO_ROOT / "data" / "cm_aasist_layer5.pt",
        help="Output checkpoint path (default: data/cm_aasist_layer5.pt)",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=5,
        help="WavLM hidden state layer index (default: 5 for Pan et al. middle layer)",
    )
    parser.add_argument(
        "--streaming-dev",
        action="store_true",
        default=False,
        help="Force streaming on-the-fly dev validation directly from raw audio",
    )
    parser.add_argument(
        "--max-dev-utts",
        type=int,
        default=None,
        help="Limit number of dev validation utterances for fast epoch evaluation",
    )
    parser.add_argument(
        "--merge-layers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Attentively merge WavLM hidden_states[0:N] instead of reading one "
            "layer (Pan et al., docs/08 [R10]). N=12 is the documented range. "
            "Overrides --layer-index. Adds only N parameters to the head."
        ),
    )
    parser.add_argument(
        "--freeze-merge",
        action="store_true",
        help=(
            "Ablation for docs/08 §12.7: hold the merge weights at their "
            "uniform init so the merge is a fixed mean. Separates 'attentive "
            "merging helps' from 'averaging helps'."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--init-weights", type=Path, default=None,
        help="Warm-start checkpoint. There is deliberately no default: it used "
             "to be data/cm_aasist.pt, which made every run a fine-tune of that "
             "one head while looking like an independent experiment (docs/10, "
             "2026-09-13). Omit the flag to train from random init.",
    )
    parser.add_argument("--max-train-utts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for torch, cuda, and DataLoader generator (default: 42)")
    parser.add_argument(
        "--unfreeze-top-n", type=int, default=0,
        help="Fine-tune the top N WavLM encoder layers (default 0 = fully frozen, "
             "today's behaviour). Switches the frontend to bf16; fp16 encoder "
             "fine-tuning diverges.",
    )
    parser.add_argument(
        "--encoder-lr", type=float, default=1e-5,
        help="Learning rate for unfrozen encoder layers (default 1e-5). "
             "Ignored when --unfreeze-top-n is 0.",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--p-codec", type=float, default=0.5,
                        help="Augmenter codec probability (default 0.5).")
    parser.add_argument("--p-noise", type=float, default=0.3,
                        help="Augmenter additive-noise probability (default 0.3).")
    parser.add_argument("--p-bandpass", type=float, default=0.3,
                        help="Augmenter band-pass probability (default 0.3).")
    parser.add_argument("--p-rir", type=float, default=0.2,
                        help="Augmenter reverberation probability (default 0.2).")
    parser.add_argument("--p-gain", type=float, default=0.3,
                        help="Augmenter gain probability (default 0.3).")
    parser.add_argument("--real-codec-p", type=float, default=0.0,
                        help="docs/16: probability of a real codec round trip (G.711, AMR) on the "
                             "raw utterance before windowing. 0 (default) = off.")
    parser.add_argument("--real-codecs", type=str, default=",".join(REAL_CODECS),
                        help="Comma-separated subset of " + ",".join(REAL_CODECS) + ".")
    parser.add_argument("--codec-aug-p", type=float, default=0.0,
                        help="docs/17: on-the-fly real-codec augmentation probability per training "
                             "sample (uniform over --codec-aug-codecs), applied to the raw utterance "
                             "before windowing. 0 (default) = off. Conflicts with --real-codec-p.")
    parser.add_argument("--codec-aug-codecs", type=str, default=",".join(REAL_CODECS),
                        help="docs/17: comma-separated subset of " + ",".join(REAL_CODECS) + ".")
    args = parser.parse_args()

    # A warm start that was asked for and did not happen is a silent experiment
    # change, which is the exact failure this flag's old default caused.
    init_w = args.init_weights
    if init_w is not None and not init_w.is_file():
        parser.error(
            f"--init-weights {init_w} does not exist. Pass a real checkpoint, or "
            "omit the flag to train from scratch."
        )

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
        max_dev_utts=args.max_dev_utts,
        n_workers=args.workers,
        layer_index=args.layer_index,
        merge_layers=args.merge_layers,
        freeze_merge=args.freeze_merge,
        streaming_dev=args.streaming_dev,
        seed=args.seed,
        unfreeze_top_n=args.unfreeze_top_n,
        encoder_lr=args.encoder_lr,
        aug_p_codec=args.p_codec,
        aug_p_noise=args.p_noise,
        aug_p_bandpass=args.p_bandpass,
        aug_p_rir=args.p_rir,
        aug_p_gain=args.p_gain,
        real_codec_p=args.real_codec_p,
        real_codecs=[c.strip() for c in args.real_codecs.split(",") if c.strip()],
        codec_aug_p=args.codec_aug_p,
        codec_aug_codecs=[c.strip() for c in args.codec_aug_codecs.split(",") if c.strip()],
    )


if __name__ == "__main__":
    main()
