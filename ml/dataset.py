"""VaniRakshak — Forensic Audio Dataset & Augmentation Pipeline.

Provides:
  - ForensicAudioDataset: PyTorch dataset supporting both file-based and in-memory
    synthetic audio streams with telephony codec augmentation (A-law, mu-law, AMR-WB, GSM-EFR).
  - Collate functions for variable-length window batching.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "vanirakshak-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import Dataset

from vanirakshak.data.augment import apply_random_pipeline
from vanirakshak.data.synthetic import synthetic_speech, synthetic_spoofed


class ForensicAudioDataset(Dataset):
    """Forensic anti-spoofing dataset with telephony channel augmentation."""

    def __init__(
        self,
        samples: Optional[List[Dict[str, Union[str, int, float]]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        synthetic_count: Optional[int] = None,
        sample_rate: int = 16000,
        window_sec: float = 4.04,
        augment: bool = True,
        feature_extractor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.window_samples = int(round(window_sec * sample_rate))
        self.augment = augment
        self.feature_extractor = feature_extractor
        self.items: List[Dict[str, Union[str, int, float]]] = []

        # 1. Synthetic generation mode
        if synthetic_count is not None and synthetic_count > 0:
            self.mode = "synthetic"
            self.synthetic_count = synthetic_count
            for i in range(synthetic_count):
                label = 1 if (i % 2 == 0) else 0  # balanced bonafide / spoof
                self.items.append({
                    "id": f"synth_{i:06d}",
                    "label": label,
                    "type": "bonafide" if label == 1 else "spoof",
                })
        # 2. Manifest-based mode
        elif manifest_path is not None:
            self.mode = "file"
            p = Path(manifest_path)
            if not p.is_file():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "files" in data:
                # Format of ground_truth.json
                base_dir = p.parent
                for fname, meta in data["files"].items():
                    fpath = base_dir / fname
                    if fpath.is_file():
                        label = 1 if meta.get("label") == "genuine" else 0
                        self.items.append({
                            "id": fname,
                            "path": str(fpath),
                            "label": label,
                            "expected_verdict": meta.get("expected_verdict", ""),
                        })
            elif isinstance(data, list):
                self.items = data
        # 3. Explicit samples list
        elif samples is not None:
            self.mode = "file"
            self.items = samples
        else:
            raise ValueError("Must provide samples, manifest_path, or synthetic_count.")

    def __len__(self) -> int:
        return len(self.items)

    def _load_or_generate_audio(self, item: Dict[str, Union[str, int, float]]) -> Tuple[np.ndarray, int]:
        label = int(item["label"])

        if self.mode == "synthetic":
            duration_sec = float(self.window_samples) / self.sample_rate
            if label == 1:
                pcm = synthetic_speech(duration_sec=duration_sec, sr=self.sample_rate)
            else:
                pcm = synthetic_spoofed(duration_sec=duration_sec, sr=self.sample_rate)
            # Normalize float32 in [-1, 1]
            x = pcm.astype(np.float32) / 32768.0
        else:
            # File loading
            path = Path(str(item["path"]))
            import wave
            with wave.open(str(path), "rb") as w:
                raw = w.readframes(w.getnframes())
                sr = w.getframerate()
                sw = w.getsampwidth()
            if sw == 2:
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                arr = np.frombuffer(raw, dtype=np.float32)

            # Center crop or pad to window_samples
            if len(arr) > self.window_samples:
                start = random.randint(0, len(arr) - self.window_samples)
                x = arr[start : start + self.window_samples]
            elif len(arr) < self.window_samples:
                pad = self.window_samples - len(arr)
                x = np.pad(arr, (0, pad), mode="constant")
            else:
                x = arr

        # Apply telephony codec augmentation
        if self.augment:
            x = apply_random_pipeline(x, self.sample_rate)

        return x, label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        x, label = self._load_or_generate_audio(item)

        # Optional feature extraction
        if self.feature_extractor is not None:
            feats = self.feature_extractor(x, self.sample_rate)
            tensor_x = torch.from_numpy(feats).float()
        else:
            # Default: 768-dim pseudo acoustic feature projection from spectral representations
            # (FFT energy profile interpolated to 768 dimensions)
            n_fft = 512
            hop = 256
            window = np.hanning(n_fft)
            num_frames = (len(x) - n_fft) // hop + 1
            if num_frames > 0:
                frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
                spec = np.abs(np.fft.rfft(frames * window, axis=1))  # (T, 257)
                mean_spec = np.mean(spec, axis=0)  # (257,)
                # Interpolate to 768
                vec_768 = np.interp(np.linspace(0, 1, 768), np.linspace(0, 1, len(mean_spec)), mean_spec)
                vec_768 = (vec_768 - np.mean(vec_768)) / (np.std(vec_768) + 1e-8)
            else:
                vec_768 = np.zeros(768, dtype=np.float32)

            tensor_x = torch.from_numpy(vec_768).float()

        tensor_y = torch.tensor(label, dtype=torch.long)
        return tensor_x, tensor_y


class FeatureDataset(Dataset):
    """Loads pre-extracted .npy feature matrices (e.g. MFCC / WavLM) from split CSV/JSON."""

    def __init__(
        self,
        split_path: Union[str, Path],
        features_dir: Union[str, Path],
        target_len: int = 300,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        import csv

        self.split_path = Path(split_path).resolve()
        self.features_dir = Path(features_dir).resolve()
        self.target_len = target_len
        self.is_train = is_train
        self.items: List[Dict[str, Any]] = []

        if not self.split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_path}")

        if self.split_path.suffix.lower() == ".csv":
            with open(self.split_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    cleaned = {k.strip(): v.strip() for k, v in r.items() if k is not None}
                    if not cleaned:
                        continue
                    fp = cleaned.get("filepath") or cleaned.get("path")
                    spk = cleaned.get("speaker_id") or cleaned.get("speaker") or ""
                    label = 1 if spk.lower() in ("anchor", "modi", "target", "1", "genuine") else 0
                    self.items.append({"filepath": fp, "speaker_id": spk, "label": label})
        else:
            data = json.loads(self.split_path.read_text(encoding="utf-8"))
            records = data.get("files", data) if isinstance(data, dict) else data
            for r in records:
                fp = r.get("path") or r.get("filepath") or r.get("id")
                spk = r.get("speaker_id") or r.get("label") or ""
                label = 1 if (r.get("label") == "genuine" or str(spk).lower() in ("anchor", "modi", "target")) else 0
                self.items.append({"filepath": fp, "speaker_id": spk, "label": label})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        rel_p = Path(item["filepath"]).with_suffix(".npy")

        feat_path = self.features_dir / rel_p
        if not feat_path.is_file():
            feat_path = self.features_dir / rel_p.name
        if not feat_path.is_file():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")

        arr = np.load(feat_path).astype(np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)

        t_len = arr.shape[0]
        if t_len > self.target_len:
            if self.is_train:
                start = random.randint(0, t_len - self.target_len)
            else:
                start = (t_len - self.target_len) // 2
            arr = arr[start : start + self.target_len]
        elif t_len < self.target_len:
            pad = self.target_len - t_len
            arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant")

        tensor_x = torch.from_numpy(arr).float()
        tensor_y = torch.tensor(item["label"], dtype=torch.long)
        return tensor_x, tensor_y


class ASVspoofDataset(Dataset):
    """Real ASVspoof corpus loader — the honest path to a countermeasure model.

    Why this exists: every checkpoint currently in ``ml/checkpoints/`` was
    trained either on ``synthetic_spoofed()`` (numpy-DSP buzz) or on
    one-speaker-vs-one-speaker identity data. Neither teaches "was this audio
    produced by a machine", so none of them can detect a real voice clone —
    see ``docs/archive/AUDIT_REPORT.md`` §12. That requires a corpus of real TTS /
    voice-conversion attacks, which is what ASVspoof provides.

    Deliberately speaker-independent: a countermeasure should learn vocoder
    artefacts in general, not one enrolled person's clones. The per-person
    half of the system is ASV (``detectors/asv_ecapa.py``), which already
    generalises to any enrolled voice without retraining.

    Protocol format (whitespace-separated), across ASVspoof 2019 LA / 2021 / 5::

        <speaker_id> <utterance_id> [<extra columns>...] <attack_id> <label>

    The column *count* differs between editions, so speaker is read from
    column 0, utterance id from column 1, and the label from the **last**
    column — the one invariant that holds for all of them.

    Labels follow this repo's existing convention (``ForensicAudioDataset``):
    ``1 = bonafide, 0 = spoof``. Note that this is inverted relative to some
    ASVspoof tooling, which scores spoof as the positive class.
    """

    def __init__(
        self,
        protocol_path: Union[str, Path],
        audio_dir: Union[str, Path],
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        target_len: int = 300,
        augment: bool = True,
        is_train: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.protocol_path = Path(protocol_path).resolve()
        self.audio_dir = Path(audio_dir).resolve()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.target_len = target_len
        self.augment = augment
        self.is_train = is_train
        self.items: List[Dict[str, Union[str, int]]] = []

        if not self.protocol_path.is_file():
            raise FileNotFoundError(f"ASVspoof protocol not found: {self.protocol_path}")
        if not self.audio_dir.is_dir():
            raise NotADirectoryError(f"ASVspoof audio directory not found: {self.audio_dir}")

        missing = 0
        for line in self.protocol_path.read_text(encoding="utf-8").splitlines():
            cols = line.split()
            if len(cols) < 3:
                continue
            utt_id, raw_label = cols[1], cols[-1].lower()
            if raw_label not in ("bonafide", "spoof"):
                continue
            path = self._resolve_audio(utt_id)
            if path is None:
                missing += 1
                continue
            self.items.append({
                "id": utt_id,
                "path": str(path),
                "speaker_id": cols[0],
                "attack_id": cols[-2] if len(cols) >= 4 else "-",
                "label": 1 if raw_label == "bonafide" else 0,
            })
            if limit is not None and len(self.items) >= limit:
                break

        if not self.items:
            raise ValueError(
                f"No usable entries from {self.protocol_path}. Checked audio in {self.audio_dir} "
                f"({missing} protocol rows had no matching .flac/.wav). Is --asvspoof-audio-dir correct?"
            )
        if missing:
            print(f"[ASVspoofDataset] {missing} protocol rows skipped (audio not found in {self.audio_dir})")

        n_bona = sum(1 for i in self.items if i["label"] == 1)
        print(f"[ASVspoofDataset] {len(self.items)} utterances — {n_bona} bonafide / {len(self.items) - n_bona} spoof")

    def _resolve_audio(self, utt_id: str) -> Optional[Path]:
        for ext in (".flac", ".wav"):
            p = self.audio_dir / f"{utt_id}{ext}"
            if p.is_file():
                return p
        return None

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, path: str) -> np.ndarray:
        import soundfile as sf

        x, sr = sf.read(path, always_2d=False)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != self.sample_rate:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(int(sr), self.sample_rate)
            x = resample_poly(x, self.sample_rate // g, int(sr) // g).astype(np.float32)
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import librosa

        item = self.items[idx]
        x = self._load_audio(str(item["path"]))

        # Same telephony codec augmentation the rest of the pipeline uses, so a
        # CM trained here survives the G.711/AMR-WB path a real call arrives on.
        if self.augment:
            x = apply_random_pipeline(x, self.sample_rate)

        # MFCC-40, matching ml/extract_features.py and vanirakshak/ml_bridge.py
        # exactly — a CM trained on different features could not be served by
        # the existing inference path.
        mfcc = librosa.feature.mfcc(y=x, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        arr = mfcc.T.astype(np.float32)  # (T, n_mfcc)

        t_len = arr.shape[0]
        if t_len > self.target_len:
            start = random.randint(0, t_len - self.target_len) if self.is_train else (t_len - self.target_len) // 2
            arr = arr[start : start + self.target_len]
        elif t_len < self.target_len:
            arr = np.pad(arr, ((0, self.target_len - t_len), (0, 0)), mode="constant")

        return torch.from_numpy(arr).float(), torch.tensor(int(item["label"]), dtype=torch.long)

