"""Bridge the trained ``ml/`` target-speaker model into the serving backend.

This module lets the backend answer the single, honest question the model was
trained for:

    "Is this the enrolled target voice (Narendra Modi), or not?"

It is NOT a general synthetic/deepfake detector -- the checkpoint was trained on
one target speaker vs one non-target speaker and has never seen cloned audio.
The Android app (and any client) should present the result as an *identity*
check, never as "this is a fake/clone".

The model ingests a 40-dim MFCC *sequence* (one frame per analysis window) and
emits a log-likelihood ratio (LLR). Per ``ml/evaluate.py`` the decision rule is
``pred_label = 1 if llr >= 0.0 else 0`` (1 = target / Modi).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# The trained model lives in the sibling ``ml/`` package at the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

_DEFAULT_CKPT = REPO_ROOT / "ml" / "checkpoints" / "best_model.pt"


class TargetSpeakerVerifier:
    """Loads ``best_model.pt`` once and scores audio chunks for target-speaker match."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self.checkpoint_path = Path(
            checkpoint_path or os.environ.get("VR_MODEL_PATH", str(_DEFAULT_CKPT))
        )
        self._lock = threading.Lock()
        self._model = None
        self._device = None
        self.available = self.checkpoint_path.is_file()

    # -- lazy load (keeps startup fast; only loads on first use) -------------
    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.available:
            return False
        with self._lock:
            if self._model is not None:
                return True
            import torch
            from ml.model import AcousticForensicClassifier

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Architecture must match the trained checkpoint (input_proj.0.weight is [128, 40]).
            model = AcousticForensicClassifier(
                input_dim=40, proj_dim=128, hidden_dim=64, embedding_dim=64
            ).to(self._device)
            state_dict = torch.load(str(self.checkpoint_path), map_location=self._device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
        return True

    # -- feature extraction (MUST match ml/extract_features.py exactly) -------
    @staticmethod
    def extract_mfcc(y: np.ndarray, sample_rate: int = 16000, n_mfcc: int = 40) -> np.ndarray:
        """Return (T, n_mfcc) MFCC matrix using the same librosa call as training."""
        import librosa

        y = np.asarray(y, dtype=np.float32).ravel()
        if y.size == 0:
            return np.zeros((1, n_mfcc), dtype=np.float32)
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
        return mfcc.T.astype(np.float32)  # (T, n_mfcc)

    # -- public API ----------------------------------------------------------
    def verify(self, pcm_float32: np.ndarray, sample_rate: int = 16000) -> Dict[str, object]:
        """Score a mono float32 waveform in [-1, 1].

        Returns a dict with: available, llr, prob_target, is_target, label.
        ``label`` is one of "modi_ji" / "not_modi_ji" / "unavailable".
        """
        if not self._ensure_loaded():
            return {
                "available": False,
                "llr": None,
                "prob_target": None,
                "is_target": False,
                "label": "unavailable",
                "reason": "model checkpoint not loaded",
            }

        import torch

        feats = self.extract_mfcc(pcm_float32, sample_rate)  # (T, 40)
        # Guard against degenerate input (silence / too-short chunk).
        if feats.shape[0] < 2:
            feats = np.vstack([feats, feats]) if feats.shape[0] == 1 else feats
        with self._lock:
            with torch.no_grad():
                x = torch.from_numpy(feats).unsqueeze(0).to(self._device)
                llr = float(self._model(x).item())
        prob_target = float(1.0 / (1.0 + np.exp(-llr)))
        is_target = llr >= 0.0  # decision rule from ml/evaluate.py
        return {
            "available": True,
            "llr": round(llr, 4),
            "prob_target": round(prob_target, 4),
            "is_target": bool(is_target),
            "label": "modi_ji" if is_target else "not_modi_ji",
        }


# Module-level singleton, lazily loaded on first request.
VERIFIER = TargetSpeakerVerifier()
