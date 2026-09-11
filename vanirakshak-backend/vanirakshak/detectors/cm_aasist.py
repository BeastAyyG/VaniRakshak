"""Trained AASIST countermeasure backend (weights NOT shipped).

This module is the landing zone for step 2 of the demo-to-working-system
roadmap (``docs/08-cm-training-plan.md``): a *trained* anti-spoofing
classifier head that replaces the hand-tuned heuristics in ``cm.py`` (DSP
stub) and ``cm_wavlm.py`` (WavLM-feature heuristic).

Status: interface + factory wiring only. No trained weights exist anywhere
in this repo, so ``try_create()`` returns ``None`` until a checkpoint
produced by the training plan is pointed at via ``VR_CM_AASIST_WEIGHTS``.
The pipeline factory (``pipeline.py::_default_cm``) treats ``None`` as
"unavailable" and falls back to the stub — selecting
``VR_CM_BACKEND=aasist`` today only changes the health-report label and a
warning, never the verdict path. Do NOT describe the CM as trained until a
checkpoint with a measured dev-set EER is actually loaded (see the
``_WEIGHTS_LOADED`` gate and the ``name`` field, which reports
``aasist_missing_weights`` in that case).

Expected checkpoint schema (written by the training script in
``docs/08-cm-training-plan.md`` §5) — a ``torch.save`` dict::

    {
      "architecture": "aasist-l",        # must match _ARCHITECTURE
      "frontend": "microsoft/wavlm-base-plus",  # frozen, must match _FRONTEND
      "sample_rate": 16000,
      "window_s": 4.0,                   # training window; inference slices/pads to this
      "state_dict": {...},               # AASIST-L parameters
      "eer_threshold": float,            # dev-set EER operating point on the logit
      "dev_eer": float,                  # measured dev EER, for the health report
      "dev_min_dcf": float,              # measured dev minDCF, for the health report
    }

``score()`` returns an LLR-shaped value (positive => genuine) by centering
the genuine logit at the dev EER threshold, so the existing fusion
(``VR_FUSION_W_CM``, ``VR_FUSION_BIAS``) and tier gates keep working —
recalibrate them with ``scripts/fit_fusion_weights.py`` after swapping in
real weights, exactly as the RUNBOOK's real-backend section describes.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("vanirakshak.detectors.cm_aasist")

_ARCHITECTURE = "aasist-l"
_FRONTEND = "microsoft/wavlm-base-plus"
_SAMPLE_RATE = 16000
_WINDOW_S = 4.0

# Set once a checkpoint passes schema validation and loads. Everything
# downstream (name label, health report) keys off this, never off the env
# var alone — requesting the backend without weights must be visible.
_WEIGHTS_LOADED = False


def weights_path() -> Path:
    """Resolve the checkpoint location. Env override wins; the default is a
    repo-relative path that is gitignored (check ``.gitignore``) so a
    ~5–20 MB weights file can never be committed by accident."""
    override = os.environ.get("VR_CM_AASIST_WEIGHTS", "").strip()
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent.parent
    return repo_root / "data" / "cm_aasist.pt"


@dataclass
class CMModelAASIST:
    """Trained AASIST-L head; same call contract as the stub ``CMModel``."""

    name: str = "aasist_missing_weights"
    eer_threshold: float = 0.0
    dev_eer: float = float("nan")
    dev_min_dcf: float = float("nan")
    natural_offset: float = field(
        default_factory=lambda: float(os.environ.get("VR_CM_NATURAL_OFFSET", "0.0"))
    )
    _model: object = field(default=None, repr=False, compare=False)
    _frontend: object = field(default=None, repr=False, compare=False)

    def _ensure_frontend(self) -> None:
        if self._frontend is not None:
            return
        import torch
        from transformers import AutoFeatureExtractor, WavLMModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(_FRONTEND, local_files_only=True)
            wavlm = WavLMModel.from_pretrained(_FRONTEND, local_files_only=True)
        except Exception:
            feature_extractor = AutoFeatureExtractor.from_pretrained(_FRONTEND)
            wavlm = WavLMModel.from_pretrained(_FRONTEND)
        wavlm.eval()
        if device == "cuda":
            wavlm = wavlm.half().to(device)
        else:
            wavlm = wavlm.to(device)
        if self._model is not None:
            self._model.to(device)
        self._frontend = (feature_extractor, wavlm, device)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # lazy: zero-dependency import of this module
        except Exception as e:
            raise RuntimeError(f"torch unavailable, cannot load AASIST weights: {e}") from e
        ckpt_path = weights_path()
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"AASIST checkpoint not found at {ckpt_path} "
                "(set VR_CM_AASIST_WEIGHTS; see docs/08-cm-training-plan.md §5)"
            )
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        _validate_checkpoint(ckpt, ckpt_path)

        # Import AASIST-L head architecture
        try:
            from ml.aasist_model import AASISTLHead
        except ImportError:
            here = Path(__file__).resolve()
            repo_root = here.parent.parent.parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from ml.aasist_model import AASISTLHead

        in_dim = 768 if ckpt.get("frontend") == _FRONTEND else 1024
        model = AASISTLHead(in_dim=in_dim)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()

        self._model = model
        self.eer_threshold = float(ckpt["eer_threshold"])
        self.dev_eer = float(ckpt["dev_eer"])
        self.dev_min_dcf = float(ckpt["dev_min_dcf"])
        self.name = f"aasist_l_wavlm_dev_eer_{self.dev_eer:.3f}"

    def score(self, x: np.ndarray, sr: int) -> float:
        """Return an LLR-shaped score (positive => genuine)."""
        self._ensure_model()
        self._ensure_frontend()
        import torch

        feature_extractor, wavlm, device = self._frontend
        audio = np.asarray(x, dtype=np.float32).ravel()

        if sr != _SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_SAMPLE_RATE)

        target_samples = int(_SAMPLE_RATE * _WINDOW_S)  # 64,000 samples for 4.0s
        if len(audio) < target_samples:
            padded = np.zeros(target_samples, dtype=np.float32)
            padded[: len(audio)] = audio
            audio = padded
        elif len(audio) > target_samples:
            # Centre crop for inference
            start = (len(audio) - target_samples) // 2
            audio = audio[start : start + target_samples]

        std = float(np.std(audio))
        if std > 1e-6:
            audio = (audio - float(np.mean(audio))) / std

        inputs = feature_extractor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(device)
        if device == "cuda":
            input_values = input_values.half()

        with torch.inference_mode():
            outputs = wavlm(input_values, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1]  # (1, T, 768)
            feat = hidden_state.to(torch.float32)
            _, logits = self._model(feat)
            # LLR = genuine_logit (index 1) - spoof_logit (index 0) - eer_threshold + natural_offset
            raw_llr = float((logits[0, 1] - logits[0, 0]).item())
            llr = raw_llr - self.eer_threshold + self.natural_offset
            return float(llr)

    @classmethod
    def try_create(cls) -> Optional["CMModelAASIST"]:
        """Construct and eagerly load; ``None`` on any failure (factory falls
        back to the stub — never raises past construction)."""
        try:
            inst = cls()
            inst._ensure_model()
            return inst
        except Exception as e:
            logger.warning("VR_CM_BACKEND=aasist requested but unavailable (%s); using stub CM", e)
            return None


CMAASISTDetector = CMModelAASIST


def _validate_checkpoint(ckpt: object, path: Path) -> None:
    """Raise unless the checkpoint matches the schema in the module docstring."""
    global _WEIGHTS_LOADED
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: checkpoint must be a dict, got {type(ckpt).__name__}")
    missing = {
        "architecture",
        "frontend",
        "sample_rate",
        "window_s",
        "state_dict",
        "eer_threshold",
        "dev_eer",
        "dev_min_dcf",
    } - set(ckpt.keys())
    if missing:
        raise ValueError(f"{path}: checkpoint missing keys {sorted(missing)}")
    if ckpt["architecture"] != _ARCHITECTURE:
        raise ValueError(
            f"{path}: architecture {ckpt['architecture']!r} != expected {_ARCHITECTURE!r}"
        )
    if ckpt["frontend"] != _FRONTEND:
        raise ValueError(f"{path}: frontend {ckpt['frontend']!r} != expected {_FRONTEND!r}")
    if int(ckpt["sample_rate"]) != _SAMPLE_RATE:
        raise ValueError(f"{path}: sample_rate {ckpt['sample_rate']} != {_SAMPLE_RATE}")
    _WEIGHTS_LOADED = True
