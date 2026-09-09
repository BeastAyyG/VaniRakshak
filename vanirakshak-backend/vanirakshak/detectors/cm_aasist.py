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
    _model: object = field(default=None, repr=False, compare=False)
    _frontend: object = field(default=None, repr=False, compare=False)

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
        # Model construction happens here once the training plan lands the
        # architecture definition; until then schema validation is the gate.
        raise RuntimeError(
            "checkpoint schema OK but the AASIST-L module definition is not "
            "implemented yet — it ships with the training run, not before it."
        )

    def score(self, x: np.ndarray, sr: int) -> float:
        """Return an LLR-shaped score (positive => genuine)."""
        if self._model is None:
            raise RuntimeError("CMModelAASIST.score() called without loaded weights")
        raise RuntimeError("unreachable: _model is only set by _ensure_model")  # pragma: no cover

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
