"""End-to-end per-window detection pipeline.

Glues together:
  * CM (countermeasure / AASIST-stub)
  * ASV (speaker verification / ECAPA-stub)
  * Channel anomaly
  * Prosody

Returns a :class:`WindowResult` that the risk fusion engine consumes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

from .cm import CMModel
from .asv import ASVModel

if TYPE_CHECKING:  # only for the _default_cm union; runtime imports stay lazy
    from .cm_wavlm import CMModelWavLM
    from .cm_aasist import CMModelAASIST
from .channel import detect as channel_detect
from .prosody import prosody_score

logger = logging.getLogger("vanirakshak.detectors.pipeline")


def _default_cm() -> "CMModel | CMModelWavLM | CMModelAASIST":
    """VR_CM_BACKEND=wavlm swaps in real self-supervised features (cm_wavlm.py);
    `aasist` selects the trained AASIST head (cm_aasist.py — returns None until
    a checkpoint from docs/08-cm-training-plan.md is pointed at via
    VR_CM_AASIST_WEIGHTS); default `stub` keeps the zero-dependency
    numpy/scipy detector. Degrades to the stub on any load failure — never
    raises past construction."""
    backend = os.environ.get("VR_CM_BACKEND", "stub").lower()
    if backend == "wavlm":
        from .cm_wavlm import CMModelWavLM

        inst = CMModelWavLM.try_create()
        if inst is not None:
            return inst
        logger.warning("VR_CM_BACKEND=wavlm requested but unavailable; using stub CM")
    elif backend == "aasist":
        from .cm_aasist import CMModelAASIST

        inst = CMModelAASIST.try_create()
        if inst is not None:
            return inst
        logger.warning("VR_CM_BACKEND=aasist requested but unavailable; using stub CM")
    return CMModel()


def _default_asv() -> ASVModel:
    """VR_ASV_BACKEND=ecapa swaps in real ECAPA-TDNN embeddings (asv_ecapa.py);
    default `stub` keeps the zero-dependency numpy/scipy detector. Degrades to
    the stub on any load failure — never raises past construction."""
    if os.environ.get("VR_ASV_BACKEND", "stub").lower() == "ecapa":
        from .asv_ecapa import ASVModelECAPA

        inst = ASVModelECAPA.try_create()
        if inst is not None:
            return inst
        logger.warning("VR_ASV_BACKEND=ecapa requested but unavailable; using stub ASV")
    return ASVModel()


@dataclass
class WindowResult:
    cm_score: float            # LLR: positive ⇒ genuine
    asv_score: float           # consistency in [0,1] (0.5 if not enrolled)
    asv_enrolled: bool
    channel_anomaly: float     # in [0,1]
    prosody_unnatural: float   # in [0,1]
    codec_guess: str
    snr_db: float
    features: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class DetectionPipeline:
    """Stateless wrapper around the four detectors."""

    def __init__(self, cm: Optional[CMModel] = None, asv: Optional[ASVModel] = None) -> None:
        self.cm = cm or _default_cm()
        self.asv = asv or _default_asv()

    @property
    def asv_is_synthetic(self) -> bool:
        """Whether the *active* ASV backend is the synthetic random-projection
        stub rather than real speaker biometrics (G7).

        Read from the instantiated backend, so it reflects what is actually
        running — a requested ``VR_ASV_BACKEND=ecapa`` that failed to load and
        degraded to the stub reports ``True`` here, which is exactly what the
        dashboard and ``GET /v1/health`` must agree on.
        """
        return bool(getattr(self.asv, "is_synthetic", True))

    def enrol(self, user_id: str, x: np.ndarray, sr: int) -> np.ndarray:
        return self.asv.enrol(user_id, x, sr)

    def is_enrolled(self, user_id: str) -> bool:
        return self.asv.is_enrolled(user_id)

    def unenroll(self, user_id: str) -> bool:
        return self.asv.unenroll(user_id)

    def analyse(self, x: np.ndarray, sr: int, claimed_user: Optional[str] = None) -> WindowResult:
        llr = self.cm.score(x, sr)
        # squash LLR to a [0,1] "genuine" probability (calibrated)
        cm_genuine_prob = float(1.0 / (1.0 + np.exp(-llr)))

        asv_consistency, enrolled = (0.5, False)
        if claimed_user:
            asv_consistency, enrolled = self.asv.consistency_score(claimed_user, x, sr)

        ch = channel_detect(x, sr)
        pr = prosody_score(x, sr)

        return WindowResult(
            cm_score=llr,
            asv_score=asv_consistency,
            asv_enrolled=enrolled,
            channel_anomaly=ch["anomaly"],
            prosody_unnatural=pr["unnatural"],
            codec_guess=ch["codec_guess"],
            snr_db=ch["snr_db"],
            features={
                "cm_genuine_prob": cm_genuine_prob,
                "asv_consistency": asv_consistency,
                "channel_anomaly": ch["anomaly"],
                "prosody_unnatural": pr["unnatural"],
                "snr_db": ch["snr_db"],
                "low_band_share": ch["low_band_share"],
                **pr,
            },
        )
