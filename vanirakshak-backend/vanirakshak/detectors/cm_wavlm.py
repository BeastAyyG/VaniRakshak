"""Real WavLM self-supervised features for the CM (anti-spoofing) signal.

Wraps ``backend/app/models/wavlm_detector.py``'s ``WavLMFeatureExtractor``
(real pretrained ``microsoft/wavlm-base-plus``) behind the ``CMModel.score()``
contract used by ``DetectionPipeline`` (``score(x, sr) -> float`` LLR,
positive => genuine). Imported via a ``sys.path`` insert — the same pattern
``asv_ecapa.py`` uses for ``ecapa_verifier.py`` — so there is one copy of
this model-loading code, not two.

Important honesty boundary — read before touching this file:
``compute_acoustic_anomaly_score()`` uses a REAL pretrained self-supervised
speech model, but the spoof/genuine decision on top of its features
(uniformity/latent-collapse penalties, computed inside that method) is still
a hand-tuned heuristic, not a trained classifier head. No pretrained
anti-spoofing (AASIST-style) classifier weights exist anywhere in this repo,
on the sibling ``backend/`` duplicate, or anywhere fetchable without a
labeled spoof-detection training corpus this project does not have. Do
**not** describe this class as "AASIST", "a trained countermeasure", or
"spoof-detection accuracy" anywhere — it is a real-feature upgrade over the
pure-DSP stub in ``cm.py``, nothing more.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("vanirakshak.detectors.cm_wavlm")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CMModelWavLM:
    """WavLM-feature heuristic; same call contract as the stub ``CMModel``."""

    name: str = "wavlm_heuristic_v1"
    natural_offset: float = 0.0
    _extractor: object = field(default=None, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _ensure_model(self) -> None:
        if self._extractor is not None:
            return
        with self._lock:
            if self._extractor is not None:
                return
            from backend.app.models.wavlm_detector import WavLMFeatureExtractor

            extractor = WavLMFeatureExtractor()
            # Warm up via the public API so try_create() actually proves the
            # real model loads, not just that the wrapper object constructs.
            extractor.compute_acoustic_anomaly_score(np.zeros(32000, dtype=np.float32), 16000)
            self._extractor = extractor

    def score(self, x: np.ndarray, sr: int) -> float:
        """Return an LLR-shaped score (positive => genuine), same sign convention as CMModel."""
        self._ensure_model()
        anomaly, _vec_768, _latency_ms = self._extractor.compute_acoustic_anomaly_score(x, sr)
        genuine_prob = float(np.clip(1.0 - anomaly, 1e-4, 1 - 1e-4))
        llr = float(np.log(genuine_prob / (1.0 - genuine_prob))) + self.natural_offset
        return llr

    @classmethod
    def try_create(cls) -> Optional["CMModelWavLM"]:
        """Construct and eagerly load the model; ``None`` on any failure."""
        try:
            inst = cls()
            inst._ensure_model()
            return inst
        except Exception as e:
            logger.warning(
                f"[CM] real WavLM backend unavailable, falling back to stub CM: {type(e).__name__}: {e}"
            )
            return None
