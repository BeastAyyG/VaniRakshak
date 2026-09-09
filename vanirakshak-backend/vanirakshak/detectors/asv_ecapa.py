"""Real ECAPA-TDNN speaker verification — drop-in replacement for ASVModel.

Overrides only :meth:`_embed_raw` to call a real pretrained speaker-embedding
model (speechbrain's ECAPA-TDNN, trained on VoxCeleb) instead of the random
projection in ``asv.py``. Everything else — ``_enrolments``,
``consistency_score``, ``calibration`` — is inherited unchanged, so this is
interface-compatible with ``DetectionPipeline`` and every existing caller
(``session.py``, ``anchor.py``, ``server/app.py``'s ``/v1/enroll``).

This is what makes multi-speaker ("family") enrollment real rather than
structural-only: the stub's embedding is a random projection seeded by
hashing ``user_id``, so it carries no actual voice biometrics. This class
embeds the real waveform through a network trained for speaker
discrimination, so two different enrolled users actually separate.

The real model-loading code lives in ``backend/app/models/ecapa_verifier.py``
(the sibling, otherwise-retired duplicate backend — see ``backend/README.md``)
and is imported here via a ``sys.path`` insert, the same pattern
``ml_bridge.py`` uses for ``ml/`` and ``semantic/signal.py`` uses for
``backend/services/asr/`` — one copy of the model code, reused, not
duplicated. Never raises past construction: :meth:`try_create` returns
``None`` on any failure (missing ``speechbrain``, no network, OOM, ...) and
the caller falls back to the plain numpy stub — the same never-raise degrade
pattern ``ml_bridge.py`` established.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .asv import ASVModel

logger = logging.getLogger("vanirakshak.detectors.asv_ecapa")

# vanirakshak-backend/vanirakshak/detectors/asv_ecapa.py -> up four levels is
# the repo root, where the sibling `backend/` package lives.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ASVModelECAPA(ASVModel):
    """Real ECAPA-TDNN embeddings; enrolment/calibration inherited from ASVModel."""

    name: str = "ecapa_tdnn_voxceleb"
    _verifier: object = field(default=None, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _ensure_model(self) -> None:
        if self._verifier is not None:
            return
        with self._lock:
            if self._verifier is not None:
                return
            from backend.app.models.ecapa_verifier import ECAPAVerifier

            verifier = ECAPAVerifier()
            # Warm up via the public embed() API (rather than reaching into
            # its private loader) so try_create() actually proves the real
            # model loads, not just that the wrapper object constructs.
            verifier.embed(np.zeros(1600, dtype=np.float32), sr=16000)
            self._verifier = verifier

    def _embed_raw(self, x: np.ndarray, sr: int, user_id: str) -> np.ndarray:
        self._ensure_model()
        wav = np.asarray(x, dtype=np.float32).ravel()
        if wav.size == 0:
            return np.zeros(192, dtype=np.float32)
        return self._verifier.embed(wav, sr=sr)

    @classmethod
    def try_create(cls) -> Optional["ASVModelECAPA"]:
        """Construct and eagerly load the model; ``None`` on any failure."""
        try:
            inst = cls()
            inst._ensure_model()
            return inst
        except Exception as e:
            logger.warning(
                f"[ASV] real ECAPA backend unavailable, falling back to stub ASV: {type(e).__name__}: {e}"
            )
            return None
