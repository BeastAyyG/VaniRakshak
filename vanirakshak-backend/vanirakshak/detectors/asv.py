"""Speaker verification stub (ECAPA-TDNN-shaped).

Maintains a per-user enrolled 192-dim "embedding" (just a 192-dim
deterministic summary of the reference audio). For a live chunk we
compute the same summary, then cosine similarity.

The 192-dim vector is computed from a fixed random projection of
hand-crafted spectral statistics. The projection is **stable across
runs** (seeded by the user_id hash), so the same user's reference
audio always embeds to a similar vector regardless of run.

Replace :meth:`ASVModel.embed` with a real ECAPA-TDNN forward pass to
upgrade. The contract (192-dim float32 vector, L2-normalised) will
not change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


EMBED_DIM = 192


def _stable_projection(seed_str: str, dim_in: int, dim_out: int = EMBED_DIM) -> np.ndarray:
    h = hashlib.sha256(seed_str.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 1.0 / np.sqrt(dim_in), size=(dim_in, dim_out)).astype(np.float32)
    return p


def _spectral_features(x: np.ndarray, sr: int, n_bins: int = 32) -> np.ndarray:
    """Return an n_bins-dim log-mel-like summary (hand-crafted, no torchaudio)."""
    from scipy.signal import welch

    nperseg = min(1024, x.size)
    # noverlap must stay < nperseg or scipy raises — which it did for any
    # input shorter than 257 samples (short uploads, tiny trailing chunks).
    noverlap = min(256, max(0, nperseg // 2))
    if nperseg < 2:
        return np.zeros(n_bins, dtype=np.float32)
    f, pxx = welch(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    if pxx.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    # log-spaced bins
    edges = np.logspace(np.log10(50.0), np.log10(sr / 2.0), n_bins + 1)
    out = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if m.any():
            out[i] = float(np.log(np.mean(pxx[m]) + 1e-12))
    # standardise per-clip
    out = (out - out.mean()) / (out.std() + 1e-6)
    return out


@dataclass
class ASVModel:
    """Enrolment + verification."""

    name: str = "synthetic_asv_v1"
    _enrolments: Dict[str, np.ndarray] = field(default_factory=dict)
    # Optional score calibration: {"mid": float, "scale": float}, produced by
    # `scripts/enroll_anchor.py`. Raw cosine from these hand-crafted features
    # only spans ~0.96-0.99, so mapping it with (cos+1)/2 yields ~0.99 for
    # every speaker — target and impostor alike. The logistic below is fitted
    # to measured target vs non-target scores so the number means something.
    calibration: Optional[Dict[str, float]] = None
    # Per-user calibrations for dynamic enrollment (W5)
    _user_calibrations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # G8: calibrated score of the anchor's worst (highest) non-target
    # enrolment match. Set by anchor.load_anchor; None when the anchor carries
    # no non-target scores.
    impostor_threshold: Optional[float] = None

    @property
    def is_synthetic(self) -> bool:
        """Whether this backend's "embedding" is a random projection rather than
        a learned voice biometric (G7).

        The stub is synthetic *by construction* — ``_stable_projection`` is a
        fixed random projection of hand-crafted spectral statistics seeded by a
        hash of ``user_id``, so a cosine "similarity" against it carries no
        speaker-identity information. :class:`ASVModelECAPA` overrides this to
        ``False``. Exposed so the health, WS and report surfaces can label the
        number honestly instead of presenting a projection as a voiceprint.

        Derived from the *instance actually running* (not the requested env
        var), so a degraded ``VR_ASV_BACKEND=ecapa`` that fell back to the stub
        still reports ``True``.
        """
        return True

    def _embed_raw(self, x: np.ndarray, sr: int, user_id: str) -> np.ndarray:
        feats = _spectral_features(x, sr, n_bins=EMBED_DIM)
        proj = _stable_projection(user_id or "anon", EMBED_DIM, EMBED_DIM)
        v = feats @ proj
        v = v / (np.linalg.norm(v) + 1e-9)
        return v.astype(np.float32)

    def embed(self, x: np.ndarray, sr: int, user_id: str = "anon") -> np.ndarray:
        return self._embed_raw(x, sr, user_id)

    def enrol(self, user_id: str, x: np.ndarray, sr: int) -> np.ndarray:
        emb = self._embed_raw(x, sr, user_id)
        self._enrolments[user_id] = emb
        return emb

    def is_enrolled(self, user_id: str) -> bool:
        return user_id in self._enrolments

    def unenroll(self, user_id: str) -> bool:
        """Erase an enrolled voiceprint and its per-user calibration (DPDP Act 2023).

        Returns True if user_id was enrolled and removed, False otherwise.
        """
        removed = False
        if user_id in self._enrolments:
            del self._enrolments[user_id]
            removed = True
        if hasattr(self, "_user_calibrations") and user_id in self._user_calibrations:
            del self._user_calibrations[user_id]
        return removed

    def enrolled_user_ids(self) -> list[str]:
        """All enrolled user_ids — lets API/UI layers list enrolled family members
        without reaching into `_enrolments` directly."""
        return sorted(self._enrolments.keys())

    def similarity(self, user_id: str, x: np.ndarray, sr: int) -> float:
        if user_id not in self._enrolments:
            return 0.0
        enrolled = self._enrolments[user_id]
        live = self._embed_raw(x, sr, user_id)
        return float(np.dot(enrolled, live) / (np.linalg.norm(enrolled) * np.linalg.norm(live) + 1e-9))

    def consistency_score(self, user_id: str, x: np.ndarray, sr: int) -> Tuple[float, bool]:
        """Return (consistency in [0,1], is_enrolled)."""
        if not self.is_enrolled(user_id):
            return 0.5, False
        cos = self.similarity(user_id, x, sr)
        cal = self._user_calibrations.get(user_id) or self.calibration
        if cal and cal.get("scale"):
            # Logistic centred between the measured target and non-target
            # score clouds; ~1.0 = matches the enrolled voice, ~0.0 = does not.
            z = (cos - float(cal["mid"])) / float(cal["scale"])
            return float(1.0 / (1.0 + np.exp(-float(np.clip(z, -30.0, 30.0))))), True
        # Uncalibrated fallback: map cosine in [-1, 1] to [0, 1].
        # Near-useless on these features (everything lands ~0.99).
        return float((cos + 1.0) / 2.0), True
