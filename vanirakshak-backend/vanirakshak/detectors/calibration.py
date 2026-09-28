"""Per-user ASV score calibration (docs/14 W5).

Enrolled users used to inherit the anchor's logistic. That logistic was fitted
for one voice; another user's target and non-target cosines can sit anywhere.
This fits the same logistic ``scripts/enroll_anchor.py`` fits for the anchor,
per enrolment:

* **target scores**, leave-one-out: the enrolment audio is split into segments
  and each segment is scored against an embedding of the others;
* **non-target scores**: a fixed cohort of other speakers scored against the
  user's full enrolment embedding;
* ``mid`` halfway between the lowest target and highest non-target score,
  ``scale`` a quarter of the gap between their means.

The cohort is read from ``VR_ASV_COHORT_DIR`` (default ``data/other_speakers``,
the directory the anchor was calibrated against). Without a cohort, or with
too little enrolment audio, the caller falls back to the anchor-seeded
calibration and records that it did.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MIN_SEGMENT_S = 3.0
MAX_SEGMENTS = 5
MIN_COHORT = 5
COHORT_LIMIT = 20
COHORT_CLIP_S = 8.0


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_cohort(directory: Path, sr: int = 16000) -> List[np.ndarray]:
    """Up to ``COHORT_LIMIT`` mono 16-bit WAV clips as float32 at ``sr``,
    each cropped to ``COHORT_CLIP_S``. Unreadable files are skipped."""
    if not directory.is_dir():
        return []
    out: List[np.ndarray] = []
    for p in sorted(directory.glob("*.wav"))[:COHORT_LIMIT]:
        try:
            with wave.open(str(p), "rb") as w:
                if w.getsampwidth() != 2:
                    continue
                raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
                if w.getnchannels() > 1:
                    raw = raw.reshape(-1, w.getnchannels()).mean(axis=1)
                x = raw.astype(np.float32) / 32768.0
                if w.getframerate() != sr:
                    from math import gcd

                    from scipy.signal import resample_poly

                    g = gcd(w.getframerate(), sr)
                    x = resample_poly(x, sr // g, w.getframerate() // g).astype(np.float32)
        except Exception:
            continue
        if x.size >= sr:
            out.append(x[: int(COHORT_CLIP_S * sr)])
    return out


class Cohort:
    """Lazily loaded non-target clips, with per-backend embedding caching
    for backends whose embedding does not depend on the user id."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._clips: Optional[List[np.ndarray]] = None
        self._cache: Dict[str, List[np.ndarray]] = {}

    def clips(self) -> List[np.ndarray]:
        if self._clips is None:
            self._clips = load_cohort(self.directory)
        return self._clips

    def embeddings(self, asv, user_id: str, sr: int) -> List[np.ndarray]:
        # The stub's projection is seeded by user_id, so its cohort must be
        # embedded per user; a real model's embedding is user-independent.
        if asv.is_synthetic:
            return [asv.embed(c, sr, user_id) for c in self.clips()]
        if asv.name not in self._cache:
            self._cache[asv.name] = [asv.embed(c, sr, user_id) for c in self.clips()]
        return self._cache[asv.name]


def fit_user_calibration(
    asv, user_id: str, x: np.ndarray, sr: int, cohort: Optional[Cohort]
) -> Tuple[Optional[Dict[str, float]], Dict[str, object]]:
    """``(calibration or None, info)``. ``info["method"]`` says what happened."""
    x = np.asarray(x, dtype=np.float32).ravel()
    n_seg = min(MAX_SEGMENTS, int(x.size // int(MIN_SEGMENT_S * sr)))
    if n_seg < 2:
        return None, {"method": "seeded_from_anchor",
                      "reason": f"need >= {2 * MIN_SEGMENT_S:.0f} s of enrolment audio"}
    cohort_embs = cohort.embeddings(asv, user_id, sr) if cohort is not None else []
    if len(cohort_embs) < MIN_COHORT:
        return None, {"method": "seeded_from_anchor",
                      "reason": f"non-target cohort has {len(cohort_embs)} clips (need >= {MIN_COHORT})"}

    segs = np.array_split(x[: n_seg * (x.size // n_seg)], n_seg)
    loo = []
    for i, seg in enumerate(segs):
        rest = np.concatenate([s for j, s in enumerate(segs) if j != i])
        loo.append(_cos(asv.embed(rest, sr, user_id), asv.embed(seg, sr, user_id)))
    full = asv.embed(x, sr, user_id)
    nontarget = [_cos(full, e) for e in cohort_embs]

    info: Dict[str, object] = {
        "n_target_segments": n_seg,
        "n_cohort": len(nontarget),
        "target_scores_loo": {"mean": round(float(np.mean(loo)), 6), "min": round(float(np.min(loo)), 6)},
        "nontarget_scores": {"mean": round(float(np.mean(nontarget)), 6), "max": round(float(np.max(nontarget)), 6)},
    }
    if np.mean(loo) <= np.mean(nontarget):
        info.update(method="seeded_from_anchor",
                    reason="target scores do not separate from the cohort")
        return None, info
    mid = (float(np.min(loo)) + float(np.max(nontarget))) / 2.0
    scale = max((float(np.mean(loo)) - float(np.mean(nontarget))) / 4.0, 1e-6)
    info["method"] = "fitted_loo_vs_cohort"
    return {"mid": round(mid, 6), "scale": round(scale, 6)}, info
