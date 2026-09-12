"""Lightweight voice-activity + zero-crossing helpers for reporting.

These are *measurement* utilities for the ``/analyze`` per-window report —
not fraud detectors. Both are deliberately simple, dependency-free NumPy
heuristics, documented as such so the report fields they feed
(``is_voiced``, ``speech_ratio``, ``zcr_variance``) are measured from the
actual window audio instead of hardcoded constants.

* VAD: short-time log-energy with a relative threshold (frames within
  35 dB of the window peak count as speech, floored by an absolute
  silence energy so digital silence never counts). ``is_voiced`` is True
  when at least 15% of frames are speech-like.
* ZCR variance: zero-crossing rate per 25 ms frame, variance across the
  window's frames. Tonal speech has low frame-to-frame ZCR variance;
  noisy/unvoiced regions raise it. Silence-only windows return 0.0.
"""

from __future__ import annotations

import numpy as np


def vad_stats(x: np.ndarray, sr: int) -> dict:
    """Return ``{"speech_ratio": float, "is_voiced": bool}`` for a window."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == 0:
        return {"speech_ratio": 0.0, "is_voiced": False}

    frame_len = max(1, int(0.025 * sr))
    n_frames = max(1, x.size // frame_len)
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    energy = np.mean(frames ** 2, axis=1)
    peak = float(np.max(energy))
    if peak <= 0.0:
        return {"speech_ratio": 0.0, "is_voiced": False}

    # Relative threshold: within 35 dB of the window peak, with an
    # absolute floor (~-60 dBFS) so pure digital silence never counts.
    thresh = max(peak * 10.0 ** (-35.0 / 10.0), 1e-6)
    speech_frames = int(np.sum(energy >= thresh))
    ratio = float(speech_frames / n_frames)
    return {"speech_ratio": round(ratio, 4), "is_voiced": bool(ratio >= 0.15)}


def zcr_variance(x: np.ndarray, sr: int) -> float:
    """Variance of per-frame zero-crossing rate across a window."""
    x = np.asarray(x, dtype=np.float32).ravel()
    frame_len = max(1, int(0.025 * sr))
    n_frames = max(1, x.size // frame_len)
    if n_frames < 2:
        return 0.0
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    # Fraction of sign changes per frame (normalised ZCR in [0, 1]).
    zc = np.sum(frames[:, 1:] * frames[:, :-1] < 0, axis=1) / max(1, frame_len - 1)
    # Ignore near-silent frames: their ZCR is quantisation noise.
    energy = np.mean(frames ** 2, axis=1)
    peak = float(np.max(energy))
    if peak <= 0.0:
        return 0.0
    voiced = zc[energy >= max(peak * 1e-4, 1e-6)]
    if voiced.size < 2:
        return 0.0
    return round(float(np.var(voiced)), 4)
