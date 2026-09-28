"""Real telephony codecs for training and evaluation (docs/16).

``ml/augment.py``'s codec branch runs *after* ``prepare_audio_window`` has
normalised the window to zero mean and unit variance (values reach about
±6). Its G.711 functions then clip ~20 % of samples, and its AMR-WB / GSM-EFR
"simulations" are filters, not codecs. Training on that is training on
distortion, not on telephony (found 2026-09-29; docs/16 §1).

These functions take the **raw** utterance (int16 or float in [-1, 1], 16 kHz)
and return it after a real codec round trip, still at 16 kHz. Apply them
before ``prepare_audio_window``. G.711 is bit-exact via ``audioop``; AMR-NB
12.2k and AMR-WB 12.65k go through ffmpeg (``libopencore_amrnb``,
``libvo_amrwbenc``), in memory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:  # deprecated in 3.12, removed in 3.13 (docs/18-agent-tasks A3)
    import audioop
except ImportError:  # pragma: no cover
    audioop = None

REAL_CODECS = ("g711u", "g711a", "amrnb_12k2", "amrwb_12k65")


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for p in base.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"):
        return str(p)
    return None


def _to16(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        return x
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def _g711(x16: np.ndarray, law: str) -> np.ndarray:
    from scipy.signal import resample_poly

    x8 = np.clip(resample_poly(x16.astype(np.float32), 1, 2), -32768, 32767).astype(np.int16)
    enc, dec = (audioop.lin2ulaw, audioop.ulaw2lin) if law == "u" else (audioop.lin2alaw, audioop.alaw2lin)
    y8 = np.frombuffer(dec(enc(x8.tobytes(), 2), 2), dtype=np.int16)
    return np.clip(resample_poly(y8.astype(np.float32), 2, 1), -32768, 32767).astype(np.int16)


def _amr(ffmpeg: str, x16: np.ndarray, codec: str) -> np.ndarray:
    if codec == "amrnb_12k2":
        enc = ["-ar", "8000", "-c:a", "libopencore_amrnb", "-b:a", "12.2k", "-f", "amr"]
    else:
        enc = ["-ar", "16000", "-c:a", "libvo_amrwbenc", "-b:a", "12.65k", "-f", "amr"]
    base = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    coded = subprocess.run(base + ["-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0"] + enc + ["pipe:1"],
                           input=x16.tobytes(), capture_output=True, check=True).stdout
    pcm = subprocess.run(base + ["-i", "pipe:0", "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
                         input=coded, capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.int16).copy()


def apply_real_codec(x: np.ndarray, codec: str, ffmpeg: Optional[str] = None) -> np.ndarray:
    """Raw 16 kHz audio -> codec round trip -> float32 in [-1, 1] at 16 kHz."""
    x16 = _to16(np.asarray(x).ravel())
    if codec in ("g711u", "g711a"):
        y = _g711(x16, codec[-1])
    elif codec in ("amrnb_12k2", "amrwb_12k65"):
        exe = ffmpeg or find_ffmpeg()
        if not exe:
            raise RuntimeError("ffmpeg with libopencore_amrnb / libvo_amrwbenc is required for AMR")
        y = _amr(exe, x16, codec)
    else:
        raise ValueError(f"unknown codec {codec!r}; expected one of {REAL_CODECS}")
    return y.astype(np.float32) / 32767.0


class RealCodecAugmenter:
    """With probability ``p``, one codec from ``codecs``, uniformly, on raw audio."""

    def __init__(self, p: float, codecs: Sequence[str] = REAL_CODECS) -> None:
        bad = [c for c in codecs if c not in REAL_CODECS]
        if bad:
            raise ValueError(f"unknown codecs {bad}")
        self.p = float(p)
        self.codecs = list(codecs)
        self.ffmpeg = find_ffmpeg() if any(c.startswith("amr") for c in self.codecs) else None
        if self.p > 0 and any(c.startswith("amr") for c in self.codecs) and not self.ffmpeg:
            raise RuntimeError("real-codec augmentation needs ffmpeg with AMR encoders")

    def __call__(self, raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.p <= 0 or rng.random() >= self.p:
            return raw
        return apply_real_codec(raw, self.codecs[int(rng.integers(len(self.codecs)))], self.ffmpeg)

    def describe(self) -> dict:
        return {"p": self.p, "codecs": self.codecs, "applied": "raw audio, before prepare_audio_window"}
