"""Neural codec augmentation for VaniRakshak CM training (Goal 2).

Real neural-codec round trips on the raw utterance (16 kHz), applied
before ``prepare_audio_window``, in the same place and with the same call
contract as ``RealCodecAugmenter`` in ``ml/real_codecs.py``. The existing
G.711 and AMR paths in ``ml/real_codecs.py`` are untouched; this module
adds Encodec, Opus, and MP3.

Fallback behaviour is the point of this module. A codec is only offered
when its backend is actually present on the machine:

- Encodec needs the ``encodec`` Python package and torch. If either is
  missing, Encodec codecs are reported unavailable and are never claimed
  to have run.
- Opus and MP3 need ffmpeg with ``libopus`` / ``libmp3lame``.
- In non-strict mode (the default) unavailable codecs are dropped with a
  warning and the augmenter runs on whatever is available. In strict
  mode, requesting an unavailable codec raises immediately, so a run
  cannot silently train without the codec it preregistered.
- If no requested codec is available at all and ``p > 0``, construction
  raises ``RuntimeError`` with the exact install step, instead of
  training unaugmented while claiming otherwise.

``describe()`` records the requested codecs, the ones actually used, and
the ones unavailable with their reason, so the checkpoint provenance
says exactly which neural codecs ran. Do not claim Encodec augmentation
if only Opus and MP3 ran.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Optional, Sequence

import numpy as np
from scipy.signal import resample_poly

try:
    from ml.real_codecs import find_ffmpeg
except ImportError:  # run as a script: python3 ml/neural_codecs.py
    from real_codecs import find_ffmpeg

logger = logging.getLogger("vanirakshak.ml.neural_codecs")

ENCODEC_BITRATES_KBPS = (24.0, 12.0, 6.0, 3.0, 1.5)
NEURAL_CODECS = (
    tuple(
        f"encodec_{int(b)}k" if b == int(b) else f"encodec_{str(b).replace('.', '_')}k"
        for b in ENCODEC_BITRATES_KBPS
    )
    + ("opus_24k", "opus_12k", "opus_6k", "mp3_32k", "mp3_16k")
)
_FFMPEG_CODECS = ("opus_24k", "opus_12k", "opus_6k", "mp3_32k", "mp3_16k")
_FFMPEG_SPEC = {
    "opus_24k": ("libopus", "24k", "ogg"),
    "opus_12k": ("libopus", "12k", "ogg"),
    "opus_6k": ("libopus", "6k", "ogg"),
    "mp3_32k": ("libmp3lame", "32k", "mp3"),
    "mp3_16k": ("libmp3lame", "16k", "mp3"),
}


def _ffmpeg_has_encoder(ffmpeg: str, encoder: str) -> bool:
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return False
    return encoder in out


def _encodec_available() -> tuple[bool, str]:
    try:
        import encodec  # noqa: F401
    except ImportError:
        return False, "encodec package is not installed (pip install encodec)"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed, so the encodec package cannot run"
    return True, ""


def available_codecs(ffmpeg: Optional[str] = None) -> dict:
    """Map every neural codec to (available, reason). Reason is empty when available."""
    exe = ffmpeg or find_ffmpeg()
    result: dict = {}
    enc_ok, enc_reason = _encodec_available()
    for codec in NEURAL_CODECS:
        if codec.startswith("encodec_"):
            result[codec] = (enc_ok, "" if enc_ok else enc_reason)
        else:
            encoder, _, _ = _FFMPEG_SPEC[codec]
            if not exe:
                result[codec] = (False, "ffmpeg is not installed")
            elif not _ffmpeg_has_encoder(exe, encoder):
                result[codec] = (False, f"ffmpeg has no {encoder} encoder")
            else:
                result[codec] = (True, "")
    return result


def _to16(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        return x
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def _trim_or_pad(y: np.ndarray, n: int) -> np.ndarray:
    if y.size == n:
        return y
    if y.size > n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - y.size, dtype=y.dtype)])


def _ffmpeg_round_trip(x16: np.ndarray, codec: str, ffmpeg: str) -> np.ndarray:
    encoder, bitrate, fmt = _FFMPEG_SPEC[codec]
    base = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    coded = subprocess.run(
        base + ["-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
                "-c:a", encoder, "-b:a", bitrate, "-f", fmt, "pipe:1"],
        input=x16.tobytes(), capture_output=True, check=True,
    ).stdout
    pcm = subprocess.run(
        base + ["-i", "pipe:0", "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
        input=coded, capture_output=True, check=True,
    ).stdout
    return _trim_or_pad(np.frombuffer(pcm, dtype=np.int16).copy(), x16.size)


def _encodec_round_trip(x: np.ndarray, bitrate_kbps: float) -> np.ndarray:
    """Encodec round trip at 24 kHz, resampled to and from 16 kHz.

    Encodec is a 24 kHz model, so the 16 kHz input is upsampled before
    encoding and the decoded output is downsampled back to 16 kHz.
    """
    import torch
    from encodec import EncodecModel

    x24 = resample_poly(np.asarray(x, dtype=np.float32).ravel(), 3, 2)
    wav = torch.from_numpy(x24).unsqueeze(0).unsqueeze(0)
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(float(bitrate_kbps))
    model.eval()
    with torch.no_grad():
        encoded = model.encode(wav)
        decoded = model.decode(encoded)
    y24 = decoded.squeeze().cpu().numpy().astype(np.float32)
    y16 = resample_poly(y24, 2, 3)
    return _trim_or_pad(y16, np.asarray(x).size).astype(np.float32)


def apply_neural_codec(
    x: np.ndarray, codec: str, ffmpeg: Optional[str] = None
) -> np.ndarray:
    """Raw 16 kHz audio -> neural codec round trip -> float32 in [-1, 1] at 16 kHz.

    Raises ``ValueError`` for an unknown codec and ``RuntimeError`` when
    the codec's backend is unavailable on this machine.
    """
    if codec not in NEURAL_CODECS:
        raise ValueError(f"unknown codec {codec!r}; expected one of {NEURAL_CODECS}")
    avail = available_codecs(ffmpeg=ffmpeg)
    ok, reason = avail[codec]
    if not ok:
        raise RuntimeError(f"neural codec {codec!r} is unavailable: {reason}")
    if codec.startswith("encodec_"):
        bitrate = float(codec[len("encodec_") : -1].replace("_", "."))
        return np.clip(_encodec_round_trip(x, bitrate), -1.0, 1.0).astype(np.float32)
    exe = ffmpeg or find_ffmpeg()
    assert exe is not None  # availability was just checked above
    y16 = _ffmpeg_round_trip(_to16(np.asarray(x).ravel()), codec, exe)
    return y16.astype(np.float32) / 32767.0


class NeuralCodecAugmenter:
    """With probability ``p``, one neural codec from ``codecs``, uniformly, on raw audio.

    Same call contract as ``RealCodecAugmenter``: ``__call__(raw, rng)``.
    Unavailable codecs are dropped with a warning unless ``strict`` is
    set, in which case requesting one raises immediately. If none of the
    requested codecs is available and ``p > 0``, construction raises
    ``RuntimeError`` instead of training unaugmented.
    """

    def __init__(
        self,
        p: float,
        codecs: Sequence[str] = NEURAL_CODECS,
        strict: bool = False,
        ffmpeg: Optional[str] = None,
    ) -> None:
        bad = [c for c in codecs if c not in NEURAL_CODECS]
        if bad:
            raise ValueError(f"unknown codecs {bad}")
        self.p = float(p)
        self.requested = list(codecs)
        self.strict = bool(strict)
        self.ffmpeg = ffmpeg or find_ffmpeg()
        avail = available_codecs(ffmpeg=self.ffmpeg)
        self.codecs = [c for c in self.requested if avail[c][0]]
        self.unavailable = {c: avail[c][1] for c in self.requested if not avail[c][0]}
        if self.unavailable:
            msg = "neural codecs unavailable: " + "; ".join(
                f"{c} ({r})" for c, r in self.unavailable.items()
            )
            if self.strict:
                raise RuntimeError(msg)
            logger.warning("%s; they will not be applied", msg)
        if self.p > 0 and not self.codecs:
            raise RuntimeError(
                "no requested neural codec is available on this machine "
                f"(requested {self.requested}). Install the encodec package "
                "for Encodec, or ffmpeg with libopus / libmp3lame for Opus "
                "and MP3, or pass --neural-codec-p 0 to train without them."
            )

    def __call__(self, raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.p <= 0 or not self.codecs or rng.random() >= self.p:
            return raw
        return apply_neural_codec(
            raw, self.codecs[int(rng.integers(len(self.codecs)))], self.ffmpeg
        )

    def describe(self) -> dict:
        return {
            "p": self.p,
            "codecs": self.codecs,
            "requested": self.requested,
            "unavailable": self.unavailable,
            "strict": self.strict,
            "applied": "raw audio, before prepare_audio_window",
        }


if __name__ == "__main__":
    print("Running ml/neural_codecs.py self-tests...")
    sr = 16000
    t = np.arange(2 * sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    avail = available_codecs()
    print(f"  available: {[c for c, (ok, _) in avail.items() if ok]}")
    print(f"  unavailable: {[c for c, (ok, _) in avail.items() if not ok]}")

    rng = np.random.default_rng(0)
    aug = NeuralCodecAugmenter(0.0)
    assert aug(tone, rng) is tone
    print("  [OK] NeuralCodecAugmenter p=0 is identity")

    try:
        apply_neural_codec(tone, "gsm")
    except ValueError:
        print("  [OK] unknown codec is rejected")
    else:
        raise AssertionError("unknown codec was not rejected")

    for codec in ("opus_24k", "mp3_32k"):
        ok, _ = avail[codec]
        if not ok:
            print(f"  [SKIP] {codec} round trip, backend unavailable here")
            continue
        y = apply_neural_codec(tone, codec)
        assert y.dtype == np.float32 and y.size == tone.size
        assert np.isfinite(y).all() and np.abs(y).max() <= 1.0
        print(f"  [OK] {codec} round trip keeps length and range")

    print("All ml/neural_codecs.py self-tests PASSED successfully!")
