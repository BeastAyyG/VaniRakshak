"""Codec & telephony audio augmentation pipeline for VaniRakshak CM training.

Implements the augmentation transforms specified in docs/08-cm-training-plan.md §4
and docs/03-codec-augmentation-and-telephony.md:
  - G.711 A-law companding
  - G.711 μ-law companding
  - AMR-WB simulation (50 Hz - 7000 Hz band-limiting + 4-bit quantisation)
  - GSM-EFR simulation (300 Hz - 3400 Hz classic PSTN bandpass)
  - Additive Gaussian noise (SNR 5-20 dB)
  - Multi-talker babble noise (SNR 5-20 dB)
  - Synthetic room impulse response (RIR) reverberation
  - Random gain jitter (±6 dB)
  - Telephony bandpass filtering

Wraps and leverages the DSP primitives in vanirakshak-backend/vanirakshak/data/augment.py.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

# Ensure vanirakshak-backend is discoverable for primitive imports
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "vanirakshak-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from vanirakshak.data.augment import (
        AUGMENT_REGISTRY,
        AugmentSpec,
        _bandlimit,
        _normalise_int16,
        add_babble_simple,
        add_noise,
        alaw_encode_decode,
        amr_wb_simulate,
        apply_augmentation,
        apply_random_pipeline,
        apply_rir,
        gsm_efr_simulate,
        random_gain,
        synthetic_rir,
        ulaw_encode_decode,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import augmentation primitives from vanirakshak.data.augment: {e}"
    ) from e

logger = logging.getLogger("vanirakshak.ml.augment")

# ---------------------------------------------------------------------------
# Telephony Bandpass Helper
# ---------------------------------------------------------------------------

def telephony_bandpass(
    x: np.ndarray,
    sr: int = 16000,
    lowcut: float = 300.0,
    highcut: float = 3400.0,
    order: int = 4,
) -> np.ndarray:
    """Apply telephone bandpass filtering (default 300-3400 Hz PSTN standard)."""
    return _bandlimit(x, sr, lowcut=lowcut, highcut=highcut, order=order)


# ---------------------------------------------------------------------------
# Audio Window Framing & Normalization
# ---------------------------------------------------------------------------

def prepare_audio_window(
    audio: Union[np.ndarray, torch.Tensor],
    sr: int = 16000,
    target_sr: int = 16000,
    target_duration: float = 4.0,
    random_crop: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Resample, frame (crop/pad to 4.0s = 64,000 samples), and normalize audio.

    Per docs/08 §4:
      - Resample to 16,000 Hz mono float32.
      - 4.0s target window (64,000 samples).
      - Random 4.0s crop during training; center crop during evaluation.
      - Zero-mean unit-variance normalization.
    """
    rng = rng or np.random.default_rng()

    if isinstance(audio, torch.Tensor):
        x = audio.detach().cpu().numpy().squeeze().astype(np.float32)
    else:
        x = np.asarray(audio, dtype=np.float32).squeeze()

    if x.ndim > 1:
        x = np.mean(x, axis=0)

    # Resample if needed
    if sr != target_sr:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    target_samples = int(target_sr * target_duration)
    n_samples = len(x)

    if n_samples < target_samples:
        padded = np.zeros(target_samples, dtype=np.float32)
        # Place audio at the start or center
        padded[:n_samples] = x
        x = padded
    elif n_samples > target_samples:
        if random_crop:
            start = rng.integers(0, n_samples - target_samples + 1)
        else:
            start = (n_samples - target_samples) // 2
        x = x[start : start + target_samples]

    # Zero-mean unit-variance normalization
    std = float(np.std(x))
    if std > 1e-6:
        x = (x - float(np.mean(x))) / std
    else:
        x = x - float(np.mean(x))

    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Stochastic Codec & Telephony Augmenter
# ---------------------------------------------------------------------------

class CodecTelephonyAugmenter:
    """Stochastic audio pipeline for telephony codec and channel simulation.

    Applies the augmentations described in docs/08 §4:
      - p_codec: 0.5 (G.711 A-law, G.711 μ-law, AMR-WB, GSM-EFR)
      - p_noise: 0.3 (Gaussian or multi-talker babble noise with SNR 5-20 dB)
      - p_bandpass: 0.3 (Telephony bandpass 300-3400 Hz)
      - p_rir: 0.2 (Synthetic room impulse response T60 100-300 ms)
      - p_gain: 0.3 (Random amplitude jitter ±6 dB)
    """

    def __init__(
        self,
        sr: int = 16000,
        p_codec: float = 0.5,
        p_noise: float = 0.3,
        p_bandpass: float = 0.3,
        p_rir: float = 0.2,
        p_gain: float = 0.3,
        codecs: Sequence[str] = ("alaw", "ulaw", "amr_wb", "gsm_efr"),
        snr_range: tuple[float, float] = (5.0, 20.0),
        seed: Optional[int] = None,
    ) -> None:
        self.sr = sr
        self.p_codec = p_codec
        self.p_noise = p_noise
        self.p_bandpass = p_bandpass
        self.p_rir = p_rir
        self.p_gain = p_gain
        self.codecs = list(codecs)
        self.snr_range = snr_range
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Apply stochastic augmentations to the given audio waveform."""
        local_rng = rng or self.rng

        if isinstance(audio, torch.Tensor):
            x = audio.detach().cpu().numpy().copy().astype(np.float32)
        else:
            x = np.asarray(audio, dtype=np.float32).copy()

        # 1. Telephony Codec Companding / Simulation (p=0.5)
        if local_rng.random() < self.p_codec and self.codecs:
            codec_choice = local_rng.choice(self.codecs)
            if codec_choice == "alaw":
                x = alaw_encode_decode(x)
            elif codec_choice == "ulaw":
                x = ulaw_encode_decode(x)
            elif codec_choice == "amr_wb":
                x = amr_wb_simulate(x, self.sr)
            elif codec_choice == "gsm_efr":
                x = gsm_efr_simulate(x, self.sr)

        # 2. Telephony Bandpass Filter (p=0.3)
        if local_rng.random() < self.p_bandpass:
            lowcut = float(local_rng.uniform(250.0, 350.0))
            highcut = float(local_rng.uniform(3200.0, 3600.0))
            x = telephony_bandpass(x, sr=self.sr, lowcut=lowcut, highcut=highcut)

        # 3. Room Impulse Response (RIR) Reverberation (p=0.2)
        if local_rng.random() < self.p_rir:
            t60 = float(local_rng.uniform(100.0, 300.0))
            rir = synthetic_rir(self.sr, t60_ms=t60, rng=local_rng)
            x = apply_rir(x, rir)

        # 4. Additive Noise / Babble (p=0.3)
        if local_rng.random() < self.p_noise:
            snr = float(local_rng.uniform(self.snr_range[0], self.snr_range[1]))
            # 50% chance Gaussian noise, 50% chance babble noise
            if local_rng.random() < 0.5:
                x = add_noise(x, snr_db=snr, rng=local_rng)
            else:
                x = add_babble_simple(x, snr_db=snr, sr=self.sr, rng=local_rng)

        # 5. Random Gain Jitter (p=0.3)
        if local_rng.random() < self.p_gain:
            x = random_gain(x, low_db=-6.0, high_db=6.0, rng=local_rng)

        return x.astype(np.float32)


def augment_audio(
    audio: Union[np.ndarray, torch.Tensor],
    sr: int = 16000,
    p_codec: float = 0.5,
    p_noise: float = 0.3,
    p_bandpass: float = 0.3,
    p_rir: float = 0.2,
    p_gain: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Convenience function for single-waveform augmentation."""
    augmenter = CodecTelephonyAugmenter(
        sr=sr,
        p_codec=p_codec,
        p_noise=p_noise,
        p_bandpass=p_bandpass,
        p_rir=p_rir,
        p_gain=p_gain,
    )
    return augmenter(audio, rng=rng)


# ---------------------------------------------------------------------------
# Self-Test CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running ml/augment.py self-tests...")
    test_sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(test_sr * duration), endpoint=False)
    # Synthetic multi-tone signal
    tone = (
        0.5 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.2 * np.sin(2 * np.pi * 1760 * t)
    ).astype(np.float32)

    # Test window preparation
    w = prepare_audio_window(tone, sr=test_sr, target_duration=4.0, random_crop=False)
    assert w.shape == (64000,), f"Expected (64000,), got {w.shape}"
    assert np.isfinite(w).all(), "NaN/Inf in prepared window"
    print("  [OK] prepare_audio_window (length, finiteness, normalization)")

    # Test each primitive transform directly
    y_alaw = alaw_encode_decode(w)
    assert y_alaw.shape == (64000,) and np.isfinite(y_alaw).all()
    print("  [OK] alaw_encode_decode")

    y_ulaw = ulaw_encode_decode(w)
    assert y_ulaw.shape == (64000,) and np.isfinite(y_ulaw).all()
    print("  [OK] ulaw_encode_decode")

    y_amr = amr_wb_simulate(w, test_sr)
    assert y_amr.shape == (64000,) and np.isfinite(y_amr).all()
    print("  [OK] amr_wb_simulate")

    y_gsm = gsm_efr_simulate(w, test_sr)
    assert y_gsm.shape == (64000,) and np.isfinite(y_gsm).all()
    print("  [OK] gsm_efr_simulate")

    y_bp = telephony_bandpass(w, test_sr)
    assert y_bp.shape == (64000,) and np.isfinite(y_bp).all()
    print("  [OK] telephony_bandpass")

    y_noise = add_noise(w, snr_db=10.0)
    assert y_noise.shape == (64000,) and np.isfinite(y_noise).all()
    print("  [OK] add_noise")

    y_babble = add_babble_simple(w, snr_db=10.0, sr=test_sr)
    assert y_babble.shape == (64000,) and np.isfinite(y_babble).all()
    print("  [OK] add_babble_simple")

    rir = synthetic_rir(test_sr, t60_ms=200.0)
    y_rir = apply_rir(w, rir)
    assert y_rir.shape == (64000,) and np.isfinite(y_rir).all()
    print("  [OK] synthetic_rir + apply_rir")

    y_gain = random_gain(w)
    assert y_gain.shape == (64000,) and np.isfinite(y_gain).all()
    print("  [OK] random_gain")

    # Test full pipeline with high probability
    aug = CodecTelephonyAugmenter(
        p_codec=1.0,
        p_noise=1.0,
        p_bandpass=1.0,
        p_rir=1.0,
        p_gain=1.0,
        seed=42,
    )
    y_full = aug(w)
    assert y_full.shape == (64000,) and np.isfinite(y_full).all()
    print("  [OK] CodecTelephonyAugmenter (full compound transform)")

    print("All ml/augment.py self-tests PASSED successfully!")
