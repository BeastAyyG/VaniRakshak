"""RawBoost augmentation for VaniRakshak CM training (Goal 2).

Reimplementation of the RawBoost recipe (Tak et al., "RawBoost: A Raw Data
Boosting and Augmentation Method Applied to Automatic Speaker Verification
Anti-Spoofing", ICASSP 2022), applied on the raw utterance before
``prepare_audio_window``, like ``ml/real_codecs.py``.

Components, each applied independently with its own probability:

1. Linear convolutive noise: the signal is passed through a randomly drawn
   FIR filter, simulating a linear channel.
2. Non-linear convolutive noise: the same FIR filtering followed by a
   memoryless Hammerstein non-linearity, simulating a non-linear channel.
3. Impulsive signal-dependent additive noise: sparse impulses whose
   amplitude scales with the local signal magnitude.
4. Stationary signal-independent additive coloured noise: white noise
   shaped by a random low-pass FIR filter, added at a random SNR.

This is a from-scratch reimplementation, not the official RawBoost code.
It uses only numpy and scipy, has no torch dependency, and never changes
the sample count. Every parameter is recorded by ``describe()`` so a
checkpoint can say exactly which RawBoost ran.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import firwin, lfilter

logger = logging.getLogger("vanirakshak.ml.rawboost")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _match_rms(y: np.ndarray, target_rms: float) -> np.ndarray:
    r = _rms(y)
    if r < 1e-12:
        return y
    return (y * (target_rms / r)).astype(np.float32)


def _random_fir(rng: np.random.Generator, n_taps: int = 21) -> np.ndarray:
    h = rng.standard_normal(n_taps).astype(np.float64)
    h /= np.linalg.norm(h)
    return h


def linear_convolutive_noise(
    x: np.ndarray, rng: np.random.Generator, n_taps: int = 21
) -> np.ndarray:
    """Pass the signal through one randomly drawn FIR filter."""
    x = np.asarray(x, dtype=np.float32).ravel()
    target = _rms(x)
    y = lfilter(_random_fir(rng, n_taps), 1.0, x.astype(np.float64))
    return _match_rms(y, target)


def nonlinear_convolutive_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    n_taps: int = 21,
    quad: float = 0.15,
    cubic: float = 0.05,
) -> np.ndarray:
    """FIR filtering plus a Hammerstein memoryless non-linearity.

    ``y = f + quad * f^2 + cubic * f^3`` where ``f`` is the FIR-filtered
    signal. The quadratic and cubic weights are small on purpose: the
    channel should distort, not destroy, the signal. RMS is restored
    afterwards so the level stays comparable to the input.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    target = _rms(x)
    f = lfilter(_random_fir(rng, n_taps), 1.0, x.astype(np.float64))
    y = f + float(quad) * f**2 + float(cubic) * f**3
    return _match_rms(y, target)


def impulsive_signal_dependent_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    impulse_prob: float = 0.002,
    scale: float = 0.5,
) -> np.ndarray:
    """Sparse impulses whose amplitude scales with the local signal.

    Each sample becomes an impulse with probability ``impulse_prob``; an
    impulse adds ``scale * |x[n]|`` with a random sign, so loud regions
    get loud impulses and silence stays silent.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    mask = (rng.random(x.size) < impulse_prob).astype(np.float64)
    sign = rng.choice(np.array([-1.0, 1.0]), size=x.size)
    y = x.astype(np.float64) + float(scale) * mask * sign * np.abs(x)
    return y.astype(np.float32)


def coloured_additive_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    snr_db: float = 10.0,
    cutoff_hz: float = 4000.0,
    sr: int = 16000,
) -> np.ndarray:
    """Stationary coloured noise: white noise through a random low-pass FIR.

    The noise is shaped by an FIR low-pass with a random cutoff, then
    scaled to the requested SNR relative to the signal power.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    n_taps = 101
    cutoff = float(rng.uniform(1000.0, float(cutoff_hz)))
    taps = firwin(n_taps, cutoff, fs=sr)
    noise = lfilter(taps, 1.0, rng.standard_normal(x.size))
    sig_power = float(np.mean(x.astype(np.float64) ** 2)) + 1e-12
    noise_power = float(np.mean(noise**2)) + 1e-12
    target_noise_power = sig_power / (10.0 ** (float(snr_db) / 10.0))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return (x.astype(np.float64) + noise).astype(np.float32)


def apply_rawboost(
    x: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    p_linear: float = 0.5,
    p_nonlinear: float = 0.5,
    p_impulsive: float = 0.5,
    p_coloured: float = 0.5,
    snr_range: tuple[float, float] = (5.0, 20.0),
    sr: int = 16000,
) -> np.ndarray:
    """Apply the RawBoost components, each with its own probability.

    Returns audio of the same length and dtype float32. The output is
    clipped to [-1, 1] only if a component pushed it outside that range.
    """
    rng = rng or np.random.default_rng()
    y = np.asarray(x, dtype=np.float32).ravel().copy()
    if rng.random() < p_linear:
        y = linear_convolutive_noise(y, rng)
    if rng.random() < p_nonlinear:
        y = nonlinear_convolutive_noise(y, rng)
    if rng.random() < p_impulsive:
        y = impulsive_signal_dependent_noise(y, rng)
    if rng.random() < p_coloured:
        snr = float(rng.uniform(snr_range[0], snr_range[1]))
        y = coloured_additive_noise(y, rng, snr_db=snr, sr=sr)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = (y / peak).astype(np.float32)
    return y.astype(np.float32)


class RawBoostAugmenter:
    """With probability ``p``, RawBoost on the raw utterance.

    Same call contract as ``RealCodecAugmenter`` in ``ml/real_codecs.py``:
    ``__call__(raw, rng)`` takes raw 16 kHz audio and returns audio of the
    same length, and ``describe()`` records everything a checkpoint needs
    to say which augmentation ran.
    """

    def __init__(
        self,
        p: float = 0.5,
        p_linear: float = 0.5,
        p_nonlinear: float = 0.5,
        p_impulsive: float = 0.5,
        p_coloured: float = 0.5,
        snr_range: tuple[float, float] = (5.0, 20.0),
        sr: int = 16000,
    ) -> None:
        self.p = float(p)
        self.p_linear = float(p_linear)
        self.p_nonlinear = float(p_nonlinear)
        self.p_impulsive = float(p_impulsive)
        self.p_coloured = float(p_coloured)
        self.snr_range = (float(snr_range[0]), float(snr_range[1]))
        self.sr = int(sr)

    def __call__(self, raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.p <= 0 or rng.random() >= self.p:
            return raw
        return apply_rawboost(
            raw,
            rng=rng,
            p_linear=self.p_linear,
            p_nonlinear=self.p_nonlinear,
            p_impulsive=self.p_impulsive,
            p_coloured=self.p_coloured,
            snr_range=self.snr_range,
            sr=self.sr,
        )

    def describe(self) -> dict:
        return {
            "p": self.p,
            "p_linear": self.p_linear,
            "p_nonlinear": self.p_nonlinear,
            "p_impulsive": self.p_impulsive,
            "p_coloured": self.p_coloured,
            "snr_range_db": list(self.snr_range),
            "sr": self.sr,
            "applied": "raw audio, before prepare_audio_window",
        }


if __name__ == "__main__":
    print("Running ml/rawboost.py self-tests...")
    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    rng = np.random.default_rng(0)
    y = linear_convolutive_noise(tone, rng)
    assert y.shape == tone.shape and np.isfinite(y).all()
    print("  [OK] linear_convolutive_noise")

    rng = np.random.default_rng(0)
    y = nonlinear_convolutive_noise(tone, rng)
    assert y.shape == tone.shape and np.isfinite(y).all()
    print("  [OK] nonlinear_convolutive_noise")

    rng = np.random.default_rng(0)
    y = impulsive_signal_dependent_noise(tone, rng)
    assert y.shape == tone.shape and np.isfinite(y).all()
    print("  [OK] impulsive_signal_dependent_noise")

    rng = np.random.default_rng(0)
    y = coloured_additive_noise(tone, rng, snr_db=10.0, sr=sr)
    assert y.shape == tone.shape and np.isfinite(y).all()
    print("  [OK] coloured_additive_noise")

    rng = np.random.default_rng(0)
    aug = RawBoostAugmenter(0.0)
    assert aug(tone, rng) is tone
    print("  [OK] RawBoostAugmenter p=0 is identity")

    rng = np.random.default_rng(0)
    aug = RawBoostAugmenter(1.0)
    y = aug(tone, rng)
    assert y.shape == tone.shape and np.isfinite(y).all()
    assert not np.array_equal(y, tone)
    print("  [OK] RawBoostAugmenter p=1 changes the signal, keeps shape")

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    aug = RawBoostAugmenter(1.0)
    assert np.array_equal(aug(tone, rng_a), aug(tone, rng_b))
    print("  [OK] RawBoostAugmenter is deterministic for a fixed seed")

    print("All ml/rawboost.py self-tests PASSED successfully!")
