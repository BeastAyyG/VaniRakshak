"""Telephony codec & channel augmentation pipeline (R1).

Implements the augmentation stack described in ``docs/03-codec-augmentation-and-telephony.md``:

    G.711 a-law, G.711 μ-law, AMR-WB simulation, band-limit to 4 kHz,
    additive Gaussian/impulse noise, simple RIR convolution, random gain.

This module runs in pure numpy so the **demo** and the **augmentation
unit tests** work on a laptop with no GPU and no ``torchaudio``.
Real training (R2) will swap in torchaudio/sox for full Opus/MP3
encoding, but the DSP the team needs to demonstrate on stage is here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


# ---------------------------------------------------------------------------
# Codec primitives
# ---------------------------------------------------------------------------

def _normalise_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def alaw_encode_decode(x: np.ndarray) -> np.ndarray:
    """G.711 A-law encode + decode (companding)."""
    A = 87.6
    x = _normalise_int16(x)
    sign = np.sign(x)
    magnitude = np.abs(x) + 1e-12
    compressed = np.where(
        magnitude < 1.0 / A,
        A * magnitude / (1.0 + np.log(A)),
        (1.0 + np.log(A * magnitude)) / (1.0 + np.log(A)),
    )
    encoded = sign * compressed
    return encoded.astype(np.float32)


def ulaw_encode_decode(x: np.ndarray) -> np.ndarray:
    """G.711 μ-law encode + decode (companding)."""
    MU = 255.0
    x = _normalise_int16(x)
    sign = np.sign(x)
    magnitude = np.abs(x)
    encoded = sign * (np.log(1.0 + MU * magnitude) / np.log(1.0 + MU))
    return encoded.astype(np.float32)


def amr_wb_simulate(x: np.ndarray, sr: int) -> np.ndarray:
    """Lightweight AMR-WB simulation: band-limit to 50 Hz–7 kHz, then
    apply a mild 4-bit amplitude quantisation step. Not bit-exact
    AMR-WB (that needs 3GPP ref code), but matches the artefact
    profile for augmentation purposes.
    """
    from scipy.signal import butter, sosfilt

    y = _bandlimit(x, sr, lowcut=50.0, highcut=7000.0, order=4)
    levels = 16
    y_q = np.round(y * (levels / 2)) / (levels / 2)
    return y_q.astype(np.float32)


def gsm_efr_simulate(x: np.ndarray, sr: int) -> np.ndarray:
    """GSM-EFR simulation: band-limit to 300–3400 Hz (classic PSTN)."""
    return _bandlimit(x, sr, lowcut=300.0, highcut=3400.0, order=4).astype(np.float32)


def _bandlimit(x: np.ndarray, sr: int, *, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    nyq = sr / 2.0
    low = max(lowcut / nyq, 1e-4)
    high = min(highcut / nyq, 0.9999)
    if low >= high:
        return x
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, x).astype(np.float32)


# ---------------------------------------------------------------------------
# Noise / RIR / gain
# ---------------------------------------------------------------------------

def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Additive Gaussian noise at a target SNR (dB)."""
    rng = rng or np.random.default_rng()
    signal_power = np.mean(x ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)


def add_babble_simple(x: np.ndarray, snr_db: float, n_talkers: int = 4, sr: int = 16000, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    babble = np.zeros_like(x, dtype=np.float32)
    for _ in range(n_talkers):
        white = rng.normal(0, 1, size=x.shape).astype(np.float32)
        babble += _bandlimit(white, sr, lowcut=200.0, highcut=3500.0, order=2)
    babble /= max(n_talkers, 1)
    # scale babble to target SNR relative to signal
    sig_power = np.mean(x ** 2) + 1e-12
    bab_power = np.mean(babble ** 2) + 1e-12
    babble *= np.sqrt(sig_power / (bab_power * (10 ** (snr_db / 10.0))))
    return (x + babble).astype(np.float32)


def apply_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with a room impulse response (small room approximation)."""
    from scipy.signal import fftconvolve
    y = fftconvolve(x, rir, mode="full")[: len(x)]
    return y.astype(np.float32)


def synthetic_rir(sr: int, t60_ms: float = 250.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a synthetic exponentially-decaying noise RIR."""
    rng = rng or np.random.default_rng()
    t60_samples = int(sr * t60_ms / 1000.0)
    decay = np.exp(-np.arange(t60_samples) / (t60_samples / 3.0))
    rir = rng.normal(0, 1, size=t60_samples).astype(np.float32) * decay
    rir[0] = 1.0
    rir /= np.max(np.abs(rir)) + 1e-9
    return rir


def random_gain(x: np.ndarray, low_db: float = -6.0, high_db: float = 6.0, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    g_db = rng.uniform(low_db, high_db)
    return (x * 10 ** (g_db / 20.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Public augmentation registry
# ---------------------------------------------------------------------------

@dataclass
class AugmentSpec:
    name: str
    fn: "object | None"


AUGMENT_REGISTRY = {
    "alaw": AugmentSpec("G.711 a-law", alaw_encode_decode),
    "ulaw": AugmentSpec("G.711 μ-law", ulaw_encode_decode),
    "amr_wb": AugmentSpec("AMR-WB (simulated)", None),
    "gsm_efr": AugmentSpec("GSM-EFR (PSTN 300-3400 Hz)", None),
    "babble_5db": AugmentSpec("Babble @ 5 dB SNR", None),
    "babble_15db": AugmentSpec("Babble @ 15 dB SNR", None),
    "rir_office": AugmentSpec("Office RIR", None),
    "gain_jitter": AugmentSpec("Random gain ±6 dB", None),
}


def apply_augmentation(x: np.ndarray, sr: int, name: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """Dispatch a named augmentation. Unknown names pass through."""
    rng = rng or np.random.default_rng()  # noqa: F841
    if name == "alaw":
        return alaw_encode_decode(x)
    if name == "ulaw":
        return ulaw_encode_decode(x)
    if name == "amr_wb":
        return amr_wb_simulate(x, sr)
    if name == "gsm_efr":
        return gsm_efr_simulate(x, sr)
    if name == "babble_5db":
        return add_babble_simple(x, snr_db=5.0, sr=sr, rng=rng)
    if name == "babble_10db":
        return add_babble_simple(x, snr_db=10.0, sr=sr, rng=rng)
    if name == "babble_15db":
        return add_babble_simple(x, snr_db=15.0, sr=sr, rng=rng)
    if name == "rir_office":
        rir = synthetic_rir(sr, t60_ms=280.0, rng=rng)
        return apply_rir(x, rir)
    if name == "gain_jitter":
        return random_gain(x, rng=rng)
    return x


def apply_random_pipeline(
    x: np.ndarray,
    sr: int,
    augments: Iterable[str] = ("alaw", "ulaw", "amr_wb", "gsm_efr", "babble_10db", "rir_office", "gain_jitter"),
    seed: int | None = None,
) -> np.ndarray:
    """Apply each augmentation in the iterable (deterministic order)."""
    rng = np.random.default_rng(seed)
    y = x.astype(np.float32, copy=True)
    for name in augments:
        y = apply_augmentation(y, sr, name, rng=rng)
    return y
