"""Telephony codec & channel anomaly detector.

Very small feature set (energy in 4 kHz–8 kHz, spectral rolloff, ratio
of high-band noise to low-band signal). Used as one of the inputs to
the calibrated risk fusion.
"""

from __future__ import annotations

import numpy as np


def detect(x: np.ndarray, sr: int) -> dict:
    from scipy.signal import welch

    x = x.astype(np.float32)
    if x.size < sr // 4:
        return {"codec_guess": "unknown", "anomaly": 0.0, "snr_db": 0.0, "low_band_share": 0.0}

    f, pxx = welch(x, fs=sr, nperseg=min(2048, x.size), noverlap=512)
    total = pxx.sum() + 1e-12
    low = pxx[(f >= 0) & (f < 4000)].sum() / total
    high = pxx[(f >= 4000) & (f < min(8000, sr / 2.0))].sum() / total
    very_low = pxx[(f >= 0) & (f < 300)].sum() / total
    very_high = pxx[(f >= 3400) & (f < min(8000, sr / 2.0))].sum() / total

    # crude codec guess
    if very_low < 0.005 and very_high < 0.005:
        codec_guess = "gsm_efr_or_amr_wb"
    elif high > 0.25:
        codec_guess = "wideband_opus"
    else:
        codec_guess = "g711_or_pstn"

    # anomaly: how much energy is in unexpected bands (spoofed TTS tends
    # to overshoot / undershoot specific bands after codec round-trips)
    expected_high = 0.10  # typical for natural speech on Opus
    anomaly = float(abs(high - expected_high))
    anomaly = min(1.0, anomaly / 0.4)

    # crude SNR estimate
    noise = max(very_low, 1e-6)
    signal = low + high
    snr_db = float(10.0 * np.log10((signal + 1e-12) / (noise + 1e-12)))

    return {"codec_guess": codec_guess, "anomaly": anomaly, "snr_db": snr_db, "low_band_share": float(low)}
