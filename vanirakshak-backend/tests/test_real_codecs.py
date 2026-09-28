"""docs/16: the real-codec augmenter works on raw audio and never clips it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.real_codecs import REAL_CODECS, RealCodecAugmenter, apply_real_codec, find_ffmpeg  # noqa: E402

SR = 16000


def _speechlike(seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return x.astype(np.float32)


@pytest.mark.parametrize("codec", ["g711u", "g711a"])
def test_g711_round_trip_is_faithful(codec):
    x = _speechlike()
    y = apply_real_codec(x, codec)
    assert y.dtype == np.float32 and abs(y.size - x.size) <= 2
    n = min(x.size, y.size)
    assert np.abs(y).max() <= 1.0
    # G.711 on in-range audio is close to transparent; the broken
    # post-normalisation path had relative error ~0.64.
    assert np.linalg.norm(y[:n] - x[:n]) / np.linalg.norm(x[:n]) < 0.1


@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg with AMR encoders not installed")
@pytest.mark.parametrize("codec", ["amrnb_12k2", "amrwb_12k65"])
def test_amr_round_trip_keeps_length_and_range(codec):
    x = _speechlike()
    y = apply_real_codec(x, codec)
    assert y.dtype == np.float32 and np.abs(y).max() <= 1.0
    assert abs(y.size - x.size) < SR // 10


def test_augmenter_off_is_identity_and_p1_always_applies():
    x = _speechlike()
    rng = np.random.default_rng(0)
    assert RealCodecAugmenter(0.0)(x, rng) is x
    y = RealCodecAugmenter(1.0, ["g711u"])(x, rng)
    assert y is not x and np.abs(y).max() <= 1.0


def test_unknown_codec_is_rejected():
    with pytest.raises(ValueError):
        RealCodecAugmenter(0.5, ["mp3"])
    with pytest.raises(ValueError):
        apply_real_codec(_speechlike(), "gsm")
    assert set(REAL_CODECS) == {"g711u", "g711a", "amrnb_12k2", "amrwb_12k65"}
