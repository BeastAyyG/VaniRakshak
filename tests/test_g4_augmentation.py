import ast

import numpy as np
import pytest
import torch

from ml.augment import (
    CodecTelephonyAugmenter,
    RawBoostAugmenter,
    add_laplacian_impulsive_noise,
    prepare_audio_window,
)


def test_same_seed_produces_identical_output():
    x = np.ones(4, dtype=np.float32)
    a = RawBoostAugmenter(p_impulsive=1.0, p_convolutive=1.0, seed=1)
    y1 = a(x, rng=np.random.default_rng(0))
    y2 = a(x, rng=np.random.default_rng(0))
    assert np.array_equal(y1, y2)


def test_different_seed_produces_different_output():
    x = np.ones(4, dtype=np.float32)
    a = RawBoostAugmenter(p_impulsive=1.0, p_convolutive=1.0, seed=1)
    y1 = a(x, rng=np.random.default_rng(42))
    y2 = a(x, rng=np.random.default_rng(43))
    assert not np.array_equal(y1, y2)


def test_zero_probability_identity_out_of_range_amplitudes():
    x = (np.linspace(-3.0, 3.0, 1000)).astype(np.float32)
    a = RawBoostAugmenter(p_impulsive=0.0, p_convolutive=0.0, seed=4)
    y = a(x, rng=np.random.default_rng(0))
    assert np.array_equal(y, x)
    assert y.dtype == np.float32


def _tone() -> np.ndarray:
    t = np.linspace(0, 1.0, 16000, endpoint=False).astype(np.float32)
    return (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_input_not_mutated():
    x = _tone()
    before = x.copy()
    a = RawBoostAugmenter(p_impulsive=1.0, p_convolutive=1.0, seed=2)
    a(x, rng=np.random.default_rng(0))
    assert np.array_equal(x, before)


def test_rejects_nonfinite_input():
    x = _tone()
    x[100] = np.nan
    a = RawBoostAugmenter(p_impulsive=0.0, p_convolutive=0.0, seed=2)
    with pytest.raises(ValueError):
        a(x, rng=np.random.default_rng(0))
    x2 = _tone()
    x2[200] = np.inf
    with pytest.raises(ValueError):
        a(x2, rng=np.random.default_rng(0))


def test_rejects_non_mono_input():
    x = np.stack([_tone(), _tone()])
    a = RawBoostAugmenter(p_impulsive=0.0, p_convolutive=0.0, seed=2)
    with pytest.raises(ValueError):
        a(x, rng=np.random.default_rng(0))


def test_invalid_snr_range_rejected():
    for bad in [(5.0, 5.0), (20.0, 10.0), (float("nan"), 15.0), (10.0, float("inf"))]:
        with pytest.raises(ValueError):
            RawBoostAugmenter(impulsive_snr_range=bad)


def test_invalid_sr_rejected():
    for bad in [0, -16000, 8000.5, float("nan")]:
        with pytest.raises(ValueError):
            RawBoostAugmenter(sr=bad)


def test_measured_snr_close_to_target():
    x = _tone()
    target = 20.0
    r = np.random.default_rng(0)
    y = add_laplacian_impulsive_noise(x, snr_db=target, rng=r)
    noise = y - x
    sig_rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    measured = 20.0 * np.log10(sig_rms / noise_rms)
    assert abs(measured - target) < 0.5


def test_impulsive_noise_preserves_silence():
    z = np.zeros(16000, dtype=np.float32)
    y = add_laplacian_impulsive_noise(z, snr_db=15.0, rng=np.random.default_rng(0))
    assert bool((y == 0).all())


def test_loader_default_has_no_rawboost():
    import ml.train_cm as tc

    ds = tc.AudioDataset.__new__(tc.AudioDataset)
    ds.samples = [("x.wav", 1, "-")]
    loader = tc.FastAudioAugmentLoader(
        dataset=ds,
        augmenter=CodecTelephonyAugmenter(seed=1),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        n_threads=1,
        queue_size=1,
    )
    assert loader.rawboost_augmenter is None


def test_loader_chains_rawboost_after_codec(monkeypatch):
    import ml.train_cm as tc

    tone = _tone()
    monkeypatch.setattr(tc.sf, "read", lambda *a, **k: (tone.copy(), 16000))
    ds = tc.AudioDataset.__new__(tc.AudioDataset)
    ds.samples = [("x.wav", 1, "-")]
    seed = 7
    loader = tc.FastAudioAugmentLoader(
        dataset=ds,
        augmenter=CodecTelephonyAugmenter(seed=seed),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        n_threads=1,
        queue_size=1,
        seed=seed,
        rawboost_augmenter=RawBoostAugmenter(seed=seed),
    )
    got, label = loader._load_and_augment((("x.wav", 1, "-"), 0, 0))
    assert label == 1
    rng = np.random.default_rng([seed, 0, 0])
    w = prepare_audio_window(tone, sr=16000, target_duration=4.0, random_crop=True, rng=rng)
    w_aug = CodecTelephonyAugmenter(seed=seed)(w, rng=rng)
    want = RawBoostAugmenter(seed=seed)(w_aug, rng=rng)
    assert np.array_equal(got, want)


def test_loader_default_off_matches_codec_only(monkeypatch):
    import ml.train_cm as tc

    tone = _tone()
    monkeypatch.setattr(tc.sf, "read", lambda *a, **k: (tone.copy(), 16000))
    ds = tc.AudioDataset.__new__(tc.AudioDataset)
    ds.samples = [("x.wav", 1, "-")]
    seed = 7
    loader = tc.FastAudioAugmentLoader(
        dataset=ds,
        augmenter=CodecTelephonyAugmenter(seed=seed),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        n_threads=1,
        queue_size=1,
        seed=seed,
    )
    got, _ = loader._load_and_augment((("x.wav", 1, "-"), 0, 0))
    rng = np.random.default_rng([seed, 0, 0])
    w = prepare_audio_window(tone, sr=16000, target_duration=4.0, random_crop=True, rng=rng)
    want = CodecTelephonyAugmenter(seed=seed)(w, rng=rng)
    assert np.array_equal(got, want)


def test_recipe_a_rejects_existing_checkpoint(tmp_path, monkeypatch):
    import ml.train_cm as tc

    out = tmp_path / "existing.pt"
    out.write_bytes(b"pretend checkpoint")
    manifest = tmp_path / "train.tsv"
    manifest.write_text("x.wav\tbonafide\t-\n", encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("must not reach dataset/model loading")

    monkeypatch.setattr(tc, "AudioDataset", _boom)
    with pytest.raises(FileExistsError):
        tc.train_recipe_a(
            train_manifest=manifest,
            dev_manifest=manifest,
            output_checkpoint=out,
            epochs=0,
            device="cpu",
        )


def test_recipe_a_checkpoint_metadata_records_flag():
    import inspect

    import ml.train_cm as tc

    src = inspect.getsource(tc.train_recipe_a)
    assert '"g4_rawboost"' in src
    assert RawBoostAugmenter.VERSION == "g4-rawboost-v1"


def test_cli_flag_wiring():
    import inspect

    import ml.train_cm as tc

    src = inspect.getsource(tc.main)
    assert "--g4-rawboost" in src
    tree = ast.parse(src)
    kwargs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "train_recipe_a":
            kwargs |= {k.arg for k in node.keywords}
    assert "g4_rawboost" in kwargs


def test_threadpool_reproducibility():
    from concurrent.futures import ThreadPoolExecutor

    x = _tone()
    n = 64
    a = RawBoostAugmenter(p_impulsive=1.0, p_convolutive=1.0, seed=13)

    def work(i):
        rng = np.random.default_rng([42, 0, i])
        return a(x, rng=rng)

    with ThreadPoolExecutor(max_workers=16) as ex:
        outs = list(ex.map(work, range(n)))
    for i in range(n):
        expected = a(x, rng=np.random.default_rng([42, 0, i]))
        assert np.array_equal(outs[i], expected)
