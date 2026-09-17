"""Tests for benchmarks/calibrate.py — affine score calibration (docs/15).

The properties here are the ones that separate a correct calibration from a
plausible-looking broken one. The most important is the invariance guard: EER
and minDCF must not move. If they do, the transform is not monotone-affine and
any actDCF improvement is meaningless.
"""

import json

import numpy as np
import pytest

from benchmarks.calibrate import (
    INVARIANCE_TOL,
    Calibration,
    TAU_BAYES,
    calibration_mask,
    evaluate,
    fit_affine_calibration,
)
from benchmarks.compute_mindcf import compute_cllr, compute_mincllr


def _scores(sep: float = 5.0, scale: float = 1.0, n: int = 4000, seed: int = 0):
    """Bonafide-high scores, which is this module's convention."""
    rng = np.random.default_rng(seed)
    return (rng.normal(sep, scale, n), rng.normal(-sep, scale, n))


# --------------------------------------------------------------------------
# the invariance guard
# --------------------------------------------------------------------------

def test_eer_and_mindcf_are_untouched_by_calibration():
    bon, spf = _scores()
    cal = fit_affine_calibration(bon, spf)
    ev = evaluate(cal, bon, spf)
    assert ev["eer_delta"] <= INVARIANCE_TOL
    assert ev["mindcf_delta"] <= INVARIANCE_TOL
    assert ev["invariance_ok"] is True


def test_invariance_holds_at_several_scales():
    """Rescaling the scores must not change anything rank-based.

    Note both the location *and* the spread are scaled together, so the two
    classes stay equally separated and only the units change.
    """
    for k in (0.1, 1.0, 17.0, 1000.0):
        rng = np.random.default_rng(0)
        bon = rng.normal(5.0 * k, 1.0 * k, 4000)
        spf = rng.normal(-5.0 * k, 1.0 * k, 4000)
        cal = fit_affine_calibration(bon, spf)
        ev = evaluate(cal, bon, spf)
        assert ev["invariance_ok"], f"invariance broken at k={k}"
        assert cal.is_monotone(), f"slope went negative at k={k}"


def test_near_separable_data_does_not_blow_up_the_newton_fit():
    """Near-separable data has no finite MLE; Newton can run away.

    The slope must stay positive and finite, and must never flip the ranking.
    The cap means the loss is ~0 either way, so this costs nothing.
    """
    rng = np.random.default_rng(0)
    bon = rng.normal(5000.0, 1000.0, 4000)
    spf = rng.normal(-5000.0, 1000.0, 4000)
    cal = fit_affine_calibration(bon, spf)
    assert np.isfinite(cal.a) and np.isfinite(cal.b)
    assert cal.is_monotone(), f"slope flipped on separable data: a={cal.a}"
    ev = evaluate(cal, bon, spf)
    assert ev["invariance_ok"] is True
    # Separable scores already sit at CLLR 0, so there is nothing to improve.
    # The requirement here is only that calibration does not make it worse.
    assert ev["cllr_after"] <= ev["cllr_before"] + 1e-9


def test_a_negative_slope_is_flagged_not_silently_used():
    """An inverted ranking must be visible, not quietly calibrated away."""
    bon, spf = _scores()
    cal = fit_affine_calibration(spf, bon)  # deliberately swapped
    assert cal.a < 0.0
    assert cal.is_monotone() is False
    assert any("negative" in n for n in cal.notes)


# --------------------------------------------------------------------------
# what calibration is actually for
# --------------------------------------------------------------------------

def test_calibration_drives_cllr_to_the_mincllr_floor():
    bon, spf = _scores()
    cal = fit_affine_calibration(bon, spf)
    assert cal.cllr_cal_after == pytest.approx(cal.mincllr_cal, abs=1e-6)
    assert cal.cllr_cal_after < cal.cllr_cal_before


def test_mincllr_itself_is_unchanged_by_calibration():
    bon, spf = _scores()
    cal = fit_affine_calibration(bon, spf)
    before = compute_mincllr(bon, spf)
    after = compute_mincllr(cal.apply(bon), cal.apply(spf))
    assert after == pytest.approx(before, abs=1e-6)


def test_actdcf_improves_when_scores_are_badly_scaled():
    """Same ranking, wrong scale: the exact defect calibration exists to fix."""
    bon, spf = _scores(sep=0.5, scale=0.1, seed=3)
    cal = fit_affine_calibration(bon, spf)
    ev = evaluate(cal, bon, spf)
    assert ev["actdcf_bayes_after"] < ev["actdcf_bayes_before"]
    assert ev["cllr_after"] < ev["cllr_before"]
    # and it must not have cheated by improving the ranking
    assert ev["invariance_ok"] is True


def test_already_calibrated_scores_are_left_essentially_alone():
    """Fitting scores that are already LLRs should move them very little."""
    rng = np.random.default_rng(7)
    # log-likelihood-ratio-like: bonafide positive, spoof negative, unit scale
    bon = rng.normal(1.2, 0.9, 20000)
    spf = rng.normal(-1.2, 0.9, 20000)
    cal = fit_affine_calibration(bon, spf)
    assert cal.cllr_cal_after == pytest.approx(cal.mincllr_cal, abs=1e-6)
    assert cal.is_monotone() is True


# --------------------------------------------------------------------------
# mechanics
# --------------------------------------------------------------------------

def test_apply_is_the_documented_affine_map():
    cal = Calibration(a=2.5, b=-1.0, mu=0.0, sd=1.0)
    s = np.array([-1.0, 0.0, 3.0])
    np.testing.assert_allclose(cal.apply(s), 2.5 * s - 1.0)


def test_fit_then_invert_recovers_the_original_scores():
    bon, spf = _scores()
    cal = fit_affine_calibration(bon, spf)
    original = np.concatenate([bon, spf])
    back = (cal.apply(original) - cal.b) / cal.a
    np.testing.assert_allclose(back, original, rtol=1e-8, atol=1e-8)


def test_constant_scores_do_not_crash_and_produce_identity_slope():
    bon = np.full(500, 3.0)
    spf = np.full(500, 3.0)
    cal = fit_affine_calibration(bon, spf)
    assert np.isfinite(cal.a) and np.isfinite(cal.b)
    assert cal.sd == 0.0
    assert cal.a == 0.0


def test_single_class_raises():
    with pytest.raises(ValueError):
        fit_affine_calibration(np.array([1.0, 2.0]), np.array([]))


def test_json_roundtrip():
    bon, spf = _scores(n=500)
    cal = fit_affine_calibration(bon, spf)
    back = Calibration.from_dict(json.loads(cal.to_json()))
    assert back.a == cal.a and back.b == cal.b
    assert back.n_bon == cal.n_bon and back.n_spf == cal.n_spf
    assert back.is_monotone() == cal.is_monotone()


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------

def test_split_is_deterministic():
    paths = [f"/corpus/utt_{i}.flac" for i in range(1000)]
    first = calibration_mask(paths)
    second = calibration_mask(paths)
    np.testing.assert_array_equal(first, second)


def test_split_is_balanced_enough_to_be_usable():
    paths = [f"/corpus/utt_{i}.flac" for i in range(2000)]
    n_cal = int(calibration_mask(paths).sum())
    assert 800 < n_cal < 1200, f"split is lopsided: {n_cal}/2000"


def test_split_ignores_ordering():
    paths = [f"/corpus/utt_{i}.flac" for i in range(500)]
    shuffled = paths[::-1]
    m1 = dict(zip(paths, calibration_mask(paths)))
    m2 = dict(zip(shuffled, calibration_mask(shuffled)))
    assert m1 == m2
