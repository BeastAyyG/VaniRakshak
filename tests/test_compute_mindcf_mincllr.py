"""Tests for minCLLR and the CLLR calibration-loss decomposition.

docs/10 (line 263, "Replacement criterion for future calibration work") makes
``CLLR - minCLLR`` the criterion for all future calibration claims, and notes
that ``benchmarks/compute_mindcf.py`` computed only ``compute_cllr``. These
tests pin the properties that make the decomposition trustworthy.

The properties below are the ones that actually discriminate a correct
implementation from a plausible-looking wrong one:

* ``minCLLR <= CLLR`` always, and ``minCLLR == CLLR`` when the scores are
  already optimally calibrated.
* Invariance under a positive affine rescaling of the input scores. minCLLR is
  a statement about *ranking*, so ``s -> c*s + d`` with ``c > 0`` must not
  change it at all. This is the sharpest available correctness check.
* The no-information floor. For scores carrying no class information, the
  optimal transform shrinks the slope to zero and minCLLR lands on 1.0, the
  prior-only CLLR. Raw noise scores sit *above* 1.0, so minCLLR < CLLR there.
* Agreement with an independent optimiser (scipy) to numerical precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.compute_mindcf import (
    compute_calibration_loss,
    compute_cllr,
    compute_mincllr,
)


def _draw(seed: int = 0, n_tar: int = 800, n_non: int = 2400):
    rng = np.random.default_rng(seed)
    return rng.normal(1.0, 1.2, n_tar), rng.normal(-1.0, 1.2, n_non)


def test_mincllr_never_exceeds_cllr():
    """The optimum can never be worse than the identity transform."""
    for seed in range(20):
        bon, spf = _draw(seed)
        assert compute_mincllr(bon, spf) <= compute_cllr(bon, spf) + 1e-12


def test_calibration_loss_is_non_negative():
    for seed in range(20):
        bon, spf = _draw(seed)
        assert compute_calibration_loss(bon, spf) >= -1e-12


def test_perfect_separation_is_zero():
    """Perfectly separable scores admit a transform with zero cost."""
    bon = np.linspace(5.0, 10.0, 500)
    spf = np.linspace(-10.0, -5.0, 1500)
    assert compute_mincllr(bon, spf) < 1e-6


def test_true_llr_scores_have_zero_calibration_loss():
    """Feed the scores that ARE the log-likelihood ratio.

    For two unit-variance Gaussians at means mu_tar and mu_non, the true LLR is
    linear with slope ``(mu_tar - mu_non) / sigma**2``. Here that slope is 4, so
    the raw draws are deliberately under-confident by a factor of 4 and only
    the scaled scores are calibrated. This test is what catches an
    implementation that silently fits the wrong parameterisation.
    """
    rng = np.random.default_rng(1)
    bon = rng.normal(2.0, 1.0, 2000)
    spf = rng.normal(-2.0, 1.0, 6000)

    assert compute_calibration_loss(4.0 * bon, 4.0 * spf) < 0.02
    assert compute_calibration_loss(bon, spf) > 0.1


def test_invariant_under_positive_affine_rescaling():
    """minCLLR measures ranking, so positive affine rescaling must be a no-op."""
    for seed in range(6):
        bon, spf = _draw(seed, 500, 1500)
        base = compute_mincllr(bon, spf)
        for c, d in ((0.01, 0.0), (7.5, -3.0), (1000.0, 500.0)):
            assert compute_mincllr(c * bon + d, c * spf + d) == pytest.approx(base, abs=1e-8)


def test_no_information_hits_the_floor():
    """Uninformative scores shrink to the prior-only CLLR of 1.0.

    Raw noise scores are *above* 1.0 (for standard-normal noise the raw CLLR is
    about 1.17), which is why CLLR = 1.0 must be reachable only through the
    optimal transform. Asserting CLLR == 1.0 on raw noise would be wrong.
    """
    rng = np.random.default_rng(2)
    bon = rng.normal(0.0, 1.0, 2000)
    spf = rng.normal(0.0, 1.0, 6000)

    raw = compute_cllr(bon, spf)
    floor = compute_mincllr(bon, spf)

    assert raw == pytest.approx(1.17, abs=0.05)
    assert floor == pytest.approx(1.0, abs=0.02)
    assert floor < raw


def test_miscalibration_moves_cllr_but_not_mincllr():
    """A pure shift is a calibration defect: CLLR rises, minCLLR does not."""
    bon, spf = _draw(3)
    base = compute_mincllr(bon, spf)
    shift = 4.0

    assert compute_mincllr(bon + shift, spf + shift) == pytest.approx(base, abs=1e-8)
    assert compute_cllr(bon + shift, spf + shift) > compute_cllr(bon, spf) + 0.1


def test_matches_independent_optimiser():
    """Cross-check against scipy's Nelder-Mead on the same objective."""
    scipy_optimize = pytest.importorskip("scipy.optimize")

    def reference(bon: np.ndarray, spf: np.ndarray) -> float:
        s_all = np.concatenate([bon, spf])
        s = (s_all - s_all.mean()) / s_all.std()
        y = np.concatenate([np.ones(bon.size), np.zeros(spf.size)])
        w = np.concatenate(
            [np.full(bon.size, 1.0 / bon.size), np.full(spf.size, 1.0 / spf.size)]
        )

        def objective(p):
            z = p[0] * s + p[1]
            return 0.5 * np.sum(w * (np.logaddexp(0.0, z) - y * z)) / np.log(2.0)

        result = scipy_optimize.minimize(
            objective,
            [1.0, 0.0],
            method="Nelder-Mead",
            options=dict(xatol=1e-11, fatol=1e-13, maxiter=40000, maxfev=40000),
        )
        return float(result.fun)

    for seed in range(4):
        bon, spf = _draw(seed + 10, 600, 1800)
        assert compute_mincllr(bon, spf) == pytest.approx(reference(bon, spf), abs=1e-6)


def test_empty_class_returns_nan():
    bon, spf = _draw(4)
    assert np.isnan(compute_mincllr(np.array([]), spf))
    assert np.isnan(compute_mincllr(bon, np.array([])))
    assert np.isnan(compute_calibration_loss(np.array([]), spf))


def test_identical_scores_are_handled():
    """A degenerate all-equal input must not raise or return a bogus value."""
    bon = np.full(500, 3.0)
    spf = np.full(1500, 3.0)
    value = compute_mincllr(bon, spf)
    assert np.isfinite(value)
    assert value == pytest.approx(1.0, abs=1e-6)


def test_evaluate_system_exposes_the_decomposition():
    """evaluate_system must surface minCLLR and the calibration loss."""
    from benchmarks.compute_mindcf import evaluate_system

    bon, spf = _draw(5)
    metrics = evaluate_system(bon, spf)

    assert "minCLLR" in metrics and "min_cllr" in metrics
    assert "CLLR_minus_minCLLR" in metrics and "cllr_calibration_loss" in metrics
    assert metrics["minCLLR"] == pytest.approx(metrics["min_cllr"])
    assert metrics["CLLR_minus_minCLLR"] == pytest.approx(
        metrics["CLLR"] - metrics["minCLLR"], abs=1e-12
    )
    assert metrics["minCLLR"] <= metrics["CLLR"] + 1e-12
