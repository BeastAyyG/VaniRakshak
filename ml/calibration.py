"""Platt calibration helpers for training checkpoints (Goal 2).

``ml/train_cm.py`` now writes a ``calibration`` block into every best
checkpoint it saves, in the same shape ``scripts/run_g2_calibration.py``
writes: ``method``, ``scale_w``, ``bias_b``, plus the threshold and the
reference to the data the fit came from. ``ml/evaluate_cm.py`` already
reads ``scale_w`` / ``bias_b`` from that block with ``--apply-calibration``.

The fit here is a small IRLS logistic regression on the dev LLRs, numpy
only, so it can be unit tested without torch or sklearn. Scores are
standardised internally before fitting, because raw LLRs can be large
(the deployed threshold is about 21.9), and the fitted parameters are
converted back to the raw-score scale.

Caveat, recorded in the block itself: the dev set is also what selects
the best checkpoint, so a calibration fitted on it is optimistic. A
held-out dev fit (the Goal 1 procedure in
``~/workspace/vanirakshak-goal1-prep/calibrate_goal1.py``) is still the
right way to fit production calibration. This block exists so a training
checkpoint always says what its score scale is and what it was fitted
on, instead of shipping with no calibration record at all.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def fit_platt(
    bonafide_scores, spoof_scores, max_iter: int = 100, tol: float = 1e-8
) -> tuple[float, float]:
    """Fit Platt scaling ``calibrated = w * raw + b`` on dev LLRs.

    Returns ``(scale_w, bias_b)`` with ``w > 0`` for a sane fit. Raises
    ``ValueError`` on empty input, a single class, or constant scores.
    """
    bon = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spf = np.asarray(spoof_scores, dtype=np.float64).ravel()
    if bon.size == 0 or spf.size == 0:
        raise ValueError("Platt fit needs both bonafide and spoof scores")
    scores = np.concatenate([bon, spf])
    if not np.isfinite(scores).all():
        raise ValueError("Platt fit needs finite scores")
    if float(np.std(scores)) < 1e-12:
        raise ValueError("Platt fit needs scores that vary")
    y = np.concatenate([np.ones(bon.size), np.zeros(spf.size)])

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    z = (scores - mean) / std

    w, b = 0.0, 0.0
    for _ in range(max_iter):
        eta = w * z + b
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        g = np.array([np.sum((p - y) * z), np.sum(p - y)])
        hess = np.array(
            [
                [np.sum(p * (1.0 - p) * z * z) + 1e-6, np.sum(p * (1.0 - p) * z)],
                [np.sum(p * (1.0 - p) * z), np.sum(p * (1.0 - p)) + 1e-6],
            ]
        )
        step = np.linalg.solve(hess, g)
        w -= float(step[0])
        b -= float(step[1])
        if float(np.max(np.abs(step))) < tol:
            break

    scale_w = w / std
    bias_b = b - w * mean / std
    if not (np.isfinite(scale_w) and np.isfinite(bias_b)):
        raise ValueError("Platt fit diverged to non-finite parameters")
    if scale_w <= 0:
        raise ValueError(
            f"Platt fit gave non-positive scale_w={scale_w:.6g}; the dev "
            "scores do not separate bonafide from spoof in the expected direction"
        )
    return float(scale_w), float(bias_b)


def compute_cllr(bonafide_scores, spoof_scores) -> float:
    """Log-likelihood-ratio cost on (optionally calibrated) LLR scores."""
    bon = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spf = np.asarray(spoof_scores, dtype=np.float64).ravel()
    if bon.size == 0 or spf.size == 0:
        raise ValueError("CLLR needs both bonafide and spoof scores")
    bon_term = float(np.mean(np.logaddexp(0.0, -bon) / np.log(2.0)))
    spf_term = float(np.mean(np.logaddexp(0.0, spf) / np.log(2.0)))
    return 0.5 * (bon_term + spf_term)


def calibration_block(
    bonafide_scores,
    spoof_scores,
    threshold: float,
    data_reference: str,
    method: str = "platt",
) -> dict:
    """Build the ``calibration`` block for a training checkpoint.

    On a fit failure the block is still written, with
    ``method: "none"`` and the reason, so a checkpoint never silently
    claims a calibration it does not have.
    """
    bon = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spf = np.asarray(spoof_scores, dtype=np.float64).ravel()
    base = {
        "threshold": float(threshold),
        "data_reference": str(data_reference),
        "fitted_on": "dev",
        "n_bonafide": int(bon.size),
        "n_spoof": int(spf.size),
        "note": (
            "Dev was also used for best-checkpoint selection, so this "
            "calibration is optimistic. Re-fit on held-out dev (Goal 1 "
            "procedure) before using it for production decisions."
        ),
    }
    try:
        scale_w, bias_b = fit_platt(bon, spf)
        cal_bon = scale_w * bon + bias_b
        cal_spf = scale_w * spf + bias_b
        return {
            "method": method,
            "scale_w": round(scale_w, 6),
            "bias_b": round(bias_b, 6),
            "calibrated_cllr": round(compute_cllr(cal_bon, cal_spf), 4),
            **base,
        }
    except ValueError as e:
        return {"method": "none", "reason": str(e), **base}


if __name__ == "__main__":
    print("Running ml/calibration.py self-tests...")
    rng = np.random.default_rng(0)
    bon = rng.normal(5.0, 1.0, 500)
    spf = rng.normal(-5.0, 1.0, 500)
    w, b = fit_platt(bon, spf)
    assert w > 0 and np.isfinite(w) and np.isfinite(b)
    print(f"  [OK] fit_platt (scale_w={w:.4f}, bias_b={b:.4f})")

    block = calibration_block(bon, spf, threshold=0.0, data_reference="dev.tsv")
    assert block["method"] == "platt" and "scale_w" in block and "bias_b" in block
    assert block["data_reference"] == "dev.tsv" and block["n_bonafide"] == 500
    assert compute_cllr(w * bon + b, w * spf + b) < compute_cllr(bon, spf)
    print("  [OK] calibration_block (keys, data reference, CLLR improves)")

    block = calibration_block([1.0, 1.0], [1.0, 1.0], 0.0, "dev.tsv")
    assert block["method"] == "none" and "reason" in block
    print("  [OK] calibration_block records a failed fit instead of crashing")

    print("All ml/calibration.py self-tests PASSED successfully!")
