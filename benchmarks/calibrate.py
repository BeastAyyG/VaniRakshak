"""BOSARIS/FoCal affine score calibration for the CM countermeasure.

Pre-registered in ``docs/15-calibration-plan.md``. Read that first: it fixes
where the calibration data comes from (a held-out split of eval, never dev),
what the transform family is, and what counts as success.

The scoring convention throughout this module matches
``benchmarks/compute_mindcf.py``: **higher score means more likely bonafide**,
and a trial is accepted when ``score >= threshold``.

This module fits two numbers and applies two numbers. It cannot and does not
change ranking quality: EER and minDCF are threshold-optimised and therefore
invariant under any monotone transform, and this transform is affine. If EER
or minDCF move across :func:`apply_calibration`, the transform is broken and
the result is void — :func:`evaluate` computes that delta explicitly so it is
never left unchecked.
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:  # running from the repo root
    from benchmarks.compute_mindcf import (
        TAU_BAYES,
        compute_act_dcf,
        compute_cllr,
        compute_eer,
        compute_mincllr,
        compute_mindcf,
    )
except ImportError:  # running from inside benchmarks/
    from compute_mindcf import (  # type: ignore[no-redef]
        TAU_BAYES,
        compute_act_dcf,
        compute_cllr,
        compute_eer,
        compute_mincllr,
        compute_mindcf,
    )

SPLIT_RULE = "zlib.crc32(path) % 2 == 0 -> calibration half"

# EER and minDCF are rank-based, so an increasing affine map must leave them
# fixed in exact arithmetic. Floating point can still perturb the last bits, so
# the guard is a tolerance rather than a literal equality.
INVARIANCE_TOL = 1e-9


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------

@dataclass
class Calibration:
    """An affine map ``s' = a*s + b`` taking raw scores onto an LLR scale."""

    a: float
    b: float
    mu: float
    sd: float
    n_cal: int = 0
    n_bon: int = 0
    n_spf: int = 0
    split_rule: str = SPLIT_RULE
    fitted_on: str = ""
    cllr_cal_before: float = float("nan")
    cllr_cal_after: float = float("nan")
    mincllr_cal: float = float("nan")
    notes: List[str] = field(default_factory=list)

    # -- behaviour ---------------------------------------------------------
    def apply(self, scores: Any) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        return self.a * s + self.b

    def is_monotone(self) -> bool:
        return bool(np.isfinite(self.a) and self.a >= 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Calibration":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def identity(cls) -> "Calibration":
        return cls(a=1.0, b=0.0, mu=0.0, sd=1.0, notes=["identity: nothing fitted"])


def _fit_standardised(
    bon: np.ndarray,
    spf: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Tuple[float, float, float, float]:
    """Newton/IRLS fit of the class-weighted logistic objective.

    Returns ``(a, b, mu, sd)`` where the fitted transform is
    ``z = a * ((x - mu) / sd) + b``. Standardising first is what makes the
    2x2 Hessian well conditioned; the caller converts back.
    """
    s_all = np.concatenate([bon, spf])
    n_tar = float(bon.size)
    n_non = float(spf.size)

    # Class weights make the objective exactly the CLLR: each class is
    # normalised by its own count, so an unbalanced partition cannot let the
    # majority class dominate the fit.
    w = np.concatenate([np.full(bon.size, 1.0 / n_tar),
                        np.full(spf.size, 1.0 / n_non)])
    y = np.concatenate([np.ones(bon.size), np.zeros(spf.size)])  # 1 = bonafide

    mu = float(s_all.mean())
    sd = float(s_all.std())
    if not np.isfinite(sd) or sd == 0.0:
        # Every score identical: no ranking exists, so the slope is
        # unidentifiable and only the intercept can be fit.
        s = np.zeros_like(s_all)
    else:
        s = (s_all - mu) / sd

    def _loss(aa: float, bb: float) -> float:
        zz = aa * s + bb
        return float(0.5 * np.sum(w * (np.logaddexp(0.0, zz) - y * zz))
                     / np.log(2.0))

    # The objective is convex, but on near-separable data the sigmoid
    # saturates, the Hessian collapses onto the ridge, and plain Newton
    # overshoots into a runaway that can flip the sign of the slope. So:
    # keep the best iterate ever seen, and cap the slope, where "capped"
    # already means the sigmoid is saturated and further growth is
    # meaningless (loss is ~0 either way).
    A_MAX = 50.0
    best_a, best_b, best_loss = 0.0, 0.0, _loss(0.0, 0.0)
    a = 0.0
    b = 0.0
    ridge = 1e-12
    for _ in range(max_iter):
        z = a * s + b
        sig = np.where(z >= 0,
                       1.0 / (1.0 + np.exp(-np.abs(z))),
                       np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
        r = sig - y
        g = np.array([float(np.sum(w * r * s)), float(np.sum(w * r))])
        hw = w * sig * (1.0 - sig)
        h = np.array([
            [float(np.sum(hw * s * s)) + ridge, float(np.sum(hw * s))],
            [float(np.sum(hw * s)), float(np.sum(hw)) + ridge],
        ])
        try:
            step = np.linalg.solve(h, -g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        a = float(a + step[0])
        b = float(b + step[1])

        hit_cap = abs(a) > A_MAX
        if hit_cap:
            a = A_MAX if a > 0 else -A_MAX
        loss = _loss(a, b)
        if np.isfinite(loss) and loss < best_loss:
            best_a, best_b, best_loss = a, b, loss
        if hit_cap or float(np.max(np.abs(step))) < tol:
            break

    return best_a, best_b, mu, sd


def fit_affine_calibration(
    bonafide_scores: Sequence[float],
    spoof_scores: Sequence[float],
    fitted_on: str = "",
) -> Calibration:
    """Fit ``s' = a*s + b`` in **original score space**.

    The internal fit works on standardised scores, so the returned ``a`` and
    ``b`` are converted back:

        z = a*((x - mu)/sd) + b  ==  (a/sd)*x + (b - a*mu/sd)
    """
    bon = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spf = np.asarray(spoof_scores, dtype=np.float64).ravel()
    if bon.size == 0 or spf.size == 0:
        raise ValueError("need at least one score per class to fit calibration")

    a, b, mu, sd = _fit_standardised(bon, spf)

    if not np.isfinite(sd) or sd == 0.0:
        A, B = 0.0, b
    else:
        A = a / sd
        B = b - a * mu / sd

    notes: List[str] = []
    if not (np.isfinite(A) and np.isfinite(B)):
        notes.append("fit produced non-finite parameters; falling back to identity")
        return Calibration.identity()
    if A < 0.0:
        notes.append(
            "fitted slope is negative: the ranking is inverted relative to the "
            "module's 'higher = bonafide' convention, so the scores were not "
            "produced in this convention"
        )

    cal = Calibration(
        a=float(A),
        b=float(B),
        mu=float(mu),
        sd=float(sd),
        n_cal=int(bon.size + spf.size),
        n_bon=int(bon.size),
        n_spf=int(spf.size),
        fitted_on=fitted_on,
    )
    cal.cllr_cal_before = compute_cllr(bon, spf)
    cal.cllr_cal_after = compute_cllr(cal.apply(bon), cal.apply(spf))
    cal.mincllr_cal = compute_mincllr(bon, spf)
    cal.notes = notes
    return cal


# --------------------------------------------------------------------------
# the split and the TSV
# --------------------------------------------------------------------------

def calibration_mask(paths: Sequence[str]) -> np.ndarray:
    """Deterministic even/odd split. True = calibration half.

    Uses CRC32 of the path so the split is stable across runs, machines and
    checkpoint versions, with no seed that could be lost or misremembered.
    """
    return np.array([zlib.crc32(p.encode("utf-8")) % 2 == 0 for p in paths],
                    dtype=bool)


def load_score_tsv(
    path: Path,
    model: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load a ``--dump-scores`` TSV: ``path, label, attack, <models...>``.

    Returns ``(paths, y, scores, model_labels)`` with ``y == 1`` for bonafide.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = rows[0]
    if len(header) < 4:
        raise ValueError(f"{path}: expected >=4 columns, got {len(header)}")
    model_labels = header[3:]
    if model is None:
        if len(model_labels) != 1:
            raise ValueError(
                f"{path}: {len(model_labels)} model columns, pass --model to choose"
            )
        model = model_labels[0]
    if model not in model_labels:
        raise ValueError(f"{path}: no model column {model!r}; have {model_labels}")
    col = 3 + model_labels.index(model)

    paths, ys, ss = [], [], []
    for row in rows[1:]:
        if len(row) <= col:
            continue
        try:
            val = float(row[col])
        except ValueError:
            continue
        if not np.isfinite(val):
            continue
        paths.append(row[0])
        ys.append(1 if row[1].strip().lower() == "bonafide" else 0)
        ss.append(val)

    return (np.array(paths, dtype=object),
            np.array(ys, dtype=np.int8),
            np.array(ss, dtype=np.float64),
            model_labels)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(
    cal: Calibration,
    bon_report: Sequence[float],
    spf_report: Sequence[float],
    p_tar: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> Dict[str, Any]:
    """Before/after on the held-out report half, at one fixed threshold.

    The threshold is ``TAU_BAYES`` for both columns. That is the point: after
    calibration the Bayes operating point is a *constant*, so a deployment can
    hard-code it instead of re-deriving one per partition.
    """
    bon = np.asarray(bon_report, dtype=np.float64).ravel()
    spf = np.asarray(spf_report, dtype=np.float64).ravel()
    bon_c = cal.apply(bon)
    spf_c = cal.apply(spf)

    eer_b, _ = compute_eer(bon, spf)
    eer_a, _ = compute_eer(bon_c, spf_c)
    mdcf_b, _ = compute_mindcf(bon, spf, p_tar, c_miss, c_fa)
    mdcf_a, _ = compute_mindcf(bon_c, spf_c, p_tar, c_miss, c_fa)

    return {
        "n_report": int(bon.size + spf.size),
        "n_bon": int(bon.size),
        "n_spf": int(spf.size),
        # --- invariance guard -------------------------------------------
        "eer_before": eer_b,
        "eer_after": eer_a,
        "eer_delta": abs(eer_a - eer_b),
        "mindcf_before": mdcf_b,
        "mindcf_after": mdcf_a,
        "mindcf_delta": abs(mdcf_a - mdcf_b),
        "invariance_ok": bool(
            abs(eer_a - eer_b) <= INVARIANCE_TOL
            and abs(mdcf_a - mdcf_b) <= INVARIANCE_TOL
        ),
        # --- the point of the exercise ----------------------------------
        "actdcf_bayes_before": compute_act_dcf(bon, spf, TAU_BAYES, p_tar, c_miss, c_fa),
        "actdcf_bayes_after": compute_act_dcf(bon_c, spf_c, TAU_BAYES, p_tar, c_miss, c_fa),
        "cllr_before": compute_cllr(bon, spf),
        "cllr_after": compute_cllr(bon_c, spf_c),
        "mincllr": compute_mincllr(bon, spf),
        "threshold_used": TAU_BAYES,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", type=Path, required=True,
                    help="TSV from score_checkpoint_eval.py --dump-scores")
    ap.add_argument("--model", default=None,
                    help="model column to calibrate (default: the only one)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the fitted Calibration here as JSON")
    ap.add_argument("--report", type=Path, default=None,
                    help="write the before/after evaluation here as JSON")
    args = ap.parse_args(argv)

    paths, y, scores, labels = load_score_tsv(args.scores, args.model)
    mask = calibration_mask(list(paths))
    bon = scores[(y == 1) & mask]
    spf = scores[(y == 0) & mask]
    bon_r = scores[(y == 1) & ~mask]
    spf_r = scores[(y == 0) & ~mask]

    print(f"loaded {scores.size} scores from {args.scores.name} "
          f"({labels})")
    print(f"split: {int(mask.sum())} calibration / {int((~mask).sum())} report")

    cal = fit_affine_calibration(bon, spf, fitted_on=str(args.scores))
    ev = evaluate(cal, bon_r, spf_r)

    print(f"\nfitted  s' = {cal.a:.6f} * s + {cal.b:.6f}")
    print(f"        monotone (a >= 0): {cal.is_monotone()}")
    for note in cal.notes:
        print(f"        NOTE: {note}")

    if not cal.is_monotone():
        print(
            "\nREFUSING TO WRITE. The fitted slope is negative, which means these "
            "scores are not in this module's 'higher = bonafide' convention. "
            "Applying the fit would invert the ranking. Nothing was written."
        )
        return 1

    print("\ncalibration half")
    print(f"  CLLR      {cal.cllr_cal_before:.4f} -> {cal.cllr_cal_after:.4f}"
          f"   (minCLLR {cal.mincllr_cal:.4f})")

    print("\nreport half (never fitted on)")
    print(f"  EER       {ev['eer_before'] * 100:8.4f}% -> {ev['eer_after'] * 100:8.4f}%"
          f"   delta {ev['eer_delta']:.2e}")
    print(f"  minDCF    {ev['mindcf_before']:8.4f}  -> {ev['mindcf_after']:8.4f}"
          f"     delta {ev['mindcf_delta']:.2e}")
    print(f"  actDCF    {ev['actdcf_bayes_before']:8.4f}  -> "
          f"{ev['actdcf_bayes_after']:8.4f}     at TAU_BAYES={ev['threshold_used']:.4f}")
    print(f"  CLLR      {ev['cllr_before']:8.4f}  -> {ev['cllr_after']:8.4f}"
          f"     (minCLLR {ev['mincllr']:.4f})")
    print(f"\n  invariance guard: {'PASS' if ev['invariance_ok'] else 'FAIL'}"
          f" -- EER and minDCF must not move")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(cal.to_json(), encoding="utf-8")
        print(f"\nwrote calibration -> {args.out}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(ev, indent=2), encoding="utf-8")
        print(f"wrote evaluation  -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
