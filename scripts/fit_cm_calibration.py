"""Fit a Platt calibration block for ANY countermeasure checkpoint.

docs/10 G2, work items 0-4. Generalises scripts/run_g2_calibration.py, which
is hardcoded to the two attentive-merge checkpoints and to merge_layers=12.
ml/train_cm.py fits no calibration, so every from-scratch / G4 checkpoint
ships with `calibration: None` and its CLLR numbers are meaningless.

Constraints taken from docs/10 G2 and honoured here:

0. Use the schema already in data/cm_aasist.pt verbatim -- "not inventing a
   new mechanism".
1. Fit on DEV scores only. Never eval: that is leakage and invalidates every
   number it touches.
2. Store the two parameters in the checkpoint alongside existing metadata.
3. DO NOT apply this in score(). This is settled, not open
   (vanirakshak/detectors/cm_aasist.py:34-47). score() already returns
   raw_llr - eer_threshold + natural_offset, which is itself a shift
   calibration, so a Platt fit applied there would STACK two calibrations and
   compress score variance, pushing genuine clips into false CHALLENGE/BLOCK.
   This script is eval/reporting path only and touches no runtime behaviour.
4. Report raw and calibrated side by side.

Usage:
    python scripts/fit_cm_calibration.py --checkpoint data/cm_g4_rawboost_s42.pt
    # live checkpoint? write to a copy so a training run cannot clobber it:
    python scripts/fit_cm_calibration.py --checkpoint X.pt --output X_cal.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.compute_mindcf import (  # noqa: E402
    TAU_BAYES,
    compute_act_dcf,
    compute_cllr,
    compute_eer,
    compute_mindcf,
)
from cm_frontend import load_frontend_for  # noqa: E402
from eval_call_channel import build_channel, load_manifest, score_all  # noqa: E402
from ml.aasist_model import AASISTLHead  # noqa: E402

DEV_MANIFEST = r"A:\cm-corpus\manifests\dev.tsv"
EVAL_MANIFEST = r"data\eval_benchmark\eval_slice_20k.tsv"


def fit_platt_dev(bon_dev: np.ndarray, spf_dev: np.ndarray):
    """Two-parameter logistic (Platt) fit on DEV scores. Never called on eval.

    Unregularised 2-parameter logistic regression, identical in substance to
    the `LogisticRegression(C=1e5, solver="lbfgs")` that
    scripts/run_g2_calibration.py used -- C=1e5 is effectively no penalty, so
    the fitted optimum is the same one.

    Implemented with scipy rather than sklearn deliberately: scipy is a
    declared dependency (requirements.txt) and is installed, whereas sklearn
    is declared nowhere and is absent from this venv -- run_g2_calibration.py
    cannot run here at all. Adding an undeclared dependency to fit a
    two-parameter model would be the wrong trade.
    """
    s = np.concatenate([bon_dev, spf_dev]).astype(np.float64)
    y = np.concatenate([np.ones_like(bon_dev), np.zeros_like(spf_dev)]).astype(np.float64)

    def nll(theta):
        z = theta[0] * s + theta[1]
        p = np.clip(expit(z), 1e-12, 1.0 - 1e-12)
        loss = -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        gw = float(np.mean((p - y) * s))
        gb = float(np.mean(p - y))
        return loss, np.array([gw, gb])

    n_pos, n_neg = float(y.sum()), float((1.0 - y).sum())
    x0 = np.array([0.0, np.log((n_pos + 1.0) / (n_neg + 1.0))])
    res = minimize(nll, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-10})
    if not res.success:
        raise RuntimeError(f"Platt fit failed to converge: {res.message}")
    return float(res.x[0]), float(res.x[1])


def report_pair(bon, spf, w=1.0, b=0.0):
    """Raw vs calibrated metrics for one score set."""
    cb, cs = bon * w + b, spf * w + b
    eer, _ = compute_eer(bon, spf)
    mindcf, _ = compute_mindcf(bon, spf)
    return {
        "raw_cllr": round(float(compute_cllr(bon, spf)), 4),
        "calibrated_cllr": round(float(compute_cllr(cb, cs)), 4),
        "raw_actdcf_bayes": round(float(compute_act_dcf(bon, spf, threshold=TAU_BAYES)), 4),
        "calibrated_actdcf_bayes": round(float(compute_act_dcf(cb, cs, threshold=TAU_BAYES)), 4),
        "eer_pct": round(float(eer) * 100, 3),
        "minDCF": round(float(mindcf), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default=None,
                    help="write the calibrated checkpoint here instead of in place "
                         "(use when training may still overwrite the source)")
    ap.add_argument("--dev-manifest", default=DEV_MANIFEST)
    ap.add_argument("--eval-manifest", default=EVAL_MANIFEST)
    ap.add_argument("--dev-limit", type=int, default=20000)
    ap.add_argument("--eval-limit", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    merge_layers = ck.get("merge_layers")
    layer_index = int(ck.get("layer_index", 5))
    freeze_merge = bool(ck.get("freeze_merge", False))
    print(f"checkpoint : {ckpt_path.name}")
    print(f"  merge_layers={merge_layers} layer_index={layer_index} seed={ck.get('seed')}")
    print(f"  existing calibration: {ck.get('calibration')}")

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    wavlm, dtype = load_frontend_for(ck, device)
    model = AASISTLHead(in_dim=768, merge_layers=merge_layers, freeze_merge=freeze_merge)
    model.load_state_dict(ck["state_dict"], strict=False)
    model.to(device).eval()

    # ASVspoof 5 is already telephone-captured: 'clean' IS the channel.
    channel = build_channel("clean")

    print(f"\n[1/4] scoring DEV ({args.dev_limit} rows) -- fit set, never eval")
    _, dev_rows = load_manifest(Path(args.dev_manifest), args.dev_limit)
    bon_d, spf_d, _ = score_all(model, wavlm, dev_rows, channel, device,
                                merge_layers, layer_index, args.batch, dtype)
    print(f"      dev bonafide={bon_d.size} spoof={spf_d.size}")

    print("[2/4] fitting Platt on DEV only")
    w, b = fit_platt_dev(bon_d, spf_d)
    print(f"      scale_w={w:.6f}  bias_b={b:.6f}")
    dev = report_pair(bon_d, spf_d, w, b)

    print(f"[3/4] scoring EVAL ({args.eval_limit} rows) -- report only, no fitting")
    _, ev_rows = load_manifest(Path(args.eval_manifest), args.eval_limit)
    bon_e, spf_e, _ = score_all(model, wavlm, ev_rows, channel, device,
                                merge_layers, layer_index, args.batch, dtype)
    ev = report_pair(bon_e, spf_e, w, b)
    ev_uncal = report_pair(bon_e, spf_e)

    block = {
        "method": "platt",
        "scale_w": round(w, 6),
        "bias_b": round(b, 6),
        "calibrated_cllr": dev["calibrated_cllr"],
        "calibrated_actdcf_bayes": dev["calibrated_actdcf_bayes"],
    }

    print("\n[4/4] results")
    print(f"  DEV   raw CLLR {dev['raw_cllr']:.4f} -> calibrated {dev['calibrated_cllr']:.4f}")
    print(f"  DEV   raw actDCF {dev['raw_actdcf_bayes']:.4f} -> {dev['calibrated_actdcf_bayes']:.4f}")
    print(f"  EVAL  raw CLLR {ev_uncal['raw_cllr']:.4f} -> calibrated {ev['calibrated_cllr']:.4f}")
    print(f"  EVAL  raw actDCF {ev_uncal['raw_actdcf_bayes']:.4f} -> {ev['calibrated_actdcf_bayes']:.4f}")
    print(f"  EVAL  EER {ev['eer_pct']:.3f}%  minDCF {ev['minDCF']:.4f}  (unchanged by design)")

    verdict = "PASS" if ev["calibrated_cllr"] < 1.0 else "FAIL"
    print(f"\n  G2 criterion (calibrated CLLR < 1.0 on eval): {verdict}")

    if not args.no_write:
        ck["calibration"] = block
        # Do NOT clobber dev_eer / dev_min_dcf. Those were measured by
        # ml/train_cm.py on the FULL dev split (140,950 records) at the best
        # epoch, and downstream comparisons read them. The numbers below come
        # from this script's own smaller dev subsample (12,000 records), so
        # they are a different measurement of a different sample -- writing
        # them over the training-time values makes a checkpoint disagree with
        # its own training log (seen: control s42 stored 16.297% vs logged
        # 15.856%). Keep both, under distinct keys.
        if "dev_eer" in ck:
            ck.setdefault("dev_eer_full_dev", ck["dev_eer"])
            ck.setdefault("dev_min_dcf_full_dev", ck["dev_min_dcf"])
        ck["calibration_dev_eer"] = float(dev["eer_pct"]) / 100.0
        ck["calibration_dev_min_dcf"] = float(dev["minDCF"])
        out = Path(args.output) if args.output else ckpt_path
        torch.save(ck, out)
        print(f"  wrote calibration block -> {out}")
        print("  NOTE: eval/reporting path only. score() is deliberately untouched.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    rep = REPO_ROOT / "reports" / f"calibration_{ckpt_path.stem}_{ts}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({
        "checkpoint": str(ckpt_path),
        "fitted_on": "dev only",
        "n_dev": int(bon_d.size + spf_d.size),
        "n_eval": int(bon_e.size + spf_e.size),
        "calibration_block": block,
        "dev": dev,
        "eval_raw": ev_uncal,
        "eval_calibrated": ev,
        "g2_criterion_cllr_lt_1": verdict,
    }, indent=2), encoding="utf-8")
    print(f"  report -> {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
