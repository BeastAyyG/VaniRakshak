#!/usr/bin/env python
"""Mechanically adjudicate run 1 against the pre-registered bars in docs/10.

Written 2026-09-15 **before** the full-partition report existed, so the verdict is
a pure function of the report and cannot be read post-hoc. The bars are copied
from docs/10; nothing here is invented.

Bars, quoted from docs/10 (line numbers as of 2026-09-15):

  G3            pooled eval EER < 11.0 %, minDCF < 0.75          (L382-385)
  criterion 1   minDCF < 0.75 on the full eval partition, on at
                least 3 of 5 seeds                               (L768)
  criterion 2   pooled eval EER < 12.666 % on all 5 seeds       (L770)
                ...and the second, "justified" reading of the same
                criterion is < 11.866 %                          (L783-785)

  Seeds in the claim: S42..S46. B42/B43 are the *no-augmentation ablation* and
  are explicitly NOT part of the 5-seed claim -- they are printed for context.

Usage:
    python scripts/check_run1_gate.py \
        --full  reports/cm_from_scratch_full_partition_2026-09-15.json \
        --slice reports/cm_from_scratch_2026-09-14.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- pre-registered bars (docs/10) -- do not edit without a dated note ---------
G3_EER = 11.0
G3_MINDCF = 0.75
CRIT2_STATED = 12.666
CRIT2_JUSTIFIED = 11.866
CLAIM_SEEDS = ("S42", "S43", "S44", "S45", "S46")
ABLATION = ("B42", "B43")
MIN_SEEDS_FOR_CRIT1 = 3

EPS = 1e-9  # a bar is "cleared" only if strictly below


def load_models(path: Path) -> dict:
    """Return {label: metrics-dict} from a score report."""
    if not path.exists():
        raise SystemExit(f"report not found: {path}")
    data = json.loads(path.read_text())
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise SystemExit(f"{path}: no 'models' block -- is this a score report?")
    out = {}
    for label, block in models.items():
        metrics = block.get("metrics", block)
        if "eer" not in metrics or "min_dcf" not in metrics:
            raise SystemExit(f"{path}: model {label} lacks eer/min_dcf")
        out[label] = metrics
    return out


def pct(x: float) -> float:
    return x * 100.0


def cleared(value: float, bar: float) -> bool:
    return value < bar - EPS


def check_label(metrics: dict) -> dict:
    """Apply every per-model bar. Bars are on percentages, metrics are fractions."""
    eer, mindcf = pct(metrics["eer"]), metrics["min_dcf"]
    return {
        "eer": eer,
        "mindcf": mindcf,
        "g3_eer": cleared(eer, G3_EER),
        "g3_mindcf": cleared(mindcf, G3_MINDCF),
        "crit2_stated": cleared(eer, CRIT2_STATED),
        "crit2_justified": cleared(eer, CRIT2_JUSTIFIED),
    }


def fmt(value: float, bar: float) -> str:
    mark = "PASS" if cleared(value, bar) else "FAIL"
    return f"{value:8.4f} vs <{bar:<7.3f} {mark}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", type=Path, required=True,
                    help="full-partition report (run 1 output)")
    ap.add_argument("--slice", type=Path, default=None,
                    help="the 20k-slice report to diff against (optional)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    full = load_models(args.full)
    slice_ = load_models(args.slice) if args.slice else {}

    print("=" * 78)
    print("RUN 1 GATE CHECK -- pre-registered bars from docs/10")
    print(f"  full : {args.full}")
    if args.slice:
        print(f"  slice: {args.slice}")
    print("=" * 78)

    rows = []
    for label in sorted(full):
        res = check_label(full[label])
        row = {"label": label, "role": "claim" if label in CLAIM_SEEDS else "ablation"}
        row.update(res)
        if label in slice_:
            s = check_label(slice_[label])
            row["slice_eer"], row["slice_mindcf"] = s["eer"], s["mindcf"]
            row["d_eer"] = res["eer"] - s["eer"]
            row["d_mindcf"] = res["mindcf"] - s["mindcf"]
        rows.append(row)

    print()
    print(f"{'label':6s} {'role':9s} {'EER %':>9s} {'minDCF':>9s}"
          + (f" {'slice EER':>10s} {'slice mD':>9s} {'dEER':>8s} {'dmD':>8s}" if slice_ else ""))
    print("-" * 78)
    for r in rows:
        line = f"{r['label']:6s} {r['role']:9s} {r['eer']:9.4f} {r['mindcf']:9.4f}"
        if slice_:
            if "slice_eer" in r:
                line += (f" {r['slice_eer']:10.4f} {r['slice_mindcf']:9.4f}"
                         f" {r['d_eer']:+8.4f} {r['d_mindcf']:+8.4f}")
            else:
                line += " " * 40
        print(line)

    print()
    print("Per-model bars")
    print("-" * 78)
    for r in rows:
        print(f"{r['label']}:")
        print(f"    G3          EER     {fmt(r['eer'], G3_EER)}")
        print(f"    G3          minDCF  {fmt(r['mindcf'], G3_MINDCF)}")
        print(f"    crit2 EER   stated  {fmt(r['eer'], CRIT2_STATED)}")
        print(f"    crit2 EER   justified {fmt(r['eer'], CRIT2_JUSTIFIED)}")

    # --- claim-level adjudication ---------------------------------------------
    claim = [r for r in rows if r["role"] == "claim"]
    n_min_dcf = sum(1 for r in claim if r["g3_mindcf"])
    n_eer_stated = sum(1 for r in claim if r["crit2_stated"])
    n_eer_just = sum(1 for r in claim if r["crit2_justified"])
    n_claim = len(claim)

    print()
    print("Claim-level (docs/10 'The headline claim', criteria 1 and 2)")
    print("-" * 78)
    print(f"  seeds present           : {n_claim} of {len(CLAIM_SEEDS)} "
          f"({', '.join(r['label'] for r in claim) or 'none'})")
    print(f"  criterion 1 minDCF<0.75 : {n_min_dcf} of {n_claim} clear "
          f"(needs >= {MIN_SEEDS_FOR_CRIT1} of {len(CLAIM_SEEDS)})")
    print(f"  criterion 2 EER<12.666  : {n_eer_stated} of {n_claim} clear "
          f"(needs all {len(CLAIM_SEEDS)})")
    print(f"  criterion 2 EER<11.866  : {n_eer_just} of {n_claim} clear "
          f"(the 'justified' reading)")

    decidable = n_claim >= len(CLAIM_SEEDS)
    if not decidable:
        print()
        print(f"  VERDICT: UNDECIDED -- only {n_claim} of {len(CLAIM_SEEDS)} seeds exist.")
        print("           Criteria 1 and 2 are defined over all 5 seeds; S44-S46 are")
        print("           not run yet. Do not report a headline pass or fail.")
    else:
        c1 = n_min_dcf >= MIN_SEEDS_FOR_CRIT1
        c2s = n_eer_stated == len(CLAIM_SEEDS)
        c2j = n_eer_just == len(CLAIM_SEEDS)
        print()
        print(f"  criterion 1: {'PASS' if c1 else 'FAIL'}")
        print(f"  criterion 2 (stated)   : {'PASS' if c2s else 'FAIL'}")
        print(f"  criterion 2 (justified): {'PASS' if c2j else 'FAIL'}")
        print("  NOTE: criteria 3 (warm-start control) and 4 (spread reported) are")
        print("        separate and are NOT adjudicated by this script.")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"rows": rows,
             "claim": {"n_seeds": n_claim, "n_min_dcf": n_min_dcf,
                       "n_eer_stated": n_eer_stated, "n_eer_justified": n_eer_just,
                       "decidable": decidable}}, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")

    # exit nonzero only if a *decidable* bar fails
    failed = any(
        (not r["g3_mindcf"] and r["label"] in CLAIM_SEEDS) or
        (not r["crit2_stated"] and r["label"] in CLAIM_SEEDS)
        for r in rows
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
