#!/usr/bin/env python
"""Adjudicate the whole pre-registered headline claim in docs/10, mechanically.

`scripts/check_run1_gate.py` covers criteria 1 and 2 only, and says so. This
script covers all five, including the two that need the warm-start control:

  1. minDCF < 0.75 on the full eval partition, on >= 3 of 5 seeds   (L768)
  2. pooled eval EER < 12.666 % on all 5 seeds                      (L770)
     ...and the "justified" reading, < 11.866 %                     (L783-785)
  3. the warm-start control is WORSE than the median from-scratch
     seed on minDCF                                                 (L772-775)
  4. seed-to-seed EVAL spread reported; > 0.8 pp updates ground rule 1 (L776)
  5. A19 eval EER < 12 % -- secondary outcome, not the claim        (L778)

Every bar is copied from docs/10. Nothing here is invented, and nothing is
graded leniently: a bar is cleared only if the value is strictly below it.

If criterion 3 fails, the claim dies by its own pre-registration -- the honest
reading is "the seeding fix, not initialisation, was worth the gap" (L799-801).
This script prints that sentence rather than letting anyone miss it.

Usage:
    python scripts/check_claim.py \
        --claim   reports/cm_from_scratch_claim_5seed_2026-09-16.json \
        --control reports/cm_warmstart_control_2026-09-16.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# --- pre-registered bars (docs/10) -- do not edit without a dated note ---------
G3_EER = 11.0
G3_MINDCF = 0.75
CRIT2_STATED = 12.666
CRIT2_JUSTIFIED = 11.866
CRIT5_A19 = 12.0            # percent, G4's own bar
SPREAD_FLOOR = 0.8          # pp, ground rule 1
CLAIM_SEEDS = ("S42", "S43", "S44", "S45", "S46")
MIN_SEEDS_FOR_CRIT1 = 3
EPS = 1e-9


def load_models(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"report not found: {path}")
    data = json.loads(path.read_text())
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise SystemExit(f"{path}: no 'models' block -- is this a score report?")
    return {
        label: {"metrics": b.get("metrics", b),
                "per_attack_eer": b.get("per_attack_eer", {}),
                "init_weights": b.get("init_weights")}
        for label, b in models.items()
    }


def pct(x: float) -> float:
    return x * 100.0


def cleared(value: float, bar: float) -> bool:
    return value < bar - EPS


def yes(b: bool) -> str:
    return "PASS" if b else "FAIL"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--claim", type=Path, required=True,
                    help="the 5-seed from-scratch report")
    ap.add_argument("--control", type=Path, default=None,
                    help="the warm-start control report (criterion 3)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    models = load_models(args.claim)
    claim = {k: v for k, v in models.items() if k in CLAIM_SEEDS}
    eers = {k: pct(v["metrics"]["eer"]) for k, v in claim.items()}
    dcfs = {k: v["metrics"]["min_dcf"] for k, v in claim.items()}

    print("=" * 78)
    print("HEADLINE CLAIM -- full adjudication, bars fixed in docs/10")
    print(f"  claim  : {args.claim}")
    print(f"  control: {args.control or '(none -- criterion 3 not adjudicated)'}")
    print("=" * 78)

    print("\nClaim seeds (full eval partition)")
    print("-" * 78)
    print(f"{'seed':6s} {'EER %':>9s} {'minDCF':>9s} {'A19 %':>8s}  c1      c2s     c2j")
    for s in CLAIM_SEEDS:
        if s not in claim:
            print(f"{s:6s} {'-':>9s} {'-':>9s} {'-':>8s}  (not run)")
            continue
        a19 = claim[s]["per_attack_eer"].get("A19")
        print(f"{s:6s} {eers[s]:9.4f} {dcfs[s]:9.4f} "
              f"{(f'{a19:8.3f}' if a19 is not None else '     -'):>8s}  "
              f"{yes(cleared(dcfs[s], G3_MINDCF)):6s} "
              f"{yes(cleared(eers[s], CRIT2_STATED)):6s} "
              f"{yes(cleared(eers[s], CRIT2_JUSTIFIED)):6s}")

    n = len(claim)
    decidable_12 = n >= len(CLAIM_SEEDS)
    c1 = sum(1 for s in claim if cleared(dcfs[s], G3_MINDCF)) >= MIN_SEEDS_FOR_CRIT1
    c2s = all(cleared(eers[s], CRIT2_STATED) for s in claim) and decidable_12
    c2j = all(cleared(eers[s], CRIT2_JUSTIFIED) for s in claim) and decidable_12

    print()
    print("Criteria")
    print("-" * 78)
    print(f"  1  minDCF < 0.75 on >= 3 of 5 seeds         : "
          f"{sum(1 for s in claim if cleared(dcfs[s], G3_MINDCF))}/{n} clear -> "
          f"{yes(c1) if decidable_12 else 'UNDECIDED'}")
    print(f"  2  EER < 12.666 on all 5 (stated)           : "
          f"{yes(c2s) if decidable_12 else 'UNDECIDED'}")
    print(f"  2  EER < 11.866 on all 5 (justified reading): "
          f"{yes(c2j) if decidable_12 else 'UNDECIDED'}")

    if not decidable_12:
        print(f"     -- only {n} of {len(CLAIM_SEEDS)} seeds exist; criteria 1 and 2 "
              f"are defined over all five.")

    # --- criterion 4: spread ---------------------------------------------------
    spread = (max(eers.values()) - min(eers.values())) if n >= 2 else None
    if spread is None:
        print("  4  seed spread                               : needs >= 2 seeds")
    else:
        print(f"  4  seed-to-seed EVAL EER spread              : {spread:.4f} pp "
              f"({'exceeds' if spread > SPREAD_FLOOR else 'within'} the "
              f"{SPREAD_FLOOR} pp floor)"
              f"{' -> ground rule 1 must be updated' if spread > SPREAD_FLOOR else ''}")

    # --- criterion 5: A19 secondary -------------------------------------------
    a19s = {s: claim[s]["per_attack_eer"].get("A19")
            for s in claim if claim[s]["per_attack_eer"].get("A19") is not None}
    if a19s:
        worst = max(a19s.values())
        print(f"  5  A19 EER < {CRIT5_A19} (secondary)              : "
              f"{yes(worst < CRIT5_A19)} (worst {worst:.3f} %)")
        print(f"     secondary only -- it is not the claim and is not graded as one")
    else:
        print("  5  A19 EER                                   : not in report")

    # --- criterion 3: the warm-start control -----------------------------------
    c3 = None
    control_label = None
    if args.control:
        cm = load_models(args.control)
        # The control is the only model in its report; take it whatever it is
        # called, but prefer a W<label> key if several exist.
        control_label = next((k for k in cm if k.startswith("W")), None) or next(iter(cm), None)
        if control_label is None:
            print("  3  warm-start control                        : report has no model")
        else:
            cdcf = cm[control_label]["metrics"]["min_dcf"]
            ceer = pct(cm[control_label]["metrics"]["eer"])
            median = statistics.median(dcfs.values()) if dcfs else None
            if median is None:
                print("  3  warm-start control                        : no claim seeds to compare")
            else:
                c3 = cdcf > median
                print(f"  3  warm-start control worse than median      : {yes(c3)}")
                print(f"     control {control_label}: minDCF {cdcf:.4f}, EER {ceer:.4f} %")
                print(f"     from-scratch median minDCF: {median:.4f} "
                      f"(spread {min(dcfs.values()):.4f}-{max(dcfs.values()):.4f})")
                print(f"     gap (control - median): {cdcf - median:+.4f}")
                if not c3:
                    print("\n     *** THE CLAIM IS DEAD AS PRE-REGISTERED (docs/10 L799-801). ***")
                    print("     The control matched or beat the from-scratch runs, so the gap is")
                    print("     NOT about initialisation. Honest conclusion: 'the seeding fix,")
                    print("     not initialisation, was worth the gap' -- still a result, per")
                    print("     ground rule 5. Do not report the headline.")
    else:
        print("  3  warm-start control                        : NOT ADJUDICATED "
              "(no --control report)")

    print()
    print("Verdict")
    print("-" * 78)
    if not decidable_12 or c3 is None:
        print("  UNDECIDED -- missing inputs (5 seeds and/or the control).")
        verdict = "undecided"
    else:
        ok = c1 and c2s and c3
        print(f"  criteria 1 and 2 (both readings) and 3: "
              f"{'ALL PASS' if ok else 'AT LEAST ONE FAILS'}")
        if ok and not c2j:
            print("  note: passes under the stated criterion 2 bar but NOT under the")
            print("        'justified' 11.866 % reading -- say which bar you are using.")
        verdict = "pass" if ok else "fail"

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "claim_report": str(args.claim),
            "control_report": str(args.control) if args.control else None,
            "seeds": {s: {"eer_pct": eers.get(s), "min_dcf": dcfs.get(s),
                          "a19_pct": claim[s]["per_attack_eer"].get("A19")}
                      for s in claim},
            "criterion1": None if not decidable_12 else c1,
            "criterion2_stated": None if not decidable_12 else c2s,
            "criterion2_justified": None if not decidable_12 else c2j,
            "criterion3": c3,
            "spread_pp": spread,
            "verdict": verdict,
        }, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")

    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
