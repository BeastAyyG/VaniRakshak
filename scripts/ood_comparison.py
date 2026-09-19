"""Assemble the cross-checkpoint out-of-domain (ITW) comparison table.

Every number is read from an existing reports/eer_ci_*.json produced by
scripts/eer_bootstrap_ci.py (2,000-rep percentile bootstrap). This script
never scores anything -- it is a pure reader/reporter, so it is cheap to
re-run and cannot drift from the underlying measurements.

The registry below is deliberately explicit rather than globbed: the report
filenames carry tags (armA, armB, pre_aug, ...) that do not by themselves say
whether codec augmentation or RawBoost was on. Getting that mapping wrong in
a published table is worse than having to maintain a dict.

Usage:
    python scripts/ood_comparison.py
    python scripts/ood_comparison.py --json reports/ood_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(r"C:\VaniRakshak")
REPORTS = REPO / "reports"

# label -> (codec_augment, rawboost, itw_clean_report, itw_full_call_report)
REGISTRY: list[tuple[str, str, str, str | None, str | None]] = [
    # label                     codec   rb    clean report                              full_call report
    ("pre-aug baseline",        "off",  "off", "eer_ci_pre_aug_ITW_clean.json",         "eer_ci_pre_aug_ITW_full_call.json"),
    ("post-aug (codec on)",     "on",   "off", "eer_ci_post_aug_ITW_clean.json",        "eer_ci_post_aug_ITW_full_call.json"),
    ("attentive merge",         "on",   "off", "eer_ci_attm_ITW_clean.json",            "eer_ci_attm_ITW_full_call.json"),
    ("G4 RawBoost s42",         "on",   "on",  "eer_ci_g4_s42_ITW.json",                "eer_ci_g4_s42_ITW_full_call.json"),
    ("G4 RawBoost s43",         "on",   "on",  "eer_ci_g4_s43_ITW.json",                "eer_ci_g4_s43_ITW_full_call.json"),
    ("arm A s42 (codec on)",    "on",   "off", "eer_ci_armA_control_s42_ITW_clean.json","eer_ci_armA_control_s42_ITW_full_call.json"),
    ("arm A s43 (codec on)",    "on",   "off", "eer_ci_armA_control_s43_ITW_clean.json","eer_ci_armA_control_s43_ITW_full_call.json"),
    ("arm B s42 (codec off)",   "off",  "off", "eer_ci_armB_s42_ITW_clean.json",        "eer_ci_armB_s42_ITW_full_call.json"),
    ("arm B s43 (codec off)",   "off",  "off", "eer_ci_armB_s43_ITW_clean.json",        "eer_ci_armB_s43_ITW_full_call.json"),
]

PRIMARY = "full_call"   # docs/20 writes its primary criterion on full_call


def load(name: str | None):
    if name is None:
        return None
    p = REPORTS / name
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    e = d.get("pooled")
    if not e:
        return None
    return {
        "report": name,
        "eer_pct": e["eer_pct"],
        "lo": e["ci95_pct"][0],
        "hi": e["ci95_pct"][1],
        "n_bonafide": d.get("n_bonafide"),
        "n_spoof": e.get("n_spoof"),
    }


def disjoint(a, b) -> str:
    """'disjoint' if the two 95% intervals do not overlap, else 'overlap'."""
    if not a or not b:
        return "n/a"
    return "disjoint" if (a["hi"] < b["lo"] or b["hi"] < a["lo"]) else "overlap"


def fmt(v) -> str:
    if v is None:
        return "     --   "
    return f"{v['eer_pct']:6.3f} [{v['lo']:5.2f},{v['hi']:5.2f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write a JSON dump here")
    args = ap.parse_args()

    rows = []
    for label, codec, rb, c_name, f_name in REGISTRY:
        rows.append({
            "label": label,
            "codec_augment": codec,
            "rawboost": rb,
            "clean": load(c_name),
            "full_call": load(f_name),
        })

    measured = [r for r in rows if r[PRIMARY] is not None]
    missing = [r["label"] for r in rows if r[PRIMARY] is None]

    print("\n=== Out-of-domain (In-the-Wild) EER %, 95% bootstrap CI ===")
    print("lower is better; n = 19,963 bonafide / 11,816 spoof\n")
    hdr = (f"{'checkpoint':<22} {'codec':<6} {'rb':<4} "
           f"{'clean':>26} {'full_call (primary)':>26}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(measured, key=lambda x: x[PRIMARY]["eer_pct"]):
        print(f"{r['label']:<22} {r['codec_augment']:<6} {r['rawboost']:<4} "
              f"{fmt(r['clean']):>26} {fmt(r['full_call']):>26}")
    for r in rows:
        if r[PRIMARY] is None:
            print(f"{r['label']:<22} {r['codec_augment']:<6} {r['rawboost']:<4} "
                  f"{fmt(r['clean']):>26} {'     NOT MEASURED':>26}")

    # ---- headline: codec on vs off, matched seeds ----------------------
    print("\n--- matched arm A (codec ON) vs arm B (codec OFF) ---")
    pairs = []
    for seed in (42, 43):
        a = next((r for r in rows if r["label"].startswith(f"arm A s{seed}")), None)
        b = next((r for r in rows if r["label"].startswith(f"arm B s{seed}")), None)
        if not a or not b:
            continue
        for cond in ("clean", "full_call"):
            av, bv = a[cond], b[cond]
            if not av or not bv:
                continue
            gap = av["eer_pct"] - bv["eer_pct"]
            pairs.append({
                "seed": seed, "condition": cond,
                "arm_a_eer": av["eer_pct"], "arm_b_eer": bv["eer_pct"],
                "gap_pp": round(gap, 3),
                "intervals": disjoint(av, bv),
                "arm_b_better": bv["eer_pct"] < av["eer_pct"],
            })
            print(f"  seed {seed} {cond:<10} A {av['eer_pct']:6.3f} -> "
                  f"B {bv['eer_pct']:6.3f}   {gap:+7.3f} pp  "
                  f"{disjoint(av, bv)}")

    payload = {
        "generated_from": "reports/eer_ci_*.json (2000-rep bootstrap)",
        "primary_condition": PRIMARY,
        "rows": rows,
        "matched_pairs": pairs,
        "not_measured": missing,
    }
    if args.json:
        out = REPO / args.json if not Path(args.json).is_absolute() else Path(args.json)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    if missing:
        print(f"\nNOT MEASURED on {PRIMARY}: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
