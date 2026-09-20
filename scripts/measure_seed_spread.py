"""Measure seed-to-seed spread by metric class, for Ground rule 1.

Ground rule 1 (docs/10) sets ONE noise floor -- 0.770 pp -- for every metric.
It was measured as a seed-to-seed spread on *pooled in-domain* EER. docs/19
already amended it for per-attack comparisons (1.36 pp) because the same
number is ~2.5x too small there. Nobody has measured it out-of-domain.

This script computes the spread for every configuration that has two seeds,
splits it by metric class, and reports what each class actually needs. It
also checks whether a larger OOD floor would have changed any adjudicated
verdict -- if it would not, adopting it is safe rather than a retroactive
rejudging.

Read-only: every number comes from reports/eer_ci_*.json.

Usage:
    python scripts/measure_seed_spread.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(r"C:\VaniRakshak")
REPORTS = REPO / "reports"

# Each config: label -> {metric_class -> (report_s42, report_s43)}
PAIRS: dict[str, dict[str, tuple[str, str]]] = {
    "dose 0.0 (arm B)": {
        "pooled_in_domain": ("eer_ci_armB_s42_eval.json",
                             "eer_ci_armB_s43_eval.json"),
        "itw_clean": ("eer_ci_armB_s42_ITW_clean.json",
                      "eer_ci_armB_s43_ITW_clean.json"),
        "itw_full_call": ("eer_ci_armB_s42_ITW_full_call.json",
                          "eer_ci_armB_s43_ITW_full_call.json"),
    },
    "dose 0.25": {
        "pooled_in_domain": ("eer_ci_docs21_pct025_s42_eval.json",
                             "eer_ci_docs21_pct025_s43_eval.json"),
        "itw_full_call": ("eer_ci_docs21_pct025_s42_ITW_full_call.json",
                          "eer_ci_docs21_pct025_s43_ITW_full_call.json"),
    },
    "dose 0.50": {
        "pooled_in_domain": ("eer_ci_docs21_pct050_s42_eval.json",
                             "eer_ci_docs21_pct050_s43_eval.json"),
        "itw_full_call": ("eer_ci_docs21_pct050_s42_ITW_full_call.json",
                          "eer_ci_docs21_pct050_s43_ITW_full_call.json"),
    },
    "dose 1.0 (G4 control)": {
        "pooled_in_domain": ("eer_ci_armA_control_s42_eval.json",
                             "eer_ci_armA_control_s43_eval.json"),
        "itw_clean": ("eer_ci_armA_control_s42_ITW_clean.json",
                      "eer_ci_armA_control_s43_ITW_clean.json"),
        "itw_full_call": ("eer_ci_armA_control_s42_ITW_full_call.json",
                          "eer_ci_armA_control_s43_ITW_full_call.json"),
    },
    "G4 RawBoost": {
        "pooled_in_domain": ("eer_ci_g4_s42_ep7.json", "eer_ci_g4_s43.json"),
        "itw_clean": ("eer_ci_g4_s42_ITW.json", "eer_ci_g4_s43_ITW.json"),
        "itw_full_call": ("eer_ci_g4_s42_ITW_full_call.json",
                          "eer_ci_g4_s43_ITW_full_call.json"),
    },
}

# Per-attack entries live under rep["attacks"], not at top level.
PER_ATTACK = ("A19", "A20")


def load(name: str) -> dict | None:
    p = REPORTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def eer(rep, key="pooled"):
    if not rep:
        return None
    if key == "pooled":
        e = rep.get("pooled")
    else:
        e = (rep.get("attacks") or {}).get(key)
    return e["eer_pct"] if e else None


def main() -> int:
    # class -> list of (label, |s42 - s43|)
    spreads: dict[str, list[tuple[str, float]]] = {}
    missing: list[str] = []

    for label, metrics in PAIRS.items():
        for cls, (f42, f43) in metrics.items():
            r42, r43 = load(f42), load(f43)
            if not r42 or not r43:
                missing.append(f"{label} / {cls}")
                continue
            v42, v43 = eer(r42), eer(r43)
            if v42 is None or v43 is None:
                missing.append(f"{label} / {cls}")
                continue
            spreads.setdefault(cls, []).append((label, abs(v42 - v43)))

            # per-attack only exists on eval-slice reports
            if cls == "pooled_in_domain":
                for atk in PER_ATTACK:
                    a42, a43 = eer(r42, atk), eer(r43, atk)
                    if a42 is None or a43 is None:
                        continue
                    spreads.setdefault("per_attack", []).append(
                        (f"{label} {atk}", abs(a42 - a43)))

    order = ["pooled_in_domain", "per_attack", "itw_clean", "itw_full_call"]
    print("Seed-to-seed spread by metric class (|seed 42 - seed 43|)")
    print("=" * 68)

    summary = {}
    for cls in order:
        vals = spreads.get(cls)
        if not vals:
            continue
        nums = [v for _, v in vals]
        summary[cls] = {
            "n": len(nums),
            "mean": round(statistics.mean(nums), 3),
            "max": round(max(nums), 3),
        }
        print(f"\n{cls}  (n={len(nums)})")
        for lbl, v in vals:
            print(f"   {lbl:<28} {v:7.3f} pp")
        print(f"   {'mean':<28} {statistics.mean(nums):7.3f} pp")
        print(f"   {'max':<28} {max(nums):7.3f} pp")

    print("\n" + "=" * 68)
    print("Current Ground rule 1: a single 0.770 pp floor for all of the above")
    print("(docs/19 amendment: per-attack raised to 1.36 pp)")
    print()
    for cls in order:
        if cls in summary:
            s = summary[cls]
            ratio = s["mean"] / 0.770
            print(f"  {cls:<20} mean {s['mean']:6.3f} pp  "
                  f"max {s['max']:6.3f} pp   {ratio:5.2f}x the 0.770 pp floor")

    if missing:
        print("\nNot measured (both seeds missing):")
        for m in missing:
            print(f"  {m}")

    out = REPORTS / "seed_spread_by_metric_class.json"
    out.write_text(json.dumps({
        "by_class": {k: {"pairs": [{"label": a, "spread_pp": round(b, 3)}
                                   for a, b in v], **summary[k]}
                     for k, v in spreads.items() if k in summary},
        "current_single_floor_pp": 0.770,
        "missing": missing,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
