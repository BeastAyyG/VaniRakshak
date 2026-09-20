"""Measure seed-to-seed spread by metric class, for Ground rule 1.

Ground rule 1 (docs/10) sets ONE noise floor -- 0.770 pp -- for every metric.
It was measured as a seed-to-seed spread on *pooled in-domain* EER. docs/19
already amended it for per-attack comparisons because the same number is
~2.5x too small there. Nobody had measured it out-of-domain until now.

This computes, for every configuration that has >= 2 seeds, the spread
(max - min across seeds) per metric class, and reports what each class
actually needs. It also separates the two-seed estimate from the three-seed
one where both exist, because a spread estimated from two points is itself a
single noisy sample.

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

SEEDS = (42, 43, 44)

# Templates: config -> metric class -> filename with {seed}.
CONFIGS: dict[str, dict[str, str]] = {
    "dose 0.0 (arm B)": {
        "pooled_in_domain": "eer_ci_armB_s{seed}_eval.json",
        "itw_clean": "eer_ci_armB_s{seed}_ITW_clean.json",
        "itw_full_call": "eer_ci_armB_s{seed}_ITW_full_call.json",
    },
    "dose 0.25": {
        "pooled_in_domain": "eer_ci_docs21_pct025_s{seed}_eval.json",
        "itw_full_call": "eer_ci_docs21_pct025_s{seed}_ITW_full_call.json",
    },
    "dose 0.50": {
        "pooled_in_domain": "eer_ci_docs21_pct050_s{seed}_eval.json",
        "itw_full_call": "eer_ci_docs21_pct050_s{seed}_ITW_full_call.json",
    },
    "dose 1.0 (G4 control)": {
        "pooled_in_domain": "eer_ci_armA_control_s{seed}_eval.json",
        "itw_clean": "eer_ci_armA_control_s{seed}_ITW_clean.json",
        "itw_full_call": "eer_ci_armA_control_s{seed}_ITW_full_call.json",
    },
}

# G4 RawBoost's report names are inconsistent -- seed 42's eval came from an
# epoch-7 run, seed 43's from its own -- so they are listed explicitly rather
# than templated. Guessing the pattern here would silently drop data.
EXPLICIT: dict[str, dict[str, dict[int, str]]] = {
    "G4 RawBoost": {
        "pooled_in_domain": {42: "eer_ci_g4_s42_ep7.json",
                             43: "eer_ci_g4_s43.json"},
        "itw_clean": {42: "eer_ci_g4_s42_ITW.json",
                      43: "eer_ci_g4_s43_ITW.json"},
        "itw_full_call": {42: "eer_ci_g4_s42_ITW_full_call.json",
                          43: "eer_ci_g4_s43_ITW_full_call.json"},
    },
}

# Per-attack entries live under rep["attacks"], not at top level.
PER_ATTACK = ("A19", "A20")


def load(name: str) -> dict | None:
    p = REPORTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a bad file must not kill the run
        return None


def eer(rep, key="pooled"):
    if not rep:
        return None
    if key == "pooled":
        e = rep.get("pooled")
    else:
        e = (rep.get("attacks") or {}).get(key)
    return e["eer_pct"] if e else None


def collect() -> tuple[dict, list]:
    """class -> [(label, [values], seeds_used)]"""
    spreads: dict[str, list] = {}
    missing: list[str] = []

    def add(cls, label, vals, seeds):
        if len(vals) < 2:
            missing.append(f"{label} / {cls} (only {len(vals)} seed)")
            return
        spreads.setdefault(cls, []).append((label, vals, seeds))

    sources = [(lbl, {c: {s: t.format(seed=s) for s in SEEDS}
                      for c, t in m.items()})
               for lbl, m in CONFIGS.items()]
    sources += list(EXPLICIT.items())

    for label, metrics in sources:
        for cls, files in metrics.items():
            vals, seeds = [], []
            for seed, fname in sorted(files.items()):
                v = eer(load(fname))
                if v is not None:
                    vals.append(v)
                    seeds.append(seed)
            add(cls, label, vals, seeds)

            # per-attack only exists on eval-slice reports
            if cls == "pooled_in_domain":
                for atk in PER_ATTACK:
                    av, asd = [], []
                    for seed, fname in sorted(files.items()):
                        v = eer(load(fname), atk)
                        if v is not None:
                            av.append(v)
                            asd.append(seed)
                    add("per_attack", f"{label} {atk}", av, asd)
    return spreads, missing


def main() -> int:
    spreads, missing = collect()
    order = ["pooled_in_domain", "per_attack", "itw_clean", "itw_full_call"]

    print("Seed-to-seed spread by metric class (max - min across seeds)")
    print("=" * 70)

    summary = {}
    for cls in order:
        rows = spreads.get(cls)
        if not rows:
            continue
        sp = [max(v) - min(v) for _, v, _ in rows]
        summary[cls] = {"n": len(sp),
                        "mean": round(statistics.mean(sp), 3),
                        "max": round(max(sp), 3)}
        print(f"\n{cls}  (n={len(sp)} configs)")
        for (label, vals, seeds), s in zip(rows, sp):
            print(f"   {label:<30} seeds {seeds}  spread {s:7.3f} pp")
        print(f"   {'mean':<30} {'':10} {statistics.mean(sp):7.3f} pp")
        print(f"   {'max':<30} {'':10} {max(sp):7.3f} pp")

    print("\n" + "=" * 70)
    print("Current Ground rule 1: a single 0.770 pp floor for all of the above")
    print("(docs/19 amendment: per-attack raised to 1.36 pp half-width)")
    print()
    for cls in order:
        if cls in summary:
            s = summary[cls]
            print(f"  {cls:<20} mean {s['mean']:6.3f} pp  "
                  f"max {s['max']:6.3f} pp   "
                  f"{s['mean'] / 0.770:5.2f}x the 0.770 pp floor")

    # Three-seed vs two-seed, where both exist. A spread from two points is
    # one noisy sample; with three we can see how much that sample moves.
    arm_b = [r for r in spreads.get("itw_full_call", []) if "arm B" in r[0]]
    if arm_b:
        vals, seeds = arm_b[0][1], arm_b[0][2]
        print(f"\nThree-seed check — arm B ITW full_call (seeds {seeds}):")
        print(f"   values {[round(v, 3) for v in vals]}")
        print(f"   spread (max-min) {max(vals) - min(vals):.3f} pp")
        if len(vals) >= 3:
            pair = abs(vals[0] - vals[1])
            print(f"   two-seed estimate was {pair:.3f} pp "
                  f"-> three-seed {max(vals) - min(vals):.3f} pp")

    if missing:
        print("\nNot measured:")
        for m in missing:
            print(f"  {m}")

    out = REPORTS / "seed_spread_by_metric_class.json"
    out.write_text(json.dumps({
        "seeds_considered": list(SEEDS),
        "by_class": {c: {"configs": [
            {"label": lbl, "seeds": sd, "values": [round(v, 3) for v in vals],
             "spread_pp": round(max(vals) - min(vals), 3)}
            for lbl, vals, sd in spreads[c]], **summary[c]}
            for c in order if c in summary},
        "current_single_floor_pp": 0.770,
        "missing": missing,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
