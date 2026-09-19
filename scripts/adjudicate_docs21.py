"""Adjudicate the docs/21 dose-response sweep against its pre-registered
criteria.

docs/21 asks whether the optimum augmentation strength is INTERIOR (some
augmentation is better than none) or at an endpoint. Criteria are fixed in
docs/21 and are NOT adjustable here. Crucially, "AT_0" -- augmentation is
simply harmful -- is a valid, publishable outcome, not a failed experiment:
docs/20 already points that way and this must be able to confirm it rather
than being built to rescue augmentation.

Decision logic lives in pure functions (`classify_optimum`, `monotonic`) so
it is testable on a CPU. Everything numeric comes from
scripts/eer_bootstrap_ci.py; existing reports are reused unless --force, so
re-running is cheap and resumable.

Usage:
    python scripts/adjudicate_docs21.py
    python scripts/adjudicate_docs21.py --force
    python scripts/adjudicate_docs21.py --no-score   # logic only, from reports
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\VaniRakshak")
PY = REPO / ".venv" / "Scripts" / "python.exe"
REPORTS = REPO / "reports"

EVAL_MANIFEST = r"data\eval_benchmark\eval_slice_20k.tsv"
ITW_MANIFEST = r"A:\cm-corpus\manifests\itw_manifest.tsv"

DOSES = (0.0, 0.25, 0.5, 1.0)
SEEDS = (42, 43)

# Endpoints are trained arms from earlier experiments, reused not retrained.
# Their report names are fixed by those experiments, so map them explicitly.
REPORT_STEM = {
    0.0: "armB_s{seed}",          # docs/20 arm B  (codec OFF)
    1.0: "armA_control_s{seed}",  # docs/10 G4 control (codec ON)
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dose_tag(strength: float) -> str:
    return f"{int(round(strength * 100)):03d}"


def stem_for(strength: float, seed: int) -> str:
    if strength in REPORT_STEM:
        return REPORT_STEM[strength].format(seed=seed)
    return f"docs21_pct{dose_tag(strength)}_s{seed}"


def checkpoint_for(strength: float, seed: int) -> str:
    if strength == 0.0:
        return f"data/cm_docs20_armb_s{seed}.pt"
    if strength == 1.0:
        return f"data/cm_g4_control_s{seed}.pt"
    return f"data/cm_docs21_pct{dose_tag(strength)}_s{seed}.pt"


# --------------------------------------------------------------------------
# pure decision logic -- no I/O, no torch
# --------------------------------------------------------------------------

def beats(a, b) -> bool:
    """True if a is better (lower) than b with non-overlapping 95% CIs.

    `a` / `b` are (point, (lo, hi)) or None. Missing data never counts as a
    win -- otherwise an unscored arm silently wins by default.
    """
    if not a or not b:
        return False
    return a[1][1] < b[1][0]


def classify_optimum(table: dict) -> tuple[str, str]:
    """Decide where the optimum sits.

    `table[dose][seed] = (point, (lo, hi)) | None`, primary condition only.

    Returns (verdict, detail). Verdicts:
      INTERIOR_<tag>  an intermediate dose beats BOTH endpoints, disjoint,
                      on every seed with data;
      AT_0            no dose beats 0.0 -> keep augmentation off;
      AT_1            no dose beats 1.0 -> keep today's setting;
      UNRESOLVED      intervals overlap or data is missing.
    """
    # Check presence explicitly. `all(...)` over an empty list is True, so a
    # table with no endpoints at all would sail past that guard and then
    # KeyError on table[0.0] -- caught by
    # tests/test_adjudicate_docs21.py::test_unresolved_when_endpoints_missing.
    if 0.0 not in table or 1.0 not in table:
        return "UNRESOLVED", "endpoint data missing"

    zero, one = table.get(0.0), table.get(1.0)
    if not zero or not one:
        return "UNRESOLVED", "endpoint data missing"
    seeds = [s for s in SEEDS if zero.get(s) and one.get(s)]
    if not seeds:
        return "UNRESOLVED", "no seed has both endpoints measured"

    for dose in sorted(d for d in table if d not in (0.0, 1.0)):
        row = table[dose]
        if not all(row.get(s) for s in seeds):
            continue
        if all(beats(row[s], zero[s]) and beats(row[s], one[s]) for s in seeds):
            return (f"INTERIOR_{dose_tag(dose)}",
                    f"dose {dose} beats both endpoints on seeds {seeds}")

    if all(beats(zero[s], one[s]) for s in seeds):
        return "AT_0", f"dose 0.0 beats 1.0 on seeds {seeds}"
    if all(beats(one[s], zero[s]) for s in seeds):
        return "AT_1", f"dose 1.0 beats 0.0 on seeds {seeds}"
    return "UNRESOLVED", "endpoints not separable; no dose beats both"


def monotonic(table: dict) -> dict:
    """Does EER rise with dose? Reported per seed; requires BOTH seeds."""
    doses = sorted(table)
    out = {}
    for seed in SEEDS:
        vals = [(d, table[d][seed][0]) for d in doses
                if table.get(d) and table[d].get(seed)]
        if len(vals) < 2:
            out[seed] = {"ok": False, "detail": "insufficient data"}
            continue
        rising = all(vals[i][1] <= vals[i + 1][1] + 1e-9
                     for i in range(len(vals) - 1))
        out[seed] = {
            "ok": rising,
            "detail": " -> ".join(f"{d}:{v:.3f}" for d, v in vals),
        }
    out["all_seeds"] = all(v.get("ok") for k, v in out.items()
                           if isinstance(k, int))
    return out


def mindcf_from_cache(npz: Path):
    if not npz.exists():
        return None
    import numpy as np

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    from benchmarks.compute_mindcf import compute_mindcf

    z = np.load(npz)
    dcf, _ = compute_mindcf(z["bon"], z["spf"])
    return round(float(dcf), 4)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def ensure_ci(ckpt: str, manifest: str, condition: str, out: Path,
              attacks, force: bool):
    if out.exists() and not force:
        log(f"reuse {out.name}")
        return json.loads(out.read_text(encoding="utf-8"))
    if not (REPO / ckpt).exists():
        log(f"MISSING {ckpt} -- skipping (dose not trained yet)")
        return None
    cmd = [
        str(PY), "-u", str(REPO / "scripts" / "eer_bootstrap_ci.py"),
        "--checkpoint", ckpt,
        "--manifest", manifest,
        "--condition", condition,
        "--attacks", *attacks,
        "--out", str(out.relative_to(REPO)),
    ]
    log(f"scoring {Path(ckpt).name} [{condition}] -> {out.name}")
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0 or not out.exists():
        log(f"  FAILED rc={rc}")
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def ci_of(rep, key="pooled"):
    """Per-attack entries live under rep['attacks'], NOT at top level."""
    if not rep:
        return None
    e = rep.get("pooled") if key == "pooled" else (rep.get("attacks") or {}).get(key)
    if not e:
        return None
    return e["eer_pct"], e["ci95_pct"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-score", action="store_true",
                    help="do not launch scoring; classify from existing reports")
    args = ap.parse_args()

    table: dict = {d: {} for d in DOSES}
    in_domain: dict = {}
    raw: dict = {}

    for dose in DOSES:
        for seed in SEEDS:
            stem = stem_for(dose, seed)
            primary = REPORTS / f"eer_ci_{stem}_ITW_full_call.json"
            ev = REPORTS / f"eer_ci_{stem}_eval.json"

            if not args.no_score:
                ensure_ci(checkpoint_for(dose, seed), ITW_MANIFEST,
                          "full_call", primary, ["ITW"], args.force)
                ensure_ci(checkpoint_for(dose, seed), EVAL_MANIFEST,
                          "clean", ev, ["A19", "A20"], args.force)

            p = json.loads(primary.read_text(encoding="utf-8")) if primary.exists() else None
            table[dose][seed] = ci_of(p)
            raw[f"{dose}_{seed}"] = {
                "itw_full_call": table[dose][seed],
                "checkpoint": checkpoint_for(dose, seed),
            }
            if ev.exists():
                e = json.loads(ev.read_text(encoding="utf-8"))
                in_domain[f"{dose}_{seed}"] = {
                    "pooled_eer_pct": e["pooled"]["eer_pct"],
                    "ci95": e["pooled"]["ci95_pct"],
                }

    verdict, detail = classify_optimum(table)
    mono = monotonic(table)

    missing = [f"dose {d} seed {s}" for d in DOSES for s in SEEDS
               if not table[d].get(s)]

    report = {
        "experiment": "docs/21 augmentation dose-response",
        "adjudicated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "primary_condition": "ITW full_call",
        "table": {str(d): {str(s): v for s, v in table[d].items()}
                  for d in DOSES},
        "raw": raw,
        "in_domain_recorded": in_domain,
        "verdict": verdict,
        "verdict_detail": detail,
        "monotonic": mono,
        "missing": missing,
    }

    print("\n=== docs/21 dose-response (ITW full_call) ===")
    for dose in DOSES:
        row = "  ".join(
            f"seed {s}: {('%.3f [%.2f, %.2f]' % (table[dose][s][0], table[dose][s][1][0], table[dose][s][1][1])) if table[dose].get(s) else '--'}"
            for s in SEEDS
        )
        print(f"  dose {dose:<5} {row}")
    print(f"\n  monotonic per seed: "
          f"{ {k: v.get('ok') for k, v in mono.items() if isinstance(k, int)} }")
    for s, v in mono.items():
        if isinstance(s, int):
            print(f"    seed {s}: {v['detail']}")
    if missing:
        print(f"\n  NOT MEASURED: {', '.join(missing)}")
    print(f"\n  VERDICT: {verdict} -- {detail}")

    out = REPORTS / f"docs21_adjudication_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
