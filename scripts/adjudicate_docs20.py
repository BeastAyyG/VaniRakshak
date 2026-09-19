"""Adjudicate the docs/20 augmentation-off control against its criteria.

docs/20 asks whether the codec/telephony augmenter causes the out-of-domain
regression. It is a matched, flag-flipped pair:

    arm A = codec ON,  RawBoost OFF   <- the docs/10 G4 control, reused
    arm B = codec OFF, RawBoost OFF   <- trained by docs20_armb_runner.py

Criteria were fixed in advance in docs/20. This script never adjusts them.
Where the data conflicts with a pre-registered number it is reported as an
OPEN ITEM, not silently patched.

  Primary   ITW full_call EER better with augmentation OFF by a
            NON-OVERLAPPING 95 % interval, on BOTH seeds.
  Secondary A19 and A20 both improve with non-overlapping intervals
            (>1.36 pp separation, per the docs/19 amendment).
  Recorded  In-domain pooled EER and minDCF. Expected to get WORSE with
            augmentation off. Reported, not pass/fail, not hidden.

This is orchestration only: every number comes from scripts/eer_bootstrap_ci.py
(2,000-rep percentile bootstrap) or is computed straight off the score cache
that script writes. Nothing is re-scored twice -- existing reports are reused
unless --force, so the script is resumable and cheap to re-run.

Usage:
    python scripts/adjudicate_docs20.py            # reuse what exists
    python scripts/adjudicate_docs20.py --force    # rescore everything
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(r"C:\VaniRakshak")
PY = REPO / ".venv" / "Scripts" / "python.exe"
REPORTS = REPO / "reports"

EVAL_MANIFEST = r"data\eval_benchmark\eval_slice_20k.tsv"
ITW_MANIFEST = r"A:\cm-corpus\manifests\itw_manifest.tsv"

ARM_A = {42: "data/cm_g4_control_s42.pt", 43: "data/cm_g4_control_s43.pt"}
ARM_B = {42: "data/cm_docs20_armb_s42.pt", 43: "data/cm_docs20_armb_s43.pt"}

PER_ATTACK_FLOOR_PP = 1.36   # docs/19 amendment: A19/A20 sampling noise


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cache_path(ckpt: str, manifest: str, condition: str) -> Path:
    stem = Path(ckpt).stem
    mstem = Path(manifest).stem
    return REPORTS / f"scores_{stem}__{mstem}__{condition}.npz"


def ensure_ci(ckpt: str, manifest: str, condition: str, out: Path,
              attacks, force: bool) -> dict | None:
    """Run eer_bootstrap_ci.py unless its report already exists."""
    if out.exists() and not force:
        log(f"reuse {out.name}")
        return json.loads(out.read_text(encoding="utf-8"))
    if not (REPO / ckpt).exists():
        log(f"MISSING {ckpt} -- skipping")
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


def mindcf_from_cache(npz: Path) -> float | None:
    """minDCF straight off the cached scores -- no rescoring."""
    if not npz.exists():
        return None
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    from benchmarks.compute_mindcf import compute_mindcf
    z = np.load(npz)
    dcf, _ = compute_mindcf(z["bon"], z["spf"])
    return round(float(dcf), 4)


def ci_of(rep: dict | None, key: str = "pooled"):
    if not rep:
        return None
    e = rep.get(key) if key != "pooled" else rep.get("pooled")
    if not e:
        return None
    return e["eer_pct"], e["ci95_pct"]


def disjoint_better(b, a) -> tuple[bool, str]:
    """True if b is better than a with non-overlapping 95 % intervals."""
    if not b or not a:
        return False, "missing data"
    b_pt, (b_lo, b_hi) = b
    a_pt, (a_lo, a_hi) = a
    ok = b_hi < a_lo
    return ok, (f"{b_pt:.3f} [{b_lo:.2f}, {b_hi:.2f}] vs "
                f"{a_pt:.3f} [{a_lo:.2f}, {a_hi:.2f}] "
                f"gap {a_pt - b_pt:+.3f} pp -> "
                f"{'disjoint' if ok else 'OVERLAP'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data: dict[str, dict] = {}

    # Filename tags deliberately match the arm-A reports already written by
    # hand (eer_ci_armA_control_s42_ITW_full_call.json etc.), so those are
    # reused rather than rescored.
    for arm_name, arms, tag_fmt in (("arm_a", ARM_A, "armA_control_s{}"),
                                    ("arm_b", ARM_B, "armB_s{}")):
        for seed, ck in arms.items():
            tag = tag_fmt.format(seed)
            data[f"{arm_name}_s{seed}"] = {}
            data[f"{arm_name}_s{seed}"]["_tag"] = tag

            slot = data[f"{arm_name}_s{seed}"]

            ev = ensure_ci(
                ck, EVAL_MANIFEST, "clean",
                REPORTS / f"eer_ci_{tag}_eval.json",
                ["A19", "A20"], args.force,
            )
            slot["eval"] = ev

            for cond in ("clean", "full_call"):
                r = ensure_ci(
                    ck, ITW_MANIFEST, cond,
                    REPORTS / f"eer_ci_{tag}_ITW_{cond}.json",
                    ["ITW"], args.force,
                )
                slot[f"itw_{cond}"] = r

            npz = cache_path(ck, EVAL_MANIFEST, "clean")
            slot["eval_mindcf"] = mindcf_from_cache(npz)
            if ev:
                slot["eval_pooled_eer"] = ev["pooled"]["eer_pct"]

    # ---- criteria -------------------------------------------------------
    criteria: dict = {}
    per_seed_primary = {}
    for seed in (42, 43):
        a = ci_of(data[f"arm_a_s{seed}"]["itw_full_call"])
        b = ci_of(data[f"arm_b_s{seed}"]["itw_full_call"])
        ok, detail = disjoint_better(b, a)
        per_seed_primary[seed] = {"ok": ok, "detail": detail}
    missing = [s for s, v in per_seed_primary.items()
               if v["detail"] == "missing data"]
    if missing:
        p_status = "UNRESOLVED"
    else:
        p_status = "PASS" if all(v["ok"] for v in per_seed_primary.values()) else "FAIL"
    criteria["primary_itw_full_call_better_off"] = {
        "status": p_status,
        "rule": "arm B better than arm A with non-overlapping 95% CI, both seeds",
        "per_seed": {str(k): v for k, v in per_seed_primary.items()},
    }

    secondary = {}
    for atk in ("A19", "A20"):
        per = {}
        for seed in (42, 43):
            a = ci_of(data[f"arm_a_s{seed}"]["eval"], atk)
            b = ci_of(data[f"arm_b_s{seed}"]["eval"], atk)
            ok, detail = disjoint_better(b, a)
            if b and a and (a[0] - b[0]) < PER_ATTACK_FLOOR_PP:
                ok = False
            per[seed] = {"ok": ok, "detail": detail}
        if any(v["detail"] == "missing data" for v in per.values()):
            s_status = "UNRESOLVED"
        else:
            s_status = "PASS" if all(v["ok"] for v in per.values()) else "FAIL"
        secondary[atk] = {
            "status": s_status,
            "floor_pp": PER_ATTACK_FLOOR_PP,
            "per_seed": {str(k): v for k, v in per.items()},
        }
    criteria["secondary_per_attack"] = secondary

    recorded = {
        tag: {
            "eval_pooled_eer_pct": d.get("eval_pooled_eer"),
            "eval_minDCF": d.get("eval_mindcf"),
        }
        for tag, d in data.items()
    }

    # Seed spread OOD -- reported against Ground rule 1, never used to change it.
    open_items = []
    for arm_name in ("arm_a", "arm_b"):
        vals = []
        for seed in (42, 43):
            c = ci_of(data[f"{arm_name}_s{seed}"]["itw_full_call"])
            if c:
                vals.append(c[0])
        if len(vals) == 2:
            spread = abs(vals[0] - vals[1])
            open_items.append(
                f"{arm_name} ITW full_call seed spread {spread:.3f} pp; the "
                f"pre-registered noise floor is 0.770 pp (Ground rule 1, "
                f"measured in-domain). Recorded as an open item; the floor was "
                f"NOT adjusted."
            )

    statuses = [criteria["primary_itw_full_call_better_off"]["status"]]
    statuses += [v["status"] for v in secondary.values()]
    if "UNRESOLVED" in statuses:
        overall = "UNRESOLVED"
    elif "FAIL" in statuses:
        overall = "FAIL"
    else:
        overall = "PASS"

    report = {
        "experiment": "docs/20 -- augmentation-off matched control",
        "adjudicated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arms": {"A": "codec ON, RawBoost OFF (docs/10 G4 control, reused)",
                 "B": "codec OFF, RawBoost OFF"},
        "manifests": {"eval": EVAL_MANIFEST, "itw": ITW_MANIFEST},
        "measurements": data,
        "criteria": criteria,
        "recorded_not_pass_fail": recorded,
        "open_items": open_items,
        "OVERALL": overall,
    }

    print("\n=== VERDICT ===")
    p = criteria["primary_itw_full_call_better_off"]
    print(f"  {p['status']:5s} primary (ITW full_call, both seeds)")
    for s, v in p["per_seed"].items():
        print(f"          seed {s}: {v['detail']}")
    for atk, v in secondary.items():
        print(f"  {v['status']:5s} secondary {atk}")
        for s, d in v["per_seed"].items():
            print(f"          seed {s}: {d['detail']}")
    print("\n  recorded (not pass/fail):")
    for tag, r in recorded.items():
        print(f"    {tag:10s} pooled {r['eval_pooled_eer_pct']}  "
              f"minDCF {r['eval_minDCF']}")
    for o in open_items:
        print(f"  OPEN ITEM: {o}")
    print(f"\n  OVERALL: {overall}")

    out = REPORTS / f"docs20_adjudication_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
