"""Adjudicate the docs/10 G4 experiment against its pre-registered criterion.

Criterion (docs/10, "G4 -- Attack diversity"), fixed in advance:

    A19 and A20 eval EER both below 12 %, without pooled EER regressing
    beyond the noise floor.

The noise floor that governs G3 and G4 is **0.770 pp** (docs/10 line 168,
Ground rule 1 -- measured between two full-corpus checkpoints computing an
identical inference function). It is NOT the 5.95 pp from the 4k/2-epoch
probe, and docs/10 explicitly warns against quoting that one.

This script never edits or reinterprets the criterion. It measures, applies
the stated thresholds, and reports. Where the data conflicts with a
pre-registered number it reports the conflict as an OPEN ITEM rather than
silently patching the threshold -- see the seed-spread check below.

Usage:
    python scripts/adjudicate_g4.py \
        --g4 data/cm_g4_rawboost_s42.pt data/cm_g4_rawboost_s43.pt \
        --control data/cm_g4_control_s42.pt data/cm_g4_control_s43.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.compute_mindcf import compute_eer, compute_mindcf  # noqa: E402
from cm_frontend import load_frontend_for  # noqa: E402
from eval_call_channel import build_channel, load_manifest, score_all  # noqa: E402
from ml.aasist_model import AASISTLHead  # noqa: E402

EER_BAR_PCT = 12.0          # "A19 and A20 eval EER both below 12 %"
DEFAULT_FLOOR_PP = 0.770    # docs/10 line 168, Ground rule 1
CRITERION_ATTACKS = ("A19", "A20")


def score_checkpoint(ckpt_path: Path, rows, device, batch, limit=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    merge_layers = ck.get("merge_layers")
    layer_index = int(ck.get("layer_index", 5))
    freeze_merge = bool(ck.get("freeze_merge", False))

    wavlm, dtype = load_frontend_for(ck, device)
    model = AASISTLHead(in_dim=768, merge_layers=merge_layers, freeze_merge=freeze_merge)
    model.load_state_dict(ck["state_dict"], strict=False)
    model.to(device).eval()

    # ASVspoof 5 was captured over real telephone networks: 'clean' IS the call.
    channel = build_channel("clean")
    bon, spf, per_atk = score_all(
        model, wavlm, rows, channel, device, merge_layers,
        layer_index, batch, dtype,
    )
    eer, thr = compute_eer(bon, spf)
    dcf, _ = compute_mindcf(bon, spf)
    atk = {}
    for k, v in per_atk.items():
        if len(v) >= 20:
            a, _ = compute_eer(bon, v)
            atk[k] = round(float(a) * 100, 3)
    del model, wavlm
    torch.cuda.empty_cache()
    return {
        "checkpoint": str(ckpt_path),
        "seed": ck.get("seed"),
        "best_epoch": ck.get("best_epoch"),
        "pooled_eer_pct": round(float(eer) * 100, 3),
        "pooled_minDCF": round(float(dcf), 4),
        "per_attack_eer_pct": atk,
        "n_bonafide": int(bon.size),
        "n_spoof": int(spf.size),
        "g4_rawboost": (ck.get("g4_rawboost") or {}).get("enabled", False),
    }


def median(xs):
    return float(np.median(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4", nargs="+", required=True)
    ap.add_argument("--control", nargs="*", default=[])
    ap.add_argument("--manifest", default=r"data\eval_benchmark\eval_slice_20k.tsv")
    ap.add_argument("--noise-floor", type=float, default=DEFAULT_FLOOR_PP)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, rows = load_manifest(Path(args.manifest), args.limit)
    print(f"manifest: {args.manifest} -> {len(rows)} rows\n")

    def run_group(paths, label):
        out = []
        for p in paths:
            pp = Path(p)
            if not pp.exists():
                print(f"[{label}] MISSING {pp} -- skipping")
                continue
            r = score_checkpoint(pp, rows, device, args.batch, args.limit)
            out.append(r)
            pa = r["per_attack_eer_pct"]
            print(
                f"[{label}] {pp.name} (seed {r['seed']}, best epoch {r['best_epoch']}): "
                f"pooled {r['pooled_eer_pct']:.3f}%  "
                f"A19 {pa.get('A19', float('nan')):.2f}%  A20 {pa.get('A20', float('nan')):.2f}%"
            )
        return out

    print("=== G4 arm (RawBoost ON) ===")
    g4 = run_group(args.g4, "g4")
    print("\n=== Control arm (RawBoost OFF) ===")
    ctl = run_group(args.control, "ctl")

    def vals(group, a):
        return [g["per_attack_eer_pct"][a] for g in group if a in g["per_attack_eer_pct"]]

    g4_pooled = [g["pooled_eer_pct"] for g in g4]
    ctl_pooled = [c["pooled_eer_pct"] for c in ctl]

    criteria = {}
    for a in CRITERION_ATTACKS:
        v = vals(g4, a)
        if not v:
            criteria[f"{a}_below_12pct"] = {"status": "UNRESOLVED", "detail": "no data"}
        else:
            worst = max(v)
            criteria[f"{a}_below_12pct"] = {
                "status": "PASS" if worst < EER_BAR_PCT else "FAIL",
                "bar_pct": EER_BAR_PCT,
                "per_seed_pct": v,
                "worst_pct": worst,
            }

    if g4_pooled and ctl_pooled:
        delta = median(g4_pooled) - median(ctl_pooled)
        criteria["pooled_eer_no_regression"] = {
            "status": "PASS" if delta <= args.noise_floor else "FAIL",
            "noise_floor_pp": args.noise_floor,
            "median_g4_pct": median(g4_pooled),
            "median_control_pct": median(ctl_pooled),
            "delta_pp": round(delta, 3),
            "note": "positive delta = G4 worse than control",
        }
    else:
        criteria["pooled_eer_no_regression"] = {
            "status": "UNRESOLVED",
            "detail": "control arm not available yet",
        }

    # Seed spread vs the pre-registered floor -- reported, never used to
    # silently change the threshold.
    open_items = []
    if len(g4_pooled) > 1:
        spread = max(g4_pooled) - min(g4_pooled)
        if spread > args.noise_floor:
            open_items.append(
                f"seed-to-seed pooled spread {spread:.3f} pp EXCEEDS the "
                f"pre-registered {args.noise_floor} pp floor (Ground rule 1). "
                "Reported as an open item; the criterion was NOT adjusted."
            )

    statuses = [c.get("status") for c in criteria.values()]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "UNRESOLVED" in statuses:
        overall = "UNRESOLVED"
    else:
        overall = "PASS"

    report = {
        "experiment": "docs/10 G4 -- attack diversity (RawBoost augmentation)",
        "criterion": "A19 and A20 eval EER both < 12% without pooled EER "
                     "regressing beyond the noise floor",
        "adjudicated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(args.manifest),
        "n_rows": len(rows),
        "noise_floor_pp": args.noise_floor,
        "g4_arm": g4,
        "control_arm": ctl,
        "criteria": criteria,
        "open_items": open_items,
        "OVERALL": overall,
    }

    print("\n=== VERDICT ===")
    for k, v in criteria.items():
        print(f"  {v['status']:11s} {k}  {json.dumps({a: b for a, b in v.items() if a != 'status'})}")
    for o in open_items:
        print(f"  OPEN ITEM: {o}")
    print(f"\n  OVERALL: {overall}")

    out = Path(args.out) if args.out else REPO_ROOT / "reports" / (
        f"g4_adjudication_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
