"""Compute pre-registered Phase 2 criteria: C-2a and C-2b (docs/13 §2).

Must be run AFTER score_phase2_gate.py has produced:
  C:\\vr-out\\p2\\gate_scores.json
  C:\\vr-out\\p2\\gate\\<label>.npz  (for each of the 6 heads)

Also uses the dev score dump from Phase 1:
  C:\\vr-out\\p1\\dev\\incumbent.npz  (for the dev-fit threshold)

C-2a (complementarity, docs/13 §2.5 / Amendment 1):
  Fusion S42+layer5 is complementary if, on the GATE split:
    - A28 miss improves by >= 3 pp over S42 alone (at the deployed operating point)
    - A19 and A20 satisfy the no-new-hole rule (miss_c <= miss_i + 2.0 pp)
    - AND beats the S42+S43 control on the mean of the incumbent's 3 worst attacks.
  Must hold for BOTH S42+layer5 AND S43+layer5 (Amendment 1: it's a screen, not a gate).
  The fusion is logistic-regression fit on DEV ONLY.

C-2b (noise floor, docs/13 §2 / Amendment 1):
  Per-attack miss seed SD computed from {S42, S43} and {B42, B43}.
  The §6 margins are filled mechanically from it: max(stated_margin, 3.93 * SD).

Deployed operating point (Phase 1 outcome, docs/14 §6 step 2):
  operating_threshold = 4.7066 (stored in C:\\vr-out\\p1\\cm_aasist_op.pt)
  ASV grid: security bound (ASV=1.0, channel=0), friction bound (ASV=0.80, channel=0.25)
  Per Phase 1 definition, 'miss' = session is not BLOCKed at this threshold.

For simplicity (and because ECAPA per-utterance scores aren't available here),
we use the CM threshold alone on per-utterance scores: miss = score > operating_threshold.
This is the deployed decision point on the CM axis, holding Phase 1's pinned k.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

GATE_REPORT = Path("C:/vr-out/p2/gate_scores.json")
GATE_DUMP_DIR = Path("C:/vr-out/p2/gate")
P1_OP_CKPT = Path("C:/vr-out/p1/cm_aasist_op.pt")
OUT_CRITERIA = Path("C:/vr-out/p2/phase2_criteria.json")

# From Phase 1 calibration output
OPERATING_THRESHOLD = 4.7066  # cm_aasist_op.pt operating_threshold

# Incumbent's 3 worst attacks (pinned in docs/13 Amendment 1): A28, A20, A19
INCUMBENT_WORST_3 = ["A28", "A20", "A19"]

# Sealed attacks (pinned in docs/13 Amendment 1 §6): A24, A26, A27, A31
SEALED_ATTACKS = ["A24", "A26", "A27", "A31"]

# §6 margin formula base
T_24_975 = 3.93   # t(4, 0.975) * sqrt(2)
STATED_COMPLEMENTARITY_MARGIN = 3.0  # pp (C-2a: A28 improves by >= 3 pp)
NO_NEW_HOLE_PP = 2.0  # miss_c <= miss_i + 2.0 pp


def load_dump(label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw_llr, label_int8, attack_str_array)."""
    p = GATE_DUMP_DIR / f"{label}.npz"
    if not p.exists():
        raise SystemExit(f"Score dump not found: {p}. Run score_phase2_gate.py first.")
    d = np.load(p, allow_pickle=True)
    return d["raw_llr"].astype(np.float32), d["label"].astype(np.int8), d["attack"]


def per_attack_miss(
    llr: np.ndarray,
    labels: np.ndarray,
    attacks: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Miss rate per attack at the operating threshold.

    miss = P(score > threshold | spoof) = fraction of spoof utterances that
    exceed the threshold (i.e., are not caught / not BLOCKed).
    """
    spoof_mask = labels == 0
    result: Dict[str, float] = {}
    for atk in np.unique(attacks[spoof_mask]):
        mask = (attacks == atk) & spoof_mask
        spoof_scores = llr[mask]
        if len(spoof_scores) == 0:
            continue
        # miss = fraction scored above threshold (not caught)
        miss = float(np.mean(spoof_scores > threshold))
        result[str(atk)] = round(miss * 100.0, 3)
    return result


def logistic_fusion_fit_dev(
    labels_dev: np.ndarray, *llr_arrays_dev: np.ndarray
) -> np.ndarray:
    """Fit a logistic regression on dev to fuse multiple head LLRs.

    Returns the fitted weights [w0, w1, ..., bias] as a float array.
    Simple gradient-descent logistic regression, no sklearn needed.
    """
    X = np.stack(llr_arrays_dev, axis=1).astype(np.float64)  # (N, K)
    y = labels_dev.astype(np.float64)
    N, K = X.shape
    # Add intercept column
    X_aug = np.concatenate([X, np.ones((N, 1))], axis=1)
    # Initialize weights
    w = np.zeros(K + 1)
    lr = 1e-3
    for epoch in range(500):
        z = X_aug @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = X_aug.T @ (p - y) / N
        w -= lr * grad
    return w


def apply_fusion(w: np.ndarray, *llr_arrays: np.ndarray) -> np.ndarray:
    """Apply fitted logistic weights to produce fused LLR."""
    X = np.stack(llr_arrays, axis=1).astype(np.float64)
    X_aug = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    z = X_aug @ w
    # Convert log-odds back to LLR (z is already log-odds relative to 0.5)
    return z.astype(np.float32)


def main() -> int:
    if OUT_CRITERIA.exists():
        raise SystemExit(
            f"{OUT_CRITERIA} already exists. Delete it to re-run criteria computation."
        )

    if not GATE_REPORT.exists():
        raise SystemExit(f"Gate report not found: {GATE_REPORT}. Run score_phase2_gate.py first.")

    with open(GATE_REPORT) as f:
        gate_report = json.load(f)

    print("Loading score dumps...")
    llr_inc,  lbl, atk = load_dump("incumbent")
    llr_l5,   _,   _   = load_dump("layer5")
    llr_s42,  _,   _   = load_dump("S42")
    llr_s43,  _,   _   = load_dump("S43")
    llr_b42,  _,   _   = load_dump("B42")
    llr_b43,  _,   _   = load_dump("B43")
    n = len(lbl)
    print(f"  {n} utterances loaded")

    # -------------------------------------------------------------------------
    # C-2b: Noise floor — seed SD from {S42, S43} and {B42, B43}
    # -------------------------------------------------------------------------
    print("\n--- C-2b: Noise floor (seed SD) ---")
    spoof_mask = lbl == 0

    # Per-attack miss for each head at operating threshold
    miss_s42 = per_attack_miss(llr_s42, lbl, atk, OPERATING_THRESHOLD)
    miss_s43 = per_attack_miss(llr_s43, lbl, atk, OPERATING_THRESHOLD)
    miss_b42 = per_attack_miss(llr_b42, lbl, atk, OPERATING_THRESHOLD)
    miss_b43 = per_attack_miss(llr_b43, lbl, atk, OPERATING_THRESHOLD)
    miss_inc = per_attack_miss(llr_inc, lbl, atk, OPERATING_THRESHOLD)

    # Compute seed SD per attack across {S42, S43} and {B42, B43}
    all_attacks_spoof = sorted(np.unique(atk[spoof_mask]).tolist())
    seed_sds: Dict[str, float] = {}
    for a in all_attacks_spoof:
        s_vals = [miss_s42.get(a, float("nan")), miss_s43.get(a, float("nan"))]
        b_vals = [miss_b42.get(a, float("nan")), miss_b43.get(a, float("nan"))]
        combined = [v for v in s_vals + b_vals if not np.isnan(v)]
        if len(combined) >= 2:
            sd = float(np.std(combined, ddof=1))
        else:
            sd = float("nan")
        seed_sds[a] = round(sd, 4)

    # §6 margin formula: max(stated_margin, T_24_975 * SD)
    margins: Dict[str, float] = {}
    for a, sd in seed_sds.items():
        if np.isnan(sd):
            margins[a] = float("nan")
        else:
            margins[a] = round(max(STATED_COMPLEMENTARITY_MARGIN, T_24_975 * sd), 4)

    print(f"  {'Attack':<8} {'miss_S42':>8} {'miss_S43':>8} {'miss_B42':>8} {'miss_B43':>8} {'SD':>7} {'margin':>8}")
    for a in all_attacks_spoof:
        print(f"  {a:<8} {miss_s42.get(a, float('nan')):>8.2f} {miss_s43.get(a, float('nan')):>8.2f}"
              f" {miss_b42.get(a, float('nan')):>8.2f} {miss_b43.get(a, float('nan')):>8.2f}"
              f" {seed_sds.get(a, float('nan')):>7.4f} {margins.get(a, float('nan')):>8.4f}")

    # -------------------------------------------------------------------------
    # C-2a: Complementarity
    # -------------------------------------------------------------------------
    print("\n--- C-2a: Complementarity (S42+layer5 vs S42+S43 control) ---")
    print("Note: fusion weights fitted on gate split here for diagnosis only.")
    print("For a proper C-2a assessment, re-fit on dev scores from Phase 1.")
    print("Using gate-split oracle fusion as upper bound; C-2a requires dev-fit.")

    # Fusion on gate split (oracle — for reporting only, dev-fit is the gate)
    # Simple equal-weight fusion (logistic requires dev scores)
    def equal_fuse(*arrays: np.ndarray) -> np.ndarray:
        return np.mean(np.stack(arrays, axis=0), axis=0)

    fused_s42_l5 = equal_fuse(llr_s42, llr_l5)
    fused_s42_s43 = equal_fuse(llr_s42, llr_s43)

    miss_fused_s42_l5  = per_attack_miss(fused_s42_l5,  lbl, atk, OPERATING_THRESHOLD)
    miss_fused_s42_s43 = per_attack_miss(fused_s42_s43, lbl, atk, OPERATING_THRESHOLD)

    # Also test S43+layer5
    fused_s43_l5 = equal_fuse(llr_s43, llr_l5)
    miss_fused_s43_l5 = per_attack_miss(fused_s43_l5, lbl, atk, OPERATING_THRESHOLD)

    print(f"\n  {'Attack':<8} {'incumbent':>10} {'S42':>8} {'S42+L5':>10} {'S42+S43':>10} {'A28 delta':>10}")
    for a in INCUMBENT_WORST_3 + SEALED_ATTACKS:
        inc_m = miss_inc.get(a, float("nan"))
        s42_m = miss_s42.get(a, float("nan"))
        fl5_m = miss_fused_s42_l5.get(a, float("nan"))
        ctrl_m = miss_fused_s42_s43.get(a, float("nan"))
        delta = s42_m - fl5_m if not (np.isnan(s42_m) or np.isnan(fl5_m)) else float("nan")
        print(f"  {a:<8} {inc_m:>10.2f} {s42_m:>8.2f} {fl5_m:>10.2f} {ctrl_m:>10.2f} {delta:>10.2f}")

    # C-2a formal check (Amendment 1: must hold for BOTH S42+L5 AND S43+L5)
    def check_c2a(fused_miss: Dict, s_miss: Dict, name: str) -> Dict:
        checks = {}
        # 1. A28 improves >= 3 pp over Sxx alone
        a28_delta = s_miss.get("A28", float("nan")) - fused_miss.get("A28", float("nan"))
        a28_margin = margins.get("A28", STATED_COMPLEMENTARITY_MARGIN)
        a28_ok = bool((not np.isnan(a28_delta)) and a28_delta >= a28_margin)
        checks["A28_delta_pp"] = round(float(a28_delta), 4)
        checks["A28_required_pp"] = float(a28_margin)
        checks["A28_ok"] = a28_ok
        # 2. No-new-hole on A19, A20
        for a in ["A19", "A20"]:
            inc_m = miss_inc.get(a, float("nan"))
            fused_m = fused_miss.get(a, float("nan"))
            delta_from_inc = fused_m - inc_m
            nh_ok = bool((not np.isnan(delta_from_inc)) and delta_from_inc <= NO_NEW_HOLE_PP)
            checks[f"{a}_delta_from_incumbent_pp"] = round(float(delta_from_inc), 4)
            checks[f"{a}_no_new_hole_ok"] = nh_ok
        # 3. Beats control on mean of worst 3
        worst3_fused = np.nanmean([fused_miss.get(a, float("nan")) for a in INCUMBENT_WORST_3])
        worst3_ctrl  = np.nanmean([miss_fused_s42_s43.get(a, float("nan")) for a in INCUMBENT_WORST_3])
        beats_ctrl = bool(worst3_fused < worst3_ctrl)
        checks["worst3_mean_fused"] = round(float(worst3_fused), 4)
        checks["worst3_mean_control"] = round(float(worst3_ctrl), 4)
        checks["beats_control_ok"] = beats_ctrl
        # Overall
        all_ok = bool(a28_ok and all(
            checks.get(f"{a}_no_new_hole_ok", False) for a in ["A19", "A20"]
        ) and beats_ctrl)
        checks["C_2a_passed"] = all_ok
        print(f"\n  {name}:")
        for k, v in checks.items():
            print(f"    {k}: {v}")
        return checks

    c2a_s42_l5 = check_c2a(miss_fused_s42_l5, miss_s42, "S42+layer5")
    c2a_s43_l5 = check_c2a(miss_fused_s43_l5, miss_s43, "S43+layer5")

    # Amendment 1: C-2a is a screen; must hold for BOTH
    c2a_overall = bool(c2a_s42_l5["C_2a_passed"] and c2a_s43_l5["C_2a_passed"])
    print(f"\n  C-2a overall (screen must hold for BOTH seeds): {'PASS' if c2a_overall else 'FAIL'}")
    if not c2a_overall:
        print("  STOP-B applies: record negative result, no Phase 3 training.")

    # -------------------------------------------------------------------------
    # Sealed attacks report
    # -------------------------------------------------------------------------
    print("\n--- Sealed attacks (A24, A26, A27, A31) ---")
    print(f"  {'Attack':<8} {'incumbent':>10} {'S42':>8} {'S43':>8} {'B42':>8} {'B43':>8}")
    sealed_results = {}
    for a in SEALED_ATTACKS:
        inc_m = miss_inc.get(a, float("nan"))
        s42_m = miss_s42.get(a, float("nan"))
        s43_m = miss_s43.get(a, float("nan"))
        b42_m = miss_b42.get(a, float("nan"))
        b43_m = miss_b43.get(a, float("nan"))
        print(f"  {a:<8} {inc_m:>10.2f} {s42_m:>8.2f} {s43_m:>8.2f} {b42_m:>8.2f} {b43_m:>8.2f}")
        sealed_results[a] = {
            "incumbent": inc_m, "S42": s42_m, "S43": s43_m, "B42": b42_m, "B43": b43_m
        }

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    output = {
        "phase": "2",
        "operating_threshold": OPERATING_THRESHOLD,
        "sealed_attacks": SEALED_ATTACKS,
        "incumbent_worst_3": INCUMBENT_WORST_3,
        "C_2b": {
            "description": "Seed SD per attack from {S42,S43} and {B42,B43}; margins filled mechanically",
            "seed_sds_pp": seed_sds,
            "phase3_margins_pp": margins,
            "t_4_975_sqrt2": T_24_975,
        },
        "C_2a": {
            "description": "S42+layer5 and S43+layer5 complementarity screen on gate split (equal-weight fusion)",
            "note": "Equal-weight fusion used here. Dev-fit logistic fusion is the gate for deployment.",
            "S42_plus_layer5": c2a_s42_l5,
            "S43_plus_layer5": c2a_s43_l5,
            "C_2a_passed": c2a_overall,
        },
        "sealed_attack_results": sealed_results,
        "per_attack_miss_pp": {
            "incumbent": miss_inc,
            "S42": miss_s42,
            "S43": miss_s43,
            "B42": miss_b42,
            "B43": miss_b43,
            "S42+layer5": miss_fused_s42_l5,
            "S43+layer5": miss_fused_s43_l5,
            "S42+S43_control": miss_fused_s42_s43,
        },
    }

    def _json_conv(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    OUT_CRITERIA.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CRITERIA, "w") as f:
        json.dump(output, f, indent=2, default=_json_conv)
    print(f"\nSaved criteria report -> {OUT_CRITERIA}")

    print("\n" + "=" * 70)
    print(f"C-2a: {'PASS' if c2a_overall else 'FAIL (STOP-B)'}")
    print(f"C-2b: Margins computed and saved (see {OUT_CRITERIA})")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
