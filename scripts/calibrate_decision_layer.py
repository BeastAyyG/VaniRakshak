"""Choose every docs/13 Phase 1 threshold from DEV scores, then measure P1-S / P1-F.

Input is a per-utterance score dump written by
``scripts/score_checkpoint_eval.py --dump-scores DIR`` (``<label>.npz`` with
``raw_llr``, ``label`` (1 = bonafide) and ``attack``). Thresholds are fixed
on the dev dump only; ``--eval-scores`` is optional and only *measured*.

What it chooses (all on dev bonafide):

1. ``operating_threshold`` (§1.3): the ``--op-percentile`` of dev bonafide raw
   LLRs, so at most that fraction of genuine windows scores below 0.
2. ``VR_CLONE_CM_LLR_MAX`` (§1.1): bisected so a genuine, enrolled caller
   (ASV at the security bound) is falsely BLOCKed in at most ``--max-false-block``
   of ``--call-seconds`` calls, measured through the real session code.
3. ``VR_CHALLENGE_CM_LLR_MAX`` (§1.2): the ``--max-false-fail`` percentile of the
   mean of ``VR_CHALLENGE_MIN_WINDOWS`` genuine windows.

What it reports: P(challenge within 30/60/120 s | genuine) and
P(BLOCK by call end | spoof), per attack, across the §1.5 ASV x channel grid,
before (incumbent settings) and after (chosen settings).

Window scores are drawn i.i.d. from the pool (the dumps carry no speaker
column yet — docs/13 Amendment 1, Fatal 1). That ignores within-call
correlation; treat the numbers as optimistic about multi-window averaging and
re-run with speaker-grouped sessions once the manifests carry speakers.

Usage:
    python scripts/calibrate_decision_layer.py \\
        --dev-scores C:/vr-out/p2/dev/incumbent.npz \\
        --eval-scores C:/vr-out/p2/gate/incumbent.npz \\
        --out reports/decision_layer_calibration.json
    python scripts/calibrate_decision_layer.py --synthetic --out /tmp/x.json   # smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "vanirakshak-backend"))

from vanirakshak.config import ChallengeConfig, FusionConfig, RiskConfig, settings  # noqa: E402
from vanirakshak.config import _REAL_BACKEND_OVERRIDES  # noqa: E402
from vanirakshak.risk.simulate import bootstrap_sessions, simulate_session, summarise  # noqa: E402

ASV_GRID = (0.80, 0.90, 0.95, 1.00)
CHANNEL_GRID = (0.0, 0.25)
# Narrowband prosody the docs/13 amendment table assumes alongside channel 0.25.
PROSODY_FOR_CHANNEL = {0.0: 0.0, 0.25: 0.5}


def load_dump(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return d["raw_llr"].astype(np.float64), d["label"].astype(np.int64), d["attack"].astype(str)


def speakers_for(dump_path: Path, manifest: Path, protocol: Path) -> np.ndarray:
    """Speaker id of every dump row: dump row_index -> manifest path ->
    utterance id -> ASVspoof 5 protocol speaker (the target speaker for spoofs)."""
    d = np.load(dump_path, allow_pickle=False)
    utt_spk = {}
    with open(protocol, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                utt_spk[parts[1]] = parts[0]
    stems = []
    with open(manifest, encoding="utf-8") as f:
        next(f)
        for line in f:
            stems.append(Path(line.split("\t")[0].replace("\\", "/")).stem)
    return np.array([utt_spk.get(stems[int(i)], "?") for i in d["row_index"]])


def group_by(values: np.ndarray, keys: np.ndarray) -> List[np.ndarray]:
    return [values[keys == k] for k in np.unique(keys)]


def synthetic_dump(rng: np.random.Generator, n_bon: int = 4000, n_spf: int = 16000):
    """Gaussian stand-in with roughly the incumbent's ~13 % EER, for smoke runs only."""
    bon = rng.normal(10.0, 3.0, n_bon)
    idx = np.arange(n_spf) % 16
    attacks = np.array([f"A{17 + i}" for i in idx])
    spf = rng.normal(3.0, 3.5, n_spf) + 1.5 * (idx % 4)
    raw = np.concatenate([bon, spf])
    label = np.concatenate([np.ones(n_bon, int), np.zeros(n_spf, int)])
    attack = np.concatenate([np.array(["bonafide"] * n_bon), attacks])
    return raw, label, attack


def real_fusion() -> FusionConfig:
    return dataclasses.replace(FusionConfig(), **_REAL_BACKEND_OVERRIDES)


def run_cell(
    pool: np.ndarray,
    *,
    asv: float,
    channel: float,
    risk_cfg: RiskConfig,
    challenge_cfg: ChallengeConfig,
    fusion_cfg: FusionConfig,
    n_sessions: int,
    call_seconds: float,
    seed: int,
    text_ok: bool,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    sessions = bootstrap_sessions(
        pool, n_sessions=n_sessions, call_seconds=call_seconds, rng=rng,
        asv=asv, asv_enrolled=True, channel_anomaly=channel,
        prosody_unnatural=PROSODY_FOR_CHANNEL[channel],
    )
    # An attacker reading the digits answers correctly; so does a genuine caller.
    outs = [
        simulate_session(
            s, risk_cfg=risk_cfg, fusion_cfg=fusion_cfg, challenge_cfg=challenge_cfg,
            challenge_responder=lambda _i: text_ok,
        )
        for s in sessions
    ]
    return summarise(outs)


def bisect_clone_threshold(
    bon: np.ndarray, *, base_risk: RiskConfig, challenge_cfg: ChallengeConfig,
    fusion_cfg: FusionConfig, max_false_block: float, n_sessions: int,
    call_seconds: float, seed: int, iters: int = 14,
) -> Tuple[float, float]:
    """Largest clone_cm_llr_max whose genuine false-BLOCK rate stays <= target.

    Measured at the worst genuine cell for this rule: ASV 1.0 (always
    qualifies) and wideband. Larger threshold = rule fires more often.
    """
    def false_block(k: float) -> float:
        rc = dataclasses.replace(base_risk, clone_cm_llr_max=k)
        return float(run_cell(
            bon, asv=1.0, channel=0.0, risk_cfg=rc, challenge_cfg=challenge_cfg,
            fusion_cfg=fusion_cfg, n_sessions=n_sessions, call_seconds=call_seconds,
            seed=seed, text_ok=True,
        )["p_block_by_end"])

    lo, hi = float(np.min(bon)) - 1.0, float(np.median(bon))
    if false_block(lo) > max_false_block:
        return lo, false_block(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if false_block(mid) <= max_false_block:
            lo = mid
        else:
            hi = mid
    return lo, false_block(lo)


def grid(
    bon: np.ndarray, spoof_by_attack: Dict[str, np.ndarray], *, risk_cfg: RiskConfig,
    challenge_cfg: ChallengeConfig, fusion_cfg: FusionConfig, n_sessions: int,
    call_seconds: float, seed: int,
) -> Dict[str, object]:
    cells: Dict[str, object] = {}
    for asv in ASV_GRID:
        for ch in CHANNEL_GRID:
            key = f"asv={asv:.2f},channel={ch:.2f}"
            genuine = run_cell(
                bon, asv=asv, channel=ch, risk_cfg=risk_cfg, challenge_cfg=challenge_cfg,
                fusion_cfg=fusion_cfg, n_sessions=n_sessions, call_seconds=call_seconds,
                seed=seed, text_ok=True,
            )
            per_attack = {
                a: run_cell(
                    s, asv=asv, channel=ch, risk_cfg=risk_cfg, challenge_cfg=challenge_cfg,
                    fusion_cfg=fusion_cfg, n_sessions=n_sessions, call_seconds=call_seconds,
                    seed=seed, text_ok=True,
                )["p_block_by_end"]
                for a, s in sorted(spoof_by_attack.items())
            }
            cells[key] = {
                "genuine": genuine,
                "spoof_p_block_by_end": per_attack,
                "spoof_p_block_worst_attack": min(per_attack.values()) if per_attack else None,
            }
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dev-scores", type=Path)
    src.add_argument("--synthetic", action="store_true", help="Gaussian stand-in; smoke test only")
    ap.add_argument("--eval-scores", type=Path, help="Measured only, never used to choose thresholds")
    ap.add_argument("--speaker-grouped", action="store_true",
                    help="Each simulated call draws all its windows from ONE speaker (needs the "
                         "manifest + protocol flags below), instead of i.i.d. from the whole pool.")
    ap.add_argument("--dev-manifest", type=Path)
    ap.add_argument("--dev-protocol", type=Path)
    ap.add_argument("--eval-manifest", type=Path)
    ap.add_argument("--eval-protocol", type=Path)
    ap.add_argument("--fixed-from", type=Path,
                    help="Reuse the thresholds 'chosen' in this calibration JSON instead of choosing "
                         "new ones; only measure. For re-measuring T1's thresholds.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--op-percentile", type=float, default=5.0)
    ap.add_argument("--max-false-block", type=float, default=0.01,
                    help="Per-call genuine false-BLOCK budget for the clone rule")
    ap.add_argument("--max-false-fail", type=float, default=0.02,
                    help="Per-challenge genuine false-fail budget for the voice-scored challenge")
    ap.add_argument("--clone-min-windows", type=int, default=3)
    ap.add_argument("--challenge-min-windows", type=int, default=2)
    ap.add_argument("--challenge-consecutive", type=int, default=1,
                    help="VR_CHALLENGE_TRIGGER_CONSECUTIVE for the 'after' settings")
    ap.add_argument("--call-seconds", type=float, default=120.0)
    ap.add_argument("--sessions", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--checkpoint", type=Path, help="CM checkpoint to copy with operating_threshold set")
    ap.add_argument("--write-checkpoint", type=Path, help="Destination for that copy (must not exist)")
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; reports are the record (docs/10 ground rule 6).")

    rng = np.random.default_rng(args.seed)
    if args.synthetic:
        raw, label, attack = synthetic_dump(rng)
        source = "synthetic (smoke test — NOT a result)"
    else:
        raw, label, attack = load_dump(args.dev_scores)
        source = str(args.dev_scores)

    bon_raw = raw[label == 1]
    if len(bon_raw) < 100:
        raise SystemExit(f"only {len(bon_raw)} bonafide scores; need >= 100 to pick percentiles")
    op_threshold = float(np.percentile(bon_raw, args.op_percentile))
    bon = bon_raw - op_threshold

    fusion_cfg = real_fusion()
    base_risk = dataclasses.replace(settings.risk, clone_min_windows=args.clone_min_windows,
                                    clone_cm_llr_max=None)
    base_challenge = dataclasses.replace(settings.challenge, min_windows=args.challenge_min_windows)

    if args.fixed_from:
        fixed = json.loads(args.fixed_from.read_text(encoding="utf-8"))["chosen"]
        op_threshold = float(fixed["operating_threshold"])
        bon = bon_raw - op_threshold
        clone_k, clone_fb = float(fixed["VR_CLONE_CM_LLR_MAX"]), fixed.get("clone_genuine_false_block_at_asv_1")
        challenge_k = float(fixed["VR_CHALLENGE_CM_LLR_MAX"])
    else:
        clone_k, clone_fb = bisect_clone_threshold(
            bon, base_risk=base_risk, challenge_cfg=base_challenge, fusion_cfg=fusion_cfg,
            max_false_block=args.max_false_block, n_sessions=args.sessions,
            call_seconds=args.call_seconds, seed=args.seed,
        )
        means = np.array([
            rng.choice(bon, size=args.challenge_min_windows, replace=True).mean() for _ in range(20000)
        ])
        challenge_k = float(np.percentile(means, 100.0 * args.max_false_fail))

    chosen_risk = dataclasses.replace(base_risk, clone_cm_llr_max=clone_k)
    chosen_challenge = dataclasses.replace(base_challenge, cm_llr_max=challenge_k,
                                           trigger_consecutive=args.challenge_consecutive)

    def spoof_pools(r, y, a, thr, spk=None):
        out = {}
        for atk in np.unique(a[y == 0]):
            m = (y == 0) & (a == atk)
            out[atk] = group_by(r[m] - thr, spk[m]) if spk is not None else r[m] - thr
        return out

    dev_spk = eval_spk = None
    if args.speaker_grouped:
        if not (args.dev_manifest and args.dev_protocol):
            raise SystemExit("--speaker-grouped needs --dev-manifest and --dev-protocol")
        dev_spk = speakers_for(args.dev_scores, args.dev_manifest, args.dev_protocol)
        if args.eval_scores:
            if not (args.eval_manifest and args.eval_protocol):
                raise SystemExit("--speaker-grouped with --eval-scores needs --eval-manifest and --eval-protocol")
            eval_spk = speakers_for(args.eval_scores, args.eval_manifest, args.eval_protocol)
    dev_bon = group_by(bon, dev_spk[label == 1]) if dev_spk is not None else bon
    eval_sets = [("dev", dev_bon, spoof_pools(raw, label, attack, op_threshold, dev_spk))]
    if args.eval_scores:
        er, el, ea = load_dump(args.eval_scores)
        eb = er[el == 1] - op_threshold
        eval_sets.append(("eval", group_by(eb, eval_spk[el == 1]) if eval_spk is not None else eb,
                          spoof_pools(er, el, ea, op_threshold, eval_spk)))

    report: Dict[str, object] = {
        "source": source,
        "assumption": ("speaker-grouped: each call draws all windows from one speaker "
                       "(ASVspoof 5 protocol speaker ids)" if args.speaker_grouped else
                       "window scores drawn i.i.d. per call (no speaker column yet); optimistic"),
        "thresholds": f"fixed from {args.fixed_from}" if args.fixed_from else "chosen on dev in this run",
        "fusion": dataclasses.asdict(fusion_cfg),
        "chosen": {
            "operating_threshold": op_threshold,
            "op_percentile": args.op_percentile,
            "VR_CLONE_CM_LLR_MAX": round(clone_k, 4),
            "VR_CLONE_MIN_WINDOWS": args.clone_min_windows,
            "clone_genuine_false_block_at_asv_1": clone_fb,
            "VR_CHALLENGE_CM_LLR_MAX": round(challenge_k, 4),
            "VR_CHALLENGE_MIN_WINDOWS": args.challenge_min_windows,
            "VR_CHALLENGE_TRIGGER_CONSECUTIVE": args.challenge_consecutive,
        },
        "results": {},
    }
    for name, b, spools in eval_sets:
        before = grid(b, spools, risk_cfg=base_risk, challenge_cfg=base_challenge, fusion_cfg=fusion_cfg,
                      n_sessions=args.sessions, call_seconds=args.call_seconds, seed=args.seed)
        after = grid(b, spools, risk_cfg=chosen_risk, challenge_cfg=chosen_challenge, fusion_cfg=fusion_cfg,
                     n_sessions=args.sessions, call_seconds=args.call_seconds, seed=args.seed)
        report["results"][name] = {"before": before, "after": after}  # type: ignore[index]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"operating_threshold      = {op_threshold:.4f}  (p{args.op_percentile:g} of dev bonafide)")
    print(f"VR_CLONE_CM_LLR_MAX      = {clone_k:.4f}  (genuine false-BLOCK {clone_fb:.3%} at ASV 1.0)")
    print(f"VR_CHALLENGE_CM_LLR_MAX  = {challenge_k:.4f}")
    for name in report["results"]:  # type: ignore[union-attr]
        for phase in ("before", "after"):
            cells = report["results"][name][phase]  # type: ignore[index]
            worst = min((c["spoof_p_block_worst_attack"] or 0.0) for c in cells.values())
            fric = max(c["genuine"]["p_challenge_within_60s"] for c in cells.values())
            print(f"[{name}/{phase}] worst-cell worst-attack P(BLOCK|spoof)={worst:.3f}  "
                  f"max P(challenge<=60s|genuine)={fric:.3f}")
    print(f"Saved {args.out}")

    if args.write_checkpoint:
        if not args.checkpoint:
            raise SystemExit("--write-checkpoint needs --checkpoint")
        if args.write_checkpoint.exists():
            raise SystemExit(f"{args.write_checkpoint} exists; refusing to overwrite a checkpoint")
        import torch

        ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
        ckpt["operating_threshold"] = op_threshold
        ckpt["operating_threshold_source"] = f"p{args.op_percentile:g} dev bonafide, {source}"
        torch.save(ckpt, str(args.write_checkpoint))
        print(f"Wrote {args.write_checkpoint} with operating_threshold={op_threshold:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
