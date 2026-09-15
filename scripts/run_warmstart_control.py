#!/usr/bin/env python
"""Run the warm-start control that docs/10's headline claim needs (criterion 3).

Why this script exists
----------------------
`scripts/run_from_scratch_experiment.py` line ~391 hard-asserts that
``--init-weights`` is absent -- correctly, because its whole point is to train
from random init. But docs/10's criterion 3 requires the *opposite* run:

    "The warm-start control run under current code (identical flags, same
     revision, --init-weights data/cm_aasist.pt) is worse than the median
     from-scratch seed on minDCF. Without this the claim is not about
     initialisation at all."

So no script in the repo could produce it. This one does, and it enforces the
things that make the comparison mean anything:

* **One variable.** Flags are identical to the S condition (same manifests,
  epochs, batch size, layer index, workers, full augmenter, same seed). The
  only difference is ``--init-weights``.
* **Never overwrites.** The checkpoint is ``data/cm_warmstart_S<seed>_L<n>.pt``
  -- a different name from ``cm_scratch_*``, so it cannot collide with, or
  silently replace, any from-scratch checkpoint. An existing file aborts.
* **GPU guard.** Refuses to start while ``handoff/experiment_state.json`` shows
  another run in flight: two trainers on one 5060 Ti would corrupt BOTH runs'
  epoch timings, and the control is only comparable if it is measured under the
  same conditions as S. Pass ``--force`` to override (prints a warning).
* **Scores on the full eval partition by default**, because criterion 1 is
  graded there. The 20k slice is available with ``--score-manifest``.

Usage:
    # after S42..S46 and their full-partition scoring have finished:
    python scripts/run_warmstart_control.py --device cuda

    # score an already-trained control checkpoint, nothing else:
    python scripts/run_warmstart_control.py --score-only

Then adjudicate everything at once:
    python scripts/check_claim.py \
        --claim reports/cm_from_scratch_claim_5seed_2026-09-16.json \
        --control reports/cm_warmstart_control_2026-09-16.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the orchestrator's proven tee/archive helpers rather than re-writing
# them: a control run whose log can be truncated by a re-run is not evidence.
from run_from_scratch_experiment import (  # noqa: E402  (sibling module)
    LOG_DIR,
    TRAIN,
    archive_existing_log,
    run_logged,
)

DEFAULT_INIT = REPO_ROOT / "data" / "cm_aasist.pt"
EXPERIMENT_STATE = REPO_ROOT / "handoff" / "experiment_state.json"
CONTROL_STATE = REPO_ROOT / "handoff" / "warmstart_control_state.json"
SCORER = REPO_ROOT / "scripts" / "score_checkpoint_eval.py"
FULL_EVAL_MANIFEST = r"A:\cm-corpus\manifests\eval_available.tsv"


def checkpoint_for(seed: int, layer_index: int) -> Path:
    """Control checkpoints live under their own prefix, never cm_scratch_*."""
    return REPO_ROOT / "data" / f"cm_warmstart_S{seed}_L{layer_index}.pt"


def gpu_busy() -> str | None:
    """Return a human-readable reason if another orchestrated run is in flight."""
    if not EXPERIMENT_STATE.is_file():
        return None
    try:
        state = json.loads(EXPERIMENT_STATE.read_text())
    except (OSError, ValueError):
        return None  # unreadable state is not evidence of a busy GPU
    for run in state.get("runs", []):
        if run.get("status") == "running":
            return f"run {run.get('label')} is still training (pid {run.get('child_pid')})"
    scoring = state.get("scoring", {})
    if scoring.get("status") == "running":
        return "the 5-seed scoring stage is still running"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--layer-index", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42,
                    help="Default 42: seed-matched to S42, so the control differs "
                         "from a claim run by exactly one variable.")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--train-manifest", type=str,
                    default=r"A:\cm-corpus\manifests\train.tsv")
    ap.add_argument("--dev-manifest", type=str,
                    default=r"A:\cm-corpus\manifests\dev.tsv")
    ap.add_argument("--init-weights", type=Path, default=DEFAULT_INIT,
                    help="The warm-start parent. Default data/cm_aasist.pt -- the "
                         "post-aug L12 model every historical checkpoint descends "
                         "from (docs/10, 2026-09-13).")
    ap.add_argument("--score-manifest", type=str, default=FULL_EVAL_MANIFEST,
                    help="Default: the 628k full eval partition, which is where "
                         "docs/10 grades the claim.")
    ap.add_argument("--out", type=str, default=None,
                    help="Report path (default reports/"
                         "cm_warmstart_control_<date>.json)")
    ap.add_argument("--score-only", action="store_true",
                    help="Skip training; score an existing control checkpoint.")
    ap.add_argument("--force", action="store_true",
                    help="Start even if another orchestrated run looks in flight.")
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    args = ap.parse_args()

    out = checkpoint_for(args.seed, args.layer_index)

    if not args.score_only:
        if not args.init_weights.is_file():
            raise SystemExit(
                f"--init-weights {args.init_weights} is not a file. The control is "
                f"meaningless without a real warm-start parent.")
        for m in (args.train_manifest, args.dev_manifest):
            if not Path(m).is_file():
                raise SystemExit(f"manifest not found: {m}")
        if out.exists():
            raise SystemExit(
                f"{out.name} already exists - refusing to overwrite. The control is "
                f"a measurement; re-running it to get a better number is exactly "
                f"what ground rule 5 forbids.")
        busy = gpu_busy()
        if busy and not args.force:
            raise SystemExit(
                f"another run is using the GPU: {busy}.\n"
                f"Two trainers on one card would slow both and the control must be "
                f"measured under the same conditions as S. Pass --force to override.")
        if busy and args.force:
            print(f"  WARNING: --force given while: {busy}", file=sys.stderr)

    state = {
        "schema_version": 1,
        "seed": args.seed,
        "layer_index": args.layer_index,
        "init_weights": str(args.init_weights),
        "checkpoint": str(out),
        "train": {"status": "skipped" if args.score_only else "pending"},
        "scoring": {"status": "pending"},
    }

    if not args.score_only:
        log_path = LOG_DIR / f"{date.today().isoformat()}-warmstart-s{args.seed}.log"
        archived = archive_existing_log(log_path)
        if archived:
            print(f"  previous log archived to {archived.name}")
        cmd = [
            sys.executable, str(TRAIN),
            "--train-manifest", args.train_manifest,
            "--dev-manifest", args.dev_manifest,
            "--output", str(out),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--layer-index", str(args.layer_index),
            "--device", args.device,
            "--seed", str(args.seed),
            "--workers", str(args.workers),
            # THE one variable. Everything above is byte-identical to the S runs.
            "--init-weights", str(args.init_weights),
        ]
        print("Warm-start control:", " ".join(cmd))
        print(f"  log: {log_path}")
        state["train"].update({"status": "running",
                               "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "cmd": cmd})
        CONTROL_STATE.write_text(json.dumps(state, indent=2))
        rc = run_logged(cmd, log_path)
        state["train"].update({"status": "done" if rc == 0 else "failed",
                               "exit_code": rc,
                               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        CONTROL_STATE.write_text(json.dumps(state, indent=2))
        if rc != 0:
            return rc
        print(f"\n  control trained in -> {out}")

    if not out.exists():
        raise SystemExit(f"{out} not found - nothing to score")

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "reports" / f"cm_warmstart_control_{date.today().isoformat()}.json"
    )
    scoring_log = LOG_DIR / f"{date.today().isoformat()}-warmstart-scoring.log"
    cmd = [sys.executable, str(SCORER),
           "--checkpoint", f"W{args.seed}={out}",
           "--out", str(out_path),
           "--device", args.device,
           "--manifest", args.score_manifest]
    print("\nScoring:", " ".join(cmd))
    state["scoring"].update({"status": "running",
                             "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "report": str(out_path)})
    CONTROL_STATE.write_text(json.dumps(state, indent=2))
    rc = run_logged(cmd, scoring_log)
    state["scoring"].update({"status": "done" if rc == 0 else "failed",
                             "exit_code": rc,
                             "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    CONTROL_STATE.write_text(json.dumps(state, indent=2))
    if rc == 0:
        print(f"\nwrote {out_path}")
        print("Adjudicate with:")
        print(f"  python scripts/check_claim.py --control {out_path} "
              f"--claim reports/cm_from_scratch_claim_5seed_<date>.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
