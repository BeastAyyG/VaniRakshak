"""Run the pre-registered from-scratch experiment (docs/10, 2026-09-13).

Four runs, fixed order, one variable:

    S42  full augmenter, from scratch, seed 42
    B42  --p-codec 0 --p-bandpass 0, seed 42
    S43  full augmenter, from scratch, seed 43
    B43  --p-codec 0 --p-bandpass 0, seed 43

Interleaved by seed so that any drift on the training box (thermals, a second
job appearing) does not line up with the condition.

Conventions this script enforces rather than leaves to memory:

* **No ``--init-weights`` is ever passed.** That is the point of the experiment;
  the flag now defaults to ``None``, but it is asserted here too.
* Checkpoints are named ``data/cm_scratch_<COND><SEED>.pt`` and are **never
  overwritten** -- an existing file aborts the run.
* Every run is teed to ``handoff/logs/<YYYY-MM-DD>-<cond><seed>.log``.
* After run 1, wall time is projected from the measured epoch time and the
  script **stops** so the projection can be reviewed. Pass ``--continue-anyway``
  to go straight through.

Usage:
    python scripts/run_from_scratch_experiment.py --device cuda
    python scripts/run_from_scratch_experiment.py --device cuda --continue-anyway
    python scripts/run_from_scratch_experiment.py --score-only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN = REPO_ROOT / "ml" / "train_cm.py"
LOG_DIR = REPO_ROOT / "handoff" / "logs"

# Epoch line from ml/train_cm.py:
#   "Epoch  1/10 [123.4s, val: 45.6s] - Loss: ... "
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)\s+\[([\d.]+)s")

CONDITIONS = {
    # label: extra flags beyond the shared base
    "S": [],
    "B": ["--p-codec", "0", "--p-bandpass", "0"],
}
SEEDS = [42, 43]
# S42, B42, S43, B43 -- interleaved by seed.
RUN_ORDER = [(cond, seed) for seed in SEEDS for cond in ("S", "B")]


def checkpoint_for(cond: str, seed: int) -> Path:
    return REPO_ROOT / "data" / f"cm_scratch_{cond}{seed}.pt"


def log_for(cond: str, seed: int) -> Path:
    return LOG_DIR / f"{date.today().isoformat()}-{cond.lower()}{seed}.log"


def run_training(cmd: list[str], log_path: Path) -> tuple[int, list[float]]:
    """Run `cmd`, teeing to `log_path`; return (exit code, per-epoch seconds)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_times: list[float] = []
    print(f"    logging to {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None

        def reader():
            for line in proc.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
                m = EPOCH_RE.search(line)
                if m:
                    epoch_times.append(float(m.group(3)))

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        proc.wait()
        t.join(timeout=30)
    return proc.returncode, epoch_times


def project(total_epochs: int, first_epoch_s: float, n_runs: int) -> None:
    per_run = first_epoch_s * total_epochs
    remaining = max(0, n_runs - 1)
    total = per_run * n_runs
    print("\n" + "=" * 68)
    print("WALL-TIME PROJECTION (from epoch 1 of run 1)")
    print("=" * 68)
    print(f"  epoch 1 duration      : {first_epoch_s / 60.0:.1f} min")
    print(f"  per run ({total_epochs} epochs) : {per_run / 3600.0:.1f} h")
    print(f"  {remaining} runs remaining     : {per_run * remaining / 3600.0:.1f} h")
    print(f"  projected total ({n_runs} runs): {total / 3600.0:.1f} h"
          f"  ({total / 86400.0:.1f} days)")
    print("=" * 68)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--layer-index", type=int, default=12)
    # train.tsv / dev.tsv are what exists in A:\cm-corpus\manifests (mirrored to
    # .corpus-manifests/). There is no train_available.tsv or dev_available.tsv.
    ap.add_argument("--train-manifest", type=str,
                    default=r"A:\cm-corpus\manifests\train.tsv")
    ap.add_argument("--dev-manifest", type=str,
                    default=r"A:\cm-corpus\manifests\dev.tsv")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--continue-anyway", action="store_true",
                    help="Do not stop after the wall-time projection.")
    ap.add_argument("--score-only", action="store_true",
                    help="Skip training; just score whatever checkpoints exist.")
    ap.add_argument("--out", type=str, default=None,
                    help="Report path (default reports/cm_from_scratch_<date>.json)")
    args = ap.parse_args()

    # The whole point of the experiment: nothing may warm-start these runs.
    assert "--init-weights" not in sys.argv, "this experiment trains from scratch"

    # Fail in seconds, not after WavLM loads, if the manifests are not here.
    if not args.score_only:
        for m in (args.train_manifest, args.dev_manifest):
            if not Path(m).is_file():
                raise SystemExit(f"manifest not found: {m}")

    # Resume, never overwrite: an existing checkpoint means that run is done.
    pending = []
    for cond, seed in RUN_ORDER:
        p = checkpoint_for(cond, seed)
        if p.exists():
            print(f"  {cond}{seed}: {p.name} already exists - skipping (never overwritten)")
        else:
            pending.append((cond, seed))
    if not args.score_only and not pending:
        print("  all four runs already have checkpoints; nothing to train")

    if not args.score_only:
        projected = False
        for i, (cond, seed) in enumerate(pending):
            out = checkpoint_for(cond, seed)
            cmd = [
                sys.executable, str(TRAIN),
                "--train-manifest", args.train_manifest,
                "--dev-manifest", args.dev_manifest,
                "--output", str(out),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--layer-index", str(args.layer_index),
                "--device", args.device,
                "--seed", str(seed),
                "--workers", str(args.workers),
                *CONDITIONS[cond],
            ]
            print(f"\n[{i + 1}/{len(pending)}] {cond}{seed} "
                  f"({' '.join(CONDITIONS[cond]) or 'full augmenter'})")
            print(f"    {' '.join(cmd)}\n")
            t0 = time.time()
            rc, epoch_times = run_training(cmd, log_for(cond, seed))
            if rc != 0:
                raise SystemExit(f"run {cond}{seed} failed with exit code {rc}")
            print(f"\n    {cond}{seed} finished in {(time.time() - t0) / 3600.0:.2f} h -> {out}")

            if not projected and epoch_times:
                projected = True
                project(args.epochs, epoch_times[0], len(pending))
                if not args.continue_anyway:
                    print("\nStopping after the first run, as pre-registered. "
                          "Review the projection above, then re-run with "
                          "--continue-anyway; the finished run is detected by its "
                          "checkpoint and will be skipped.\n")
                    return 0

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "reports" / f"cm_from_scratch_{date.today().isoformat()}.json"
    )
    specs = []
    for cond, seed in RUN_ORDER:
        p = checkpoint_for(cond, seed)
        if p.exists():
            specs += ["--checkpoint", f"{cond}{seed}={p}"]
        else:
            print(f"  skipping {cond}{seed}: {p} not found")
    if not specs:
        raise SystemExit("no checkpoints to score")

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "score_checkpoint_eval.py"),
           *specs, "--out", str(out_path), "--device", args.device]
    print("\nScoring:", " ".join(cmd), "\n")
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
