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
* Every run is teed to ``handoff/logs/<YYYY-MM-DD>-<cond><seed>.log``. An
  existing log is **archived** to ``<stem>.prev-<HHMMSS>.log`` rather than
  truncated, so re-running a crashed condition cannot destroy the traceback.
* The scoring stage is teed too, to ``handoff/logs/<date>-scoring.log`` --
  otherwise the final eval step is a black box that emits nothing for minutes.
* After run 1, wall time is projected from the measured epoch time and the
  script **stops** so the projection can be reviewed. Pass ``--continue-anyway``
  to go straight through.
* Progress is written to ``handoff/experiment_state.json`` for the two
  companion tools; nothing else reads or depends on it.

Companion tools (both read-only against the experiment):

    python scripts/experiment_progress.py --watch 30   # bar + ETA, whole process
    python scripts/experiment_control.py pause|resume  # freeze/resume, lossless

Usage:
    python scripts/run_from_scratch_experiment.py --device cuda
    python scripts/run_from_scratch_experiment.py --device cuda --continue-anyway
    python scripts/run_from_scratch_experiment.py --score-only
    python scripts/run_from_scratch_experiment.py --continue-anyway --respect-pause
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_state as xs  # noqa: E402  (sibling module, not a package)

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


def _run_order(seeds_raw: str | None, conds_raw: str | None) -> list[tuple[str, int]]:
    """Resolve the run list from CLI overrides, defaulting to the pre-registered set.

    Interleaved by seed, not grouped by condition, so any drift on the training
    box does not line up with the condition.

    This exists because the seed set was hardcoded to [42, 43] while docs/10's
    pre-registered claim is defined over FIVE seeds (S42-S46). The claim could
    therefore never be adjudicated: S44-S46 were unreachable however many runs
    were spent. Widening the seed set is a scope decision, so it belongs in the
    launch command and the run log rather than in a silent source edit.
    """
    def _split(raw: str, flag: str) -> list[str]:
        items = [x.strip() for x in raw.split(",") if x.strip()]
        if not items:
            raise SystemExit(f"{flag}: no values given")
        return items

    seeds = list(SEEDS)
    if seeds_raw is not None:
        parts = _split(seeds_raw, "--seeds")
        try:
            seeds = [int(p) for p in parts]
        except ValueError:
            raise SystemExit(f"--seeds: not integers: {seeds_raw!r}")
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"--seeds: duplicate seeds in {seeds_raw!r}")

    conds = list(CONDITIONS)
    if conds_raw is not None:
        conds = _split(conds_raw, "--conditions")
        unknown = [c for c in conds if c not in CONDITIONS]
        if unknown:
            raise SystemExit(
                f"--conditions: unknown {unknown}; known: {list(CONDITIONS)}"
            )
        if len(set(conds)) != len(conds):
            raise SystemExit(f"--conditions: duplicates in {conds_raw!r}")

    return [(cond, seed) for seed in seeds for cond in conds]

# Must match ml/train_cm.py's --patience default. Used ONLY to project wall
# time (early stopping ends a run at best_epoch + patience). It is deliberately
# not forwarded to the trainer, because that would change the recipe.
PATIENCE = 5


# The layer index the 2026-09-14 run used. It is the default so that the
# existing checkpoint names (cm_scratch_S42.pt, ...) do not move: this is a
# backward-compatible change, not a rename.
DEFAULT_LAYER_INDEX = 12


def checkpoint_for(cond: str, seed: int, layer_index: int = DEFAULT_LAYER_INDEX) -> Path:
    """Path for a condition's checkpoint, keyed by layer index when non-default.

    Why the layer index is in the name: the checkpoint path used to depend only
    on (condition, seed), so a run at a different --layer-index targeted the
    SAME file as the layer-12 run. Two bad outcomes followed, and the second is
    destructive:

      1. `--layer-index 5` silently did nothing -- every checkpoint already
         existed, so the "resume, never overwrite" check skipped all four runs
         and the command exited having trained nothing. docs/10's run plan
         ("from scratch at layer 5", the headline candidate) is written this
         way, so the plan as written could not run.
      2. Anyone who cleared the existing checkpoints to force the run would have
         had the layer-5 output land on data/cm_scratch_S42.pt, destroying the
         layer-12 S42 result the whole project is built on.

    With the suffix, a layer-5 run gets its own file and cannot collide.
    Default-12 runs keep their original names.
    """
    suffix = "" if layer_index == DEFAULT_LAYER_INDEX else f"_L{layer_index}"
    return REPO_ROOT / "data" / f"cm_scratch_{cond}{seed}{suffix}.pt"


def log_for(cond: str, seed: int) -> Path:
    return LOG_DIR / f"{date.today().isoformat()}-{cond.lower()}{seed}.log"


def archive_existing_log(log_path: Path) -> Path | None:
    """Move an existing log aside instead of truncating it.

    ``run_logged`` opens each log with ``"w"``. The runbook flags the
    consequence: if a run crashes and is re-invoked the same day, that
    truncation destroys the only copy of the traceback. Archiving keeps the
    evidence while leaving the live path parseable by the progress tool.
    """
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    stamp = datetime.now().strftime("%H%M%S")
    archived = log_path.with_name(f"{log_path.stem}.prev-{stamp}.log")
    n = 1
    while archived.exists():
        archived = log_path.with_name(f"{log_path.stem}.prev-{stamp}-{n}.log")
        n += 1
    shutil.move(str(log_path), str(archived))
    return archived


def run_logged(cmd: list[str], log_path: Path, on_line=None, on_spawn=None) -> int:
    """Run ``cmd`` with cwd=REPO_ROOT, teeing stdout+stderr to ``log_path``.

    Returns the exit code. Shared by the training loop and the scoring stage so
    that every stage of the experiment leaves a log behind -- the scoring stage
    previously ran with an inherited stdout and produced no artefact at all.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    archived = archive_existing_log(log_path)
    if archived:
        print(f"    archived previous log -> {archived.name}")
    print(f"    logging to {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        if on_spawn:
            on_spawn(proc.pid)
        assert proc.stdout is not None

        def reader():
            for line in proc.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
                if on_line:
                    on_line(line)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        proc.wait()
        t.join(timeout=30)
    return proc.returncode


def run_training(cmd: list[str], log_path: Path,
                 on_spawn=None) -> tuple[int, list[float]]:
    """Run `cmd`, teeing to `log_path`; return (exit code, per-epoch seconds)."""
    epoch_times: list[float] = []

    def collect(line: str) -> None:
        m = EPOCH_RE.search(line)
        if m:
            epoch_times.append(float(m.group(3)))

    rc = run_logged(cmd, log_path, on_line=collect, on_spawn=on_spawn)
    return rc, epoch_times


def project(rp: xs.RunProgress, n_runs: int, patience: int = 5) -> None:
    """Print the wall-time projection from what the first run actually did.

    Mean epoch time, not epoch 1 alone -- epoch 1 carries the one-off warm-up.
    And the horizon is ``best_epoch + patience``, not ``--epochs``: early
    stopping is the normal ending here (S42 stopped at 9 of 10), so assuming 10
    would overstate every remaining run by about 11%.
    """
    times = rp.epoch_seconds or []
    mean_epoch_s = (sum(times) / len(times)) if times else xs.DEFAULT_EPOCH_SECONDS
    planned = rp.total_epochs or xs.EPOCHS_PLANNED_DEFAULT
    n_expected = xs.expected_epochs(planned, rp.best_epoch, patience)
    per_run_s = mean_epoch_s * n_expected
    remaining = max(0, n_runs - 1)
    total_s = per_run_s * n_runs + xs.DEFAULT_SCORING_SECONDS
    eta = datetime.now() + timedelta(
        seconds=per_run_s * remaining + xs.DEFAULT_SCORING_SECONDS)

    actual_s = sum(times)
    print("\n" + "=" * 68)
    print("WALL-TIME PROJECTION (from what run 1 actually did)")
    print("=" * 68)
    print(f"  measured epoch         : {mean_epoch_s / 60.0:.1f} min"
          f"  (mean of {len(times)} epoch(s))")
    if actual_s:
        print(f"  run 1 took             : {actual_s / 3600.0:.2f} h"
              f"  ({rp.epochs_completed} epochs, best epoch {rp.best_epoch})")
    print(f"  expected per run       : {per_run_s / 3600.0:.2f} h"
          f"  ({n_expected} epochs x {mean_epoch_s / 60.0:.1f} min)")
    print(f"  {remaining} runs remaining       : "
          f"{per_run_s * remaining / 3600.0:.2f} h")
    print(f"  scoring (4 checkpoints): {xs.DEFAULT_SCORING_SECONDS / 3600.0:.2f} h")
    print(f"  projected total        : {total_s / 3600.0:.2f} h"
          f"  ({total_s / 86400.0:.2f} days)")
    print(f"  projected finish       : {eta:%Y-%m-%d %H:%M}")
    print("=" * 68)
    print("  track it live:  python scripts/experiment_progress.py --watch 30")


def wait_for_pause_gate(poll_s: float = 15.0) -> None:
    """Hold before starting a stage while ``handoff/PAUSE`` exists.

    Zero-risk by construction: the four runs share no state -- separate
    processes, separate seeds, separate checkpoints -- so an arbitrary delay
    between them cannot change any result. Opt-in via ``--respect-pause`` so a
    scheduled or unattended invocation can never be stalled by a stale gate.
    """
    if not xs.GATE_PATH.exists():
        return
    note = ""
    try:
        note = json.loads(xs.GATE_PATH.read_text(encoding="utf-8")).get("note", "")
    except (OSError, json.JSONDecodeError):
        pass
    print(f"\n  PAUSE gate present ({xs.GATE_PATH}); holding before the next stage."
          + (f"\n    note: {note}" if note else ""))
    print("    clear with: python scripts/experiment_control.py resume")
    while xs.GATE_PATH.exists():
        time.sleep(poll_s)
    print("  gate cleared; continuing.")


def preflight(device: str, allow_cpu_fallback: bool) -> None:
    """Fail in seconds, not after WavLM loads, if ``--device cuda`` cannot work.

    ``ml/train_cm.py`` silently falls back to CPU when CUDA is unavailable
    (``dev = torch.device(device if ... torch.cuda.is_available() else "cpu")``),
    so a CUDA run on a broken environment becomes a multi-day CPU run that still
    emits plausible numbers. The runbook makes this a manual check; doing it here
    means a pre-registered run cannot be lost to it. torch is imported lazily --
    the CPU-only dev box has no torch and must not be broken by this.
    """
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  preflight: torch not importable ({exc}); the trainer will "
              "report the real error.")
        return
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown device"
        print(f"  preflight: CUDA OK -- {name}")
        return
    message = (
        "--device cuda was requested but torch.cuda.is_available() is False.\n"
        "  A driver being present is not the same as a usable CUDA build, and\n"
        "  ml/train_cm.py would silently fall back to CPU -- turning this into a\n"
        "  multi-day run whose wall time is not comparable to the others.\n"
        "  Fix the environment, or pass --allow-cpu-fallback if you truly want CPU."
    )
    if allow_cpu_fallback:
        print(f"  preflight WARNING: {message}")
    else:
        raise SystemExit(message)


def update_run(state: dict, label: str, **fields) -> None:
    """Record per-run progress. Bookkeeping must never break the experiment."""
    try:
        for run in state.get("runs", []):
            if run.get("label") == label:
                run.update(fields)
                break
        xs.update_state(state)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  (state bookkeeping skipped: {exc})", file=sys.stderr)


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
    ap.add_argument("--score-manifest", type=str, default=None,
                    help="Manifest for the final scoring stage. Default is the "
                         "SCORER's default, which is the 20k slice "
                         "(data/eval_benchmark/eval_slice_20k.tsv) -- NOT the full "
                         "eval partition. docs/10's criterion 1 is 'minDCF below "
                         "0.75 on the full eval partition', so grading the claim "
                         "requires --score-manifest A:/cm-corpus/manifests/"
                         "eval_available.tsv. Passed through verbatim.")
    ap.add_argument("--respect-pause", action="store_true",
                    help="Hold before each stage while handoff/PAUSE exists. Opt-in, "
                         "so a scheduled run can never be stalled by a stale gate.")
    ap.add_argument("--allow-cpu-fallback", action="store_true",
                    help="Permit --device cuda where CUDA is unavailable "
                         "(ml/train_cm.py would silently run the whole thing on CPU).")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds, overriding the pre-registered "
                         f"{','.join(str(s) for s in SEEDS)}. Widening this is a "
                         "SCOPE change, not a convenience: docs/10's headline claim "
                         "is defined over five seeds, S42-S46, and this script "
                         "could not previously produce S44-S46 at all.")
    ap.add_argument("--conditions", type=str, default=None,
                    help="Comma-separated condition labels from "
                         f"{{{','.join(CONDITIONS)}}} (default: "
                         f"{','.join(CONDITIONS)}). 'S' alone runs the claim's seeds "
                         "without the no-augmentation ablation, which is not part "
                         "of the claim.")
    args = ap.parse_args()

    run_order = _run_order(args.seeds, args.conditions)

    # The whole point of the experiment: nothing may warm-start these runs.
    assert "--init-weights" not in sys.argv, "this experiment trains from scratch"

    # Fail in seconds, not after WavLM loads, if the environment is wrong.
    if not args.score_only:
        for m in (args.train_manifest, args.dev_manifest):
            if not Path(m).is_file():
                raise SystemExit(f"manifest not found: {m}")
        preflight(args.device, args.allow_cpu_fallback)

    # Resume, never overwrite: an existing checkpoint means that run is done.
    # Always state the layer index, because it is part of the checkpoint's
    # identity now -- "nothing to train" at layer 5 is a different statement
    # from "nothing to train" at layer 12, and silently conflating them is how
    # the layer-5 run plan came to do nothing at all.
    print(f"  layer index: {args.layer_index}"
          f"{' (the 2026-09-14 default)' if args.layer_index == DEFAULT_LAYER_INDEX else ''}")
    pending = []
    for cond, seed in run_order:
        p = checkpoint_for(cond, seed, args.layer_index)
        if p.exists():
            print(f"  {cond}{seed}: {p.name} already exists - skipping (never overwritten)")
        else:
            pending.append((cond, seed))
    if not args.score_only and not pending:
        print(f"  all {len(run_order)} layer-{args.layer_index} runs already have "
              f"checkpoints; nothing to train")
        if args.layer_index == DEFAULT_LAYER_INDEX:
            print("  (to run a different layer index, pass --layer-index N; "
                  "outputs are named cm_scratch_<cond><seed>_L<N>.pt)")

    state = xs.new_state(os.getpid(), sys.argv, runs=run_order)
    xs.update_state(state)

    if not args.score_only:
        projected = False
        for i, (cond, seed) in enumerate(pending):
            if args.respect_pause:
                wait_for_pause_gate()
            out = checkpoint_for(cond, seed, args.layer_index)
            label = f"{cond}{seed}"
            log_path = log_for(cond, seed)
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
            print(f"\n[{i + 1}/{len(pending)}] {label} "
                  f"({' '.join(CONDITIONS[cond]) or 'full augmenter'})")
            print(f"    {' '.join(cmd)}\n")
            update_run(state, label, status="running",
                       started_at=datetime.now().isoformat(timespec="seconds"))

            def on_spawn(pid: int, label: str = label) -> None:
                update_run(state, label, child_pid=pid)

            t0 = time.time()
            rc, _ = run_training(cmd, log_path, on_spawn=on_spawn)
            if rc != 0:
                update_run(state, label, status="failed", exit_code=rc,
                           finished_at=datetime.now().isoformat(timespec="seconds"))
                raise SystemExit(f"run {label} failed with exit code {rc}")
            print(f"\n    {label} finished in {(time.time() - t0) / 3600.0:.2f} h -> {out}")
            update_run(state, label, status="done", exit_code=rc,
                       finished_at=datetime.now().isoformat(timespec="seconds"))

            if not projected:
                projected = True
                # Parse the log rather than trusting the tee'd epoch times: this
                # also yields best_epoch, which the projection needs.
                project(xs.parse_run_log(log_path), len(pending), PATIENCE)
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
    for cond, seed in run_order:
        p = checkpoint_for(cond, seed, args.layer_index)
        if p.exists():
            specs += ["--checkpoint", f"{cond}{seed}={p}"]
        else:
            print(f"  skipping {cond}{seed}: {p} not found")
    if not specs:
        raise SystemExit("no checkpoints to score")

    if args.respect_pause:
        wait_for_pause_gate()

    # Tee the scoring stage as well. It used to inherit stdout, so the last and
    # least predictable step of the experiment left no log at all.
    scoring_log = LOG_DIR / f"{date.today().isoformat()}-scoring.log"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "score_checkpoint_eval.py"),
           *specs, "--out", str(out_path), "--device", args.device]
    if args.score_manifest:
        cmd += ["--manifest", args.score_manifest]
        print(f"  scoring manifest: {args.score_manifest}")
    else:
        print("  scoring manifest: the scorer's default (the 20k slice). "
              "Pass --score-manifest for the full eval partition.")
    print("\nScoring:", " ".join(cmd), "\n")
    state["scoring"].update({
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "report": str(out_path),
    })
    xs.update_state(state)

    rc = run_logged(cmd, scoring_log)
    state["scoring"].update({
        "status": "done" if rc == 0 else "failed",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "exit_code": rc,
    })
    xs.update_state(state)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
