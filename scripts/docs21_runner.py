"""Train the intermediate augmentation doses for the docs/21 sweep.

docs/21 asks whether the optimum augmentation strength is interior or at an
endpoint. The endpoints already exist and are NOT retrained here:

    dose 0.0  -> docs/20 arm B      (cm_docs20_armb_s{42,43}.pt)
    dose 1.0  -> docs/10 G4 control (cm_g4_control_s{42,43}.pt)

so only intermediate doses cost GPU time. Passing 0.0 or 1.0 is rejected
rather than silently duplicating an existing arm.

Only ONE variable moves between arms (ground rule 3): --codec-augment-strength.
Everything else is inherited from the G4 recipe:
  * --merge-layers 12  (mandatory per docs/10)
  * --unfreeze-top-n 0 (fine-tuning is a separate lever)
  * --g4-rawboost off  (separate lever again)
  * same seeds, init weights, data, epochs

Usage:
    python scripts/docs21_runner.py --dry-run
    python scripts/docs21_runner.py --strengths 0.25 0.5 --seeds 42 43

Checkpoints land in the repo (gitignored) and are copied to C:\\vr-save\\ckpt
on success -- untracked files inside the repo have been destroyed by git
operations before.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\VaniRakshak")
PY = REPO / ".venv" / "Scripts" / "python.exe"
LOG_DIR = Path(r"C:\vr-save\g4-logs")
CKPT_BACKUP = Path(r"C:\vr-save\ckpt")

STATUS = LOG_DIR / "docs21_status.json"

# Already-trained endpoints. Documented here so the runner can refuse to
# duplicate them and so the adjudicator can find them without guesswork.
ENDPOINTS = {
    0.0: "docs/20 arm B -- data/cm_docs20_armb_s{seed}.pt",
    1.0: "docs/10 G4 control -- data/cm_g4_control_s{seed}.pt",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dose_tag(strength: float) -> str:
    """0.25 -> '025'. Used in filenames, so it must be filesystem-safe and
    sortable -- a bare '0.25' in a filename invites locale/decimal bugs."""
    return f"{int(round(strength * 100)):03d}"


def checkpoint_path(strength: float, seed: int) -> Path:
    return REPO / "data" / f"cm_docs21_pct{dose_tag(strength)}_s{seed}.pt"


def write_status(state: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_one(strength: float, seed: int, state: dict) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_BACKUP.mkdir(parents=True, exist_ok=True)

    out_ckpt = checkpoint_path(strength, seed)
    tag = f"pct{dose_tag(strength)}-s{seed}"
    log_file = LOG_DIR / f"docs21-{tag}.log"

    key = f"{strength}_{seed}"
    state["runs"][key] = {
        "status": "running",
        "strength": strength,
        "seed": seed,
        "checkpoint": str(out_ckpt),
        "log": str(log_file),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_status(state)

    cmd = [
        str(PY), "-u", "-m", "ml.train_cm",
        "--merge-layers", "12",
        "--seed", str(seed),
        "--codec-augment-strength", str(strength),
        "--output", str(out_ckpt),
    ]
    log(f"launching dose {strength} seed {seed}: {' '.join(cmd)}")

    with open(log_file, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=fh,
                              stderr=subprocess.STDOUT)
    rc = proc.returncode

    e = state["runs"][key]
    e["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    e["returncode"] = rc
    if rc == 0 and out_ckpt.exists():
        backup = CKPT_BACKUP / f"VaniRakshak__{out_ckpt.name}"
        shutil.copy2(out_ckpt, backup)
        e.update(status="ok", backup=str(backup),
                 bytes=out_ckpt.stat().st_size)
        log(f"dose {strength} seed {seed} OK -> {backup}")
    else:
        e["status"] = "failed"
        log(f"dose {strength} seed {seed} FAILED rc={rc}")
    write_status(state)
    return e["status"] == "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without training")
    args = ap.parse_args()

    # Reject bad doses before writing a status file or launching anything.
    for s in args.strengths:
        if not 0.0 < s < 1.0:
            where = ENDPOINTS.get(s)
            hint = (f" that is the {where} endpoint -- reuse it, do not "
                    f"retrain it") if where else " must be in (0, 1)"
            print(f"ERROR: strength {s}{hint}", file=sys.stderr)
            return 2

    plan = [(s, d) for s in args.strengths for d in args.seeds]
    n = len(plan)
    print(f"docs/21 dose-response: {n} run(s) to train")
    for s, d in plan:
        print(f"  dose {s:<5} seed {d}  ->  {checkpoint_path(s, d).name}")
    print("\nReused, not retrained:")
    for dose, where in sorted(ENDPOINTS.items()):
        print(f"  dose {dose:<5} {where}")
    # Measured on docs/20's early-stopped runs: 2h39m and 3h24m.
    print(f"\nEstimated GPU: {n} x ~2.5-3.5 h = ~{n * 3:.0f} h "
          f"(rough; early stopping shortens it)")

    if args.dry_run:
        print("\n--dry-run: nothing launched.")
        return 0

    state = {
        "experiment": "docs/21 augmentation dose-response",
        "variable": "--codec-augment-strength",
        "endpoints_reused": {str(k): v for k, v in ENDPOINTS.items()},
        "identical": ["merge_layers=12", "unfreeze_top_n=0", "seeds", "data",
                      "init weights", "epochs", "g4_rawboost=off"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": {},
    }

    # Merge, do not reset. Doses are launched as separate invocations (one per
    # strength) so they can be chained and observed between runs; starting
    # from an empty "runs" would silently discard the earlier dose's status
    # and make a completed run look like it never happened.
    if STATUS.exists():
        try:
            prev = json.loads(STATUS.read_text(encoding="utf-8"))
            state["runs"] = prev.get("runs", {})
            state["created_at"] = prev.get("created_at", state["created_at"])
            log(f"resuming status with {len(state['runs'])} prior run(s)")
        except Exception as exc:  # noqa: BLE001 - corrupt status is not fatal
            log(f"could not read prior status ({exc}); starting fresh")

    write_status(state)

    ok = True
    for s, d in plan:
        if not run_one(s, d, state):
            ok = False

    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["all_ok"] = ok
    write_status(state)
    log(f"docs/21 sweep complete all_ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
