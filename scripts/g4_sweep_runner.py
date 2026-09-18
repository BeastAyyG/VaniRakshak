"""Detached G4 RawBoost training sweep runner.

Runs the pre-registered docs/10 G4 experiment for seeds 42 and 43 back to back.
Each seed is a fully independent `ml.train_cm` process so that a failure in one
cannot corrupt the other.

`--merge-layers 12` is NOT optional. docs/10: "Every model in the G2 table is a
12-layer merge; omitting it trains a different architecture and the comparison
is void." The G4 criterion is measured against the Layer-12 baseline, so a
run without it produces numbers that cannot be compared to anything.
`--unfreeze-top-n` stays at its default 0: G4 is one variable (augmentation),
per ground rule 3.

Safety properties (learned the hard way, see MEMORY):
  * Writes logs OUTSIDE the repo (C:\\vr-save\\g4-logs) - untracked files inside
    the repo have been destroyed before by concurrent git operations.
  * Copies each finished checkpoint to C:\\vr-save\\ckpt immediately on success.
  * Emits a machine-readable status file after every stage.

Usage:
    python scripts/g4_sweep_runner.py [--seeds 42 43]
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


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_seed(seed: int, status_path: Path, state: dict) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_BACKUP.mkdir(parents=True, exist_ok=True)

    out_ckpt = REPO / "data" / f"cm_g4_rawboost_s{seed}.pt"
    log_file = LOG_DIR / f"g4-s{seed}.log"

    state["seeds"][str(seed)] = {
        "status": "running",
        "checkpoint": str(out_ckpt),
        "log": str(log_file),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_status(status_path, state)

    cmd = [
        str(PY), "-u", "-m", "ml.train_cm",
        "--g4-rawboost",
        "--merge-layers", "12",
        "--seed", str(seed),
        "--output", str(out_ckpt),
    ]

    print(f"[runner] launching seed {seed}: {' '.join(cmd)}", flush=True)
    with open(log_file, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(
            cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
        )
    rc = proc.returncode

    entry = state["seeds"][str(seed)]
    entry["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    entry["returncode"] = rc

    if rc == 0 and out_ckpt.exists():
        backup = CKPT_BACKUP / f"VaniRakshak-main__cm_g4_rawboost_s{seed}.pt"
        shutil.copy2(out_ckpt, backup)
        entry["status"] = "ok"
        entry["backup"] = str(backup)
        entry["bytes"] = out_ckpt.stat().st_size
        print(f"[runner] seed {seed} OK -> {backup}", flush=True)
    else:
        entry["status"] = "failed"
        print(f"[runner] seed {seed} FAILED rc={rc}", flush=True)

    write_status(status_path, state)
    return entry["status"] == "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = ap.parse_args()

    status_path = LOG_DIR / "g4_sweep_status.json"
    state = {
        "experiment": "docs/10 G4 RawBoost augmentation",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": {},
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_status(status_path, state)

    ok_all = True
    for seed in args.seeds:
        if not run_seed(seed, status_path, state):
            ok_all = False

    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["all_ok"] = ok_all
    write_status(status_path, state)
    print(f"[runner] sweep complete all_ok={ok_all}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
