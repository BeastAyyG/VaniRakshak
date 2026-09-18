"""Chained control run for the docs/10 G4 experiment.

G4 is `RawBoost augmentation ON`. Without an otherwise-identical run with it
OFF, the experiment cannot be read at all: you cannot tell whether any
movement in A19/A20 came from the augmentation or from the seed.

So this waits for the G4 sweep to release the GPU, then trains the same
architecture, same seeds, same data -- with `--g4-rawboost` omitted.

Only ONE variable differs, per ground rule 3: augmentation.
  * --merge-layers 12  (docs/10: mandatory, or "the comparison is void")
  * --unfreeze-top-n 0 (default -- fine-tuning is a separate lever)
  * same seed(s), same init weights, same epochs

Usage:
    python scripts/g4_control_runner.py [--seeds 42 43] [--wait-hours 24]
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

G4_STATUS = LOG_DIR / "g4_sweep_status.json"
CONTROL_STATUS = LOG_DIR / "control_status.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_g4(wait_hours: float, poll: int = 60) -> bool:
    """Block until the G4 sweep writes its finished marker."""
    deadline = time.time() + wait_hours * 3600.0
    log(f"waiting for G4 sweep to finish (status: {G4_STATUS})")
    while time.time() < deadline:
        try:
            st = json.loads(G4_STATUS.read_text(encoding="utf-8"))
        except Exception:
            st = {}
        if st.get("finished_at"):
            log(f"G4 sweep finished at {st['finished_at']} (all_ok={st.get('all_ok')})")
            return True
        time.sleep(poll)
    log("G4 wait timed out -- proceeding anyway")
    return False


def run_seed(seed: int, state: dict) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_BACKUP.mkdir(parents=True, exist_ok=True)

    out_ckpt = REPO / "data" / f"cm_g4_control_s{seed}.pt"
    log_file = LOG_DIR / f"control-s{seed}.log"

    state["seeds"][str(seed)] = {
        "status": "running",
        "checkpoint": str(out_ckpt),
        "log": str(log_file),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_status(state)

    # NOTE: no --g4-rawboost. That single omission is the entire experiment.
    cmd = [
        str(PY), "-u", "-m", "ml.train_cm",
        "--merge-layers", "12",
        "--seed", str(seed),
        "--output", str(out_ckpt),
    ]
    log(f"launching control seed {seed}: {' '.join(cmd)}")

    with open(log_file, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
    rc = proc.returncode

    e = state["seeds"][str(seed)]
    e["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    e["returncode"] = rc
    if rc == 0 and out_ckpt.exists():
        backup = CKPT_BACKUP / f"VaniRakshak-main__cm_g4_control_s{seed}.pt"
        shutil.copy2(out_ckpt, backup)
        e.update(status="ok", backup=str(backup), bytes=out_ckpt.stat().st_size)
        log(f"control seed {seed} OK -> {backup}")
    else:
        e["status"] = "failed"
        log(f"control seed {seed} FAILED rc={rc}")
    write_status(state)
    return e["status"] == "ok"


def write_status(state: dict) -> None:
    CONTROL_STATUS.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_STATUS.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--wait-hours", type=float, default=24.0)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    state = {
        "experiment": "docs/10 G4 CONTROL -- everything identical, augmentation OFF",
        "differs_from_g4_by": ["g4_rawboost disabled"],
        "identical_to_g4": ["merge_layers=12", "unfreeze_top_n=0", "seeds", "data", "init weights"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": {},
    }
    write_status(state)

    if not args.no_wait:
        wait_for_g4(args.wait_hours)

    ok = True
    for seed in args.seeds:
        if not run_seed(seed, state):
            ok = False

    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["all_ok"] = ok
    write_status(state)
    log(f"control sweep complete all_ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
