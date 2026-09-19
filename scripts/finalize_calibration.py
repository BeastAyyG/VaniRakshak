"""Chain: wait for the G4 + control sweep, then calibrate every final checkpoint.

Completes the pipeline end to end without a human in the loop:

    train G4 (s42, s43) -> train control (s42, s43) -> calibrate all four
                                                    -> adjudicate_g4.py

Calibration is deliberately LAST. Platt coefficients are fitted to the scores
of one specific checkpoint, so calibrating mid-training produces numbers that
are stale the moment the next epoch improves minDCF (which it did twice during
this run: epoch 6 then epoch 7). Writing in place is safe only once training
has stopped -- so this waits, and that ordering is the whole point.

Usage:
    python scripts/finalize_calibration.py [--wait-hours 40] [--no-wait]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\VaniRakshak")
PY = REPO / ".venv" / "Scripts" / "python.exe"
LOG_DIR = Path(r"C:\vr-save\g4-logs")
CONTROL_STATUS = LOG_DIR / "control_status.json"

TARGETS = [
    REPO / "data" / "cm_g4_rawboost_s42.pt",
    REPO / "data" / "cm_g4_rawboost_s43.pt",
    REPO / "data" / "cm_g4_control_s42.pt",
    REPO / "data" / "cm_g4_control_s43.pt",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_control(wait_hours: float, poll: int = 120) -> bool:
    deadline = time.time() + wait_hours * 3600.0
    log("waiting for the G4 + control sweep to finish")
    while time.time() < deadline:
        try:
            st = json.loads(CONTROL_STATUS.read_text(encoding="utf-8"))
        except Exception:
            st = {}
        if st.get("finished_at"):
            log(f"control sweep finished at {st['finished_at']} (all_ok={st.get('all_ok')})")
            return True
        time.sleep(poll)
    log("wait timed out -- calibrating whatever exists")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-hours", type=float, default=40.0)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--dev-limit", type=int, default=12000)
    ap.add_argument("--eval-limit", type=int, default=20000)
    args = ap.parse_args()

    if not args.no_wait:
        wait_for_control(args.wait_hours)

    # Give the last training process a moment to flush its final save.
    time.sleep(60)

    done, missing = [], []
    for ck in TARGETS:
        if not ck.exists():
            log(f"SKIP (absent) {ck.name}")
            missing.append(ck.name)
            continue
        log(f"calibrating {ck.name} (in place -- training has stopped)")
        rc = subprocess.run(
            [str(PY), "-u", str(REPO / "scripts" / "fit_cm_calibration.py"),
             "--checkpoint", str(ck),
             "--dev-limit", str(args.dev_limit),
             "--eval-limit", str(args.eval_limit)],
            cwd=str(REPO),
        ).returncode
        (done if rc == 0 else missing).append(ck.name)
        log(f"  {ck.name} -> rc={rc}")

    log(f"calibrated: {done}")
    if missing:
        log(f"not calibrated: {missing}")

    # -- Adjudicate -------------------------------------------------------
    # Calibration cannot move EER or minDCF (it is a monotone map of the LLR),
    # so the verdict is identical before and after this step. What changes is
    # CLLR/actDCF, which is exactly the G2 criterion. Run the verdict anyway so
    # the pipeline ends with a number rather than a suggested command.
    g4 = [c for c in TARGETS[:2] if c.exists()]
    control = [c for c in TARGETS[2:] if c.exists()]
    if not g4:
        log("no G4 checkpoints -- nothing to adjudicate")
        return 1

    log("adjudicating G4 against the pre-registered criterion (full 20k eval)")
    rc = subprocess.run(
        [str(PY), "-u", str(REPO / "scripts" / "adjudicate_g4.py"),
         "--g4", *[str(c) for c in g4],
         "--control", *[str(c) for c in control]],
        cwd=str(REPO),
    ).returncode
    log(f"adjudicate_g4.py -> rc={rc}")

    # Persist the verdict next to the other artifacts.
    verdicts = sorted(
        (REPO / "reports").glob("g4_adjudication_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if verdicts:
        latest = json.loads(verdicts[-1].read_text(encoding="utf-8"))
        log(f"OVERALL: {latest.get('OVERALL')}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
