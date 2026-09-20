"""Chained arm-B run for the docs/20 augmentation-off control.

docs/20 asks whether the codec/telephony augmenter causes the out-of-domain
regression measured in docs/18. It is a matched, flag-flipped pair:

    arm A = codec ON,  RawBoost OFF   <- ALREADY TRAINED. This is exactly the
                                         docs/10 G4 control (cm_g4_control_s42/
                                         s43.pt): identical architecture, seeds,
                                         data and recipe, only --g4-rawboost
                                         omitted. Reused, not retrained.
    arm B = codec OFF, RawBoost OFF   <- this script.

So the whole experiment costs two runs, not four.

Only ONE variable differs between the arms, per ground rule 3: the
codec/telephony augmenter. Everything else is inherited from the G4 recipe:
  * --merge-layers 12  (docs/10: mandatory, or "the comparison is void")
  * --unfreeze-top-n 0 (default -- fine-tuning is a separate lever)
  * same seed(s), same init weights, same epochs

`--no-codec-augment` is additive: default is ON, so passing nothing reproduces
today's behaviour exactly. It also stamps `codec_augment` into the checkpoint,
which is how the arms stay distinguishable after the fact -- arm A predates the
flag and carries no such key, which must be read as "codec was ON", not as
"unknown".

Usage:
    python scripts/docs20_armb_runner.py [--seeds 42 43]

Checkpoints are written inside the repo (gitignored) and copied to
C:\\vr-save\\ckpt immediately on success -- untracked files inside the repo have
been destroyed by git operations before.
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

STATUS = LOG_DIR / "docs20_status.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_status(state: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_seed(seed: int, state: dict) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_BACKUP.mkdir(parents=True, exist_ok=True)

    out_ckpt = REPO / "data" / f"cm_docs20_armb_s{seed}.pt"
    log_file = LOG_DIR / f"docs20-s{seed}.log"

    state["seeds"][str(seed)] = {
        "status": "running",
        "checkpoint": str(out_ckpt),
        "log": str(log_file),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_status(state)

    # The single variable: --no-codec-augment. RawBoost is off in both arms.
    cmd = [
        str(PY), "-u", "-m", "ml.train_cm",
        "--merge-layers", "12",
        "--seed", str(seed),
        "--no-codec-augment",
        "--output", str(out_ckpt),
    ]
    log(f"launching arm B seed {seed}: {' '.join(cmd)}")

    with open(log_file, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
    rc = proc.returncode

    e = state["seeds"][str(seed)]
    e["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    e["returncode"] = rc
    if rc == 0 and out_ckpt.exists():
        backup = CKPT_BACKUP / f"VaniRakshak-main__cm_docs20_armb_s{seed}.pt"
        shutil.copy2(out_ckpt, backup)
        e.update(status="ok", backup=str(backup), bytes=out_ckpt.stat().st_size)
        log(f"arm B seed {seed} OK -> {backup}")
    else:
        e["status"] = "failed"
        log(f"arm B seed {seed} FAILED rc={rc}")
    write_status(state)
    return e["status"] == "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = ap.parse_args()

    state = {
        "experiment": "docs/20 augmentation-off control -- arm B",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": {},
        "arm_a": {
            "source": "reused, not retrained: the docs/10 G4 control",
            "checkpoints": [
                str(REPO / "data" / "cm_g4_control_s42.pt"),
                str(REPO / "data" / "cm_g4_control_s43.pt"),
            ],
            "codec_augment": True,
            "note": "predates the --no-codec-augment flag, so the key is "
                    "absent from these checkpoints; absent means ON",
        },
        "arm_b": {"codec_augment": False, "g4_rawboost": False},
        "differs_by": ["codec/telephony augmenter"],
        "identical": ["merge_layers=12", "unfreeze_top_n=0", "seeds",
                      "data", "init weights", "epochs", "g4_rawboost=off"],
        "seeds": {},
    }

    # Merge, do not reset. Extra seeds (44+) are launched as later separate
    # invocations; starting from an empty "seeds" would discard the original
    # 42/43 entries that the dashboard reads.
    if STATUS.exists():
        try:
            prev = json.loads(STATUS.read_text(encoding="utf-8"))
            state["seeds"] = prev.get("seeds", {})
            state["created_at"] = prev.get("created_at", state["created_at"])
            log(f"resuming status with {len(state['seeds'])} prior seed(s)")
        except Exception as exc:  # noqa: BLE001 - corrupt status is not fatal
            log(f"could not read prior status ({exc}); starting fresh")

    write_status(state)

    ok = True
    for seed in args.seeds:
        if not run_seed(seed, state):
            ok = False

    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["all_ok"] = ok
    write_status(state)
    log(f"arm B sweep complete all_ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
