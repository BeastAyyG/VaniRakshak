"""Tiny local dashboard for the G4 / control pipeline.

Reads the live status files and training logs and serves them as JSON to a
single-page dashboard, so progress can be seen without tailing logs.

This is a local dev/ops tool, NOT part of the product frontend (that is
``frontend/``). Stdlib only, no dependencies.

Usage:
    python scripts/serve_dashboard.py [--port 8765]

Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(r"C:\VaniRakshak")
LOG_DIR = Path(r"C:\vr-save\g4-logs")
HTML = REPO / "progress-dashboard.html"

G4_STATUS = LOG_DIR / "g4_sweep_status.json"
CONTROL_STATUS = LOG_DIR / "control_status.json"

# Measured: seed 42 ran 10 epochs in 4 h 15 m.
AVG_EPOCH_S = 1560.0
FINALIZE_S = 2400.0
EPOCH_BUDGET = 10

RE_EPOCH_DONE = re.compile(r"Epoch\s+(\d+)/10\s+\[")
RE_STEP = re.compile(r"Epoch\s+(\d+):\s+step\s+(\d+)/(\d+)")
# NOTE: the two log lines do not use the same separator. "New best checkpoint"
# writes `minDCF=0.9997, EER=16.189%` while the per-epoch summary writes
# `Dev minDCF: 0.9997, EER: 16.189%`. Accept either.
RE_BEST = re.compile(
    r"New best checkpoint at epoch (\d+) saved.*?minDCF[:=]\s*([\d.]+),\s*EER[:=]\s*([\d.]+)%"
)
RE_FINISH = re.compile(
    r"Training finished\. Best epoch (\d+): dev minDCF[:=]\s*([\d.]+),\s*dev EER[:=]\s*([\d.]+)%"
)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def tail_text(path: Path, max_bytes: int = 200_000) -> str:
    try:
        data = path.read_bytes()[-max_bytes:]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_log(path: Path) -> dict:
    """Return progress + best-epoch info parsed from a training log."""
    txt = tail_text(path)
    done = [int(m) for m in RE_EPOCH_DONE.findall(txt)]
    epochs_done = max(done) if done else 0

    steps = RE_STEP.findall(txt)
    frac = 0.0
    if steps:
        cur_epoch, cur_step, total = int(steps[-1][0]), int(steps[-1][1]), int(steps[-1][2])
        # "Epoch M: step S" is logged *during* epoch M. If we have completed
        # M-1 epochs, then M is the one in flight.
        if cur_epoch == epochs_done + 1:
            frac = min(1.0, cur_step / max(1, total))

    fin = RE_FINISH.search(txt)
    best = None
    if fin:
        best = {
            "epoch": int(fin.group(1)),
            "dev_mindcf": float(fin.group(2)),
            "dev_eer": float(fin.group(3)),
        }
    else:
        b = RE_BEST.findall(txt)
        if b:
            best = {
                "epoch": int(b[-1][0]),
                "dev_mindcf": float(b[-1][1]),
                "dev_eer": float(b[-1][2]),
            }

    early = "Early stopping" in txt
    return {
        "epochs_done": epochs_done,
        "frac": frac,
        "progress": (epochs_done + frac) / EPOCH_BUDGET,
        "best": best,
        "early_stop": early,
    }


def build_stage(name: str, group: str, seed: int, status_obj: dict,
                log_path: Path) -> dict:
    entry = (status_obj.get("seeds") or {}).get(str(seed), {})
    st = entry.get("status", "pending")
    lg = parse_log(log_path) if log_path.exists() else {
        "epochs_done": 0, "frac": 0.0, "progress": 0.0, "best": None,
        "early_stop": False,
    }

    if st == "ok":
        status, progress = "done", 1.0
    elif st == "failed":
        status, progress = "failed", lg["progress"]
    elif st == "running":
        status, progress = "running", lg["progress"]
    else:
        status, progress = "pending", 0.0

    return {
        "name": name,
        "group": group,
        "seed": seed,
        "status": status,
        "progress": round(progress, 4),
        "epochs_done": lg["epochs_done"],
        "epoch_frac": round(lg["frac"], 3),
        "best": lg["best"],
        "early_stop": lg["early_stop"] and status == "done",
        "finished_at": entry.get("finished_at"),
        "log": str(log_path),
    }


def build_status() -> dict:
    g4 = read_json(G4_STATUS)
    ctl = read_json(CONTROL_STATUS)

    stages = [
        build_stage("G4 seed 42", "g4", 42, g4, LOG_DIR / "g4-s42.log"),
        build_stage("G4 seed 43", "g4", 43, g4, LOG_DIR / "g4-s43.log"),
        build_stage("Control seed 42", "control", 42, ctl, LOG_DIR / "control-s42.log"),
        build_stage("Control seed 43", "control", 43, ctl, LOG_DIR / "control-s43.log"),
    ]

    train_done = all(s["status"] == "done" for s in stages)
    finalize = {
        "name": "Calibrate + adjudicate",
        "group": "finalize",
        "seed": None,
        "status": "done" if (train_done and any(
            (REPO / "reports").glob("g4_adjudication_2seed*.json"))) else
                  ("running" if train_done else "pending"),
        "progress": 1.0 if train_done else 0.0,
        "epochs_done": EPOCH_BUDGET if train_done else 0,
        "epoch_frac": 0.0,
        "best": None,
        "early_stop": False,
        "finished_at": None,
        "log": None,
    }
    stages.append(finalize)

    # Overall: 4 training runs carry the weight, finalize is a small tail.
    train_p = sum(s["progress"] for s in stages[:4]) / 4.0
    overall = train_p * 0.93 + finalize["progress"] * 0.07

    # ETA from measured epoch cost.
    remaining_epochs = sum(
        max(0.0, EPOCH_BUDGET - (s["epochs_done"] + s["epoch_frac"]))
        for s in stages[:4] if s["status"] != "done"
    )
    eta_s = remaining_epochs * AVG_EPOCH_S
    if finalize["status"] == "pending":
        eta_s += FINALIZE_S
    eta = datetime.now() + timedelta(seconds=eta_s)

    results = {"g4_seeds": [], "ood": [], "verdict": None}
    adj = REPO / "reports" / "g4_adjudication_2seed_20260919.json"
    if adj.exists():
        d = read_json(adj)
        results["verdict"] = d.get("OVERALL")
        for g in d.get("g4_arm", []):
            results["g4_seeds"].append({
                "seed": g.get("seed"),
                "epoch": g.get("best_epoch"),
                "pooled": g.get("pooled_eer_pct"),
                "mindcf": g.get("pooled_minDCF"),
                "a19": g.get("per_attack_eer_pct", {}).get("A19"),
                "a20": g.get("per_attack_eer_pct", {}).get("A20"),
            })

    ood_files = {
        "Pre-Aug L12 (none)": ("eer_ci_pre_aug_ITW_clean.json",
                               "eer_ci_pre_aug_ITW_full_call.json"),
        "Post-Aug L12 (codec)": ("eer_ci_post_aug_ITW_clean.json",
                                 "eer_ci_post_aug_ITW_full_call.json"),
        "AttM (codec)": ("eer_ci_attm_ITW_clean.json",
                         "eer_ci_attm_ITW_full_call.json"),
        "G4 s42 (codec+RawBoost)": ("eer_ci_g4_s42_ITW.json",
                                    "eer_ci_g4_s42_ITW_full_call.json"),
    }
    for label, (fc, ff) in ood_files.items():
        a = read_json(REPO / "reports" / fc).get("pooled")
        b = read_json(REPO / "reports" / ff).get("pooled")
        if a and b:
            results["ood"].append({
                "model": label,
                "clean": a.get("eer_pct"),
                "clean_ci": a.get("ci95_pct"),
                "full_call": b.get("eer_pct"),
                "full_call_ci": b.get("ci95_pct"),
            })

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_percent": round(overall * 100, 1),
        "eta": eta.strftime("%H:%M, %d %b"),
        "remaining_hours": round(eta_s / 3600.0, 1),
        "stages": stages,
        "results": results,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/status"):
            body = json.dumps(build_status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        try:
            body = HTML.read_bytes()
        except Exception:
            body = b"<h1>progress-dashboard.html missing</h1>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard: http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
