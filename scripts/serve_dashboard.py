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
DOCS20_STATUS = LOG_DIR / "docs20_status.json"

# Measured: seed 42 ran 10 epochs in 4 h 15 m.
# Measured, not guessed: docs/20 arm B epoch 1 = 1369.7s wall clock
# (17:15:50 -> 17:38:53). Note the log bracket is [total_including_val, val] --
# adding 1369.7 + 589.5 gives 32.7 min, which is WRONG and will over-project
# the ETA by ~45 %.
AVG_EPOCH_S = 1370.0
FINALIZE_S = 2400.0
# docs/20 tail: 8 scorings (2 arms x 2 seeds x {eval, ITW clean, ITW full_call}
# minus the 4 arm-A ITW ones already cached) at roughly 7 min each.
DOCS20_SCORE_S = 3600.0
# docs/21 tail: 8 scorings (4 new checkpoints x {eval, ITW full_call}); the
# two endpoint doses already have cached reports. Measured ~5 min per ITW
# full_call scoring plus bootstrap.
DOCS21_SCORE_S = 2700.0
EPOCH_BUDGET = 10

DOCS21_STATUS = LOG_DIR / "docs21_status.json"

# Endpoint dev figures, read from the earlier experiments' training logs
# (full 140,950-record dev split). They are deliberately NOT read from the
# checkpoints: fit_cm_calibration.py used to overwrite dev_eer/dev_min_dcf
# with its own 12,000-record subsample measurement, so the G4-control
# checkpoints store 16.297% where the log says 15.856%.
DOCS21_ENDPOINT_DEV = {
    0.0: {42: {"epoch": 2, "dev_mindcf": 0.7105, "dev_eer": 6.728},
          43: {"epoch": 4, "dev_mindcf": 0.7843, "dev_eer": 7.297}},
    1.0: {42: {"epoch": 2, "dev_mindcf": 0.9965, "dev_eer": 15.856},
          43: {"epoch": 3, "dev_mindcf": 0.9920, "dev_eer": 13.983}},
}

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


# The four checkpoints finalize_calibration.py walks, in order.
CALIB_STEMS = (
    "cm_g4_rawboost_s42",
    "cm_g4_rawboost_s43",
    "cm_g4_control_s42",
    "cm_g4_control_s43",
)


def finalize_state(ctl: dict, train_done: bool) -> dict:
    """Derive the finalize stage from artifacts it actually produces.

    Previously this keyed off `any(g4_adjudication_2seed*.json)`, which matched
    the *pre-control* report written hours earlier -- so the stage read "done"
    the instant training stopped, while calibration had not even started.

    Real signals, both written after training ends:
      reports/calibration_<stem>_<YYYYMMDD_HHMMSS>.json  (one per checkpoint)
      reports/g4_adjudication_<YYYYMMDD>_<HHMMSS>.json   (one, last)
    Both are only counted if newer than the control sweep's finished_at, so a
    stale run from a previous experiment can never satisfy this stage.
    """
    reports = REPO / "reports"
    raw_fin = ctl.get("finished_at")
    try:
        cutoff = datetime.strptime(raw_fin, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        cutoff = None

    def fresh(paths):
        out = []
        for p in paths:
            m = datetime.fromtimestamp(p.stat().st_mtime)
            if cutoff is None or m >= cutoff:
                out.append((m, p))
        return sorted(out, key=lambda t: t[0])

    # calibration_<stem>_20260919_051642.json -> <stem>
    calibrated = set()
    for _m, p in fresh(reports.glob("calibration_*.json")):
        stem = p.name[len("calibration_"):-len("_20260919_051642.json")]
        if stem in CALIB_STEMS:
            calibrated.add(stem)

    adj = fresh(reports.glob("g4_adjudication_*.json"))
    adj_done = adj[-1] if adj else None

    steps = len(CALIB_STEMS) + 1          # 4 calibrations + 1 adjudication
    done_steps = len(calibrated) + (1 if adj_done else 0)

    if not train_done:
        status, progress = "pending", 0.0
    elif adj_done:
        status, progress = "done", 1.0
    else:
        status, progress = "running", done_steps / steps

    return {
        "name": "Calibrate + adjudicate",
        "group": "finalize",
        "seed": None,
        "status": status,
        "progress": round(progress, 4),
        "epochs_done": done_steps,
        "epochs_total": steps,
        "epoch_frac": 0.0,
        "best": None,
        "early_stop": False,
        "finished_at": (adj_done[0].strftime("%Y-%m-%d %H:%M:%S")
                        if adj_done else None),
        "log": None,
        "calibrated": sorted(calibrated),
    }


def docs21_stages() -> list:
    """docs/21 dose-response stages.

    docs21_status.json keys runs as "<dose>_<seed>" rather than nesting them
    under a "seeds" dict, so build_stage() cannot be reused here.

    Each stage carries its live dev minDCF/EER (parsed from the log), which
    is the interesting part mid-sweep: the dose-response curve is visible in
    the dev numbers hours before any In-the-Wild scoring exists.
    """
    st = read_json(DOCS21_STATUS)
    runs = st.get("runs") or {}
    empty = {"epochs_done": 0, "frac": 0.0, "progress": 0.0, "best": None,
             "early_stop": False}
    out = []

    def sort_key(k):
        try:
            dose_s, seed_s = k.split("_")
            return (float(dose_s), int(seed_s))
        except (ValueError, AttributeError):
            return (999.0, 0)

    for key in sorted(runs, key=sort_key):
        try:
            dose_s, seed_s = key.split("_")
            dose, seed = float(dose_s), int(seed_s)
        except (ValueError, AttributeError):
            continue
        entry = runs[key]
        log_path = Path(entry.get("log") or "")
        lg = parse_log(log_path) if log_path.exists() else empty

        s = entry.get("status")
        if s == "ok":
            status, progress = "done", 1.0
        elif s == "failed":
            status, progress = "failed", lg["progress"]
        elif s == "running":
            status, progress = "running", lg["progress"]
        else:
            status, progress = "pending", 0.0

        out.append({
            "name": f"docs/21 dose {dose_s} seed {seed_s}",
            "group": "docs21",
            "seed": seed,
            "dose": dose,
            "status": status,
            "progress": round(progress, 4),
            "epochs_done": lg["epochs_done"],
            "epoch_frac": round(lg["frac"], 3),
            "best": lg["best"],
            "early_stop": lg["early_stop"] and status == "done",
            "finished_at": entry.get("finished_at"),
            "log": str(log_path) if log_path.exists() else None,
        })

    # Scoring + adjudication. Counted from the artifact it actually emits,
    # not from the training status -- see the finalize_state() note.
    adj = sorted((REPO / "reports").glob("docs21_adjudication_*.json"),
                 key=lambda p: p.stat().st_mtime)
    train_stages = [s for s in out if s["seed"] is not None]
    all_trained = bool(train_stages) and all(
        s["status"] == "done" for s in train_stages)

    if adj:
        status, progress = "done", 1.0
    elif all_trained:
        status, progress = "running", 0.15
    else:
        status, progress = "pending", 0.0

    out.append({
        "name": "docs/21 score + adjudicate",
        "group": "docs21",
        "seed": None,
        "dose": None,
        "status": status,
        "progress": progress,
        "epochs_done": 0,
        "epoch_frac": 0.0,
        "best": None,
        "early_stop": False,
        "finished_at": (datetime.fromtimestamp(adj[-1].stat().st_mtime)
                        .strftime("%Y-%m-%d %H:%M:%S") if adj else None),
        "log": None,
    })
    return out


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
    finalize = finalize_state(ctl, train_done)
    stages.append(finalize)

    # -- docs/20 (augmentation-off control) -------------------------------
    # Arm A was NOT retrained: the docs/10 G4 control *is* arm A (codec ON,
    # RawBoost OFF). Shown explicitly so the dashboard does not imply two more
    # runs are owed.
    d20 = read_json(DOCS20_STATUS)
    stages20 = [{
        "name": "Docs20 arm A (reused, not retrained)",
        "group": "docs20", "seed": None, "status": "done", "progress": 1.0,
        "epochs_done": 0, "epoch_frac": 0.0, "best": None, "early_stop": False,
        "finished_at": "2026-09-19 16:21:49", "log": None,
    }]
    # Seeds are read from the status file, not hardcoded: extra seeds (44+)
    # are launched as later invocations and would otherwise be invisible
    # while training -- the dashboard would sit at 100% with a run in flight.
    d20_seeds = sorted(int(s) for s in (d20.get("seeds") or {})
                       if str(s).isdigit()) or [42, 43]
    arm_b = [
        build_stage(f"Docs20 arm B seed {s}", "docs20", s, d20,
                    LOG_DIR / f"docs20-s{s}.log")
        for s in d20_seeds
    ]
    stages20 += arm_b
    d20_adj = sorted((REPO / "reports").glob("docs20_adjudication_*.json"),
                     key=lambda p: p.stat().st_mtime)
    stages20.append({
        "name": "Docs20 score + adjudicate", "group": "docs20", "seed": None,
        "status": "done" if d20_adj else (
            "running" if all(s["status"] == "done" for s in arm_b) else "pending"),
        "progress": 1.0 if d20_adj else 0.0,
        "epochs_done": 0, "epoch_frac": 0.0, "best": None, "early_stop": False,
        "finished_at": None, "log": None,
    })
    stages += stages20

    d21 = docs21_stages()
    d21_train = [s for s in d21 if s["seed"] is not None]
    stages += d21

    # Overall: weighted by GPU cost -- G4 4 runs, docs/20 2 runs, docs/21
    # 4 runs (each ~3.7 h measured, not the 2.5-3.5 h first assumed).
    train_p = sum(s["progress"] for s in stages[:4]) / 4.0
    g4_p = train_p * 0.93 + finalize["progress"] * 0.07
    d20_train_p = (sum(s["progress"] for s in arm_b) / len(arm_b)
                   if arm_b else 0.0)
    d20_p = d20_train_p * 0.9 + stages20[-1]["progress"] * 0.1
    d21_p = ((sum(s["progress"] for s in d21_train) / len(d21_train)) * 0.9
             + d21[-1]["progress"] * 0.1) if d21_train else 0.0
    overall = g4_p * 0.4 + d20_p * 0.2 + d21_p * 0.4

    # ETA from measured epoch cost.
    remaining_epochs = sum(
        max(0.0, EPOCH_BUDGET - (s["epochs_done"] + s["epoch_frac"]))
        for s in stages[:4] + arm_b + d21_train if s["status"] != "done"
    )
    eta_s = remaining_epochs * AVG_EPOCH_S
    if finalize["status"] != "done":
        # Only the not-yet-finished part of finalize counts.
        eta_s += FINALIZE_S * (1.0 - finalize["progress"])
    if stages20[-1]["status"] != "done":
        eta_s += DOCS20_SCORE_S * (1.0 - stages20[-1]["progress"])
    if d21 and d21[-1]["status"] != "done":
        eta_s += DOCS21_SCORE_S * (1.0 - d21[-1]["progress"])
    eta = datetime.now() + timedelta(seconds=eta_s)

    results = {"g4_seeds": [], "ood": [], "verdict": None}
    # Newest adjudication wins -- the run has been re-adjudicated as arms
    # landed, and a hardcoded filename would keep serving the stale one.
    adj_all = sorted((REPO / "reports").glob("g4_adjudication_*.json"),
                     key=lambda p: p.stat().st_mtime)
    adj = adj_all[-1] if adj_all else None
    if adj is not None and adj.exists():
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

    # docs/20 verdict is surfaced next to G4's, because both are real: G4 FAILs
    # its own bar while docs/20 PASSes. Showing only one would misrepresent the
    # state of the work.
    d20_adj_all = sorted((REPO / "reports").glob("docs20_adjudication_*.json"),
                         key=lambda p: p.stat().st_mtime)
    d20_rep = read_json(d20_adj_all[-1]) if d20_adj_all else None
    if d20_rep:
        results["docs20_verdict"] = d20_rep.get("OVERALL")
        arms = []
        for key in ("arm_a_s42", "arm_b_s42", "arm_a_s43", "arm_b_s43"):
            m = d20_rep.get("measurements", {}).get(key) or {}
            itw_c = (m.get("itw_clean") or {}).get("pooled")
            itw_f = (m.get("itw_full_call") or {}).get("pooled")
            arms.append({
                "arm": key.replace("arm_a", "arm A").replace("arm_b", "arm B")
                          .replace("s42", "seed 42").replace("s43", "seed 43"),
                "clean": (itw_c or {}).get("eer_pct"),
                "clean_ci": (itw_c or {}).get("ci95_pct"),
                "full_call": (itw_f or {}).get("eer_pct"),
                "full_call_ci": (itw_f or {}).get("ci95_pct"),
                "pooled": m.get("eval_pooled_eer"),
                "mindcf": m.get("eval_mindcf"),
            })
        results["d20_arms"] = arms

    # Labels carry the augmentation setting: the whole point of the table is
    # that the codec/none split predicts the out-of-domain ranking perfectly.
    ood_files = {
        "Arm B s42 (none)": ("eer_ci_armB_s42_ITW_clean.json",
                             "eer_ci_armB_s42_ITW_full_call.json"),
        "Pre-Aug L12 (none)": ("eer_ci_pre_aug_ITW_clean.json",
                               "eer_ci_pre_aug_ITW_full_call.json"),
        "Arm B s43 (none)": ("eer_ci_armB_s43_ITW_clean.json",
                             "eer_ci_armB_s43_ITW_full_call.json"),
        "Arm A s43 (codec)": ("eer_ci_armA_control_s43_ITW_clean.json",
                              "eer_ci_armA_control_s43_ITW_full_call.json"),
        "Post-Aug L12 (codec)": ("eer_ci_post_aug_ITW_clean.json",
                                 "eer_ci_post_aug_ITW_full_call.json"),
        "G4 s43 (codec+RawBoost)": ("eer_ci_g4_s43_ITW.json",
                                    "eer_ci_g4_s43_ITW_full_call.json"),
        "Arm A s42 (codec)": ("eer_ci_armA_control_s42_ITW_clean.json",
                              "eer_ci_armA_control_s42_ITW_full_call.json"),
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

    # docs/21: live dose table straight off the training logs (no scoring
    # needed, so the curve is visible hours before the ITW numbers exist),
    # plus the verdict once adjudication has written one.
    #
    # Endpoints are prepended from constants: they were never run by this
    # sweep, so they are absent from docs21_status.json -- but a dose table
    # without them is unreadable, since the whole point is the shape of the
    # curve across 0.0 -> 1.0.
    doses = []
    for dose in sorted(DOCS21_ENDPOINT_DEV):
        for seed in sorted(DOCS21_ENDPOINT_DEV[dose]):
            e = DOCS21_ENDPOINT_DEV[dose][seed]
            doses.append({
                "dose": dose, "seed": seed, "status": "reused",
                "epoch": e["epoch"], "dev_mindcf": e["dev_mindcf"],
                "dev_eer": e["dev_eer"],
            })
    doses += [{
        "dose": s.get("dose"),
        "seed": s.get("seed"),
        "status": s["status"],
        "epoch": (s.get("best") or {}).get("epoch"),
        "dev_mindcf": (s.get("best") or {}).get("dev_mindcf"),
        "dev_eer": (s.get("best") or {}).get("dev_eer"),
    } for s in d21_train]
    # Endpoints and trained doses are collected separately, so sort by dose or
    # the table reads 0.0, 1.0, 0.25, 0.5 -- which hides the shape of the
    # curve, the one thing this table exists to show.
    doses.sort(key=lambda x: (float(x["dose"]), int(x["seed"])))
    results["d21_doses"] = doses

    d21_adj_all = sorted((REPO / "reports").glob("docs21_adjudication_*.json"),
                         key=lambda p: p.stat().st_mtime)
    if d21_adj_all:
        d21_rep = read_json(d21_adj_all[-1])
        results["docs21_verdict"] = d21_rep.get("verdict")
        results["docs21_detail"] = d21_rep.get("verdict_detail")
        results["docs21_missing"] = d21_rep.get("missing") or []

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
            # See the note in serve_results_dashboard: the page may be opened
            # from another origin (preview proxy), so allow the cross-origin
            # fallback fetch.
            self.send_header("Access-Control-Allow-Origin", "*")
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
