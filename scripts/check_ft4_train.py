"""Live checker for the G3 seed-42 fine-tune run (data/cm_aasist_ft4_s42.pt).

Single shot (default) prints RUNNING / STALLED / COMPLETED / DEAD and exits
non-zero unless the run looks healthy or finished. ``--watch`` loops until the
training processes exit, then validates the checkpoint -- that is the mode to
leave running overnight.

Usage:
  python scripts/check_ft4_train.py                       # single check
  python scripts/check_ft4_train.py --watch --interval 300
  python scripts/check_ft4_train.py --log training_s42.log # also parse log tail

Follows docs/10-quality-and-hardening-plan.md G3 (two seeds, bf16,
--merge-layers 12, eval on data/eval_benchmark/eval_slice_20k.tsv).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    print("FAILED: psutil not installed in this interpreter.")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "data" / "cm_aasist_ft4_s42.pt"
TRAIN_TAG = "train_cm.py"
OUTPUT_TAG = "ft4_s42"
TOTAL_TRAIN_UTTS = 182_357
EPOCHS = 10
TRAIN_SPEED = 70.7    # measured utts/s from the smoke log
DEV_UTTS = 140_950    # full dev (this run launched WITHOUT --max-dev-utts)
DEV_SPEED = 300.0     # assumed forward-only utts/s, +/-50%
TRAIN_S_PER_EPOCH = TOTAL_TRAIN_UTTS / TRAIN_SPEED
DEV_S_PER_EPOCH = DEV_UTTS / DEV_SPEED
CYCLE_S = TRAIN_S_PER_EPOCH + DEV_S_PER_EPOCH
TOTAL_S = CYCLE_S * EPOCHS

for p in (str(REPO_ROOT), str(REPO_ROOT / "vanirakshak-backend")):
    if p not in sys.path:
        sys.path.insert(0, p)


def find_train_procs() -> list:
    """All live processes whose cmdline is the ft4_s42 training run."""
    hits = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            name = (proc.info["name"] or "").lower()
            if "python" not in name:
                continue  # skip the parent powershell, whose cmdline also contains the command
            cmd = " ".join(proc.info["cmdline"] or [])
            if TRAIN_TAG in cmd and OUTPUT_TAG in cmd:
                hits.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return hits


def gpu_metrics() -> str:
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            u, mu, tot, temp = [p.strip() for p in res.stdout.strip().split(",")]
            return f"GPU {u}% | VRAM {mu}/{tot} MiB | {temp}C"
    except Exception as e:
        return f"GPU query failed: {e}"
    return "GPU metrics unavailable"


def parse_log_tail(log_path: Path, n_lines: int = 60) -> dict:
    """Extract last step/speed and last epoch summary from a training log."""
    info: dict = {"log_exists": log_path.is_file()}
    if not info["log_exists"]:
        return info
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-max(n_lines, 200):]
    except Exception as e:
        info["log_error"] = str(e)
        return info
    step_re = re.compile(r"Epoch\s+(\d+): step\s+(\d+)/(\d+).*loss:\s+([\d.]+).*speed:\s+([\d.]+) utts/s")
    epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\[([\d.]+)s, val:\s+([\d.]+)s\]\s+-\s+Loss:\s+([\d.]+).*Dev minDCF:\s+([\d.]+), EER:\s+([\d.]+)%")
    for line in lines:
        m = step_re.search(line)
        if m:
            info["last_step"] = {"epoch": int(m.group(1)), "step": int(m.group(2)),
                                 "of": int(m.group(3)), "loss": float(m.group(4)),
                                 "speed": float(m.group(5)), "line": line.strip()[-160:]}
        m = epoch_re.search(line)
        if m:
            info["last_epoch"] = {"epoch": int(m.group(1)), "epochs": int(m.group(2)),
                                  "train_s": float(m.group(3)), "val_s": float(m.group(4)),
                                  "loss": float(m.group(5)), "mindcf": float(m.group(6)),
                                  "eer": float(m.group(7)), "line": line.strip()[-200:]}
    info["mtime_age_s"] = time.time() - log_path.stat().st_mtime
    return info


def check_checkpoint(ckpt_path: Path) -> dict:
    """Validate the best-checkpoint file if it exists yet."""
    info: dict = {"exists": ckpt_path.is_file()}
    if not info["exists"]:
        return info
    try:
        info["size_mb"] = round(ckpt_path.stat().st_size / 1024 / 1024, 2)
        info["mtime_age_s"] = round(time.time() - ckpt_path.stat().st_mtime, 1)
        import torch

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        from vanirakshak.detectors.cm_aasist import _validate_checkpoint

        _validate_checkpoint(ckpt, ckpt_path)
        info["valid"] = True
        for k in ("dev_eer", "dev_min_dcf", "eer_threshold", "seed",
                  "unfreeze_top_n", "encoder_lr", "merge_layers", "frontend_dtype"):
            if k in ckpt:
                info[k] = ckpt[k]
        fe = ckpt.get("frontend_state_dict")
        info["frontend_tensors"] = len(fe) if isinstance(fe, dict) else 0
        info["frontend_layers"] = ckpt.get("frontend_layer_indices")
    except Exception as e:
        info["valid"] = False
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def render_bar(frac: float, width: int = 30) -> str:
    frac = min(1.0, max(0.0, frac))
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {frac * 100:5.1f}%"


def progress_snapshot() -> str:
    """One-line wall-clock progress estimate for the training run."""
    procs = find_train_procs()
    ckpt = check_checkpoint(DEFAULT_CHECKPOINT if _cli_ckpt is None else _cli_ckpt)
    gpu = gpu_metrics()
    if not procs:
        if ckpt.get("valid"):
            return f"{gpu} | COMPLETED -- checkpoint validates ({ckpt.get('dev_eer', float('nan')) * 100:.3f}% dev EER)."
        return f"{gpu} | no training process and no checkpoint -- nothing to show."
    start = min(p.create_time() for p in procs)
    elapsed = time.time() - start
    frac = min(elapsed / TOTAL_S, 0.99)
    epoch = min(int(elapsed / CYCLE_S) + 1, EPOCHS)
    in_cycle = elapsed % CYCLE_S
    phase = "train" if in_cycle < TRAIN_S_PER_EPOCH else "dev-eval"
    eta = TOTAL_S - elapsed
    done_note = ""
    if ckpt.get("exists") and ckpt.get("valid"):
        done_note = f" | ckpt best: EER {ckpt['dev_eer'] * 100:.2f}% minDCF {ckpt['dev_min_dcf']:.4f}"
    return (
        f"{render_bar(frac)} | epoch ~{epoch}/{EPOCHS} ({phase}) | "
        f"elapsed {fmt_hms(elapsed)} | ETA ~{fmt_hms(eta)} | {gpu}{done_note}"
    )


_cli_ckpt: Path | None = None


def single_check(ckpt_path: Path, log_path: Path | None, stall_mins: float) -> tuple[str, int]:
    """Return (report_text, exit_code). 0 = healthy or completed."""
    now = datetime.now().strftime("%H:%M:%S")
    procs = find_train_procs()
    gpu = gpu_metrics()
    ckpt = check_checkpoint(ckpt_path)
    log = parse_log_tail(log_path, 60) if log_path else {"log_exists": None}

    lines = [f"[{now}] ft4_s42 check | {gpu}"]
    global _cli_ckpt
    _cli_ckpt = ckpt_path
    lines.append("  " + progress_snapshot())
    if procs:
        for p in procs:
            try:
                age_h = (time.time() - p.create_time()) / 3600
                cpu = p.cpu_percent(interval=1)
                mem = round(p.memory_info().rss / 1024 / 1024)
                lines.append(f"  proc PID {p.pid} alive, age {age_h:.1f}h, CPU {cpu:.0f}%, MEM {mem}MB")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                lines.append(f"  proc PID {p.pid} unreadable: {e}")
    else:
        lines.append("  proc: NO live train_cm.py/ft4_s42 process")

    if log.get("log_exists") is True:
        s, e = log.get("last_step"), log.get("last_epoch")
        if s:
            lines.append(f"  log tail: epoch {s['epoch']} step {s['step']}/{s['of']} loss {s['loss']:.4f} @ {s['speed']:.1f} utts/s")
        if e:
            lines.append(f"  last epoch: {e['epoch']}/{e['epochs']} loss {e['loss']:.4f} dev minDCF {e['mindcf']:.4f} EER {e['eer']:.2f}% (train {e['train_s']:.0f}s + val {e['val_s']:.0f}s)")
        lines.append(f"  log mtime age: {log['mtime_age_s']:.0f}s")
    elif log.get("log_exists") is False and log_path:
        lines.append(f"  log: {log_path} not found (training stdout is in its own terminal)")

    if ckpt["exists"]:
        if ckpt.get("valid"):
            lines.append(
                f"  checkpoint: {ckpt_path.name} {ckpt['size_mb']}MB, "
                f"updated {ckpt['mtime_age_s']:.0f}s ago, dev EER {ckpt.get('dev_eer', float('nan')) * 100:.3f}% "
                f"minDCF {ckpt.get('dev_min_dcf', float('nan')):.4f}, "
                f"encoder tensors {ckpt.get('frontend_tensors')}"
            )
        else:
            lines.append(f"  checkpoint: EXISTS but INVALID: {ckpt.get('error')}")
    else:
        lines.append("  checkpoint: not yet written (no epoch has beaten best -- normal in epoch 1)")

    # ---- verdict ----
    if not procs and ckpt.get("valid"):
        lines.append("VERDICT: COMPLETED -- process exited and checkpoint validates. Run the eval slice next; seed 43 only if close to target.")
        return "\n".join(lines), 0
    if not procs and not ckpt["exists"]:
        lines.append("VERDICT: DEAD -- no process and no checkpoint. Relaunch the seed-42 command.")
        return "\n".join(lines), 1
    if ckpt.get("exists") and ckpt.get("valid") is False:
        lines.append("VERDICT: BROKEN -- checkpoint fails schema validation. Inspect before eval.")
        return "\n".join(lines), 1
    # stalled? process alive but nothing moving
    log_age = log.get("mtime_age_s")
    if procs and log_age is not None and log_age > stall_mins * 60:
        lines.append(f"VERDICT: STALLED? -- log untouched for {log_age / 60:.0f}min (threshold {stall_mins:.0f}min). Check the training terminal for traceback/OOM.")
        return "\n".join(lines), 1
    lines.append("VERDICT: RUNNING -- process alive, GPU busy. Leave it; re-run this checker for updates.")
    return "\n".join(lines), 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the ft4_s42 training run.")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--log", type=Path, default=None, help="Optional training stdout log to tail")
    ap.add_argument("--watch", action="store_true", help="Loop until the run exits, then validate")
    ap.add_argument("--progress", action="store_true",
                    help="Live progress bar (redraws every --interval seconds, default 30)")
    ap.add_argument("--interval", type=int, default=300, help="Seconds between watch/progress updates")
    ap.add_argument("--stall-mins", type=float, default=20.0)
    args = ap.parse_args()
    global _cli_ckpt
    _cli_ckpt = args.checkpoint

    if args.progress:
        interval = args.interval if args.interval != 300 else 30
        print(f"ft4_s42 progress -- ~{fmt_hms(TOTAL_S)} total assumed ({EPOCHS} epochs; "
              f"early stopping may finish sooner). One line per update. Ctrl+C to stop.")
        try:
            while True:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] " + progress_snapshot(), flush=True)
                if not find_train_procs():
                    print()
                    ckpt = check_checkpoint(args.checkpoint)
                    if ckpt.get("valid"):
                        print(f"COMPLETED -- dev EER {ckpt['dev_eer'] * 100:.3f}%, "
                              f"minDCF {ckpt['dev_min_dcf']:.4f}. Eval slice next.")
                        sys.exit(0)
                    print("Process gone with no valid checkpoint -- see verdict via single check.")
                    sys.exit(1)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped watching (training keeps running).")
            sys.exit(0)

    if not args.watch:
        report, code = single_check(args.checkpoint, args.log, args.stall_mins)
        print(report)
        sys.exit(code)

    print(f"Watching ft4_s42 every {args.interval}s -- Ctrl+C to stop.")
    while True:
        report, code = single_check(args.checkpoint, args.log, args.stall_mins)
        print(report, flush=True)
        procs = find_train_procs()
        if not procs:
            sys.exit(code)  # COMPLETED or DEAD, message already printed
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
