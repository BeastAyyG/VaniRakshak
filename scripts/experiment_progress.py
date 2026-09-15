"""Progress bar for the whole from-scratch CM experiment.

Shows one bar for the *entire* process -- S42, B42, S43, B43, then scoring --
plus a sub-bar per run, the current epoch/step, and a wall-clock ETA.

It is a pure reader. It never writes to a log, a checkpoint, or a report, so it
is safe to run at any point, including while a run is mid-epoch. All of its
input comes from :mod:`experiment_state`.

    python scripts/experiment_progress.py              # one snapshot
    python scripts/experiment_progress.py --watch 30   # refresh every 30 s
    python scripts/experiment_progress.py --json       # machine-readable

Estimates are honest rather than optimistic: a run's total is projected from
``best_epoch + patience`` (early stopping), because that is what S42 actually
did -- assuming all 10 epochs would overstate every remaining run by ~11%.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_state as xs  # noqa: E402

# ---------------------------------------------------------------------------
# Glyphs
# ---------------------------------------------------------------------------

UNICODE = {
    "filled": "\u2588",   # full block
    "empty": "\u2591",    # light shade
    "done": "\u2713",     # check
    "running": "\u25b6",  # right triangle
    "queued": "\u00b7",   # middle dot
    "warn": "!",
    "rule": "\u2501",
    "sep": "\u00b7",
}

ASCII = {
    "filled": "#",
    "empty": "-",
    "done": "[x]",
    "running": "[>]",
    "queued": "[ ]",
    "warn": "[!]",
    "rule": "=",
    "sep": "|",
}

ANSI = {
    "reset": "\x1b[0m",
    "dim": "\x1b[2m",
    "bold": "\x1b[1m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
    "red": "\x1b[31m",
}


def _use_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" not in enc:
        return False
    try:
        "\u2588".encode(enc)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, _ = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d {h:02d}h {m:02d}m"
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def _bar(fraction: float, width: int, g: dict) -> str:
    width = max(4, width)
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return g["filled"] * filled + g["empty"] * (width - filled)


def _pct(fraction: float) -> str:
    return f"{max(0.0, min(1.0, fraction)) * 100:5.1f}%"


def _paint(text: str, colour: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{ANSI[colour]}{text}{ANSI['reset']}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _run_detail(rp: xs.RunProgress, timing: xs.RunTiming, sep: str = "\u00b7") -> str:
    """Right-hand status text for one run row."""
    if rp.state == "done":
        bits = []
        if rp.finished:
            stop = "early-stop" if rp.early_stopped_at else "full"
            when = rp.last_activity[11:16] if rp.last_activity else ""
            bits.append(f"done {when} ({stop} @ ep {rp.best_epoch})".strip())
        else:
            bits.append("checkpoint present")
        if rp.best_dev_eer is not None:
            bits.append(f"dev EER {rp.best_dev_eer:.3f}%")
            bits.append(f"minDCF {rp.best_dev_mindcf:.4f}")
        return f" {sep} ".join(b for b in bits if b)

    if rp.state == "queued":
        return "queued"

    if rp.stalled():
        return f"NO OUTPUT FOR >{xs.STALL_SECONDS // 60} min -- check the process"

    if rp.validating:
        where = f"validating epoch {rp.current_epoch}/{rp.total_epochs} (dev, 141k utts)"
    elif rp.steps_total:
        where = (f"epoch {rp.current_epoch}/{rp.total_epochs} {sep} "
                 f"step {rp.steps_done}/{rp.steps_total}")
    else:
        where = "starting up"
    bits = [where]
    if rp.speed_utts_s:
        bits.append(f"{rp.speed_utts_s:.0f} utts/s")
    if rp.last_dev_eer is not None:
        bits.append(f"last dev EER {rp.last_dev_eer:.3f}%")
    return f" {sep} ".join(bits)


def render(prog: xs.ExperimentProgress, width: int = 0, unicode: bool = True,
           colour: bool = False) -> str:
    g = UNICODE if unicode else ASCII
    width = width or min(110, max(64, shutil.get_terminal_size((100, 24)).columns))
    width = max(60, width)

    out: list[str] = []
    now = xs._parse_dt(prog.generated_at) or datetime.now()

    title = f"VaniRakshak {g['sep']} CM from-scratch experiment"
    stamp = f"{now:%Y-%m-%d %H:%M:%S}"
    pad = max(1, width - len(title) - len(stamp))
    out.append(_paint(f"{title}{' ' * pad}{stamp}", "bold", colour))
    out.append(g["rule"] * width)

    # --- overall ---------------------------------------------------------
    pct = prog.fraction
    overall_bar = _bar(pct, width - 9, g)
    colour_key = "green" if prog.complete else "cyan"
    out.append(f"OVERALL  {_paint(overall_bar, colour_key, colour)} {_pct(pct)}")

    if prog.complete:
        out.append("         complete -- report at "
                   f"{prog.scoring.report_path or '(missing)'}")
    else:
        eta = prog.eta
        flag = ""
        if prog.paused:
            flag = "  " + _paint("[PAUSED]", "yellow", colour)
        elif any(r.stalled() for r in prog.runs):
            flag = "  " + _paint("[STALLED]", "red", colour)
        out.append(f"         ETA {eta:%Y-%m-%d %H:%M}  {g['sep']}  "
                   f"in {_fmt_duration(prog.remaining_seconds)}  {g['sep']}  "
                   f"4 runs + scoring{flag}")
    out.append(g["rule"] * width)

    # --- per run ---------------------------------------------------------
    bar_w = width - 22
    for rp, timing in zip(prog.runs, prog.timings):
        if rp.state == "done":
            marker = _paint(g["done"], "green", colour)
        elif rp.state == "running":
            marker = _paint(g["running"], "cyan", colour)
        else:
            marker = g["queued"]
        row = f"  {marker} {rp.label:<4} {_bar(timing.fraction, bar_w, g)} {_pct(timing.fraction)}"
        detail = _run_detail(rp, timing, g["sep"])
        out.append(f"{row}  {_paint(detail, 'dim', colour)}")

    # --- scoring ---------------------------------------------------------
    sp = prog.scoring
    if sp.state == "done":
        marker = _paint(g["done"], "green", colour)
        frac = 1.0
        detail = f"report {Path(sp.report_path).name}" if sp.report_path else "complete"
    elif sp.state == "running":
        marker = _paint(g["running"], "cyan", colour)
        frac = (sp.done / sp.total) if sp.total else 0.0
        detail = f"scoring {sp.done}/{sp.total} utterances on 4 checkpoints"
    else:
        marker = g["queued"]
        frac = 0.0
        detail = f"queued {g['sep']} runs after all four checkpoints exist"
    row = f"  {marker} eval {_bar(frac, bar_w, g)} {_pct(frac)}"
    out.append(f"{row}  {_paint(detail, 'dim', colour)}")

    out.append(g["rule"] * width)

    # --- footer ----------------------------------------------------------
    finished = [rp for rp in prog.runs if rp.state == "done"]
    running = next((rp for rp in prog.runs if rp.state == "running"), None)
    bits = [f"{len(finished)}/4 runs done"]
    if running is not None:
        bits.append(f"{running.label} running")
    if prog.paused:
        bits.append("pause requested")
    out.append(_paint("  " + f" {g['sep']} ".join(bits), "dim", colour))

    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=float, nargs="?", const=30.0, default=None,
                    metavar="SECONDS", help="Refresh continuously (default 30 s).")
    ap.add_argument("--json", action="store_true", help="Emit state as JSON instead.")
    ap.add_argument("--width", type=int, default=0, help="Override render width.")
    ap.add_argument("--ascii", action="store_true", help="Force ASCII glyphs.")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colour.")
    args = ap.parse_args()

    if args.json:
        prog = xs.collect()
        payload = {
            "generated_at": prog.generated_at,
            "paused": prog.paused,
            "overall": {
                "fraction": prog.fraction,
                "done_seconds": prog.done_seconds,
                "total_seconds": prog.total_seconds,
                "remaining_seconds": prog.remaining_seconds,
                "eta": prog.eta.isoformat(timespec="seconds"),
                "complete": prog.complete,
            },
            "runs": [
                {
                    "label": rp.label,
                    "state": rp.state,
                    "fraction": t.fraction,
                    "epochs_completed": rp.epochs_completed,
                    "total_epochs": rp.total_epochs,
                    "current_epoch": rp.current_epoch,
                    "steps_done": rp.steps_done,
                    "steps_total": rp.steps_total,
                    "speed_utts_s": rp.speed_utts_s,
                    "validating": rp.validating,
                    "best_dev_eer": rp.best_dev_eer,
                    "best_dev_mindcf": rp.best_dev_mindcf,
                    "early_stopped_at": rp.early_stopped_at,
                    "stalled": rp.stalled(),
                }
                for rp, t in zip(prog.runs, prog.timings)
            ],
            "scoring": {
                "state": prog.scoring.state,
                "done": prog.scoring.done,
                "total": prog.scoring.total,
                "report": prog.scoring.report_path,
            },
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
        return 0

    unicode = not args.ascii and _use_unicode()
    colour = not args.no_color and sys.stdout.isatty()

    if args.watch is None:
        print(render(xs.collect(), args.width, unicode, colour))
        return 0

    try:
        while True:
            prog = xs.collect()
            frame = render(prog, args.width, unicode, colour)
            # Redraw in place: home the cursor and clear to end of screen, so
            # the bar does not scroll away on a long run.
            sys.stdout.write("\x1b[H\x1b[2J" if sys.stdout.isatty() else "")
            sys.stdout.write(frame + "\n")
            sys.stdout.flush()
            if prog.complete:
                print("\nExperiment complete.")
                return 0
            time.sleep(max(2.0, args.watch))
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
