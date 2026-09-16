"""Shared state model for the from-scratch CM experiment (docs/10, 2026-09-13).

One source of truth for "what is this experiment doing right now", shared by:

* ``scripts/run_from_scratch_experiment.py`` -- the orchestrator (writes state)
* ``scripts/experiment_progress.py``          -- the progress bar (reads)
* ``scripts/experiment_control.py``           -- pause / resume (reads)

Design constraints, in order of importance:

1. **Read-only against the training run.** Nothing here writes to ``data/``,
   ``reports/``, or a run log. The progress bar must be safe to run against a
   job that is mid-flight; the only file it ever creates is
   ``handoff/experiment_state.json``, and only when explicitly asked.
2. **No heavy imports.** No torch, no transformers, no numpy, no psutil. The
   dev box is CPU-only and cannot import ``ml/train_cm.py`` at all; the progress
   bar still has to work there, because that is where someone will run it.
3. **Derive, do not duplicate.** The logs and checkpoints are the record. The
   state file is an optimisation (exact start times, planned run list), never a
   prerequisite -- if it is missing or stale, everything still works.

Log lines this parses, all emitted by ``ml/train_cm.py``::

    20:18:22 [INFO] Initializing Recipe A Retraining on device: cuda (...)
    20:19:34 [INFO] Epoch  1: step  200/2849 (7.0%) - loss: 0.4794 - speed: 195.4 utts/s
    19:17:14 [INFO] Evaluating epoch 9 on dev set (140950 records)...
    19:27:27 [INFO] Epoch  9/10 [1436.1s, val: 609.5s] - Loss: 0.0816 | Dev minDCF: 0.9462, EER: 13.765%, ...
    19:27:27 [INFO] Early stopping triggered at epoch 9
    19:27:27 [INFO] Training finished. Best epoch 4: dev minDCF=0.9425, dev EER=12.749%, threshold=8.3789

Note on the epoch summary: the bracket is ``[total_epoch_seconds, val_seconds]``,
so **training time is ``total - val``**. Reading the first number as train time
overstates a run by ~40%; every wall-clock number in this module depends on
getting that right.

Usage:
    python scripts/experiment_state.py --selftest   # validate the parser
    python scripts/experiment_state.py --json       # dump derived state
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_DIR = REPO_ROOT / "handoff"
LOG_DIR = HANDOFF_DIR / "logs"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"

STATE_PATH = HANDOFF_DIR / "experiment_state.json"
CONTROL_PATH = HANDOFF_DIR / "experiment_control.json"
GATE_PATH = HANDOFF_DIR / "PAUSE"

# ---------------------------------------------------------------------------
# The experiment, as pre-registered in docs/10-quality-and-hardening-plan.md
# ---------------------------------------------------------------------------

# label: extra flags beyond the shared base (see CONDITIONS in the orchestrator)
CONDITIONS: Dict[str, List[str]] = {
    "S": [],
    "B": ["--p-codec", "0", "--p-bandpass", "0"],
}
COND_DESCRIPTION: Dict[str, str] = {
    "S": "full augmenter",
    "B": "no band-limiting",
}
SEEDS: Tuple[int, ...] = (42, 43)
# S42, B42, S43, B43 -- interleaved by seed, not grouped by condition.
RUN_ORDER: List[Tuple[str, int]] = [(cond, seed) for seed in SEEDS for cond in ("S", "B")]

EPOCHS_PLANNED_DEFAULT = 10
PATIENCE_DEFAULT = 5
# S42 stopped at epoch 9 (best 4 + patience 5). Until a run says otherwise,
# assume the same: it is a far better prior than "runs all 10 epochs", which
# overstates every remaining run by 11%.
DEFAULT_EXPECTED_EPOCHS = 9
# Measured on S42: ~1435 s/epoch. Only used before any run has reported.
DEFAULT_EPOCH_SECONDS = 1440.0
DEFAULT_VAL_FRACTION = 0.30
# Four checkpoints x 20k utterances. Coarse; replaced by the real thing as soon
# as the scoring log reports a rate.
DEFAULT_SCORING_SECONDS = 480.0
# A run whose log has been silent this long outside validation is presumed dead.
STALL_SECONDS = 20 * 60

# ---------------------------------------------------------------------------
# Log grammar
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\s")
_DATE_IN_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_STEP_RE = re.compile(
    r"Epoch\s+(\d+):\s+step\s+(\d+)/(\d+)\s+\(([\d.]+)%\)"
    r"\s+-\s+loss:\s*([\d.eE+-]+)\s+-\s+speed:\s*([\d.]+)\s+utts/s"
)
_EVAL_START_RE = re.compile(r"Evaluating epoch (\d+) on dev set")
_EPOCH_SUMMARY_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\[([\d.]+)s,\s*val:\s*([\d.]+)s\]"
    r".*?Dev minDCF:\s*([\d.]+),\s*EER:\s*([\d.]+)%"
)
_EARLY_STOP_RE = re.compile(r"Early stopping triggered at epoch\s+(\d+)")
_DONE_RE = re.compile(
    r"Training finished\.\s+Best epoch\s+(\d+):\s*"
    r"dev minDCF=([\d.]+),\s*dev EER=([\d.]+)%"
)
_TRAIN_START_RE = re.compile(r"Starting training loop:\s*(\d+)\s*epochs,\s*(\d+)\s*batches/epoch")
_NEW_BEST_RE = re.compile(r"New best checkpoint at epoch\s+(\d+)")
_INIT_RE = re.compile(r"Initializing Recipe A Retraining on device:\s*(\S+)")
_DEV_MODE_RE = re.compile(r"Dev validation configured in (\w+) mode")

# Scoring log, emitted by score_checkpoint_eval.py
_SCORE_TOTAL_RE = re.compile(r"Scoring\s+(\d+)\s+utterances")
_SCORE_PROGRESS_RE = re.compile(r"scored\s+(\d+)/(\d+)")
_SCORE_MODEL_RE = re.compile(r"^=== (\S+) ::")


def _now() -> datetime:
    return datetime.now()


# ---------------------------------------------------------------------------
# Per-run model
# ---------------------------------------------------------------------------


@dataclass
class RunProgress:
    """Everything derivable about one of the four runs."""

    cond: str
    seed: int
    checkpoint: Optional[str] = None
    log_path: Optional[str] = None

    checkpoint_exists: bool = False
    started_at: Optional[str] = None
    last_activity: Optional[str] = None

    device: Optional[str] = None
    dev_validation_mode: Optional[str] = None

    total_epochs: int = EPOCHS_PLANNED_DEFAULT
    epochs_completed: int = 0
    epoch_seconds: List[float] = field(default_factory=list)
    val_seconds: List[float] = field(default_factory=list)
    steps_total: int = 0
    current_epoch: int = 0
    steps_done: int = 0
    speed_utts_s: float = 0.0
    validating: bool = False
    val_started_at: Optional[str] = None

    finished: bool = False
    early_stopped_at: Optional[int] = None
    best_epoch: Optional[int] = None
    best_dev_mindcf: Optional[float] = None
    best_dev_eer: Optional[float] = None
    last_dev_mindcf: Optional[float] = None
    last_dev_eer: Optional[float] = None

    @property
    def label(self) -> str:
        return f"{self.cond}{self.seed}"

    @property
    def state(self) -> str:
        """One of ``done``, ``running``, ``queued``.

        Order matters. ``ml/train_cm.py`` saves the checkpoint the first time
        dev minDCF improves -- typically epoch 1 -- so ``checkpoint_exists`` is
        true for a run that is still training. Only ``Training finished.`` means
        done while a log is present.
        """
        if self.finished:
            return "done"
        if self.log_path:
            return "running"
        if self.checkpoint_exists:
            return "done"
        return "queued"

    def stalled(self, now: Optional[datetime] = None) -> bool:
        if self.state != "running" or self.validating:
            return False
        last = _parse_dt(self.last_activity)
        if last is None:
            return False
        return ((now or _now()) - last).total_seconds() > STALL_SECONDS


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def checkpoint_for(cond: str, seed: int, root: Path = REPO_ROOT) -> Path:
    return root / "data" / f"cm_scratch_{cond}{seed}.pt"


def find_run_log(cond: str, seed: int, log_dir: Path = LOG_DIR) -> Optional[Path]:
    """Newest ``<date>-<cond><seed>.log``, ignoring ``.prev-*`` archives.

    The suffix match is exact (``-s42``), so an archived
    ``...-s42-partial-aborted-epoch1.log`` is never mistaken for the live log.
    """
    if not log_dir.is_dir():
        return None
    want = f"-{cond.lower()}{seed}"
    hits = [p for p in log_dir.glob("*.log") if p.stem.endswith(want)]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def find_scoring_log(log_dir: Path = LOG_DIR) -> Optional[Path]:
    if not log_dir.is_dir():
        return None
    hits = [p for p in log_dir.glob("*-scoring.log") if ".prev-" not in p.name]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def find_report(reports_dir: Path = REPORTS_DIR) -> Optional[Path]:
    """Newest from-scratch report, excluding smoke tests."""
    if not reports_dir.is_dir():
        return None
    hits = [
        p
        for p in reports_dir.glob("cm_from_scratch_*.json")
        if "smoketest" not in p.name.lower()
    ]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def parse_run_log(path: Path) -> RunProgress:
    """Parse one run log into a :class:`RunProgress` (never raises on bad data).

    Timestamps in the log are bare ``HH:MM:SS``; the date comes from the
    filename and midnight rollover is handled by accumulating whole days, so a
    run that starts at 20:18 and finishes at 00:05 parses with a sane duration
    instead of a negative one.
    """
    m = _DATE_IN_NAME_RE.match(path.name)
    base_date = (
        datetime.strptime(m.group(1), "%Y-%m-%d").date() if m else datetime.fromtimestamp(
            path.stat().st_mtime
        ).date()
    )

    cond, seed = _run_from_name(path.stem)
    rp = RunProgress(cond=cond or "?", seed=seed or 0, log_path=str(path))

    text = path.read_text(encoding="utf-8", errors="replace")
    prev_dt: Optional[datetime] = None
    day_offset = 0

    for line in text.splitlines():
        ts = _TS_RE.match(line)
        if ts:
            hh, mm, ss = (int(g) for g in ts.groups())
            dt = datetime.combine(base_date + timedelta(days=day_offset), time(hh, mm, ss))
            if prev_dt is not None and dt < prev_dt:
                day_offset += 1
                dt = datetime.combine(base_date + timedelta(days=day_offset), time(hh, mm, ss))
            prev_dt = dt
            rp.last_activity = dt.isoformat(timespec="seconds")
            if rp.started_at is None:
                rp.started_at = rp.last_activity

        if (mm2 := _INIT_RE.search(line)) and rp.device is None:
            rp.device = mm2.group(1)
        if (mm2 := _DEV_MODE_RE.search(line)) and rp.dev_validation_mode is None:
            rp.dev_validation_mode = mm2.group(1).lower()
        if (mm2 := _TRAIN_START_RE.search(line)):
            rp.total_epochs = int(mm2.group(1))
            rp.steps_total = int(mm2.group(2))
        if (mm2 := _STEP_RE.search(line)):
            rp.current_epoch = int(mm2.group(1))
            rp.steps_done = int(mm2.group(2))
            rp.steps_total = int(mm2.group(3))
            rp.speed_utts_s = float(mm2.group(6))
            rp.validating = False
        if (mm2 := _EVAL_START_RE.search(line)):
            rp.validating = True
            rp.val_started_at = rp.last_activity
        if (mm2 := _EPOCH_SUMMARY_RE.search(line)):
            rp.epochs_completed += 1
            rp.total_epochs = int(mm2.group(2))
            rp.epoch_seconds.append(float(mm2.group(3)))
            rp.val_seconds.append(float(mm2.group(4)))
            rp.last_dev_mindcf = float(mm2.group(5))
            rp.last_dev_eer = float(mm2.group(6))
            rp.validating = False
            rp.val_started_at = None
        if (mm2 := _EARLY_STOP_RE.search(line)):
            rp.early_stopped_at = int(mm2.group(1))
        if (mm2 := _DONE_RE.search(line)):
            rp.finished = True
            rp.best_epoch = int(mm2.group(1))
            rp.best_dev_mindcf = float(mm2.group(2))
            rp.best_dev_eer = float(mm2.group(3))
            rp.validating = False
            rp.val_started_at = None
        if mm2 := _NEW_BEST_RE.search(line):
            rp.best_epoch = int(mm2.group(1))

    return rp


def _run_from_name(stem: str) -> Tuple[Optional[str], Optional[int]]:
    """``2026-09-13-b42`` -> ``("B", 42)``."""
    tail = stem.rsplit("-", 1)[-1].lower()
    m = re.fullmatch(r"([sb])(\d+)", tail)
    if not m:
        return None, None
    return m.group(1).upper(), int(m.group(2))


@dataclass
class ScoringProgress:
    log_path: Optional[str] = None
    report_path: Optional[str] = None
    total: int = 0
    done: int = 0
    models_done: int = 0
    models_total: int = 0
    finished: bool = False
    started_at: Optional[str] = None
    last_activity: Optional[str] = None

    @property
    def state(self) -> str:
        if self.finished:
            return "done"
        if self.log_path:
            return "running"
        return "queued"


def parse_scoring_log(path: Path) -> ScoringProgress:
    sp = ScoringProgress(log_path=str(path))
    text = path.read_text(encoding="utf-8", errors="replace")
    prev: Optional[datetime] = None
    day_offset = 0
    m = _DATE_IN_NAME_RE.match(path.name)
    base_date = (
        datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if m
        else datetime.fromtimestamp(path.stat().st_mtime).date()
    )
    for line in text.splitlines():
        ts = _TS_RE.match(line)
        if ts:
            hh, mm, ss = (int(g) for g in ts.groups())
            dt = datetime.combine(base_date + timedelta(days=day_offset), time(hh, mm, ss))
            if prev is not None and dt < prev:
                day_offset += 1
                dt = datetime.combine(base_date + timedelta(days=day_offset), time(hh, mm, ss))
            prev = dt
            sp.last_activity = dt.isoformat(timespec="seconds")
            if sp.started_at is None:
                sp.started_at = sp.last_activity
        if (mm2 := _SCORE_TOTAL_RE.search(line)):
            sp.total = int(mm2.group(1))
        if (mm2 := _SCORE_PROGRESS_RE.search(line)):
            sp.done = int(mm2.group(1))
            sp.total = int(mm2.group(2))
        if _SCORE_MODEL_RE.search(line):
            sp.models_done += 1
        if line.startswith("Saved "):
            sp.finished = True
            sp.done = sp.total
    return sp


# ---------------------------------------------------------------------------
# Wall-clock model
# ---------------------------------------------------------------------------


def expected_epochs(planned: int, best_epoch: Optional[int],
                    patience: int = PATIENCE_DEFAULT,
                    floor: int = DEFAULT_EXPECTED_EPOCHS) -> int:
    """How many epochs a run is likely to actually execute.

    Early stopping fires ``patience`` epochs after the best one, so
    ``best_epoch + patience`` is where the run is heading. That is floored at
    ``floor`` -- the horizon completed runs actually took -- because a run one
    epoch in always looks like it will stop at ``1 + patience``, which would
    read as a much shorter job than this recipe has ever produced. As the best
    epoch moves later the estimate grows on its own.
    """
    if best_epoch:
        return max(1, min(planned, max(best_epoch + patience, floor)))
    return min(planned, floor)


@dataclass
class RunTiming:
    done_seconds: float = 0.0
    estimated_seconds: float = 0.0
    mean_epoch_seconds: float = DEFAULT_EPOCH_SECONDS

    @property
    def fraction(self) -> float:
        if self.estimated_seconds <= 0:
            return 1.0
        return min(1.0, self.done_seconds / self.estimated_seconds)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.estimated_seconds - self.done_seconds)


def run_timing(rp: RunProgress, fallback_epoch_seconds: float,
               horizon_floor: int = DEFAULT_EXPECTED_EPOCHS) -> RunTiming:
    """Wall-clock done/estimated for one run, in seconds.

    A finished run contributes its *actual* duration to both sides, so it is
    always exactly 100% and the overall bar never goes backwards.
    """
    obs = rp.epoch_seconds
    mean_epoch_s = statistics.fmean(obs) if obs else fallback_epoch_seconds
    mean_val_s = statistics.fmean(rp.val_seconds) if rp.val_seconds else mean_epoch_s * DEFAULT_VAL_FRACTION
    mean_val_s = min(mean_val_s, mean_epoch_s * 0.95)
    train_s = mean_epoch_s - mean_val_s

    done = sum(obs)
    if rp.finished:
        return RunTiming(done, done, mean_epoch_s)

    # Count a partial epoch ONLY while the epoch being worked on has no summary
    # yet. Once its summary lands the epoch is already inside ``obs``, so adding
    # a partial for it double-counts its training time -- the bar leaps forward
    # at the summary, then snaps backwards when the next epoch's first step line
    # arrives. ``current_epoch`` is the epoch of the last step line.
    in_flight = rp.current_epoch > rp.epochs_completed
    partial = 0.0
    if in_flight:
        if rp.validating and rp.val_started_at and rp.last_activity:
            # Validation emits no step lines, so use wall time since the
            # "Evaluating epoch N" line rather than assuming the epoch is done.
            v0 = _parse_dt(rp.val_started_at)
            v1 = _parse_dt(rp.last_activity)
            partial = train_s + max(0.0, (v1 - v0).total_seconds()) if v0 and v1 else train_s
        elif rp.steps_total:
            partial = (rp.steps_done / rp.steps_total) * train_s

    done += max(0.0, partial)
    n_expected = expected_epochs(rp.total_epochs or EPOCHS_PLANNED_DEFAULT,
                                 rp.best_epoch, floor=horizon_floor)
    est = max(n_expected * mean_epoch_s, done)
    return RunTiming(done, est, mean_epoch_s)


@dataclass
class ExperimentProgress:
    runs: List[RunProgress] = field(default_factory=list)
    timings: List[RunTiming] = field(default_factory=list)
    scoring: ScoringProgress = field(default_factory=ScoringProgress)
    scoring_seconds: float = DEFAULT_SCORING_SECONDS
    generated_at: str = ""
    paused: bool = False

    @property
    def total_seconds(self) -> float:
        return sum(t.estimated_seconds for t in self.timings) + self.scoring_seconds

    @property
    def done_seconds(self) -> float:
        done = sum(t.done_seconds for t in self.timings)
        if self.scoring.state == "done":
            done += self.scoring_seconds
        elif self.scoring.state == "running" and self.scoring.total:
            done += self.scoring_seconds * (self.scoring.done / self.scoring.total)
        return done

    @property
    def fraction(self) -> float:
        total = self.total_seconds
        if total <= 0:
            return 0.0
        return min(1.0, self.done_seconds / total)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.total_seconds - self.done_seconds)

    @property
    def eta(self) -> datetime:
        base = _parse_dt(self.generated_at) or _now()
        return base + timedelta(seconds=self.remaining_seconds)

    @property
    def complete(self) -> bool:
        return all(r.state == "done" for r in self.runs) and self.scoring.state == "done"


def collect(
    root: Path = REPO_ROOT,
    now: Optional[datetime] = None,
    runs: Sequence[Tuple[str, int]] = tuple(RUN_ORDER),
) -> ExperimentProgress:
    """Build the full picture from checkpoints, logs, and the report."""
    now = now or _now()
    log_dir = root / "handoff" / "logs"

    parsed: List[RunProgress] = []
    for cond, seed in runs:
        ckpt = checkpoint_for(cond, seed, root)
        log = find_run_log(cond, seed, log_dir)
        rp = parse_run_log(log) if log else RunProgress(cond=cond, seed=seed)
        rp.cond, rp.seed = cond, seed
        rp.checkpoint = str(ckpt)
        rp.checkpoint_exists = ckpt.exists()
        parsed.append(rp)

    # A run that has no log yet still needs a sane epoch estimate; borrow the
    # median of whatever has actually been observed.
    observed = [s for rp in parsed for s in rp.epoch_seconds]
    fallback = statistics.median(observed) if observed else DEFAULT_EPOCH_SECONDS

    # Empirical early-stop horizon: how long completed runs of this recipe
    # actually ran. Falls back to the observed default until one exists.
    done_epochs = [rp.epochs_completed for rp in parsed
                   if rp.finished and rp.epochs_completed]
    horizon = (int(statistics.median(done_epochs)) if done_epochs
               else DEFAULT_EXPECTED_EPOCHS)

    scoring_log = find_scoring_log(log_dir)
    scoring = parse_scoring_log(scoring_log) if scoring_log else ScoringProgress()
    report = find_report(root / "reports")
    if report:
        scoring.finished = True
        scoring.report_path = str(report)
    scoring.models_total = scoring.models_total or len(parsed)

    return ExperimentProgress(
        runs=parsed,
        timings=[run_timing(rp, fallback, horizon) for rp in parsed],
        scoring=scoring,
        generated_at=now.isoformat(timespec="seconds"),
        paused=(root / "handoff" / "PAUSE").exists(),
    )


# ---------------------------------------------------------------------------
# State file (written by the orchestrator; optional everywhere)
# ---------------------------------------------------------------------------


def write_state(payload: Dict[str, object], path: Path = STATE_PATH) -> None:
    """Best-effort state write. Never let bookkeeping break the experiment."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        tmp.replace(path)
    except OSError as exc:  # pragma: no cover - disk full / permissions
        print(f"[experiment_state] could not write {path}: {exc}", file=sys.stderr)


def read_state(path: Path = STATE_PATH) -> Dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def new_state(pid: int, argv: Sequence[str],
              runs: Optional[Sequence[Tuple[str, int]]] = None) -> Dict[str, object]:
    """Initialise the state file.

    ``runs`` defaults to the pre-registered RUN_ORDER, but the orchestrator now
    accepts --seeds/--conditions, so it passes the resolved order in. Without
    that the state file would claim 4 runs were queued while 3 were running.
    """
    if runs is None:
        runs = RUN_ORDER
    return {
        "schema_version": SCHEMA_VERSION,
        "orchestrator_pid": pid,
        "argv": list(argv),
        "started_at": _now().isoformat(timespec="seconds"),
        "updated_at": _now().isoformat(timespec="seconds"),
        "runs": [
            {"label": f"{c}{s}", "cond": c, "seed": s, "status": "pending",
             "started_at": None, "finished_at": None, "exit_code": None,
             "child_pid": None}
            for c, s in runs
        ],
        "scoring": {"status": "pending", "started_at": None, "finished_at": None,
                    "report": None, "exit_code": None},
    }


def update_state(state: Dict[str, object], path: Path = STATE_PATH, **changes: object) -> Dict[str, object]:
    state.update(changes)
    state["updated_at"] = _now().isoformat(timespec="seconds")
    write_state(state, path)
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def selftest() -> int:
    """Assert the parser against the real S42 log, if it is on this machine."""
    log = find_run_log("S", 42)
    if log is None:
        print("SKIP: no S42 log on this machine (expected on the training box)")
        return 0
    rp = parse_run_log(log)
    checks = [
        ("total_epochs", rp.total_epochs, 10),
        ("epochs_completed", rp.epochs_completed, 9),
        ("finished", rp.finished, True),
        ("early_stopped_at", rp.early_stopped_at, 9),
        ("best_epoch", rp.best_epoch, 4),
        ("steps_total", rp.steps_total, 2849),
        ("validating", rp.validating, False),
    ]
    failures = []
    for name, got, want in checks:
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r} (want {want!r})")
        if not ok:
            failures.append(name)

    if rp.epoch_seconds:
        got = rp.epoch_seconds[0]
        ok = abs(got - 1499.8) < 0.05
        print(f"  {'ok  ' if ok else 'FAIL'} epoch_seconds[0]: {got} (want 1499.8)")
        if not ok:
            failures.append("epoch_seconds[0]")
    if rp.best_dev_eer is not None:
        ok = abs(rp.best_dev_eer - 12.749) < 0.001
        print(f"  {'ok  ' if ok else 'FAIL'} best_dev_eer: {rp.best_dev_eer} (want 12.749)")
        if not ok:
            failures.append("best_dev_eer")

    t = run_timing(rp, DEFAULT_EPOCH_SECONDS)
    ok = abs(t.fraction - 1.0) < 1e-9
    print(f"  {'ok  ' if ok else 'FAIL'} finished run fraction: {t.fraction}")
    if not ok:
        failures.append("fraction")

    print(f"\n{len(checks) + 3 - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="Dump derived state as JSON.")
    ap.add_argument("--selftest", action="store_true", help="Validate the log parser.")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    prog = collect()
    if args.json:
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
                    **asdict(rp),
                    "state": rp.state,
                    "stalled": rp.stalled(),
                    "timing": {
                        **asdict(t),
                        "fraction": t.fraction,
                        "remaining_seconds": t.remaining_seconds,
                    },
                }
                for rp, t in zip(prog.runs, prog.timings)
            ],
            "scoring": {**asdict(prog.scoring), "state": prog.scoring.state},
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
        return 0

    print(f"overall {prog.fraction * 100:.1f}%  "
          f"remaining {_fmt_hms(prog.remaining_seconds)}  "
          f"eta {prog.eta:%Y-%m-%d %H:%M}")
    for rp, t in zip(prog.runs, prog.timings):
        print(f"  {rp.label:<4} {rp.state:<8} {t.fraction * 100:5.1f}%  "
              f"epochs {rp.epochs_completed}/{rp.total_epochs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
