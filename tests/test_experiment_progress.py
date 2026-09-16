"""Regression tests for the from-scratch experiment tooling.

Covers ``scripts/experiment_state.py`` (log parsing + progress math),
``scripts/experiment_progress.py`` (rendering), and the log-archiving / pause
gate added to ``scripts/run_from_scratch_experiment.py``.

Deliberately stdlib-only and fixture-based: no torch, no transformers, no real
corpus, no real run logs. These must pass on the CPU-only dev box, because that
is where someone will change the parsing code.

Not part of the default ``pytest`` run (``pytest.ini`` sets
``testpaths = vanirakshak-backend/tests``); run them explicitly:

    pytest tests/test_experiment_progress.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:  # standalone mode -- see the runner at the bottom
    class _Approx:
        def __init__(self, expected, rel=1e-6, abs=1e-12):  # noqa: A002
            self.expected = expected
            self.rel = rel
            self.abs = abs

        def __eq__(self, other):
            try:
                return abs(other - self.expected) <= max(
                    self.abs, self.rel * abs(self.expected))
            except TypeError:
                return NotImplemented

        def __repr__(self):
            return f"approx({self.expected!r})"

    class _PytestShim:
        @staticmethod
        def approx(expected, rel=1e-6, abs=1e-12):  # noqa: A002
            return _Approx(expected, rel, abs)

    pytest = _PytestShim()  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import experiment_state as xs  # noqa: E402
import experiment_progress as xp  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: synthetic logs in the exact format ml/train_cm.py emits
# ---------------------------------------------------------------------------

COMPLETED_RUN = """\
15:51:23 [INFO] Initializing Recipe A Retraining on device: cuda (WavLM layer_index=12, seed=42)
15:51:25 [INFO] Loaded 182357 audio records from train.tsv (bonafide=18797, spoof=163560, ratio=0.1149)
15:51:25 [INFO] Augmenter probabilities: codec=0.50 noise=0.30 bandpass=0.30 rir=0.20 gain=0.30
15:51:25 [INFO] Dev validation configured in STREAMING mode directly from audio (dev.tsv)
15:51:25 [INFO] No --init-weights given: training from random init.
15:51:25 [INFO] Starting training loop: 10 epochs, 2849 batches/epoch, lr=5.00e-04, patience=5, layer_index=12
15:52:00 [INFO] Epoch  1: step  200/2849 (7.0%) - loss: 0.4794 - speed: 195.4 utts/s
16:16:33 [INFO] Epoch  1/10 [1499.8s, val: 624.3s] - Loss: 0.2221 | Dev minDCF: 0.9457, EER: 13.851%, tau: 4.6502 | Demo01 (Genuine): LLR=-1.55, Risk=64.5 (CHALLENGE)
16:40:28 [INFO] Epoch  2/10 [1435.2s, val: 605.4s] - Loss: 0.1440 | Dev minDCF: 0.9899, EER: 15.334%, tau: 7.5793 | Demo01 (Genuine): LLR=+4.39, Risk=20.5 (ALLOW)
17:28:05 [INFO] Epoch  4/10 [1427.5s, val: 600.7s] - Loss: 0.1097 | Dev minDCF: 0.9425, EER: 12.749%, tau: 8.3789 | Demo01 (Genuine): LLR=+3.19, Risk=21.6 (ALLOW)
19:27:27 [INFO] Epoch  9/10 [1436.1s, val: 609.5s] - Loss: 0.0816 | Dev minDCF: 0.9462, EER: 13.765%, tau: 9.1881 | Demo01 (Genuine): LLR=+4.68, Risk=20.4 (ALLOW)
19:27:27 [INFO] Dev minDCF did not improve (patience: 5/5)
19:27:27 [INFO] Early stopping triggered at epoch 9
19:27:27 [INFO] Training finished. Best epoch 4: dev minDCF=0.9425, dev EER=12.749%, threshold=8.3789
"""

IN_PROGRESS_RUN = """\
20:18:22 [INFO] Initializing Recipe A Retraining on device: cuda (WavLM layer_index=12, seed=42)
20:18:28 [INFO] No --init-weights given: training from random init.
20:18:28 [INFO] Starting training loop: 10 epochs, 2849 batches/epoch, lr=5.00e-04, patience=5, layer_index=12
20:19:34 [INFO] Epoch  1: step  200/2849 (7.0%) - loss: 0.4794 - speed: 195.4 utts/s
20:33:55 [INFO] Epoch  1: step 2849/2849 (100.0%) - loss: 0.1634 - speed: 196.9 utts/s
20:33:55 [INFO] Evaluating epoch 1 on dev set (140950 records)...
"""

MIDNIGHT_RUN = """\
23:50:00 [INFO] Starting training loop: 10 epochs, 2849 batches/epoch, lr=5.00e-04, patience=5, layer_index=12
00:15:00 [INFO] Epoch  1/10 [1400.0s, val: 600.0s] - Loss: 0.2000 | Dev minDCF: 0.9500, EER: 14.000%, tau: 4.0 | Demo01 (Genuine): LLR=-1.00, Risk=60.0 (CHALLENGE)
"""


def write_log(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parses_completed_run(tmp_path):
    rp = xs.parse_run_log(write_log(tmp_path, "2026-09-13-s42.log", COMPLETED_RUN))
    assert (rp.cond, rp.seed) == ("S", 42)
    assert rp.total_epochs == 10
    assert rp.steps_total == 2849
    assert rp.epochs_completed == 4
    assert rp.finished is True
    assert rp.early_stopped_at == 9
    assert rp.best_epoch == 4
    assert rp.best_dev_eer == pytest.approx(12.749)
    assert rp.best_dev_mindcf == pytest.approx(0.9425)
    assert rp.validating is False
    assert rp.device == "cuda"
    assert rp.dev_validation_mode == "streaming"


def test_epoch_bracket_is_total_not_train(tmp_path):
    """The bracket is [total_epoch_seconds, val_seconds] -- train is total - val.

    Misreading it as train time overstates a run by ~40%, which is exactly the
    error the wall-clock estimate must not make.
    """
    rp = xs.parse_run_log(write_log(tmp_path, "2026-09-13-s42.log", COMPLETED_RUN))
    assert rp.epoch_seconds[0] == pytest.approx(1499.8)
    assert rp.val_seconds[0] == pytest.approx(624.3)
    assert rp.val_seconds[0] < rp.epoch_seconds[0]


def test_parses_in_progress_run(tmp_path):
    rp = xs.parse_run_log(write_log(tmp_path, "2026-09-13-b42.log", IN_PROGRESS_RUN))
    assert rp.finished is False
    assert rp.state == "running"
    assert rp.validating is True
    assert rp.steps_done == 2849
    assert rp.speed_utts_s == pytest.approx(196.9)
    assert rp.epochs_completed == 0


def test_midnight_rollover_gives_positive_duration(tmp_path):
    rp = xs.parse_run_log(write_log(tmp_path, "2026-09-13-b43.log", MIDNIGHT_RUN))
    start = xs._parse_dt(rp.started_at)
    last = xs._parse_dt(rp.last_activity)
    assert start is not None and last is not None
    assert last > start
    assert (last - start).total_seconds() == pytest.approx(25 * 60)


def test_archived_logs_are_not_picked_up_as_live(tmp_path):
    write_log(tmp_path, "2026-09-13-s42.log", COMPLETED_RUN)
    write_log(tmp_path, "2026-09-13-s42-partial-aborted-epoch1.log", COMPLETED_RUN)
    write_log(tmp_path, "2026-09-13-s42.prev-120000.log", COMPLETED_RUN)
    found = xs.find_run_log("S", 42, tmp_path)
    assert found is not None
    assert found.name == "2026-09-13-s42.log"


# ---------------------------------------------------------------------------
# Progress maths
# ---------------------------------------------------------------------------


def test_expected_epochs_uses_early_stop_horizon():
    # best_epoch + patience, capped at the planned epoch count.
    assert xs.expected_epochs(10, 4, 5) == 9
    assert xs.expected_epochs(10, 8, 5) == 10
    # Nothing has improved yet -> the observed default, not the plan.
    assert xs.expected_epochs(10, None, 5) == xs.DEFAULT_EXPECTED_EPOCHS
    # A run one epoch in must not project a 6-epoch job when this recipe has
    # never stopped before epoch 9; the estimate grows once the best moves.
    assert xs.expected_epochs(10, 1, 5, floor=9) == 9
    assert xs.expected_epochs(10, 6, 5, floor=9) == 10


def test_finished_run_is_exactly_complete(tmp_path):
    rp = xs.parse_run_log(write_log(tmp_path, "2026-09-13-s42.log", COMPLETED_RUN))
    t = xs.run_timing(rp, xs.DEFAULT_EPOCH_SECONDS)
    assert t.fraction == pytest.approx(1.0)
    assert t.remaining_seconds == 0.0
    assert t.done_seconds == pytest.approx(sum(rp.epoch_seconds))


def test_in_progress_run_is_partial(tmp_path):
    rp = xs.parse_run_log(write_log(tmp_path, "2026-09-13-b42.log", IN_PROGRESS_RUN))
    t = xs.run_timing(rp, xs.DEFAULT_EPOCH_SECONDS)
    assert 0.0 < t.fraction < 1.0
    assert t.remaining_seconds > 0


def test_checkpoint_present_mid_run_still_reads_as_running(tmp_path):
    """train_cm.py saves on the first improvement, so a checkpoint does not
    mean the run is finished -- the log is what says so."""
    log = write_log(tmp_path, "2026-09-13-b42.log", IN_PROGRESS_RUN)
    rp = xs.parse_run_log(log)
    rp.checkpoint_exists = True
    assert rp.state == "running"


def test_overall_fraction_is_monotonic_and_bounded(tmp_path):
    logs = tmp_path / "handoff" / "logs"
    logs.mkdir(parents=True)
    (logs / "2026-09-13-s42.log").write_text(COMPLETED_RUN, encoding="utf-8")
    (logs / "2026-09-13-b42.log").write_text(IN_PROGRESS_RUN, encoding="utf-8")

    prog = xs.collect(root=tmp_path)
    assert 0.0 < prog.fraction < 1.0
    assert prog.complete is False
    assert [r.state for r in prog.runs] == ["done", "running", "queued", "queued"]
    assert prog.remaining_seconds > 0


def test_progress_never_goes_backwards(tmp_path):
    """Regression: an epoch's summary used to be double-counted.

    The bar leapt forward the moment a summary landed, then snapped backwards
    when the next epoch's first step line arrived -- which reads as the run
    losing work.
    """
    header = ("20:00:00 [INFO] Starting training loop: 10 epochs, 2849 "
              "batches/epoch, lr=5.00e-04, patience=5, layer_index=12\n")
    step_200 = ("20:01:00 [INFO] Epoch  1: step  200/2849 (7.0%) - loss: 0.4000 "
                "- speed: 200.0 utts/s\n")
    step_end = ("20:12:00 [INFO] Epoch  1: step 2849/2849 (100.0%) - loss: 0.2000 "
                "- speed: 200.0 utts/s\n")
    evaluating = "20:12:00 [INFO] Evaluating epoch 1 on dev set (140950 records)...\n"
    ep1_summary = ("20:24:00 [INFO] Epoch  1/10 [1440.0s, val: 600.0s] - Loss: 0.2000 "
                   "| Dev minDCF: 0.9500, EER: 14.000%, tau: 4.0 | Demo01 (Genuine): "
                   "LLR=-1.00, Risk=60.0 (CHALLENGE)\n")
    ep2_step_200 = ("20:25:00 [INFO] Epoch  2: step  200/2849 (7.0%) - loss: 0.1500 "
                    "- speed: 200.0 utts/s\n")

    before_summary = header + step_200 + step_end
    after_summary = before_summary + evaluating + ep1_summary
    frames = [
        header,
        header + step_200,
        before_summary,
        after_summary,
        after_summary + ep2_step_200,
    ]

    log = tmp_path / "2026-09-13-s42.log"
    fractions = []
    for text in frames:
        log.write_text(text, encoding="utf-8")
        rp = xs.parse_run_log(log)
        fractions.append(xs.run_timing(rp, xs.DEFAULT_EPOCH_SECONDS).fraction)

    for prev, cur in zip(fractions, fractions[1:]):
        assert cur >= prev - 1e-9, f"progress went backwards: {fractions}"
    assert fractions[-1] > 0


def test_scoring_progress_partial_and_finished(tmp_path):
    partial = tmp_path / "2026-09-13-scoring.log"
    partial.write_text(
        "Scoring 20000 utterances from eval_slice_20k.tsv on cuda\n"
        "    scored 5000/20000 (100.0 utts/s)\n",
        encoding="utf-8",
    )
    sp = xs.parse_scoring_log(partial)
    assert sp.done == 5000 and sp.total == 20000 and not sp.finished

    done = tmp_path / "2026-09-14-scoring.log"
    done.write_text(
        "Scoring 20000 utterances from eval_slice_20k.tsv on cuda\n"
        "    scored 20000/20000 (100.0 utts/s)\n"
        "Saved reports/cm_from_scratch_2026-09-14.json\n",
        encoding="utf-8",
    )
    sp2 = xs.parse_scoring_log(done)
    assert sp2.finished and sp2.done == sp2.total == 20000


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_contains_bar_and_eta(tmp_path):
    logs = tmp_path / "handoff" / "logs"
    logs.mkdir(parents=True)
    (logs / "2026-09-13-s42.log").write_text(COMPLETED_RUN, encoding="utf-8")
    prog = xs.collect(root=tmp_path)
    out = xp.render(prog, width=90, unicode=True, colour=False)
    assert "OVERALL" in out
    assert "ETA" in out
    assert "S42" in out
    assert "\u2588" in out  # filled block
    assert "%" in out


def test_render_ascii_fallback_has_no_unicode_blocks(tmp_path):
    logs = tmp_path / "handoff" / "logs"
    logs.mkdir(parents=True)
    (logs / "2026-09-13-s42.log").write_text(COMPLETED_RUN, encoding="utf-8")
    prog = xs.collect(root=tmp_path)
    out = xp.render(prog, width=90, unicode=False, colour=False)
    assert "#" in out
    assert "\u2588" not in out
    assert "\u2591" not in out


# ---------------------------------------------------------------------------
# Orchestrator helpers
# ---------------------------------------------------------------------------


def test_archive_existing_log_preserves_evidence(tmp_path):
    import run_from_scratch_experiment as orch

    log = write_log(tmp_path, "2026-09-13-b42.log", "Traceback (most recent call last):\n")
    archived = orch.archive_existing_log(log)
    assert archived is not None and archived.exists()
    assert "Traceback" in archived.read_text(encoding="utf-8")
    assert not log.exists()
    assert ".prev-" in archived.name


def test_archive_skips_empty_and_missing_logs(tmp_path):
    import run_from_scratch_experiment as orch

    assert orch.archive_existing_log(tmp_path / "nope.log") is None
    empty = write_log(tmp_path, "empty.log", "")
    assert orch.archive_existing_log(empty) is None


def test_pause_gate_holds_then_releases(tmp_path, monkeypatch):
    import run_from_scratch_experiment as orch

    gate = tmp_path / "PAUSE"
    monkeypatch.setattr(xs, "GATE_PATH", gate)

    # No gate -> immediate return.
    orch.wait_for_pause_gate(poll_s=0.01)

    gate.write_text('{"note": "test"}', encoding="utf-8")
    gate.unlink()
    orch.wait_for_pause_gate(poll_s=0.01)
    assert not gate.exists()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
# pytest is declared in requirements.txt but is not installed in every
# environment -- and it must not be installed into the project venv while a
# training run is using it. So this file also runs under plain python:
#
#     python tests/test_experiment_progress.py
#
# It provides just enough of tmp_path / monkeypatch to run the suite.

if __name__ == "__main__":
    import inspect
    import shutil
    import tempfile
    import traceback

    class _MonkeyPatch:
        def __init__(self):
            self._undo = []

        def setattr(self, target, name, value):
            self._undo.append((target, name, getattr(target, name)))
            setattr(target, name, value)

        def undo(self):
            for target, name, old in reversed(self._undo):
                setattr(target, name, old)
            self._undo.clear()

    suite = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in suite:
        params = set(inspect.signature(fn).parameters)
        kwargs = {}
        mp = None
        if "tmp_path" in params:
            kwargs["tmp_path"] = Path(tempfile.mkdtemp(prefix=f"vr-test-{name}-"))
        if "monkeypatch" in params:
            mp = _MonkeyPatch()
            kwargs["monkeypatch"] = mp
        try:
            fn(**kwargs)
            print(f"  ok   {name}")
        except Exception:
            failures += 1
            print(f"  FAIL {name}")
            traceback.print_exc()
        finally:
            if mp is not None:
                mp.undo()
            if "tmp_path" in kwargs:
                shutil.rmtree(kwargs["tmp_path"], ignore_errors=True)

    print(f"\n{len(suite) - failures} passed, {failures} failed")
    raise SystemExit(1 if failures else 0)
