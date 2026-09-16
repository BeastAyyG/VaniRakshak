"""Tests for the warm-start control runner (docs/10 criterion 3).

docs/10's headline claim needs a control run that `run_from_scratch_experiment.py`
structurally cannot produce (it asserts --init-weights is absent). This script is
that control, and the things worth pinning are the ones that would quietly
destroy the comparison if they drifted:

* the checkpoint must NOT be named cm_scratch_* — a collision would let the
  control overwrite a from-scratch checkpoint, or vice versa;
* the GPU guard must fire while another orchestrated run is in flight. Two
  trainers on one card change both runs' epoch timings, and the control is only
  meaningful if it is measured under the same conditions as S.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "warmstart_under_test", REPO_ROOT / "scripts" / "run_warmstart_control.py"
)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_checkpoint_name_cannot_collide_with_from_scratch():
    p = mod.checkpoint_for(42, 12)
    assert p.name == "cm_warmstart_S42_L12.pt"
    assert not p.name.startswith("cm_scratch_")
    # the thing the collision would break: a from-scratch run at the same
    # seed/layer resolves to a different path
    scratch = REPO_ROOT / "data" / "cm_scratch_S42.pt"
    assert p != scratch


def test_checkpoint_name_varies_with_seed_and_layer():
    assert mod.checkpoint_for(46, 12) != mod.checkpoint_for(42, 12)
    assert mod.checkpoint_for(42, 5).name == "cm_warmstart_S42_L5.pt"


def _write_state(tmp_path: Path, runs: list, scoring: dict | None = None) -> Path:
    p = tmp_path / "experiment_state.json"
    p.write_text(json.dumps({
        "schema_version": 1,
        "runs": runs,
        "scoring": scoring or {"status": "pending"},
    }))
    return p


def test_gpu_busy_detects_a_running_training_run(tmp_path, monkeypatch):
    state = _write_state(tmp_path, [
        {"label": "S42", "status": "done"},
        {"label": "S44", "status": "running", "child_pid": 13168},
    ])
    monkeypatch.setattr(mod, "EXPERIMENT_STATE", state)
    reason = mod.gpu_busy()
    assert reason is not None and "S44" in reason


def test_gpu_busy_detects_a_running_scoring_stage(tmp_path, monkeypatch):
    state = _write_state(tmp_path, [{"label": "S44", "status": "done"}],
                         scoring={"status": "running"})
    monkeypatch.setattr(mod, "EXPERIMENT_STATE", state)
    assert mod.gpu_busy() is not None


def test_gpu_busy_is_none_when_everything_is_finished(tmp_path, monkeypatch):
    state = _write_state(tmp_path, [
        {"label": "S42", "status": "done"},
        {"label": "S46", "status": "done"},
    ], scoring={"status": "done"})
    monkeypatch.setattr(mod, "EXPERIMENT_STATE", state)
    assert mod.gpu_busy() is None


def test_gpu_busy_is_none_without_a_state_file(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "EXPERIMENT_STATE", tmp_path / "nope.json")
    assert mod.gpu_busy() is None


def test_archive_partial_renames_and_keeps_the_bytes(tmp_path):
    """A killed trainer leaves a valid but under-trained checkpoint at --output.

    Scoring it would produce a number the control never earned -- biased towards
    'worse', which is exactly what criterion 3 wants. So the partial must be
    moved aside, not overwritten and not deleted.
    """
    ckpt = tmp_path / "cm_warmstart_S42_L12.pt"
    ckpt.write_bytes(b"PARTIAL-WEIGHTS")
    dest = mod.archive_partial(ckpt)
    assert not ckpt.exists()
    assert ".partial-" in dest.name
    assert dest.read_bytes() == b"PARTIAL-WEIGHTS"


def test_score_only_refuses_an_interrupted_checkpoint(tmp_path, monkeypatch):
    ckpt = tmp_path / "cm_warmstart_S42_L12.pt"
    ckpt.write_bytes(b"PARTIAL")
    state = tmp_path / "warmstart_control_state.json"
    state.write_text(json.dumps({"train": {"status": "running"}}))
    monkeypatch.setattr(mod, "CONTROL_STATE", state)
    monkeypatch.setattr(mod, "checkpoint_for", lambda seed, layer: ckpt)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--score-only"])
    assert "interrupted" in str(exc.value)


def test_a_completed_checkpoint_is_not_treated_as_interrupted(tmp_path, monkeypatch):
    """...but the guard must not fire on a genuinely finished control."""
    state = tmp_path / "warmstart_control_state.json"
    state.write_text(json.dumps({"train": {"status": "done"}}))
    monkeypatch.setattr(mod, "CONTROL_STATE", state)
    monkeypatch.setattr(mod, "checkpoint_for",
                        lambda seed, layer: tmp_path / "missing.pt")
    with pytest.raises(SystemExit) as exc:
        mod.main(["--score-only"])
    assert "interrupted" not in str(exc.value)


def test_default_init_weights_is_the_warm_start_lineage():
    # docs/10 (2026-09-13): every historical checkpoint is a warm start of THIS
    # file. The control is only the right control if it starts from it.
    assert mod.DEFAULT_INIT.name == "cm_aasist.pt"
