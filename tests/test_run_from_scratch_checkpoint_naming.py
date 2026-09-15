"""Tests for the from-scratch orchestrator's checkpoint naming.

Guards the fix for a trap that made docs/10's headline run plan unexecutable.

The checkpoint path used to depend only on (condition, seed), so a run at a
different ``--layer-index`` targeted the SAME file as the layer-12 run:

* ``--layer-index 5`` silently trained nothing, because all four checkpoints
  already existed and the "resume, never overwrite" check skipped every run.
  docs/10's run 2 -- "from scratch at layer 5", the headline candidate -- is
  written that way.
* Forcing it by clearing the checkpoints would have written the layer-5 output
  onto ``data/cm_scratch_S42.pt``, destroying the layer-12 S42 result the whole
  project rests on.

These tests pin both halves of the fix: existing names do not move, and a
non-default layer index cannot collide.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ORCHESTRATOR = Path(__file__).resolve().parents[1] / "scripts" / "run_from_scratch_experiment.py"

# Loaded lazily and cached at module level rather than through a pytest fixture,
# so this file also runs under minimal test runners that lack fixture support.
_ORCH = None


def orch():
    """Import the orchestrator without executing its __main__ block."""
    global _ORCH
    if _ORCH is None:
        spec = importlib.util.spec_from_file_location("_orfse_under_test", ORCHESTRATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ORCH = module
    return _ORCH


def test_default_layer_keeps_original_names():
    """The 2026-09-14 checkpoints must not be renamed by this change."""
    o = orch()
    assert o.DEFAULT_LAYER_INDEX == 12
    assert o.checkpoint_for("S", 42).name == "cm_scratch_S42.pt"
    assert o.checkpoint_for("B", 43).name == "cm_scratch_B43.pt"
    assert o.checkpoint_for("S", 42, 12).name == "cm_scratch_S42.pt"


def test_non_default_layer_is_suffixed():
    o = orch()
    assert o.checkpoint_for("S", 42, 5).name == "cm_scratch_S42_L5.pt"
    assert o.checkpoint_for("B", 43, 5).name == "cm_scratch_B43_L5.pt"


def test_layer_index_cannot_collide_with_the_default():
    """The whole point: a layer-5 run must never target the layer-12 file."""
    o = orch()
    for cond in ("S", "B"):
        for seed in (42, 43):
            assert o.checkpoint_for(cond, seed, 5) != o.checkpoint_for(cond, seed, 12)
            assert o.checkpoint_for(cond, seed, 5) != o.checkpoint_for(cond, seed)


def test_all_run_order_entries_get_distinct_paths():
    o = orch()
    paths = [o.checkpoint_for(c, s) for c, s in o.RUN_ORDER]
    assert len(paths) == len(set(paths)) == 4


def test_layer5_paths_are_distinct_across_conditions():
    o = orch()
    paths = [o.checkpoint_for(c, s, 5) for c, s in o.RUN_ORDER]
    assert len(paths) == len(set(paths)) == 4
    assert all(p.name.endswith("_L5.pt") for p in paths)


def test_paths_live_in_the_data_directory():
    o = orch()
    assert o.checkpoint_for("S", 42, 5).parent == o.REPO_ROOT / "data"
