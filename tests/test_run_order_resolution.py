"""Tests for the --seeds / --conditions run-order resolution.

The seed set used to be hardcoded to [42, 43] while docs/10's pre-registered
claim is defined over five seeds (S42-S46), which made the claim impossible to
adjudicate: S44-S46 were unreachable however many runs were spent. These tests
pin both halves of the fix -- that the default is unchanged, and that the claim's
seed set is now actually producible.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "rfs_under_test", REPO_ROOT / "scripts" / "run_from_scratch_experiment.py"
)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


# --------------------------------------------------------------------------
# Backward compatibility -- the default must not move
# --------------------------------------------------------------------------
def test_default_is_still_the_pre_registered_four():
    assert mod._run_order(None, None) == [("S", 42), ("B", 42), ("S", 43), ("B", 43)]


def test_default_matches_the_module_constant():
    assert mod._run_order(None, None) == mod.RUN_ORDER


def test_module_seeds_are_unchanged_by_the_new_flags():
    # The flags must be additive. If someone "fixes" the claim by editing SEEDS,
    # the default run set changes silently and every historical number's context
    # shifts with it.
    assert mod.SEEDS == [42, 43]


# --------------------------------------------------------------------------
# The new capability
# --------------------------------------------------------------------------
def test_conditions_s_only_drops_the_ablation():
    assert mod._run_order(None, "S") == [("S", 42), ("S", 43)]


def test_seeds_44_46_gives_three_S_runs():
    assert mod._run_order("44,45,46", "S") == [("S", 44), ("S", 45), ("S", 46)]


def test_the_five_seed_claim_is_now_producible():
    # The point of the whole change: docs/10's claim is over S42-S46, and
    # scripts/check_run1_gate.py refuses to grade it without all five. This
    # asserts the orchestrator can actually produce exactly that set.
    order = mod._run_order("42,43,44,45,46", "S")
    assert [f"{c}{s}" for c, s in order] == ["S42", "S43", "S44", "S45", "S46"]


def test_full_matrix_for_five_seeds_is_ten_runs():
    order = mod._run_order("42,43,44,45,46", None)
    assert len(order) == 10
    assert order == [(c, s) for s in (42, 43, 44, 45, 46) for c in ("S", "B")]


def test_order_is_interleaved_by_seed_not_grouped_by_condition():
    # Grouping by condition would let training-box drift line up with the
    # condition, which is the thing the interleaving exists to prevent.
    order = mod._run_order("42,43", None)
    assert order == [("S", 42), ("B", 42), ("S", 43), ("B", 43)]
    assert order != [("S", 42), ("S", 43), ("B", 42), ("B", 43)]


def test_seeds_are_ints_not_strings():
    order = mod._run_order("7,8", "S")
    assert all(isinstance(s, int) for _, s in order)
    assert order == [("S", 7), ("S", 8)]


def test_whitespace_is_tolerated():
    assert mod._run_order(" 44 , 45 ", " S ") == [("S", 44), ("S", 45)]


# --------------------------------------------------------------------------
# Rejections -- a typo must not silently run the wrong experiment
# --------------------------------------------------------------------------
def test_non_integer_seed_is_rejected():
    with pytest.raises(SystemExit):
        mod._run_order("a", None)


def test_unknown_condition_is_rejected():
    with pytest.raises(SystemExit):
        mod._run_order(None, "X")


def test_duplicate_seeds_are_rejected():
    # Silently de-duplicating would train fewer runs than the operator believes.
    with pytest.raises(SystemExit):
        mod._run_order("42,42", None)


def test_duplicate_conditions_are_rejected():
    with pytest.raises(SystemExit):
        mod._run_order(None, "S,S")


def test_empty_seeds_is_rejected():
    with pytest.raises(SystemExit):
        mod._run_order("", None)


def test_empty_conditions_is_rejected():
    with pytest.raises(SystemExit):
        mod._run_order(None, ",")


# --------------------------------------------------------------------------
# State file truthfulness
# --------------------------------------------------------------------------
def test_new_state_records_the_run_list_it_was_given():
    import experiment_state as xs

    state = xs.new_state(1234, ["prog"], runs=[("S", 44), ("S", 45)])
    assert [r["label"] for r in state["runs"]] == ["S44", "S45"]


def test_new_state_defaults_to_the_pre_registered_order():
    import experiment_state as xs

    state = xs.new_state(1234, ["prog"])
    assert [r["label"] for r in state["runs"]] == ["S42", "B42", "S43", "B43"]
