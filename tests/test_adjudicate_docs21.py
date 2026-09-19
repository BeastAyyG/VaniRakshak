"""Decision logic for the docs/21 dose-response adjudicator.

These are pure-function tests: no GPU, no torch, no scoring. They exist so
the criteria in docs/21 (fixed before any result) are pinned by code rather
than by memory -- and so "AT_0" stays a reachable verdict. docs/21 must be
able to confirm that augmentation is simply harmful; if someone later
restructures the logic so only an interior optimum can win, these fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from adjudicate_docs21 import (  # noqa: E402
    beats,
    checkpoint_for,
    classify_optimum,
    dose_tag,
    monotonic,
    stem_for,
)


def mkeer(point: float, half: float = 0.5):
    """(point, (lo, hi)) -- the shape eer_bootstrap_ci.py reports."""
    return (point, (point - half, point + half))


def mk(zero, one, inter: dict | None = None):
    t = {0.0: {42: zero, 43: zero}, 1.0: {42: one, 43: one}}
    for dose, val in (inter or {}).items():
        t[dose] = {42: val, 43: val}
    return t


# --------------------------------------------------------------------------
# beats()
# --------------------------------------------------------------------------

def test_beats_when_strictly_lower_and_disjoint():
    assert beats(mkeer(20), mkeer(30)) is True


def test_no_beat_when_intervals_overlap():
    assert beats(mkeer(20, 15), mkeer(30, 15)) is False


def test_no_beat_when_worse():
    assert beats(mkeer(40), mkeer(30)) is False


def test_missing_data_never_beats():
    """An unscored arm must not win by default -- the exact way a
    half-finished sweep would otherwise look like a result."""
    assert beats(None, mkeer(30)) is False
    assert beats(mkeer(20), None) is False
    assert beats(None, None) is False


# --------------------------------------------------------------------------
# classify_optimum()
# --------------------------------------------------------------------------

def test_interior_wins_when_a_dose_beats_both_endpoints():
    v, _ = classify_optimum(mk(mkeer(30), mkeer(45), {0.25: mkeer(20)}))
    assert v == "INTERIOR_025"


def test_at_zero_when_no_dose_beats_off():
    v, _ = classify_optimum(mk(mkeer(30), mkeer(45)))
    assert v == "AT_0"


def test_at_one_when_full_augmentation_wins():
    v, _ = classify_optimum(mk(mkeer(45), mkeer(30)))
    assert v == "AT_1"


def test_unresolved_when_endpoints_overlap():
    v, _ = classify_optimum(mk(mkeer(30), mkeer(31)))
    assert v == "UNRESOLVED"


def test_unresolved_when_endpoints_missing():
    v, d = classify_optimum({0.25: {42: mkeer(20), 43: mkeer(20)}})
    assert v == "UNRESOLVED" and "missing" in d


def test_dose_beating_only_one_endpoint_is_not_interior():
    """Beats 1.0 but not 0.0 -> the optimum is still at 0, not interior."""
    v, _ = classify_optimum(mk(mkeer(30), mkeer(45), {0.5: mkeer(40)}))
    assert v == "AT_0"


def test_interior_needs_every_seed():
    table = mk(mkeer(30), mkeer(45))
    table[0.25] = {42: mkeer(20), 43: None}  # one seed not scored
    v, _ = classify_optimum(table)
    assert v == "AT_0"


# --------------------------------------------------------------------------
# monotonic()
# --------------------------------------------------------------------------

def test_monotonic_rising_across_doses():
    table = {0.0: {42: mkeer(30), 43: mkeer(30)},
             0.25: {42: mkeer(33), 43: mkeer(33)},
             1.0: {42: mkeer(45), 43: mkeer(45)}}
    assert monotonic(table)["all_seeds"] is True


def test_non_monotonic_is_reported_not_hidden():
    table = {0.0: {42: mkeer(40), 43: mkeer(40)},
             0.25: {42: mkeer(30), 43: mkeer(30)},
             1.0: {42: mkeer(45), 43: mkeer(45)}}
    assert monotonic(table)["all_seeds"] is False


def test_monotonic_needs_more_than_one_dose():
    table = {0.0: {42: mkeer(30), 43: mkeer(30)}}
    assert monotonic(table)[42]["ok"] is False


# --------------------------------------------------------------------------
# wiring: endpoints reused, intermediate doses new
# --------------------------------------------------------------------------

def test_dose_tag_is_filesystem_safe():
    assert dose_tag(0.25) == "025"
    assert dose_tag(0.5) == "050"
    assert dose_tag(1.0) == "100"


def test_endpoints_map_to_existing_arms():
    """Retraining an endpoint would waste ~3h and break comparability."""
    assert checkpoint_for(0.0, 42) == "data/cm_docs20_armb_s42.pt"
    assert checkpoint_for(1.0, 43) == "data/cm_g4_control_s43.pt"


def test_intermediate_dose_gets_its_own_checkpoint():
    assert checkpoint_for(0.5, 42) == "data/cm_docs21_pct050_s42.pt"


def test_endpoint_report_stems_match_earlier_experiments():
    assert stem_for(0.0, 42) == "armB_s42"
    assert stem_for(1.0, 43) == "armA_control_s43"
    assert stem_for(0.25, 42) == "docs21_pct025_s42"
