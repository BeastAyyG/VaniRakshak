"""Unit tests for the docs/20 adjudicator's decision logic.

These cover the pure parts only -- no GPU, no scoring. The point is that the
rules the verdict rests on are tested before they are applied to a real run,
rather than discovered to be wrong afterwards.

The criterion this encodes (docs/20, fixed in advance): arm B must be better
than arm A by a NON-OVERLAPPING 95 % interval. That is deliberately stricter
than "the point estimate improved", because docs/20 showed the out-of-domain
seed spread alone is 6-8 pp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from adjudicate_docs20 import PER_ATTACK_FLOOR_PP, ci_of, disjoint_better  # noqa: E402


def _m(eer, lo, hi):
    """Shape returned by ci_of: (point, [lo, hi])."""
    return (eer, [lo, hi])


def test_better_and_disjoint_passes():
    ok, detail = disjoint_better(_m(30.0, 29.5, 30.5), _m(44.0, 43.5, 44.5))
    assert ok is True
    assert "disjoint" in detail


def test_better_but_overlapping_fails():
    # Point estimate improves a lot, intervals still touch -> must FAIL.
    ok, detail = disjoint_better(_m(40.0, 38.0, 42.0), _m(44.0, 41.0, 47.0))
    assert ok is False
    assert "OVERLAP" in detail


def test_disjoint_but_worse_fails():
    # Perfectly separated, but in the wrong direction.
    ok, _ = disjoint_better(_m(50.0, 49.5, 50.5), _m(30.0, 29.5, 30.5))
    assert ok is False


def test_identical_intervals_fail():
    ok, _ = disjoint_better(_m(40.0, 39.0, 41.0), _m(40.0, 39.0, 41.0))
    assert ok is False


def test_missing_data_is_not_a_pass():
    # The failure mode this guards against: an absent arm B report silently
    # reading as "arm A wins" and therefore "criterion satisfied".
    ok, detail = disjoint_better(None, _m(44.0, 43.5, 44.5))
    assert ok is False
    assert detail == "missing data"
    ok2, detail2 = disjoint_better(_m(30.0, 29.5, 30.5), None)
    assert ok2 is False
    assert detail2 == "missing data"


def test_ci_of_reads_pooled_and_attacks():
    rep = {
        "pooled": {"eer_pct": 15.43, "ci95_pct": [14.9, 15.9]},
        "attacks": {"A19": {"eer_pct": 22.376, "ci95_pct": [21.0, 23.8]}},
    }
    assert ci_of(rep) == (15.43, [14.9, 15.9])
    assert ci_of(rep, "A19") == (22.376, [21.0, 23.8])


def test_ci_of_missing_is_none():
    assert ci_of(None) is None
    assert ci_of({"pooled": None}) is None
    assert ci_of({"pooled": {"eer_pct": 1.0, "ci95_pct": [0.5, 1.5]}}, "A19") is None


def test_per_attack_floor_matches_docs19_amendment():
    # docs/19 raised the per-attack bar from 0.770 pp to 1.36 pp because A19/A20
    # have only ~935/920 spoof utterances. If this drifts, the criterion drifts.
    assert PER_ATTACK_FLOOR_PP == 1.36


def test_just_above_the_floor_is_still_not_enough():
    # Guards the interaction: disjoint, but the gap is under the per-attack
    # floor, so the secondary criterion must reject it.
    # Narrow intervals, genuinely disjoint: b's upper 19.8 < a's lower 20.2.
    b = _m(20.0, 19.4, 19.8)
    a = _m(21.0, 20.2, 21.8)
    ok, _ = disjoint_better(b, a)
    assert ok is True                            # interval rule alone says yes
    assert (a[0] - b[0]) < PER_ATTACK_FLOOR_PP   # but the 1.36 pp floor rejects it


def test_arm_a_reports_are_the_expected_shape():
    """The four hand-scored arm-A ITW reports must be readable by ci_of.

    These were produced by scripts/eer_bootstrap_ci.py, not by this module, so
    a schema drift there would silently break the adjudication.
    """
    import json

    for seed in (42, 43):
        for cond in ("clean", "full_call"):
            p = REPO / "reports" / f"eer_ci_armA_control_s{seed}_ITW_{cond}.json"
            if not p.exists():
                pytest.skip(f"{p.name} not written yet")
            d = json.loads(p.read_text(encoding="utf-8"))
            eer, ci = ci_of(d)
            assert 0.0 < eer < 100.0
            assert len(ci) == 2 and ci[0] < ci[1]
