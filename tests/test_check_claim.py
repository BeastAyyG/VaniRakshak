"""Tests for the full claim adjudicator.

`check_run1_gate.py` covers criteria 1 and 2 and explicitly does not adjudicate
criterion 3. Criterion 3 is the one that can kill the claim: if the warm-start
control is not WORSE than the median from-scratch seed, the gap was never about
initialisation. That judgement has to be a pure function of the two reports, so
that the sentence "the claim is dead" is something the code says -- not something
someone decides after seeing the numbers.

Also pinned: the strict comparison (a bar is cleared only if strictly below) and
the default of the warm-start parent, which is what makes the control a control.
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
    "check_claim_under_test", REPO_ROOT / "scripts" / "check_claim.py"
)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

CLAIM_SEEDS = ["S42", "S43", "S44", "S45", "S46"]


def _report(path: Path, seeds, eer=0.115, mindcf=0.65, a19=6.0, prefix="S") -> Path:
    models = {}
    for s in seeds:
        models[f"{prefix}{s}"] = {
            "metrics": {"eer": eer, "min_dcf": mindcf},
            "per_attack_eer": {"A19": a19},
            "init_weights": None,
        }
    path.write_text(json.dumps({"models": models}))
    return path


def test_cleared_is_strictly_below():
    assert mod.cleared(0.7499, 0.75)
    assert not mod.cleared(0.75, 0.75)      # equal is NOT a clear
    assert not mod.cleared(0.7501, 0.75)


def test_pct_converts_fractions():
    assert mod.pct(0.115) == pytest.approx(11.5)


def test_criterion3_passes_when_control_is_worse(tmp_path):
    claim = _report(tmp_path / "claim.json", [42, 43, 44, 45, 46], mindcf=0.65)
    control = _report(tmp_path / "control.json", [42], mindcf=0.90, prefix="W")
    out = tmp_path / "verdict.json"
    rc = mod.main(["--claim", str(claim), "--control", str(control),
                   "--json-out", str(out)])
    v = json.loads(out.read_text())
    assert v["criterion3"] is True
    assert v["criterion1"] is True
    assert v["criterion2_stated"] is True
    assert v["verdict"] == "pass"
    assert rc == 0


def test_criterion3_kills_the_claim_when_control_wins(tmp_path):
    """docs/10 L799-801: if the control matches or beats scratch, the claim is dead."""
    claim = _report(tmp_path / "claim.json", [42, 43, 44, 45, 46], mindcf=0.65)
    control = _report(tmp_path / "control.json", [42], mindcf=0.50, prefix="W")
    out = tmp_path / "verdict.json"
    rc = mod.main(["--claim", str(claim), "--control", str(control),
                   "--json-out", str(out)])
    v = json.loads(out.read_text())
    assert v["criterion3"] is False
    assert v["verdict"] == "fail"
    assert rc == 1


def test_undecided_with_only_two_seeds(tmp_path):
    claim = _report(tmp_path / "claim.json", [42, 43], mindcf=0.65)
    out = tmp_path / "verdict.json"
    rc = mod.main(["--claim", str(claim), "--json-out", str(out)])
    v = json.loads(out.read_text())
    assert v["verdict"] == "undecided"
    assert v["criterion1"] is None
    assert v["criterion3"] is None
    assert rc == 1


def test_without_a_control_report_criterion3_is_not_graded(tmp_path):
    claim = _report(tmp_path / "claim.json", [42, 43, 44, 45, 46], mindcf=0.65)
    out = tmp_path / "verdict.json"
    mod.main(["--claim", str(claim), "--json-out", str(out)])
    v = json.loads(out.read_text())
    # Absent evidence is not a pass.
    assert v["criterion3"] is None
    assert v["verdict"] == "undecided"


def test_spread_is_reported_and_flagged_against_the_floor(tmp_path):
    path = tmp_path / "claim.json"
    models = {
        "S42": {"metrics": {"eer": 0.1100, "min_dcf": 0.65}, "per_attack_eer": {}},
        "S43": {"metrics": {"eer": 0.1200, "min_dcf": 0.70}, "per_attack_eer": {}},
    }
    path.write_text(json.dumps({"models": models}))
    out = tmp_path / "verdict.json"
    mod.main(["--claim", str(path), "--json-out", str(out)])
    v = json.loads(out.read_text())
    assert v["spread_pp"] == pytest.approx(1.0)
    assert v["spread_pp"] > mod.SPREAD_FLOOR   # would trigger the ground-rule-1 update


def test_control_with_a_non_W_label_is_still_found(tmp_path):
    """The scorer labels whatever it is given; do not depend on the W prefix."""
    claim = _report(tmp_path / "claim.json", [42, 43, 44, 45, 46], mindcf=0.65)
    control = _report(tmp_path / "control.json", [42], mindcf=0.90, prefix="CTRL")
    out = tmp_path / "verdict.json"
    mod.main(["--claim", str(claim), "--control", str(control), "--json-out", str(out)])
    assert json.loads(out.read_text())["criterion3"] is True
