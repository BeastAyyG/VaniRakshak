"""Tests for the run-1 gate checker.

The bars in `check_run1_gate.py` are *pre-registered*: they were copied from
docs/10 before the full-partition result existed, precisely so the verdict could
not be read post-hoc. That makes the constants themselves load-bearing — if one
drifts, a headline claim silently changes grade. These tests pin them, pin the
strict-comparison boundary, and check the verdict on the numbers already on
record (the 20 k slice), where the answers are known.
"""

from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "check_run1_gate", REPO_ROOT / "scripts" / "check_run1_gate.py"
)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

SLICE_REPORT = REPO_ROOT / "reports" / "cm_from_scratch_2026-09-14.json"

# Metrics exactly as recorded in reports/cm_from_scratch_2026-09-14.json.
RECORDED = {
    "S42": {"eer": 0.113356, "min_dcf": 0.651505},
    "B42": {"eer": 0.137749, "min_dcf": 0.868772},
    "S43": {"eer": 0.118555, "min_dcf": 0.726110},
    "B43": {"eer": 0.120503, "min_dcf": 0.819576},
}


# --------------------------------------------------------------------------
# The pre-registered bars
# --------------------------------------------------------------------------
def test_bars_match_docs10():
    assert mod.G3_EER == 11.0
    assert mod.G3_MINDCF == 0.75
    assert mod.CRIT2_STATED == 12.666
    assert mod.CRIT2_JUSTIFIED == 11.866


def test_criterion_1_needs_three_of_five():
    assert mod.MIN_SEEDS_FOR_CRIT1 == 3
    assert len(mod.CLAIM_SEEDS) == 5


def test_the_five_claim_seeds_are_the_S_condition_only():
    # B42/B43 are the no-augmentation ablation and are explicitly NOT part of
    # the 5-seed claim. Conflating them would inflate the apparent seed count.
    assert mod.CLAIM_SEEDS == ("S42", "S43", "S44", "S45", "S46")
    assert mod.ABLATION == ("B42", "B43")
    assert not set(mod.CLAIM_SEEDS) & set(mod.ABLATION)


# --------------------------------------------------------------------------
# Comparison semantics
# --------------------------------------------------------------------------
def test_cleared_is_strictly_below():
    # "below 0.75" must not be satisfied by exactly 0.75.
    assert mod.cleared(0.7499, 0.75) is True
    assert mod.cleared(0.75, 0.75) is False
    assert mod.cleared(0.7501, 0.75) is False


def test_pct_converts_fraction_to_percent():
    assert mod.pct(0.113356) == pytest.approx(11.3356)


def test_check_label_reports_percentages_and_booleans():
    r = mod.check_label(RECORDED["S42"])
    assert r["eer"] == pytest.approx(11.3356)
    assert r["mindcf"] == pytest.approx(0.651505)
    assert r["g3_mindcf"] is True
    assert r["g3_eer"] is False          # 11.3356 is not below 11.0
    assert r["crit2_stated"] is True
    assert r["crit2_justified"] is True


# --------------------------------------------------------------------------
# The numbers already on record
# --------------------------------------------------------------------------
def test_S43_clears_the_justified_reading_by_only_a_hair():
    r = mod.check_label(RECORDED["S43"])
    # 11.8555 vs 11.866 -- a 0.011 pp margin, well inside noise. It passes,
    # but it must not look comfortable.
    assert r["crit2_justified"] is True
    assert mod.CRIT2_JUSTIFIED - r["eer"] == pytest.approx(0.0105, abs=5e-4)
    assert r["g3_mindcf"] is True
    assert mod.G3_MINDCF - r["mindcf"] == pytest.approx(0.0239, abs=5e-4)


def test_S42_clears_mindcf_with_room():
    r = mod.check_label(RECORDED["S42"])
    assert r["g3_mindcf"] is True
    assert mod.G3_MINDCF - r["mindcf"] == pytest.approx(0.0985, abs=5e-4)


def test_the_no_augmentation_ablation_fails_every_bar():
    for label in ("B42", "B43"):
        r = mod.check_label(RECORDED[label])
        assert r["g3_mindcf"] is False, label
        assert r["crit2_justified"] is False, label


def test_B43_would_pass_only_the_loosest_reading():
    # B43 (12.0503 %) sneaks under the stated 12.666 % but not the justified
    # 11.866 %. Recording this stops the stated reading being used to make the
    # ablation look better than it is.
    r = mod.check_label(RECORDED["B43"])
    assert r["crit2_stated"] is True
    assert r["crit2_justified"] is False


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def test_load_models_reads_the_committed_slice_report():
    models = mod.load_models(SLICE_REPORT)
    assert set(models) == {"S42", "B42", "S43", "B43"}
    # The report stores full precision (0.11335633316904879); 0.113356 is the
    # 6-dp value quoted in docs. Compare at that precision, not at approx's
    # default rel=1e-6, which is tighter than 6 dp for these magnitudes.
    assert models["S42"]["eer"] == pytest.approx(0.113356, abs=1e-6)


def test_load_models_accepts_metrics_inlined_in_the_block(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"models": {"S1": {"eer": 0.1, "min_dcf": 0.5}}}))
    assert mod.load_models(p)["S1"]["min_dcf"] == 0.5


def test_load_models_rejects_a_report_without_models(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"n_records": 5}))
    with pytest.raises(SystemExit):
        mod.load_models(p)


def test_load_models_rejects_a_model_missing_its_metrics(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"models": {"S1": {"eer": 0.1}}}))
    with pytest.raises(SystemExit):
        mod.load_models(p)


def test_load_models_rejects_a_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        mod.load_models(tmp_path / "nope.json")


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
def test_main_reports_undecided_while_only_two_seeds_exist():
    # The claim is defined over 5 seeds. With 2 present the script must refuse
    # to grade it, rather than reporting a partial pass as a pass.
    rc = mod.main(["--full", str(SLICE_REPORT)])
    assert rc == 0


def test_main_writes_a_json_summary(tmp_path):
    out = tmp_path / "summary.json"
    mod.main(["--full", str(SLICE_REPORT), "--slice", str(SLICE_REPORT),
              "--json-out", str(out)])
    data = json.loads(out.read_text())
    assert data["claim"]["n_seeds"] == 2
    assert data["claim"]["decidable"] is False
    assert data["claim"]["n_min_dcf"] == 2
    assert {r["label"] for r in data["rows"]} == {"S42", "B42", "S43", "B43"}


def test_main_flags_a_claim_seed_that_fails_a_bar(tmp_path):
    # If a claim seed misses the stated EER bar the script must exit nonzero.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"models": {
        "S42": {"eer": 0.99, "min_dcf": 0.99},
    }}))
    assert mod.main(["--full", str(bad)]) == 1
