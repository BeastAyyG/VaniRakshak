"""The ASV term must not penalise a caller who claimed no identity.

Regression guard for a real defect: when nobody claims an identity the ASV
cannot run and returns the neutral 0.5, which ``calibrate()`` used to score as
a 0.5 mismatch. That added a flat ``w_asv * 0.5`` (0.8 at the default weight)
to every anonymous caller — enough on its own to move genuine audio from ALLOW
(risk ~21) to CHALLENGE (risk ~37).

The fix distinguishes three states, so all three are pinned here. Cases 2 and 3
both produce ``(asv_score=0.5, asv_enrolled=False)``; only ``asv_claimed``
separates them, which is why that field exists.
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

from vanirakshak.config import FusionConfig, RiskConfig, settings
from vanirakshak.detectors.pipeline import WindowResult
from vanirakshak.risk.fusion import calibrate, to_tier


def _window(asv_score: float, asv_enrolled: bool, asv_claimed: bool) -> WindowResult:
    """A clean, low-anomaly window; only the identity state varies."""
    return WindowResult(
        cm_score=3.0,
        asv_score=asv_score,
        asv_enrolled=asv_enrolled,
        asv_claimed=asv_claimed,
        channel_anomaly=0.0,
        prosody_unnatural=0.0,
        codec_guess="none",
        snr_db=20.0,
        features={"cm_genuine_prob": float(1.0 / (1.0 + np.exp(-3.0)))},
    )


def _risk(window: WindowResult) -> float:
    return 100.0 * calibrate(window, context_risk=0.0, fusion=settings.fusion)


def test_no_claim_contributes_no_asv_evidence() -> None:
    """Case 3: no claim at all -> the ASV term must be exactly zero."""
    anon = _window(0.5, asv_enrolled=False, asv_claimed=False)
    # Same window, but with the ASV term forced off by hand.
    expected_z = settings.fusion.intercept + settings.fusion.w_cm * (
        1.0 - anon.features["cm_genuine_prob"]
    )
    got_z = float(np.log(calibrate(anon, 0.0, settings.fusion) / (1.0 - calibrate(anon, 0.0, settings.fusion))))
    assert abs(got_z - expected_z) < 1e-9, (
        f"anonymous caller picked up an ASV term: z={got_z} vs {expected_z}"
    )


def test_unverifiable_claim_keeps_the_half_mismatch_policy() -> None:
    """Case 2: a claim nobody enrolled stays at the neutral 0.5 (unchanged)."""
    claimed = _window(0.5, asv_enrolled=False, asv_claimed=True)
    z = float(np.log(calibrate(claimed, 0.0, settings.fusion) / (1.0 - calibrate(claimed, 0.0, settings.fusion))))
    expected = (
        settings.fusion.intercept
        + settings.fusion.w_cm * (1.0 - claimed.features["cm_genuine_prob"])
        + settings.fusion.w_asv * 0.5
    )
    assert abs(z - expected) < 1e-9, f"unenrolled claim changed policy: z={z} vs {expected}"


def test_anonymous_genuine_is_not_pushed_into_challenge() -> None:
    """The user-visible symptom: genuine audio must stay ALLOW anonymously."""
    anon = _risk(_window(0.5, asv_enrolled=False, asv_claimed=False))
    claimed = _risk(_window(0.5, asv_enrolled=False, asv_claimed=True))
    assert anon < claimed, f"anonymous risk {anon:.2f} should be below claim risk {claimed:.2f}"
    assert to_tier(anon, settings.risk) == "ALLOW", f"anonymous genuine landed in {to_tier(anon, settings.risk)}"


def test_enrolled_identity_still_counts() -> None:
    """Case 1: a real enrolled mismatch must still raise risk (no regression)."""
    match = _risk(_window(0.78, asv_enrolled=True, asv_claimed=True))
    mismatch = _risk(_window(0.01, asv_enrolled=True, asv_claimed=True))
    assert mismatch > match, f"impostor {mismatch:.2f} should exceed match {match:.2f}"


def test_window_result_exposes_asv_claimed() -> None:
    """The flag calibrates on is part of the contract, not an implementation detail."""
    w = _window(0.5, asv_enrolled=False, asv_claimed=True)
    assert "asv_claimed" in w.to_dict(), "asv_claimed missing from the serialised window"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [ok] {name}")
    print("\nAll fusion no-identity tests passed")
