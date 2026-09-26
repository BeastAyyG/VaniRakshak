"""Decision layer (docs/13 Phase 1): fusion pin, clone rule, voice-scored and
re-armable challenge, trigger persistence, session simulator. Torch-free."""

import dataclasses

import pytest

from vanirakshak.config import (
    _REAL_BACKEND_OVERRIDES,
    ChallengeConfig,
    FusionConfig,
    RiskConfig,
)
from vanirakshak.risk.fusion import RiskFusionEngine, calibrate
from vanirakshak.risk.simulate import WindowScore, simulate_session, summarise

REAL_FUSION = dataclasses.replace(FusionConfig(), **_REAL_BACKEND_OVERRIDES)
BASE_RISK = RiskConfig()


def _risk(**kw) -> RiskConfig:
    kw.setdefault("clone_cm_llr_max", None)
    return dataclasses.replace(BASE_RISK, **kw)


def _challenge(**kw) -> ChallengeConfig:
    return dataclasses.replace(
        ChallengeConfig(cm_llr_max=None, asv_min=None, min_windows=2, require_audio=False,
                        max_per_session=1, trigger_consecutive=1),
        **kw,
    )


# -- 1.5 fusion pin ----------------------------------------------------------
# docs/13 Amendment 1 table: a CM certain the audio is fake (cm_spoof = 1).
# A fusion refit that moves these fails here instead of silently
# invalidating every Phase 1 gate.
@pytest.mark.parametrize(
    "asv,channel,prosody,expected",
    [
        (1.00, 0.0, 0.0, 47.5),
        (0.95, 0.0, 0.0, 50.0),
        (0.80, 0.0, 0.0, 57.4),
        (1.00, 0.25, 0.5, 61.7),
        (0.95, 0.25, 0.5, 64.0),
        (0.80, 0.25, 0.5, 70.6),
    ],
)
def test_fusion_operating_points_pinned(asv, channel, prosody, expected):
    w = WindowScore(cm_llr=-50.0, asv=asv, asv_enrolled=True,
                    channel_anomaly=channel, prosody_unnatural=prosody).to_result()
    w.features["cm_genuine_prob"] = 0.0
    assert round(100 * calibrate(w, 0.0, REAL_FUSION), 1) == expected


def test_real_backend_weights_pinned():
    assert (REAL_FUSION.intercept, REAL_FUSION.w_cm, REAL_FUSION.w_asv) == (-1.6, 1.5, 2.0)


# -- 1.1 clone rule ------------------------------------------------------------
def _clone_window(llr=-6.0, asv=0.97, enrolled=True):
    return WindowScore(cm_llr=llr, asv=asv, asv_enrolled=enrolled).to_result()


def test_perfect_clone_is_not_blocked_without_the_rule():
    eng = RiskFusionEngine(_risk(), REAL_FUSION)
    tiers = {eng.update(_clone_window()).tier for _ in range(10)}
    assert "BLOCK" not in tiers


def test_clone_rule_blocks_after_min_windows():
    eng = RiskFusionEngine(_risk(clone_cm_llr_max=-2.0, clone_min_windows=3), REAL_FUSION)
    d1, d2, d3 = (eng.update(_clone_window()) for _ in range(3))
    assert d1.tier != "BLOCK" and d2.tier != "BLOCK"
    assert d3.tier == "BLOCK" and d3.override == "clone"
    assert d3.risk >= BASE_RISK.challenge_threshold


def test_clone_rule_ignores_unenrolled_and_resets_on_low_asv():
    eng = RiskFusionEngine(_risk(clone_cm_llr_max=-2.0, clone_min_windows=2), REAL_FUSION)
    for _ in range(5):
        assert eng.update(_clone_window(asv=0.5, enrolled=False)).override is None
    eng.update(_clone_window())
    eng.update(_clone_window(asv=0.05))  # impostor window; breaks the clone streak
    assert eng.update(_clone_window()).override is None
    assert eng.update(_clone_window()).override == "clone"


def test_clone_rule_spares_genuine_scores():
    eng = RiskFusionEngine(_risk(clone_cm_llr_max=-2.0, clone_min_windows=3), REAL_FUSION)
    for llr in (3.0, -4.0, 2.5, -3.0, 4.0, 1.0):  # occasional low windows, mean well above -2
        assert eng.update(_clone_window(llr=llr)).override != "clone"


def test_impostor_override_takes_precedence():
    eng = RiskFusionEngine(_risk(clone_cm_llr_max=10.0, clone_min_windows=1), REAL_FUSION)
    d = eng.update(_clone_window(asv=0.05))
    assert d.tier == "BLOCK" and d.override == "impostor"


# -- 1.2 challenge: scored response audio, re-arm, persistence ----------------
def _run(windows, *, challenge_cfg, responder=lambda _i: True, response_windows=3):
    return simulate_session(
        windows, risk_cfg=_risk(), fusion_cfg=REAL_FUSION, challenge_cfg=challenge_cfg,
        challenge_responder=responder, response_windows=response_windows,
    )


BORDERLINE = WindowScore(cm_llr=-1.0, asv=0.9, asv_enrolled=True)  # CHALLENGE tier
GOOD = WindowScore(cm_llr=6.0, asv=0.95, asv_enrolled=True)       # ALLOW tier
SYNTH = WindowScore(cm_llr=-5.0, asv=0.95, asv_enrolled=True)      # clone-like response


def test_borderline_window_is_challenge_tier():
    eng = RiskFusionEngine(_risk(), REAL_FUSION)
    assert eng.update(BORDERLINE.to_result()).tier == "CHALLENGE"
    eng.reset()
    assert eng.update(GOOD.to_result()).tier == "ALLOW"


def test_text_only_grading_passes_a_synthetic_response_by_default():
    out = _run([BORDERLINE] + [SYNTH] * 4, challenge_cfg=_challenge())
    assert out.challenges_issued == 1 and not out.blocked


def test_synthetic_response_audio_fails_the_challenge():
    out = _run([BORDERLINE] + [SYNTH] * 4, challenge_cfg=_challenge(cm_llr_max=-2.0))
    assert out.blocked


def test_genuine_response_audio_passes_the_challenge():
    out = _run([BORDERLINE] + [GOOD] * 4, challenge_cfg=_challenge(cm_llr_max=-2.0))
    assert not out.blocked


def test_mismatched_response_voice_fails_the_challenge():
    other = WindowScore(cm_llr=6.0, asv=0.3, asv_enrolled=True)
    out = _run([BORDERLINE] + [other] * 4, challenge_cfg=_challenge(asv_min=0.6))
    assert out.blocked


def test_require_audio_fails_a_challenge_graded_without_response_audio():
    from vanirakshak.session import CallSession, SessionContext
    from vanirakshak.risk.simulate import _ScriptedPipeline

    sess = CallSession("t", SessionContext(), pipeline=_ScriptedPipeline([BORDERLINE]),
                       fusion_engine=RiskFusionEngine(_risk(), REAL_FUSION),
                       challenge_cfg=_challenge(require_audio=True))
    import numpy as np
    cr = None
    while cr is None:
        cr = sess.push_chunk(np.zeros(16000, dtype=np.int16))
    assert cr.trigger_challenge
    res = sess.grade_challenge(" ".join(cr.challenge["expected_digits"]), 900)
    assert res["ok"] is False and res["reason"] == "no_response_audio"
    assert res["acoustic"]["windows"] == 0 and sess.blocked


def test_one_shot_by_default_rearm_when_enabled():
    windows = [BORDERLINE] + [GOOD] * 5 + [BORDERLINE] * 2 + [GOOD] * 4
    assert _run(windows, challenge_cfg=_challenge()).challenges_issued == 1
    assert _run(windows, challenge_cfg=_challenge(max_per_session=3)).challenges_issued == 2


def test_trigger_consecutive_filters_single_borderline_windows():
    flicker = [BORDERLINE, GOOD, GOOD, GOOD] * 5
    assert _run(flicker, challenge_cfg=_challenge()).challenges_issued == 1
    assert _run(flicker, challenge_cfg=_challenge(trigger_consecutive=3)).challenges_issued == 0
    sustained = [BORDERLINE] * 4 + [GOOD] * 4
    assert _run(sustained, challenge_cfg=_challenge(trigger_consecutive=3)).challenges_issued == 1


# -- 1.4 simulator ---------------------------------------------------------------
def test_simulator_timing_and_summary():
    # EMA: the first borderline window after ALLOW lands at 34 (ALLOW), the second at 39.5.
    out = _run([GOOD] * 3 + [BORDERLINE] * 3, challenge_cfg=_challenge(), responder=None)
    assert out.first_challenge_s == pytest.approx(4.04 + 4)
    s = summarise([out, _run([GOOD] * 7, challenge_cfg=_challenge(), responder=None)])
    assert s["sessions"] == 2 and s["p_challenge_within_30s"] == 0.5 and s["p_block_by_end"] == 0.0
