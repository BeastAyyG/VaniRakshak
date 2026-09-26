"""Tests for W2 / P1-C: Unforgeable challenge response.

Exit criterion P1-C (docs/14 W2 / docs/13 §1.2):
A scripted client that sends the correct digits over silent or synthetic/TTS
response audio fails the challenge in >= 90% of trials.
"""

import dataclasses
import numpy as np
import pytest

from vanirakshak.config import ChallengeConfig, FusionConfig, RiskConfig, _REAL_BACKEND_OVERRIDES
from vanirakshak.risk.fusion import RiskFusionEngine
from vanirakshak.risk.simulate import WindowScore, _ScriptedPipeline
from vanirakshak.session import CallSession, SessionContext

REAL_FUSION = dataclasses.replace(FusionConfig(), **_REAL_BACKEND_OVERRIDES)
BORDERLINE = WindowScore(cm_llr=-1.0, asv=0.9, asv_enrolled=True)
SYNTH = WindowScore(cm_llr=-5.0, asv=0.95, asv_enrolled=True)
GENUINE = WindowScore(cm_llr=6.0, asv=0.95, asv_enrolled=True)


def _challenge_cfg(**kw) -> ChallengeConfig:
    defaults = {
        "cm_llr_max": 0.5757,  # fitted dev threshold
        "asv_min": None,
        "min_windows": 2,
        "require_audio": True,
        "max_per_session": 1,
        "trigger_consecutive": 1,
    }
    defaults.update(kw)
    return ChallengeConfig(**defaults)


def _push_window(sess: CallSession):
    """Push 1s chunks until a window is produced."""
    cr = None
    while cr is None:
        cr = sess.push_chunk(np.zeros(16000, dtype=np.int16))
    return cr


def test_scripted_client_over_silent_audio_fails_p1_c():
    """A scripted client sending correct digits without spoken response audio fails."""
    trials = 100
    failures = 0

    for i in range(trials):
        sess = CallSession(
            f"test-silent-{i}",
            SessionContext(claimed_identity="user_1"),
            pipeline=_ScriptedPipeline([BORDERLINE]),
            fusion_engine=RiskFusionEngine(RiskConfig(), REAL_FUSION),
            challenge_cfg=_challenge_cfg(require_audio=True, min_windows=2),
        )
        cr = _push_window(sess)
        assert cr.trigger_challenge
        digits = sess._active_challenge.expected_digits if sess._active_challenge else []
        correct_text = " ".join(digits) + " uh ok"

        # Client grades without sending any response audio chunks
        grade = sess.grade_challenge(correct_text, 950)
        if not grade["ok"] and grade["reason"] == "no_response_audio" and sess.blocked:
            failures += 1

    fail_rate = failures / trials
    assert fail_rate >= 0.90, f"Expected silent failure rate >= 0.90, got {fail_rate:.2f}"


def test_scripted_client_over_synthetic_audio_fails_p1_c():
    """A scripted client sending correct digits over synthetic/cloned response audio fails."""
    trials = 100
    failures = 0
    rng = np.random.default_rng(2026)

    for i in range(trials):
        synth_windows = [
            WindowScore(cm_llr=float(rng.normal(-4.5, 1.0)), asv=0.95, asv_enrolled=True)
            for _ in range(3)
        ]
        sess = CallSession(
            f"test-synth-{i}",
            SessionContext(claimed_identity="user_1"),
            pipeline=_ScriptedPipeline([BORDERLINE] + synth_windows),
            fusion_engine=RiskFusionEngine(RiskConfig(), REAL_FUSION),
            challenge_cfg=_challenge_cfg(cm_llr_max=0.5757, min_windows=2, require_audio=True),
        )

        cr = _push_window(sess)
        assert cr.trigger_challenge
        digits = sess._active_challenge.expected_digits if sess._active_challenge else []
        correct_text = " ".join(digits) + " um okay"

        # Stream response windows
        for _ in range(3):
            _push_window(sess)

        # Client grades with perfect text and human latency
        grade = sess.grade_challenge(correct_text, 1100)
        if (
            not grade["ok"]
            and grade["reason"] == "response_voice_synthetic"
            and sess.blocked
        ):
            failures += 1

    fail_rate = failures / trials
    assert fail_rate >= 0.90, f"Expected synthetic failure rate >= 0.90, got {fail_rate:.2f}"


def test_genuine_response_passes_challenge():
    """A genuine user speaking correct digits with authentic voice passes."""
    sess = CallSession(
        "test-genuine",
        SessionContext(claimed_identity="user_1"),
        pipeline=_ScriptedPipeline([BORDERLINE, GENUINE, GENUINE]),
        fusion_engine=RiskFusionEngine(RiskConfig(), REAL_FUSION),
        challenge_cfg=_challenge_cfg(cm_llr_max=0.5757, min_windows=2, require_audio=True),
    )
    cr = _push_window(sess)
    assert cr.trigger_challenge
    digits = sess._active_challenge.expected_digits if sess._active_challenge else []
    correct_text = " ".join(digits) + " uh ok"

    # Push 2 response audio windows
    _push_window(sess)
    _push_window(sess)

    grade = sess.grade_challenge(correct_text, 1200)
    assert grade["ok"] is True
    assert grade["reason"] == "ok"
    assert grade["acoustic"]["verdict"] == "pass"
    assert not sess.blocked
