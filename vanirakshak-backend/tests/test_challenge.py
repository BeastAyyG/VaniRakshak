"""Tests for W2 / P1-C: Unforgeable challenge response.

Exit criterion P1-C (docs/14 W2 / docs/13 §1.2, docs/15 N2):
A scripted client that sends the correct digits over silent or synthetic/TTS
response audio fails the challenge in >= 90% of trials.

Digits are graded from the server's transcript of the response audio, so the
client's ``response_text`` carries no weight. These tests inject a transcriber
that returns a fixed "heard" text; ``scripts/challenge_forgery_harness.py``
runs the same attacks against the real local ASR engine.
"""

import dataclasses
import numpy as np
import pytest

from vanirakshak.challenge.engine import ENGLISH_WORDS, HINDI_WORDS, extract_digits
from vanirakshak.challenge.transcribe import ChallengeTranscriber
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


class HeardTranscriber:
    """Returns ``heard(session)`` as the transcript and records each call."""

    def __init__(self, heard=lambda sess: ""):
        self.heard = heard
        self.sess = None
        self.calls = []

    def transcribe(self, pcm_int16, sample_rate):
        self.calls.append((pcm_int16.size, sample_rate))
        return self.heard(self.sess), None


def _digits_said(sess) -> str:
    """What a TTS clone or a genuine caller says: the right digits."""
    return " ".join(sess._grading_digits)


def _session(name, windows, transcriber=None, **cfg):
    tr = transcriber or HeardTranscriber()
    sess = CallSession(
        name,
        SessionContext(claimed_identity="user_1"),
        pipeline=_ScriptedPipeline(windows),
        fusion_engine=RiskFusionEngine(RiskConfig(), REAL_FUSION),
        challenge_cfg=_challenge_cfg(**cfg),
        challenge_transcriber=tr,
    )
    tr.sess = sess
    return sess


def _push_window(sess: CallSession):
    """Push 1s chunks until a window is produced."""
    cr = None
    while cr is None:
        cr = sess.push_chunk(np.zeros(16000, dtype=np.int16))
    return cr


def _answer_window(sess: CallSession):
    """Push one window of response audio that contains a voice: the first
    answer chunk opens with 0.8 s of silence, so the server-measured voice
    onset lands at ~800 ms, inside the human band (docs/15 W2.b)."""
    t = np.arange(16000) / 16000.0
    tone = (6000 * np.sin(2 * np.pi * 180 * t)).astype(np.int16)
    cr = None
    while cr is None:
        chunk = tone.copy()
        if not getattr(sess, "_answer_started", False):
            chunk[:12800] = 0
            sess._answer_started = True
        cr = sess.push_chunk(chunk)
    return cr


def _issue(sess) -> str:
    """Trigger the challenge; return the correct answer a scripted client sends."""
    cr = _push_window(sess)
    assert cr.trigger_challenge
    # expected_digits is not on the wire (W2); read the internal challenge.
    # Kept on the session for the "heard" lambdas, since grading clears it.
    sess._grading_digits = list(sess._active_challenge.expected_digits)
    return " ".join(sess._grading_digits)


def test_scripted_client_without_response_audio_fails_p1_c():
    """Correct digits in response_text and no response audio at all."""
    trials, failures = 100, 0
    for i in range(trials):
        sess = _session(f"no-audio-{i}", [BORDERLINE])
        correct = _issue(sess)
        grade = sess.grade_challenge(correct + " uh ok", 950)
        if not grade["ok"] and grade["reason"] == "no_response_audio" and sess.blocked:
            failures += 1
    assert failures / trials >= 0.90


def test_scripted_client_over_silent_response_audio_fails_p1_c():
    """Correct digits in response_text, silence on the wire: nothing is heard."""
    trials, failures = 100, 0
    for i in range(trials):
        sess = _session(f"silent-{i}", [BORDERLINE] + [GENUINE] * 3)
        correct = _issue(sess)
        for _ in range(3):
            _push_window(sess)
        grade = sess.grade_challenge(correct + " uh ok", 950)
        if not grade["ok"] and grade["reason"] == "no_voice_onset" and sess.blocked:
            failures += 1
    assert failures / trials >= 0.90


def test_scripted_client_over_wrong_speech_fails_p1_c():
    """Correct digits in response_text, audio saying other digits: the audio wins."""
    trials, failures = 100, 0
    for i in range(trials):
        wrong = HeardTranscriber(lambda s: " ".join(str((int(d) + 1) % 10) for d in s._grading_digits))
        sess = _session(f"wrong-{i}", [BORDERLINE] + [GENUINE] * 3, transcriber=wrong)
        correct = _issue(sess)
        for _ in range(3):
            _answer_window(sess)
        grade = sess.grade_challenge(correct, 1000)
        if not grade["ok"] and grade["reason"] == "digit_order_mismatch" and sess.blocked:
            failures += 1
    assert failures / trials >= 0.90


def test_scripted_client_over_synthetic_audio_fails_p1_c():
    """A TTS clone that says the right digits passes the transcript; the
    response audio's CM evidence must stop it."""
    trials, failures = 100, 0
    rng = np.random.default_rng(2026)
    for i in range(trials):
        synth_windows = [
            WindowScore(cm_llr=float(rng.normal(-4.5, 1.0)), asv=0.95, asv_enrolled=True)
            for _ in range(3)
        ]
        sess = _session(f"synth-{i}", [BORDERLINE] + synth_windows,
                        transcriber=HeardTranscriber(_digits_said))
        correct = _issue(sess)
        for _ in range(3):
            _answer_window(sess)
        grade = sess.grade_challenge(correct + " um okay", 1100)
        if not grade["ok"] and grade["reason"] == "response_voice_synthetic" and sess.blocked:
            failures += 1
    assert failures / trials >= 0.90


def test_genuine_response_passes_challenge():
    """A genuine user speaking correct digits with authentic voice passes."""
    sess = _session("genuine", [BORDERLINE, GENUINE, GENUINE],
                    transcriber=HeardTranscriber(_digits_said))
    _issue(sess)
    _answer_window(sess)
    _answer_window(sess)
    # The client's text is ignored: it can be empty and still pass.
    grade = sess.grade_challenge("", 1200)
    assert grade["ok"] is True, grade
    assert grade["reason"] == "ok"
    assert grade["grade_source"] == "server_asr"
    assert grade["transcript"] == " ".join(sess._grading_digits)
    assert grade["acoustic"]["verdict"] == "pass"
    assert not sess.blocked


def test_no_local_engine_fails_closed():
    """Without a transcript there is no evidence the caller said the digits."""
    # No challenge_transcriber: conftest's NoEngineTranscriber is the default.
    sess = CallSession(
        "no-engine",
        SessionContext(claimed_identity="user_1"),
        pipeline=_ScriptedPipeline([BORDERLINE, GENUINE, GENUINE]),
        fusion_engine=RiskFusionEngine(RiskConfig(), REAL_FUSION),
        challenge_cfg=_challenge_cfg(),
    )
    correct = _issue(sess)
    _answer_window(sess)
    _answer_window(sess)
    grade = sess.grade_challenge(correct, 1000)
    assert grade["ok"] is False
    assert grade["reason"] == "challenge_transcriber_unavailable"
    assert grade["asr_error"]
    assert sess.blocked


def test_client_text_drill_grades_response_text():
    """The drill flag restores the old, forgeable behaviour, and says so."""
    tr = HeardTranscriber()
    sess = _session("drill", [BORDERLINE, GENUINE, GENUINE], transcriber=tr, client_text_drill=True)
    correct = _issue(sess)
    _answer_window(sess)
    _answer_window(sess)
    grade = sess.grade_challenge(correct, 1000)
    assert grade["ok"] is True
    assert grade["grade_source"] == "client_text_drill"
    assert grade["transcript"] is None
    assert tr.calls == []


def test_response_audio_is_buffered_from_issue_and_capped():
    tr = HeardTranscriber(_digits_said)
    sess = _session("buffer", [BORDERLINE] + [GENUINE] * 8, transcriber=tr, response_max_seconds=2.5)
    _issue(sess)
    assert sess._challenge_audio_samples == 0  # audio before issue is not the answer
    for _ in range(4):
        _answer_window(sess)
    assert sess._challenge_audio_samples == int(2.5 * 16000)
    sess.grade_challenge("", 1000)
    assert tr.calls == [(int(2.5 * 16000), 16000)]
    assert sess._challenge_audio == [] and sess._challenge_audio_samples == 0


def test_transcriber_refuses_cloud_provider():
    tr = ChallengeTranscriber(provider="groq")
    text, err = tr.transcribe(np.ones(16000, dtype=np.int16), 16000)
    assert text is None
    assert "not a local engine" in err


@pytest.mark.parametrize("text,digits", [
    ("3 5", ["3", "5"]),
    ("three, five.", ["3", "5"]),
    ("Teen paanch", ["3", "5"]),
    ("तीन पाँच", ["3", "5"]),  # तीन पाँच
    ("३५", ["3", "5"]),  # ३५
    ("Thank you.", []),
    ("seventy", []),
])
def test_extract_digits(text, digits):
    assert extract_digits(text) == digits


def test_phrase_words_are_not_number_words():
    """A distractor word must never be read as a spoken digit."""
    assert extract_digits(" ".join(HINDI_WORDS + ENGLISH_WORDS)) == []


def test_health_reports_challenge_grading_mode():
    import importlib
    from fastapi.testclient import TestClient

    app_module = importlib.import_module("vanirakshak.server.app")
    grading = TestClient(app_module.app).get("/v1/health").json()["challenge_grading"]
    assert grading["source"] == "server_asr"
    assert grading["asr_provider"] == "local"


def test_latency_is_measured_by_the_server_not_the_client():
    """W2.b: the client's response_latency_ms has no effect under server grading."""
    sess = _session("latency", [BORDERLINE, GENUINE, GENUINE],
                    transcriber=HeardTranscriber(_digits_said))
    _issue(sess)
    _answer_window(sess)
    _answer_window(sess)
    # 50 ms would fail the band if the client's number were used.
    grade = sess.grade_challenge("", 50)
    assert grade["ok"] is True, grade
    assert 750 <= grade["latency_ms"] <= 850


def test_instant_answer_fails_server_latency():
    """Voice from the very first sample (a pre-queued TTS) is below the band."""
    t = np.arange(16000) / 16000.0
    tone = (6000 * np.sin(2 * np.pi * 180 * t)).astype(np.int16)
    sess = _session("instant", [BORDERLINE, GENUINE, GENUINE],
                    transcriber=HeardTranscriber(_digits_said))
    _issue(sess)
    for _ in range(2):
        cr = None
        while cr is None:
            cr = sess.push_chunk(tone)
    grade = sess.grade_challenge("", 1000)
    assert grade["ok"] is False
    assert grade["reason"] == "latency_suspicious"
    assert grade["latency_ms"] == 0


def test_answer_is_decoded_once_per_expected_language():
    """W2.c: detection on a two-word Hindi answer picks English and spells
    "chhe teen" as "Chae deen". Each expected language is forced in turn."""
    from vanirakshak.challenge.transcribe import _FasterWhisperDigits

    class Seg:
        def __init__(self, text):
            self.text = text

    class FakeModel:
        def __init__(self):
            self.languages = []

        def transcribe(self, audio, language=None, **kw):
            self.languages.append(language)
            return [Seg({"hi": "छह तीन", "en": "Chae deen."}[language])], None

    eng = _FasterWhisperDigits.__new__(_FasterWhisperDigits)
    eng.model, eng.languages = FakeModel(), ["hi", "en"]
    readings = eng.candidates(np.zeros(16000, dtype=np.float32))
    assert eng.model.languages == ["hi", "en"]
    assert [extract_digits(r) for r in readings] == [["6", "3"], ["6", "3"]]


class ReadingsTranscriber(HeardTranscriber):
    """Two readings, like the real engine: only the second has the digits."""

    def transcribe_candidates(self, pcm_int16, sample_rate):
        self.calls.append((pcm_int16.size, sample_rate))
        return ["garbled", self.heard(self.sess)], None


def test_any_reading_with_the_digits_passes():
    sess = _session("readings", [BORDERLINE, GENUINE, GENUINE],
                    transcriber=ReadingsTranscriber(_digits_said))
    _issue(sess)
    _answer_window(sess)
    _answer_window(sess)
    grade = sess.grade_challenge("", 1000)
    assert grade["ok"] is True, grade
    assert grade["transcript"].split("\n")[0] == "garbled"


def test_no_reading_with_the_digits_fails():
    wrong = ReadingsTranscriber(lambda s: " ".join(str((int(d) + 1) % 10) for d in s._grading_digits))
    sess = _session("readings-wrong", [BORDERLINE, GENUINE, GENUINE], transcriber=wrong)
    _issue(sess)
    _answer_window(sess)
    _answer_window(sess)
    grade = sess.grade_challenge("", 1000)
    assert grade["ok"] is False and grade["reason"] == "digit_order_mismatch"


def test_devanagari_spelling_variants_parse():
    """ASR spells the same spoken digit many ways; all must read as it."""
    cases = {
        "शूँनि दो": ["0", "2"],   # shoonni do
        "शुने पाच": ["0", "5"],   # shune paach
        "तीन साथ": ["3", "7"],          # teen saath
        "चार तीं": ["4", "3"],          # char teen (anusvara)
        "इक नों": ["1", "9"],                 # ik non
        "आठ छह": ["8", "6"],                        # aath chhah
    }
    for text, digits in cases.items():
        assert extract_digits(text) == digits, text
