"""docs/14 W5: per-user ASV calibration fitted against a non-target cohort."""

from __future__ import annotations

import importlib
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vanirakshak.data.synthetic import synthetic_speech
from vanirakshak.detectors.asv import ASVModel
from vanirakshak.detectors.calibration import Cohort, fit_user_calibration, load_cohort

SR = 16000


def _other_speakers(n=8):
    """Stand-in cohort: signals spectrally unlike the enrolled voice."""
    rng = np.random.default_rng(5)
    t = np.arange(4 * SR) / SR
    out = []
    for i in range(n):
        tone = 0.3 * np.sin(2 * np.pi * (300 + 150 * i) * t)
        out.append((tone + 0.05 * rng.normal(size=t.size)).astype(np.float32))
    return out


class ListCohort(Cohort):
    def __init__(self, clips):
        super().__init__(directory=__import__("pathlib").Path("unused"))
        self._clips = clips


def _write_wav(path, x, sr=SR):
    with wave.open(path if hasattr(path, "write") else str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def test_fit_separates_target_from_cohort():
    asv = ASVModel()
    voice = synthetic_speech(duration_sec=12.0, sr=SR, rng=np.random.default_rng(1))
    cal, info = fit_user_calibration(asv, "u1", voice, SR, ListCohort(_other_speakers()))
    assert info["method"] == "fitted_loo_vs_cohort", info
    assert cal is not None and cal["scale"] > 0
    assert info["target_scores_loo"]["mean"] > info["nontarget_scores"]["mean"]
    lo, hi = sorted((info["nontarget_scores"]["max"], info["target_scores_loo"]["min"]))
    assert lo <= cal["mid"] <= hi

    # With the fitted logistic, the user's own voice reads as a match and a
    # cohort speaker does not.
    asv.enrol("u1", voice, SR)
    asv._user_calibrations["u1"] = cal
    own, _ = asv.consistency_score("u1", synthetic_speech(duration_sec=4.0, sr=SR, rng=np.random.default_rng(2)), SR)
    other, _ = asv.consistency_score("u1", _other_speakers(1)[0], SR)
    assert own > 0.5 > other


def test_short_enrolment_falls_back_to_anchor_seed():
    voice = synthetic_speech(duration_sec=4.0, sr=SR, rng=np.random.default_rng(1))
    cal, info = fit_user_calibration(ASVModel(), "u1", voice, SR, ListCohort(_other_speakers()))
    assert cal is None
    assert info["method"] == "seeded_from_anchor"
    assert "enrolment audio" in info["reason"]


def test_missing_cohort_falls_back_to_anchor_seed(tmp_path):
    voice = synthetic_speech(duration_sec=12.0, sr=SR, rng=np.random.default_rng(1))
    cal, info = fit_user_calibration(ASVModel(), "u1", voice, SR, Cohort(tmp_path / "absent"))
    assert cal is None
    assert info["method"] == "seeded_from_anchor"
    assert "cohort" in info["reason"]


def test_load_cohort_reads_resamples_and_crops(tmp_path):
    long = np.zeros(12 * 8000, dtype=np.float32) + 0.1  # 12 s at 8 kHz
    _write_wav(tmp_path / "a.wav", long, sr=8000)
    _write_wav(tmp_path / "b.wav", np.zeros(SR // 4, dtype=np.float32))  # too short
    (tmp_path / "c.wav").write_bytes(b"not a wav")
    clips = load_cohort(tmp_path)
    assert len(clips) == 1
    assert clips[0].size == 8 * SR  # resampled to 16 kHz, cropped to 8 s


def test_enroll_reports_and_stores_calibration_method(monkeypatch):
    app_module = importlib.import_module("vanirakshak.server.app")
    from vanirakshak.server.auth import Authenticator

    monkeypatch.setattr(app_module, "AUTH", Authenticator(raw_keys="", require_auth=False,
                                                          rate_limits={"analyze": 0, "enroll": 0}))
    monkeypatch.setattr(app_module, "ASV_COHORT", ListCohort(_other_speakers()))
    monkeypatch.setenv("VR_ENROLL_BYPASS_QA", "1")
    voice = synthetic_speech(duration_sec=12.0, sr=SR, rng=np.random.default_rng(3))
    buf = io.BytesIO()
    _write_wav(buf, voice)
    r = TestClient(app_module.app).post(
        "/v1/enroll", data={"user_id": "w5_user"},
        files={"audio": ("a.wav", buf.getvalue(), "audio/wav")},
    )
    try:
        assert r.status_code == 200, r.text
        assert r.json()["calibration_method"] == "fitted_loo_vs_cohort"
        cal = app_module.PIPELINE.asv._user_calibrations["w5_user"]
        assert set(cal) == {"mid", "scale"}
    finally:
        app_module.PIPELINE.unenroll("w5_user")
