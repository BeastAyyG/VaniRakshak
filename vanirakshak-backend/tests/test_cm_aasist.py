"""Tests for the trained-AASIST landing zone (detectors/cm_aasist.py).

No trained weights exist (see docs/08-cm-training-plan.md), so every test
here pins the *absence* behavior: selecting VR_CM_BACKEND=aasist without a
checkpoint must fall back to the stub and never crash the pipeline. The day
real weights land, these tests get a sibling that loads them.
"""

import numpy as np

from vanirakshak.detectors.cm import CMModel
from vanirakshak.detectors.cm_aasist import (
    CMModelAASIST,
    _validate_checkpoint,
    weights_path,
)


def _valid_schema():
    return {
        "architecture": "aasist-l",
        "frontend": "microsoft/wavlm-base-plus",
        "sample_rate": 16000,
        "window_s": 4.0,
        "state_dict": {"head.weight": [0.1]},
        "eer_threshold": 0.25,
        "dev_eer": 0.031,
        "dev_min_dcf": 0.087,
    }


def test_try_create_without_weights_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("VR_CM_AASIST_WEIGHTS", str(tmp_path / "missing.pt"))
    assert CMModelAASIST.try_create() is None


def test_factory_falls_back_to_stub_without_weights(tmp_path, monkeypatch):
    from vanirakshak.detectors.pipeline import DetectionPipeline, _default_cm

    monkeypatch.setenv("VR_CM_BACKEND", "aasist")
    monkeypatch.setenv("VR_CM_AASIST_WEIGHTS", str(tmp_path / "missing.pt"))
    cm = _default_cm()
    assert isinstance(cm, CMModel)
    # Full pipeline still scores end to end on the fallback.
    pipe = DetectionPipeline()
    x = (np.random.default_rng(0).normal(0, 0.05, 16000)).astype(np.float32)
    assert np.isfinite(pipe.analyse(x, 16000).cm_score)


def test_weights_path_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VR_CM_AASIST_WEIGHTS", str(tmp_path / "custom.pt"))
    assert weights_path().name == "custom.pt"


def test_weights_path_default_name(monkeypatch):
    monkeypatch.delenv("VR_CM_AASIST_WEIGHTS", raising=False)
    assert weights_path().name == "cm_aasist.pt"


def test_validate_checkpoint_accepts_schema():
    assert _validate_checkpoint(_valid_schema(), weights_path()) is None


def test_validate_checkpoint_rejects_missing_keys():
    bad = _valid_schema()
    del bad["eer_threshold"]
    try:
        _validate_checkpoint(bad, weights_path())
    except ValueError as e:
        assert "eer_threshold" in str(e) or "missing keys" in str(e)
    else:
        raise AssertionError("expected ValueError for missing keys")


def test_validate_checkpoint_rejects_wrong_architecture():
    bad = _valid_schema()
    bad["architecture"] = "resnet18"
    try:
        _validate_checkpoint(bad, weights_path())
    except ValueError as e:
        assert "aasist-l" in str(e)
    else:
        raise AssertionError("expected ValueError for wrong architecture")


def test_validate_checkpoint_rejects_non_dict():
    try:
        _validate_checkpoint([1, 2, 3], weights_path())
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-dict checkpoint")


def test_try_create_with_valid_checkpoint(tmp_path, monkeypatch):
    try:
        import torch
    except ImportError:
        return  # Torch not installed yet, skip

    from ml.aasist_model import AASISTLHead

    ckpt_file = tmp_path / "test_aasist.pt"
    head = AASISTLHead(in_dim=768)
    payload = _valid_schema()
    payload["state_dict"] = head.state_dict()
    torch.save(payload, str(ckpt_file))

    monkeypatch.setenv("VR_CM_BACKEND", "aasist")
    monkeypatch.setenv("VR_CM_AASIST_WEIGHTS", str(ckpt_file))

    cm = CMModelAASIST.try_create()
    assert cm is not None
    assert cm.dev_eer == 0.031
    assert cm.dev_min_dcf == 0.087
    assert cm.eer_threshold == 0.25
    assert "aasist_l_wavlm" in cm.name

