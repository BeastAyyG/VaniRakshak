"""Guards for the deployed countermeasure.

The failure mode this file exists to prevent: `cm_aasist.try_create()`
returns None when no weights are present, the pipeline silently falls back
to the stub, and the deployment looks healthy while a heuristic is deciding.
Nothing else in the suite would notice.

Resolution tests are pure filesystem logic and stay fast. The end-to-end
detector test loads a 95M-parameter frontend, so it is opt-in via
VR_CM_DEPLOY_E2E=1 -- the default suite stays quick.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "vanirakshak-backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from vanirakshak.detectors import backend_selection as bs  # noqa: E402
from vanirakshak.detectors import cm_aasist as cm  # noqa: E402


# --------------------------------------------------------------------------
# weights_path() precedence -- pure logic, no torch
# --------------------------------------------------------------------------

def test_env_override_wins(monkeypatch, tmp_path):
    pinned = tmp_path / "pinned.pt"
    pinned.write_bytes(b"x")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cm_docs20_armb_s42_cal.pt").write_bytes(b"x")

    monkeypatch.setenv("VR_CM_AASIST_WEIGHTS", str(pinned))
    monkeypatch.setattr(cm, "repo_root", lambda: tmp_path)
    assert cm.weights_path() == pinned


def test_preferred_used_when_present(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    preferred = tmp_path / "data" / "cm_docs20_armb_s42_cal.pt"
    preferred.write_bytes(b"x")
    (tmp_path / "data" / "cm_aasist.pt").write_bytes(b"x")

    monkeypatch.delenv("VR_CM_AASIST_WEIGHTS", raising=False)
    monkeypatch.setattr(cm, "repo_root", lambda: tmp_path)
    assert cm.weights_path() == preferred


def test_falls_back_to_legacy_when_preferred_absent(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    legacy = tmp_path / "data" / "cm_aasist.pt"
    legacy.write_bytes(b"x")

    monkeypatch.delenv("VR_CM_AASIST_WEIGHTS", raising=False)
    monkeypatch.setattr(cm, "repo_root", lambda: tmp_path)
    assert cm.weights_path() == legacy


def test_preferred_returned_when_nothing_exists(monkeypatch, tmp_path):
    """With no file at all, name the path we WANT, not one we merely
    tolerate -- so the operator sees an actionable error."""
    (tmp_path / "data").mkdir()

    monkeypatch.delenv("VR_CM_AASIST_WEIGHTS", raising=False)
    monkeypatch.setattr(cm, "repo_root", lambda: tmp_path)
    resolved = cm.weights_path()
    assert resolved.name == "cm_docs20_armb_s42_cal.pt"


def test_preferred_is_the_docs20_arm_b_winner():
    """Pins the ship decision. If someone repoints the default, this fails
    and forces them to say why."""
    assert cm.PREFERRED_WEIGHTS_RELATIVE.parts == (
        "data", "cm_docs20_armb_s42_cal.pt",
    )


# --------------------------------------------------------------------------
# the actually-resolved checkpoint
# --------------------------------------------------------------------------

def _resolved():
    return cm.weights_path()


@pytest.mark.skipif(not _resolved().exists(),
                    reason=f"weights not on this box: {_resolved()}")
def test_resolved_checkpoint_passes_backend_validation():
    import torch

    path = _resolved()
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cm._validate_checkpoint(ckpt, path)  # raises if the schema is wrong


@pytest.mark.skipif(not _resolved().exists(),
                    reason=f"weights not on this box: {_resolved()}")
def test_resolved_checkpoint_is_calibrated():
    import torch

    ckpt = torch.load(str(_resolved()), map_location="cpu", weights_only=False)
    assert ckpt.get("calibration"), (
        "deployed checkpoint carries no calibration block -- scores are raw "
        "LLRs and the decision threshold is wrong (EER/minDCF are unaffected "
        "by calibration, so this will not show up in any accuracy metric)"
    )


# --------------------------------------------------------------------------
# end-to-end -- opt in with VR_CM_DEPLOY_E2E=1
# --------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("VR_CM_DEPLOY_E2E") != "1",
                    reason="set VR_CM_DEPLOY_E2E=1 to load the real detector")
def test_detector_is_not_the_stub():
    """The stub has name 'synthetic_cm_v1', NOT 'aasist_missing_weights' --
    asserting on the latter alone passes happily while the heuristic runs.
    Assert a real detector is what got built."""
    from vanirakshak.detectors import backend_selection as bs
    from vanirakshak.detectors.pipeline import DetectionPipeline

    pipe = DetectionPipeline()
    assert pipe.cm.name.startswith(bs.REAL_CM_BACKENDS), (
        f"detector is {pipe.cm.name!r} -- a stub, not the trained "
        f"countermeasure. Real CM names start with {bs.REAL_CM_BACKENDS}. "
        f"Check VR_CM_BACKEND and that weights_path() resolves to a file."
    )


# --------------------------------------------------------------------------
# backend selection -- the auto-select path, no torch needed
# --------------------------------------------------------------------------

def test_resolve_cm_backend_env_wins(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cm_docs20_armb_s42_cal.pt").write_bytes(b"x")
    monkeypatch.setattr(cm, "repo_root", lambda: tmp_path)

    monkeypatch.setenv("VR_CM_BACKEND", "stub")
    assert bs.resolve_cm_backend() == "stub"


def test_resolve_cm_backend_auto_selects_aasist_when_weights_present(
        monkeypatch, tmp_path):
    """With no env var and a checkpoint on disk, the trained head must win by
    default -- otherwise every deployment silently runs the heuristic."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cm_docs20_armb_s42_cal.pt").write_bytes(b"x")
    monkeypatch.setattr(cm, "repo_root", lambda: tmp_path)
    monkeypatch.delenv("VR_CM_BACKEND", raising=False)

    assert bs.resolve_cm_backend() == "aasist"
    assert bs.real_backend_active() is True


def test_resolve_cm_backend_falls_back_to_stub_without_weights(
        monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(cm, "repo_root", lambda: tmp_path)
    monkeypatch.delenv("VR_CM_BACKEND", raising=False)
    monkeypatch.delenv("VR_ASV_BACKEND", raising=False)

    assert bs.resolve_cm_backend() == "stub"
    assert bs.real_backend_active() is False


def test_aasist_requested_without_weights_is_not_real(monkeypatch, tmp_path):
    """Asking for aasist without a checkpoint must NOT claim a real backend
    is active -- the pipeline falls back to the stub, so the real-backend
    fusion preset would be applied to stub scores."""
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(cm, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("VR_CM_BACKEND", "aasist")
    monkeypatch.delenv("VR_ASV_BACKEND", raising=False)

    assert bs.resolve_cm_backend() == "aasist"
    assert bs.real_backend_active() is False
