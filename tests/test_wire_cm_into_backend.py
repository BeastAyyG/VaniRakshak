"""Tests for the checkpoint -> backend wiring check.

The failure this guards against is quiet and total: `cm_aasist.py` is an empty
landing zone, so a deployment can set VR_CM_BACKEND=aasist, get no error, and
keep serving the numpy/scipy stub while the health report says
`aasist_missing_weights`. These tests pin the two things that must never drift:
that the check uses the *backend's own* validator rather than a copy, and that
the emitted config actually names the checkpoint.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "wire_under_test", REPO_ROOT / "scripts" / "wire_cm_into_backend.py"
)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class _FakeBackend:
    """Stands in for vanirakshak.detectors.cm_aasist."""

    def __init__(self, strict: bool = True):
        self.strict = strict

    def _validate_checkpoint(self, ckpt, path):
        if not self.strict:
            return
        import torch
        ckpt = torch.load(str(path), map_location="cpu")
        if ckpt.get("architecture") != "aasist-l":
            raise ValueError(f"{path}: architecture {ckpt.get('architecture')!r} "
                             f"!= expected 'aasist-l'")


def _write_ckpt(tmp_path: Path, **over) -> Path:
    import torch
    ckpt = {
        "architecture": "aasist-l",
        "frontend": "microsoft/wavlm-base-plus",
        "sample_rate": 16000,
        "window_s": 4.0,
        "state_dict": {},
        "eer_threshold": 8.3789,
        "dev_eer": 0.12749,
        "dev_min_dcf": 0.94245,
        "seed": 42,
        "layer_index": 12,
        "init_weights": None,
    }
    ckpt.update(over)
    p = tmp_path / "ck.pt"
    torch.save(ckpt, str(p))
    return p


def test_env_block_names_the_checkpoint_and_the_backend(tmp_path):
    p = _write_ckpt(tmp_path)
    block = mod.env_block(p)
    assert "VR_CM_BACKEND=aasist" in block
    assert f"VR_CM_AASIST_WEIGHTS={p}" in block


def test_env_block_warns_that_fusion_must_be_refitted(tmp_path):
    """The fusion weights were fitted to a different head's LLR distribution."""
    block = mod.env_block(_write_ckpt(tmp_path))
    assert "Re-fit" in block


def test_env_block_does_not_export_fusion_weights(tmp_path):
    """Exporting VR_FUSION_W_CM would DISABLE the config's real-backend preset.

    config.py's _REAL_BACKEND_OVERRIDES supplies w_cm=1.5/w_asv=2.0 whenever a
    real backend is active, but only for keys absent from the environment
    (config.py:229). The stub-era w_cm=2.4 is documented at config.py:126 as
    having "pushed genuine real clips into CHALLENGE", so emitting it here would
    re-introduce that false alarm. Measured on data/demo with cm_scratch_S42.pt:
    w_cm=2.4 gives demo_01 (genuine) risk 43.2, the preset's 1.5 gives 38.3.
    """
    block = mod.env_block(_write_ckpt(tmp_path))
    # Match *export lines*, not the bare name: the explanatory comment
    # legitimately names the variables it is declining to export.
    exported = [
        ln.strip()
        for ln in block.splitlines()
        if ln.strip().startswith("export VR_FUSION")
    ]
    assert exported == [], f"env block must not export fusion weights: {exported}"
    # The block should still say WHY it stays silent, not just omit the keys.
    assert "config.py" in block


def test_a_valid_checkpoint_passes_and_emits_config(tmp_path, monkeypatch):
    p = _write_ckpt(tmp_path)
    monkeypatch.setattr(mod, "load_backend_detector", lambda root: _FakeBackend())
    out = tmp_path / "deploy.env"
    rc = mod.main(["--checkpoint", str(p), "--out", str(out)])
    assert rc == 0
    assert "VR_CM_BACKEND=aasist" in out.read_text()


def test_a_bad_checkpoint_fails_without_emitting_config(tmp_path, monkeypatch):
    p = _write_ckpt(tmp_path, architecture="resnet")
    monkeypatch.setattr(mod, "load_backend_detector", lambda root: _FakeBackend())
    out = tmp_path / "deploy.env"
    rc = mod.main(["--checkpoint", str(p), "--out", str(out)])
    assert rc == 1
    assert not out.exists()


def test_missing_checkpoint_is_reported_not_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_backend_detector", lambda root: _FakeBackend())
    with pytest.raises(SystemExit) as exc:
        mod.main(["--checkpoint", str(tmp_path / "nope.pt")])
    assert "not found" in str(exc.value)


def test_a_missing_backend_checkout_is_an_error_not_a_silent_pass(tmp_path, monkeypatch):
    """If the backend is not there, the check cannot be run -- say so."""
    with pytest.raises(SystemExit) as exc:
        mod.load_backend_detector(tmp_path / "no-such-backend")
    assert "backend root not found" in str(exc.value)
