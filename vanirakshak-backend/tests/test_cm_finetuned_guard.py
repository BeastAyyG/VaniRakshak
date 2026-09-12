"""A fine-tuned head must never be scored against a stock frontend.

`cm_aasist.py` pins ``_FRONTEND`` and loads stock WavLM weights. A head trained
on fine-tuned features and scored against that stock frontend produces
*silently wrong* numbers: no exception, no missing key, and the integration
check still passes. These tests pin the fail-closed contract from
docs/10-quality-and-hardening-plan.md G3.

Lives under vanirakshak-backend/tests/ because pytest.ini sets
``testpaths = vanirakshak-backend/tests`` -- a file anywhere else is never
collected, so the guard would look tested while never running.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vanirakshak.detectors.cm_aasist import CMModelAASIST  # noqa: E402


def _minimal_ckpt(**extra):
    """Checkpoint metadata shaped like train_cm.py's payload.

    ``state_dict`` is empty: the guard must fire before any head weights are
    consulted, so an empty one is the sharper test.
    """
    payload = {
        "architecture": "aasist-l",
        "frontend": "microsoft/wavlm-base-plus",
        "sample_rate": 16000,
        "window_s": 4.0,
        "state_dict": {},
        "eer_threshold": 10.0,
        "dev_eer": 0.14,
        "dev_min_dcf": 0.97,
        "layer_index": -1,
        "merge_layers": 12,
        "seed": 42,
    }
    payload.update(extra)
    return payload


def test_finetuned_checkpoint_without_encoder_weights_raises(tmp_path, monkeypatch):
    """unfreeze_top_n > 0 with no frontend_state_dict must refuse to load."""
    ckpt = _minimal_ckpt(unfreeze_top_n=2)
    path = tmp_path / "finetuned_missing_encoder.pt"
    torch.save(ckpt, path)
    monkeypatch.setenv("VR_CM_AASIST_WEIGHTS", str(path))

    with pytest.raises(RuntimeError, match="frontend_state_dict"):
        CMModelAASIST()._ensure_model()


def test_finetuned_checkpoint_with_empty_encoder_dict_raises(tmp_path, monkeypatch):
    """An empty dict is falsy and must be treated as absent, not as 'zero layers'."""
    ckpt = _minimal_ckpt(unfreeze_top_n=2, frontend_state_dict={})
    path = tmp_path / "finetuned_empty_encoder.pt"
    torch.save(ckpt, path)
    monkeypatch.setenv("VR_CM_AASIST_WEIGHTS", str(path))

    with pytest.raises(RuntimeError, match="frontend_state_dict"):
        CMModelAASIST()._ensure_model()


def test_finetuned_checkpoint_records_expected_layers(tmp_path, monkeypatch):
    """With encoder weights present, load succeeds and records which layers.

    This stops at metadata: actually applying the weights needs the real WavLM
    download, which `_ensure_frontend` handles and which the integration check
    exercises. What is asserted here is that the guard does not fire on a
    well-formed checkpoint -- a guard that rejected everything would pass the
    two tests above while being useless.
    """
    fsd = {
        "encoder.layers.10.attention.q_proj.weight": torch.zeros(2, 2),
        "encoder.layers.11.attention.q_proj.weight": torch.zeros(2, 2),
    }
    ckpt = _minimal_ckpt(
        unfreeze_top_n=2,
        frontend_state_dict=fsd,
        frontend_layer_indices=[10, 11],
    )
    path = tmp_path / "finetuned_ok.pt"
    torch.save(ckpt, path)
    monkeypatch.setenv("VR_CM_AASIST_WEIGHTS", str(path))

    cm = CMModelAASIST()
    cm._ensure_model()
    assert cm.unfreeze_top_n == 2
    assert cm._frontend_layer_indices == [10, 11]
    # The stale stock frontend must have been dropped so it gets rebuilt with
    # the fine-tuned weights on the next score().
    assert cm._frontend is None


def test_frozen_checkpoint_still_loads_unchanged(tmp_path, monkeypatch):
    """The frozen-encoder path is the one every published benchmark used."""
    ckpt = _minimal_ckpt()  # no unfreeze_top_n key at all, like older checkpoints
    path = tmp_path / "frozen.pt"
    torch.save(ckpt, path)
    monkeypatch.setenv("VR_CM_AASIST_WEIGHTS", str(path))

    cm = CMModelAASIST()
    cm._ensure_model()
    assert cm.unfreeze_top_n == 0
    assert cm._frontend_state_dict is None
