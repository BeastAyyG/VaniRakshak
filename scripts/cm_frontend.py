"""Shared WavLM frontend loader for the offline scoring harnesses.

A head trained with ``--unfreeze-top-n`` expects a *fine-tuned* frontend. Score
it against the stock one and you get silently wrong numbers: no exception, no
missing key, just a benchmark row that looks plausible and isn't.

``vanirakshak/detectors/cm_aasist.py`` already fails closed on this. The six
``scripts/evaluate_*.py`` harnesses did not -- they each build a stock fp16
WavLM unconditionally. This module is the one place that knows how to do it
correctly, so a harness either uses it or refuses.

Usage:

    from cm_frontend import load_frontend_for, assert_frozen_checkpoint

    # fine-tune aware:
    wavlm, dtype = load_frontend_for(ckpt, device)

    # or, for a harness that only ever handled frozen checkpoints:
    assert_frozen_checkpoint(ckpt, "scripts/evaluate_4way_benchmark.py")
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers import WavLMModel

_FRONTEND = "microsoft/wavlm-base-plus"


def is_finetuned(ckpt: Dict[str, Any]) -> bool:
    """True when the checkpoint's head was trained on fine-tuned features."""
    return int(ckpt.get("unfreeze_top_n", 0) or 0) > 0


def assert_frozen_checkpoint(ckpt: Dict[str, Any], harness: str) -> None:
    """Refuse, loudly, rather than score a fine-tuned head on a stock frontend.

    For harnesses that only know how to build the stock frontend. Cheaper than
    teaching every one of them the fine-tuned path, and it closes the trap.
    """
    if is_finetuned(ckpt):
        raise RuntimeError(
            f"{harness} builds a stock {_FRONTEND} frontend, but this checkpoint "
            f"records unfreeze_top_n={ckpt['unfreeze_top_n']}. Its head was trained "
            "on fine-tuned features; scoring it here would produce silently wrong "
            "numbers. Use scripts/evaluate_finetune_benchmark.py instead."
        )


def load_frontend_for(
    ckpt: Dict[str, Any], device: str
) -> Tuple[WavLMModel, torch.dtype]:
    """Build the frontend this checkpoint was actually trained against.

    Returns the model and its dtype. fp16 for frozen checkpoints -- the
    numerics every published benchmark used, which must not move. bf16 for
    fine-tuned ones, matching how they were trained (docs/10 G3: one dtype
    across training, eval and runtime).
    """
    try:
        wavlm = WavLMModel.from_pretrained(_FRONTEND, local_files_only=True)
    except Exception:
        wavlm = WavLMModel.from_pretrained(_FRONTEND)

    if not is_finetuned(ckpt):
        dtype = torch.float16 if device == "cuda" else torch.float32
        wavlm = wavlm.to(dtype).to(device)
        wavlm.eval()
        return wavlm, dtype

    fsd = ckpt.get("frontend_state_dict")
    if not fsd:
        raise RuntimeError(
            f"checkpoint records unfreeze_top_n={ckpt['unfreeze_top_n']} but carries "
            "no 'frontend_state_dict'; refusing to score against a stock frontend"
        )

    expected = list(ckpt.get("frontend_layer_indices") or sorted(
        {int(k.split(".")[2]) for k in fsd if k.startswith("encoder.layers.")}
    ))
    res = wavlm.load_state_dict(fsd, strict=False)
    # strict=False silently drops names that don't match, which would make this
    # whole guard vacuous -- verify the weights actually landed.
    unexpected = list(res.unexpected_keys)
    applied = set(fsd) - set(unexpected)
    got = sorted({int(k.split(".")[2]) for k in applied if k.startswith("encoder.layers.")})
    if unexpected or got != expected:
        raise RuntimeError(
            f"fine-tuned encoder weights did not apply cleanly "
            f"(unexpected={unexpected[:5]}, applied layers={got}, expected={expected})"
        )

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    wavlm = wavlm.to(dtype).to(device)
    wavlm.eval()
    print(
        f"  frontend: fine-tuned layers {got} applied "
        f"({len(applied)} tensors, dtype={dtype})"
    )
    return wavlm, dtype
