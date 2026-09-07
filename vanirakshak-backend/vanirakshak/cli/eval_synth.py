"""Synthetic ASVspoof 5-style evaluation harness.

Builds a labelled set of bonafide / spoof windows from
``synthetic_speech`` and ``synthetic_spoofed`` (with the codec
augmentation stack applied), scores them with the current
``DetectionPipeline`` + ``RiskFusionEngine``, and reports the same
minDCF / actDCF / CLLR / EER metric set as
``benchmarks/compute_mindcf.py``.

This is **NOT** a real benchmark — it uses synthetic audio and a
synthetic-pretrained detector. But the **metric formulas** are the
official ASVspoof 5 ones, so the printed numbers behave correctly
(perfect separation → minDCF 0; noisy overlap → reasonable minDCF).

Run::

    python -m vanirakshak eval
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# allow running the script directly
_PKG_PARENT = Path(__file__).resolve().parents[2]
_REPO = _PKG_PARENT.parent
sys.path.insert(0, str(_REPO))

from vanirakshak.config import settings  # noqa: E402
from vanirakshak.data.synthetic import synthetic_speech, synthetic_spoofed  # noqa: E402
from vanirakshak.data.augment import apply_random_pipeline  # noqa: E402
from vanirakshak.detectors.pipeline import DetectionPipeline  # noqa: E402


def _load_metric_harness():
    # LIVE harness first. The vanirakshak-backend/benchmarks/ copy is a dead
    # duplicate; preferring it silently pinned this eval to stale metric code.
    candidates = [
        _REPO / "benchmarks" / "compute_mindcf.py",
        _PKG_PARENT / "benchmarks" / "compute_mindcf.py",
    ]
    for c in candidates:
        if c.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("compute_mindcf", c)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("could not find benchmarks/compute_mindcf.py")


def run_eval(n_bonafide: int = 80, n_spoof: int = 80, sr: int = 16000, seed: int = 0) -> int:
    harness = _load_metric_harness()
    rng = np.random.default_rng(seed)
    pipe = DetectionPipeline()

    print(f"Building synthetic evaluation set: {n_bonafide} bonafide + {n_spoof} spoof")
    bonafide_scores: list[float] = []
    spoof_scores: list[float] = []

    for i in range(n_bonafide):
        clip = synthetic_speech(duration_sec=4.0, sr=sr, rng=np.random.default_rng(rng.integers(0, 2 ** 32 - 1)))
        clip = apply_random_pipeline(clip, sr, seed=rng.integers(0, 2 ** 32 - 1))
        w = pipe.analyse(clip, sr, claimed_user=None)
        bonafide_scores.append(w.cm_score)

    for i in range(n_spoof):
        clip = synthetic_spoofed(duration_sec=4.0, sr=sr, rng=np.random.default_rng(rng.integers(0, 2 ** 32 - 1)))
        clip = apply_random_pipeline(clip, sr, seed=rng.integers(0, 2 ** 32 - 1))
        w = pipe.analyse(clip, sr, claimed_user=None)
        spoof_scores.append(w.cm_score)

    res = harness.evaluate_system(np.array(bonafide_scores), np.array(spoof_scores))
    harness.print_metrics(res)
    print("\n[WARNING] Honest caveat: this is a SYNTHETIC eval using stub detectors.")
    print("   The metric formulas are official ASVspoof 5; the scores themselves")
    print("   are not representative of any trained model. Use this only to")
    print("   verify the harness is wired up correctly end-to-end.")
    return 0
