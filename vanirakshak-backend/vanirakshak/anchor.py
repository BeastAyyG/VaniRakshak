"""Enrolled target-speaker ("anchor") loading, shared by the server and the benchmarks.

Why this lives in its own module: the anchor has to be loaded into the *serving*
pipeline for `asv_consistency` to mean anything. Without it the ASV returns a
hard-coded 0.5 for every caller and the target-conditioned half of the system is
inert. `benchmarks/run_benchmarks.py` used to build a bare `CallSession` with no
anchor, so it was measuring a different (strictly weaker) system than the one
the dashboard talks to — its numbers could not be compared to the server's.

Both entry points now call `load_anchor()` so there is one code path and one set
of numbers.

The embedding is produced by `python scripts/enroll_anchor.py`. Only the vector is
stored — no audio is retained (DPDP Act 2023 §8(7)).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .detectors.pipeline import DetectionPipeline

# vanirakshak-backend/vanirakshak/anchor.py -> up three levels is the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Each ASV backend embeds into its own vector space (the stub's hand-crafted
# random projection vs. real ECAPA-TDNN), so each has its own anchor file —
# picked by `pipeline.asv.name`, exactly what `scripts/enroll_anchor.py`
# tags every anchor it writes with. Loading the wrong one silently misfires:
# cosine similarities in the wrong space land near the calibration's
# non-target cluster, which the impostor-block override then reads as every
# caller being an impostor.
_ANCHOR_PATHS_BY_BACKEND = {
    "synthetic_asv_v1": _REPO_ROOT / "data" / "anchor_asv_embedding.json",
    "ecapa_tdnn_voxceleb": _REPO_ROOT / "data" / "anchor_asv_embedding_ecapa.json",
}
DEFAULT_ANCHOR_PATH = _ANCHOR_PATHS_BY_BACKEND["synthetic_asv_v1"]


def load_anchor(
    pipeline: DetectionPipeline,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the enrolled anchor vector (and its calibration) into `pipeline`.

    Never raises: a missing or corrupt anchor must degrade the system, not stop
    it. Returns a state dict suitable for reporting over `/api/v1/health`.
    """
    backend_default = _ANCHOR_PATHS_BY_BACKEND.get(pipeline.asv.name, DEFAULT_ANCHOR_PATH)
    resolved = Path(str(path or os.environ.get("VR_ANCHOR_EMBEDDING") or backend_default))
    state: Dict[str, Any] = {
        "user_id": None,
        "enrolled": False,
        "source_clips": 0,
        "calibrated": False,
        "path": str(resolved),
        "error": None,
    }

    if not resolved.is_file():
        state["error"] = f"anchor embedding not found at {resolved}"
        return state

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        anchor_model = str(payload.get("model", ""))
        if anchor_model != pipeline.asv.name:
            state["error"] = (
                f"anchor at {resolved} was embedded by '{anchor_model}' but the running "
                f"ASV backend is '{pipeline.asv.name}' — refusing to load (incompatible "
                "vector spaces). Run scripts/enroll_anchor.py with the matching --backend."
            )
            return state
        user_id = str(payload["user_id"])
        vec = np.asarray(payload["embedding"], dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        pipeline.asv._enrolments[user_id] = vec

        # Score calibration is optional but essential in practice. Raw cosine
        # similarities from the ASV all sit in ~0.96-0.99, so without the fitted
        # logistic every speaker (target or not) scores ~0.99 and the detector
        # carries no information.
        cal = payload.get("calibration")
        if isinstance(cal, dict) and cal.get("scale"):
            pipeline.asv.calibration = {
                "mid": float(cal["mid"]),
                "scale": float(cal["scale"]),
            }

        state.update(
            {
                "user_id": user_id,
                "enrolled": True,
                "source_clips": len(payload.get("source_clips", [])),
                "calibrated": bool(cal),
                "error": None,
            }
        )
    except Exception as e:  # a broken anchor must never stop the service
        state["error"] = f"anchor load failed: {type(e).__name__}: {e}"

    return state
