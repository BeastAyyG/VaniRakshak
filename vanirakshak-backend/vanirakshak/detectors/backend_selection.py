"""Single source of truth for which detector backends are actually active.

Two places need this answer and used to compute it differently:

* ``detectors/pipeline.py`` picks which CM object to construct.
* ``config.py`` decides whether to apply ``_REAL_BACKEND_OVERRIDES`` (the
  fusion weights fitted for real detector score distributions).

``config.py`` previously read ``VR_CM_BACKEND`` straight from the
environment. That breaks the moment the backend is chosen *automatically*
rather than by an env var: the pipeline would run the trained AASIST head
while the fusion kept the stub-era weights, which is a silent
misconfiguration — the fused score would be wrong with nothing reporting it.

So both call into here. The rule is "a real backend is one that is actually
running", not "one someone asked for by name".

``cm_aasist`` is imported lazily inside the helper: it is cheap at module
level, but keeping the import local means this module stays importable in
contexts where the detector package is only partly available.
"""

from __future__ import annotations

import os

# Backends that replace the zero-dependency numpy/scipy heuristic.
REAL_CM_BACKENDS = ("wavlm", "aasist")
REAL_ASV_BACKENDS = ("ecapa",)


def _cm_weights_available() -> bool:
    """True when a real AASIST checkpoint resolves to a file that exists."""
    try:
        from .cm_aasist import weights_path
    except Exception:  # noqa: BLE001 - absence is a valid answer here
        return False
    try:
        return weights_path().exists()
    except Exception:  # noqa: BLE001
        return False


def resolve_cm_backend() -> str:
    """Which CM backend should be constructed.

    An explicit ``VR_CM_BACKEND`` always wins, so an operator can pin the
    stub (or wavlm) on purpose.

    When it is unset, prefer the trained AASIST head **if a checkpoint is
    actually present**, otherwise fall back to the stub. That makes a
    correctly-provisioned box run the real model by default instead of
    requiring whoever deploys it to remember an env var — which is exactly
    how a deployment ends up silently running the heuristic.
    """
    explicit = os.environ.get("VR_CM_BACKEND", "").strip().lower()
    if explicit:
        return explicit
    return "aasist" if _cm_weights_available() else "stub"


def real_backend_active() -> bool:
    """True when at least one real (non-heuristic) detector will run.

    For ``aasist`` this requires the checkpoint to exist: asking for aasist
    without weights makes the pipeline fall back to the stub, and applying
    real-backend fusion weights to stub scores would be wrong. ``wavlm``
    carries no weights file, so its availability is decided at construction
    and we take the request at face value.
    """
    if os.environ.get("VR_ASV_BACKEND", "stub").strip().lower() in REAL_ASV_BACKENDS:
        return True

    cm = resolve_cm_backend()
    if cm == "aasist":
        return _cm_weights_available()
    return cm in REAL_CM_BACKENDS
