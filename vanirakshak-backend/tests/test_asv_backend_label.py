"""G7 — the ASV backend must announce whether it is a real voiceprint or the
synthetic stub, and ``GET /v1/health`` must agree with what is actually running.

The default stub (``detectors/asv.py``) embeds a *fixed random projection* of
hand-crafted spectral statistics — a cosine "similarity" against it carries no
speaker-identity information. The dashboard relabels that number
"(SYNTHETIC) / Demo projection — NOT a voiceprint"; these tests guard the
backend half of that contract so the label cannot silently drift out of sync
with the backend it describes.

Run::

    python -m pytest vanirakshak-backend/tests/test_asv_backend_label.py -v
"""

from __future__ import annotations

import os
import sys

# allow running this file directly
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fastapi.testclient import TestClient  # noqa: E402

from vanirakshak.detectors.asv import ASVModel  # noqa: E402
from vanirakshak.detectors.asv_ecapa import ASVModelECAPA  # noqa: E402
from vanirakshak.detectors.pipeline import DetectionPipeline  # noqa: E402
from vanirakshak.server.app import PIPELINE, app  # noqa: E402


def test_stub_labels_itself_synthetic() -> None:
    assert ASVModel().is_synthetic is True


def test_ecapa_labels_itself_real() -> None:
    # `is_synthetic` is a plain property, so this neither loads the model nor
    # needs speechbrain/network — it only asserts the class tells the truth.
    assert ASVModelECAPA().is_synthetic is False


def test_pipeline_reports_the_active_backend() -> None:
    # Default construction → whatever backend the factory actually resolved
    # (stub unless VR_ASV_BACKEND=ecapa is set). The property must track the
    # instance, not the env var, so a degraded ecapa→stub fallback still reads
    # synthetic.
    pipe = DetectionPipeline()
    assert pipe.asv_is_synthetic is bool(getattr(pipe.asv, "is_synthetic", True))


def test_health_agrees_with_running_backend() -> None:
    client = TestClient(app)
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    models = r.json()["models"]

    assert "asv_synthetic" in models, "health must expose models.asv_synthetic (G7)"
    # health describes the instance actually loaded — not the requested env var.
    assert models["asv_synthetic"] is PIPELINE.asv_is_synthetic
    # The name and the synthetic flag must not contradict each other.
    if models["asv_backend"] == "ecapa_tdnn_voxceleb":
        assert models["asv_synthetic"] is False
    else:
        assert models["asv_synthetic"] is True
