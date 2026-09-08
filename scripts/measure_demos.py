#!/usr/bin/env python
"""Measure what the pipeline actually returns for each demo clip.

Not part of the integration harness — this exists so the UI copy and
ground_truth.json can quote *measured* numbers instead of invented ones.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "vanirakshak-backend"
sys.path.insert(0, str(BACKEND_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from vanirakshak.server.app import app  # noqa: E402

DEMOS = [
    "demo_01_genuine.wav",
    "demo_02_degraded_copy.wav",
    "demo_03_partial_splice.wav",
    "demo_04_impostor.wav",
]
DEMO_DIR = REPO_ROOT / "frontend" / "public" / "demo"

c = TestClient(app)
print(f"{'file':32} {'risk':>6} {'verdict':<26} {'spk_sim':>8} {'anom':>6} {'target':>28}")
print("-" * 112)

rows = {}
for name in DEMOS:
    p = DEMO_DIR / name
    if not p.is_file():
        print(f"{name:32} MISSING")
        continue
    with p.open("rb") as fh:
        r = c.post(
            "/api/v1/analyze/audio",
            files={"audio": (name, fh.read(), "audio/wav")},
            data={"claimed_identity": "narendra_modi"},
        )
    if r.status_code != 200:
        print(f"{name:32} HTTP {r.status_code}: {r.text[:200]}")
        continue
    d = r.json()
    ts = d.get("target_speaker") or {}
    if ts.get("available"):
        tgt = f"{ts.get('decision')} p={ts.get('probability', 0):.3f}"
    else:
        tgt = "UNAVAILABLE"
    print(
        f"{name:32} {d['overall_risk_score']:6.1f} {d['verdict']:<26} "
        f"{d['mean_speaker_sim']:8.4f} {d['mean_acoustic_anomaly']:6.3f} {d['tier']:>28}"
    )
    rows[name] = {
        "risk": d["overall_risk_score"],
        "verdict": d["verdict"],
        "tier": d["tier"],
        "mean_speaker_sim": d["mean_speaker_sim"],
        "mean_acoustic_anomaly": d["mean_acoustic_anomaly"],
        "target_speaker": tgt,
    }

if rows:
    risks = [v["risk"] for v in rows.values()]
    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spread": round(max(risks) - min(risks), 1),
        "files": rows,
    }
    for d in (REPO_ROOT / "frontend" / "public" / "demo", REPO_ROOT / "data" / "demo"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "baseline.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nRisk spread {payload['spread']} points; wrote baseline.json to 2 locations.")
