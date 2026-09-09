"""ARCHIVED 2026-09-09 — no longer runnable as-is.

`backend.app.core.forensic_signal.ForensicSignalProcessor`, which this script
scores, was deleted when the standalone `backend/` server was retired (see
`backend/README.md`). The DSP it measured was never wired into the serving
path this repo actually ships (see `vanirakshak-backend/`), so nothing today
depends on this script running. Kept for historical reference alongside
`AUDIT_REPORT.md` §5, which this script's output supported.

--- Original docstring below ---

Phase 4: score the component the published benchmark NEVER measures.

`benchmarks/run_benchmarks.py` imports only `vanirakshak.*` (the deterministic
stubs). It never imports `backend.app.*` — so the 429-line real DSP
`ForensicSignalProcessor`, the ECAPA-TDNN verifier and the WavLM extractor have
never been scored against any ground truth.

This script scores `backend.app.core.forensic_signal.ForensicSignalProcessor`
on EXACTLY the ground truth the published benchmark 4 used
(data/demo/ground_truth.json), so the result is directly comparable.

Mapping from the engine's output vocabulary to the task's label set is written
out explicitly below; unmapped outputs stay visible instead of being forced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vanirakshak-backend"))

from backend.app.core.forensic_signal import ForensicSignalProcessor  # noqa: E402

# ---------------------------------------------------------------------------
# Explicit mapping: engine vocabulary -> ground-truth label set.
# IMPOSTOR_SPEAKER is deliberately UNMAPPED: impostor detection is a speaker-
# identity task (ASV/ECAPA), not an acoustic-vocoder task. The DSP path cannot
# express it. We keep that visible rather than pretending it maps.
# ---------------------------------------------------------------------------
ENGINE_TO_LABEL = {
    "NATURAL_ACOUSTIC_PROFILE": "AUTHENTIC_TARGET",
    "SYNTHETIC_VOCODER_DETECTED": "SYNTHETIC_VOICE_CLONE",
    "SUSPICIOUS_HIGH_FREQUENCY_FLATNESS": "SYNTHETIC_VOICE_CLONE",
    "SUSPICIOUS_PHASE_DISCONTINUITY": "SUSPICIOUS_SPLICED_AUDIO",
}
UNREACHABLE = {"IMPOSTOR_SPEAKER": "needs speaker-identity (ASV), not acoustic DSP"}

gt = json.loads((REPO / "data" / "demo" / "ground_truth.json").read_text(encoding="utf-8"))
proc = ForensicSignalProcessor(target_sr=16000)

records = []
print("=" * 78)
print("HEAVY BACKEND (backend.app.core.forensic_signal) vs published ground truth")
print("=" * 78)
print(f"{'file':<30}{'expected':<28}{'engine verdict':<40}{'mapped':<26}")
print("-" * 124)
for fname, meta in sorted(gt["files"].items()):
    fpath = REPO / "data" / "demo" / fname
    expected = meta["expected_verdict"]
    try:
        rep = proc.analyze_stream(fpath, orig_sr=16000)
        verdict = rep.verdict_indicator
        mapped = ENGINE_TO_LABEL.get(verdict, "<UNMAPPED>")
        risk = rep.composite_vocoder_risk_score
        err = None
    except Exception as e:
        verdict, mapped, risk, err = f"ERROR", "<ERROR>", None, f"{type(e).__name__}: {e}"
    records.append({"file": fname, "expected": expected, "engine_verdict": verdict,
                    "mapped": mapped, "composite_risk_pct": risk, "error": err})
    print(f"{fname:<30}{expected:<28}{verdict:<40}{mapped:<26}")
    if err:
        print(f"    !! {err}")

print()
print("=" * 78)
print("SCORE (n=4 — treat as a signal, not a publishable result)")
print("=" * 78)
ok = [r for r in records if not r["error"]]
print(f"  parse/coverage        : {len(ok)}/{len(records)} files analysed without error")

mapped_ok = [r for r in ok if r["mapped"] != "<UNMAPPED>"]
correct = [r for r in mapped_ok if r["mapped"] == r["expected"]]
print(f"  mappable              : {len(mapped_ok)}/{len(ok)} "
      f"(IMPOSTOR_SPEAKER is structurally unreachable by this path)")
print(f"  correct               : {len(correct)}/{len(mapped_ok)}")
if mapped_ok:
    print(f"  accuracy (mappable)   : {len(correct)/len(mapped_ok):.3f}")
print(f"  accuracy (all 4)      : {len(correct)}/{len(records)} = {len(correct)/len(records):.3f}")

# macro-F1 over the 4 ground-truth classes
labels = sorted({r["expected"] for r in records})
print()
print("  per-class F1 over the FULL 4-class label set (unreachable class included):")
f1s = []
for lab in labels:
    tp = sum(1 for r in ok if r["mapped"] == lab and r["expected"] == lab)
    fp = sum(1 for r in ok if r["mapped"] == lab and r["expected"] != lab)
    fn = sum(1 for r in ok if r["expected"] == lab and r["mapped"] != lab)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    f1s.append(f1)
    print(f"    {lab:<28} tp={tp} fp={fp} fn={fn}  P={prec:.2f} R={rec:.2f} F1={f1:.3f}")
print(f"    {'MACRO-F1 (4 classes)':<28} = {sum(f1s)/len(f1s):.3f}")

# and the flattering version: only over classes the path can express
reachable = [l for l in labels if l not in UNREACHABLE]
f1r = []
for lab in reachable:
    tp = sum(1 for r in ok if r["mapped"] == lab and r["expected"] == lab)
    fp = sum(1 for r in ok if r["mapped"] == lab and r["expected"] != lab)
    fn = sum(1 for r in ok if r["expected"] == lab and r["mapped"] != lab)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1r.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
print(f"    {'MACRO-F1 (3 reachable)':<28} = {sum(f1r)/len(f1r):.3f}   <- the flattering number")

print()
for lab, why in UNREACHABLE.items():
    print(f"  NOTE: '{lab}' is structurally unreachable here — {why}.")

(REPO / "audit_heavy_backend_predictions.json").write_text(
    json.dumps({"mapping": ENGINE_TO_LABEL, "unreachable": UNREACHABLE,
                "records": records}, indent=2), encoding="utf-8")
print("\n[written] audit_heavy_backend_predictions.json")
