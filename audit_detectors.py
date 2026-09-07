"""Audit helper: instrument what the VaniRakshak detectors ACTUALLY emit.

Answers three questions with measurements, not reading:
  1. Does each of the 4 layers (CM / ASV / channel / prosody) vary at all?
  2. Is the reported EER=0.0 real separation, or an artifact of the
     synthetic generator being built to be separable?
  3. Does the heavy `backend/` path ever get scored? (scored separately,
     only if torch is importable)
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vanirakshak-backend"))

from vanirakshak.data.synthetic import synthetic_speech, synthetic_spoofed  # noqa: E402
from vanirakshak.detectors.pipeline import DetectionPipeline  # noqa: E402

SR = 16000


def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def describe(name: str, vals: list[float]) -> None:
    a = np.array(vals, dtype=float)
    print(f"    {name:<20s} n={a.size:4d} min={a.min():+.4f} max={a.max():+.4f} "
          f"mean={a.mean():+.4f} std={a.std():.4f} unique={len(np.unique(np.round(a, 6)))}")


print("=" * 78)
print("PART A - Do the four detector layers actually VARY on the demo corpus?")
print("=" * 78)
pipe = DetectionPipeline()
gt = json.loads((REPO / "data" / "demo" / "ground_truth.json").read_text())

rows = []
for fname in sorted(gt["files"]):
    pcm = load_wav(REPO / "data" / "demo" / fname).astype(np.float32) / 32768.0
    # score the whole 10 s clip the way the CM stub does
    r = pipe.analyse(pcm, SR)
    rows.append((fname, gt["files"][fname]["expected_verdict"], r))
    print(f"\n  {fname}  (expected={gt['files'][fname]['expected_verdict']})")
    print(f"    cm_score(LLR)      = {r.cm_score:+.4f}")
    print(f"    cm_genuine_prob    = {r.features['cm_genuine_prob']:.4f}")
    print(f"    asv_score          = {r.asv_score:.4f}   enrolled={r.asv_enrolled}")
    print(f"    channel_anomaly    = {r.channel_anomaly:.4f}   codec={r.codec_guess}")
    print(f"    prosody_unnatural  = {r.prosody_unnatural:.4f}")
    print(f"    snr_db             = {r.snr_db:.2f}")

print()
print("  --- spread across the 4 demo files (should be WIDE if discriminative) ---")
for key, getter in [
    ("cm_score", lambda r: r.cm_score),
    ("cm_genuine_prob", lambda r: r.features["cm_genuine_prob"]),
    ("asv_score", lambda r: r.asv_score),
    ("channel_anomaly", lambda r: r.channel_anomaly),
    ("prosody_unnatural", lambda r: r.prosody_unnatural),
]:
    describe(key, [getter(r) for _, _, r in rows])

print()
print("=" * 78)
print("PART B - Is EER=0.0 real separation? Score the synthetic pair.")
print("=" * 78)
rng = np.random.default_rng(42)
bon, spf = [], []
for _ in range(100):
    bon.append(pipe.analyse(synthetic_speech(4.0, sr=SR, rng=rng).astype(np.float32) / 32768.0, SR).cm_score)
    spf.append(pipe.analyse(synthetic_spoofed(4.0, sr=SR, rng=rng).astype(np.float32) / 32768.0, SR).cm_score)
bon, spf = np.array(bon), np.array(spf)
describe("bona fide LLR", list(bon))
describe("spoof LLR", list(spf))
print(f"    separation gap   = {bon.mean() - spf.mean():+.4f}  (bona fide mean - spoof mean)")
print(f"    overlap?         = {'YES' if bon.min() < spf.max() else 'NO -> perfectly separable'}")
print(f"    bon range [{bon.min():+.4f}, {bon.max():+.4f}]   spf range [{spf.min():+.4f}, {spf.max():+.4f}]")

# The headline number is EER, but the deployed system thresholds at 0.5 (actDCF).
print()
print("    --- what happens at the threshold the system ACTUALLY ships (0.5)? ---")
THRESH = 0.5
bon_pred_spoof = (bon < THRESH).sum()
spf_pred_spoof = (spf < THRESH).sum()
print(f"    threshold={THRESH}: bona fide flagged spoof = {bon_pred_spoof}/100, "
      f"spoof caught = {spf_pred_spoof}/100")

print()
print("=" * 78)
print("PART C - Is the synthetic generator's separability engineered?")
print("=" * 78)
print("  Reading vanirakshak/data/synthetic.py docstring, verbatim:")
for line in (
    (REPO / "vanirakshak-backend" / "vanirakshak" / "data" / "synthetic.py")
    .read_text(encoding="utf-8")
    .splitlines()[:24][3:]
):
    print("   ", line)

print()
print("=" * 78)
print("PART D - Does the heavy `backend/` path ever run? (needs torch)")
print("=" * 78)
try:
    import torch  # noqa: F401

    sys.path.insert(0, str(REPO))
    from backend.app.core.forensic_signal import ForensicSignalProcessor  # noqa: E402
    from backend.app.models.ecapa_verifier import ECAPAVerifier  # noqa: E402
    from backend.app.models.wavlm_detector import WavLMFeatureExtractor  # noqa: E402

    print("  torch is available - heavy backend importable.")
    HAVE = True
except Exception as e:
    print(f"  torch/heavy backend NOT available: {type(e).__name__}: {e}")
    HAVE = False

if HAVE:
    fsp = ForensicSignalProcessor(target_sr=16000)
    print()
    print("  --- ForensicSignalProcessor (real DSP) on the demo corpus ---")
    for fname in sorted(gt["files"]):
        pcm = load_wav(REPO / "data" / "demo" / fname)
        try:
            tensor, arr, sha, dur = fsp.standardize_audio(pcm, orig_sr=SR, peak=0.95)
            feats = fsp.extract_forensic_features(arr, SR) if hasattr(fsp, "extract_forensic_features") else None
            print(f"    {fname:<30s} dur={dur:.2f}s sha={sha[:12]}... feats={feats}")
        except Exception as e:
            print(f"    {fname:<30s} FAILED: {type(e).__name__}: {e}")

print()
print("=" * 78)
print("PART E - per-file demo predictions (raw, for the owner to check)")
print("=" * 78)
out = {
    fname: {
        "expected": exp,
        "cm_score": round(r.cm_score, 4),
        "cm_genuine_prob": round(r.features["cm_genuine_prob"], 4),
        "asv_score": r.asv_score,
        "asv_enrolled": r.asv_enrolled,
        "channel_anomaly": round(r.channel_anomaly, 4),
        "prosody_unnatural": round(r.prosody_unnatural, 4),
    }
    for fname, exp, r in rows
}
(REPO / "audit_demo_predictions.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
print("\n[written] audit_demo_predictions.json")
