#!/usr/bin/env python
"""Rebuild the standardized benchmark suite from real, documented audio.

Why this exists: the previous `demo_0*.wav` files were unverifiable. AUDIT_REPORT.md
found they correlate +0.9965 / +0.9814 with each other (derived by `tanh(x*1.5)`,
resampling and copy-paste), and after the anchor was enrolled this tool showed the
"genuine" clip scoring 0.9880 — squarely in the non-target range — while the "clone"
scored higher. A benchmark whose ground truth is invented cannot validate anything.

This rebuilds all four clips from audio with known provenance:

  demo_01_genuine        target speaker, HELD OUT of enrolment (never seen by
                         the anchor embedding)
  demo_02_degraded_copy  signal-degraded copy of the target voice (tanh saturation
                         + 6% resample). NOT a neural clone — labelled as such.
  demo_03_partial_splice target + a different speaker spliced at 4.0 s and 8.0 s
  demo_04_impostor       a genuinely different, unenrolled speaker

Run `scripts/enroll_anchor.py` first (it holds out the last 3 anchor clips for
exactly this purpose).

Usage:
    python scripts/build_demo_suite.py            # writes wavs + ground_truth.json
    python scripts/build_demo_suite.py --dry-run  # show the plan only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from enroll_anchor import (  # noqa: E402
    ANCHOR_DIR,
    OTHER_DIR,
    TARGET_SR,
    USER_ID,
    load_dir,
)

OUT_DIRS = [REPO_ROOT / "frontend" / "public" / "demo", REPO_ROOT / "data" / "demo"]
DURATION_SEC = 10.0
HOLDOUT = 3  # must match `enroll_anchor.py --holdout`


def _fit(x: np.ndarray, seconds: float = DURATION_SEC, sr: int = TARGET_SR) -> np.ndarray:
    """Trim or zero-pad to exactly `seconds`."""
    n = int(seconds * sr)
    if x.size >= n:
        return x[:n].astype(np.float32)
    return np.pad(x.astype(np.float32), (0, n - x.size), mode="constant")


def _concat(clips: List[Tuple[str, np.ndarray]]) -> np.ndarray:
    return np.concatenate([x for _, x in clips]) if clips else np.zeros(0, dtype=np.float32)


def _degrade(x: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Signal-degrade a copy: tanh saturation + ~6% resample.

    This is a *distortion* of the target voice, not a neural voice clone. It is
    included so the pipeline has a non-bona-fide, target-matching sample to chew
    on — and it is labelled honestly because a real clone detector must be tested
    against real clones, which this is not.
    """
    from scipy.signal import resample_poly

    y = np.tanh(1.5 * x).astype(np.float32)
    y = resample_poly(y, 1000, 1060).astype(np.float32)  # ~6% slower/lower
    return _fit(y, sr=sr)


def _write_wav(path: Path, x: np.ndarray, sr: int = TARGET_SR) -> str:
    import soundfile as sf

    pcm = np.clip(x, -1.0, 1.0)
    sf.write(str(path), pcm, sr, subtype="PCM_16")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    anchor_clips = load_dir(ANCHOR_DIR)
    other_clips = load_dir(OTHER_DIR, limit=6)
    if len(anchor_clips) < HOLDOUT + 2:
        print(f"Need at least {HOLDOUT + 2} readable anchor recordings in {ANCHOR_DIR}")
        return 1
    if not other_clips:
        print(f"Need at least one non-target recording in {OTHER_DIR}")
        return 1

    # The last HOLDOUT clips are what enroll_anchor.py excluded from the anchor.
    held_out = anchor_clips[len(anchor_clips) - HOLDOUT :]
    print("Building from audio the anchor embedding has never seen:")
    for n, x in held_out:
        print(f"  target  {n}: {x.size / TARGET_SR:.1f}s")
    for n, x in other_clips[:3]:
        print(f"  other   {n}: {x.size / TARGET_SR:.1f}s")

    genuine = _fit(_concat(held_out))
    impostor_src = _concat(other_clips[:3])
    impostor = _fit(impostor_src)
    degraded = _degrade(genuine)
    splice = _fit(
        np.concatenate([genuine[: 4 * TARGET_SR], impostor[: 4 * TARGET_SR], genuine[4 * TARGET_SR : 6 * TARGET_SR]])
    )

    plan = [
        ("demo_01_genuine.wav", genuine, "genuine", "Target speaker, held out of enrolment"),
        ("demo_02_degraded_copy.wav", degraded, "degraded_copy", "tanh saturation + 6% resample of the target voice"),
        ("demo_03_partial_splice.wav", splice, "spliced", "Target / non-target spliced at 4.0s and 8.0s"),
        ("demo_04_impostor.wav", impostor, "impostor", "Different, unenrolled speaker"),
    ]

    if args.dry_run:
        for name, x, label, note in plan:
            print(f"  would write {name}: {x.size / TARGET_SR:.1f}s [{label}] {note}")
        return 0

    files = {}
    for name, x, label, note in plan:
        digest = None
        for d in OUT_DIRS:
            d.mkdir(parents=True, exist_ok=True)
            digest = _write_wav(d / name, x)
        files[name] = {
            "label": label,
            "target_speaker": USER_ID if label != "impostor" else "non-target speaker",
            "is_real_clone": False,
            "is_spliced": label == "spliced",
            "provenance": note,
            "duration_sec": round(float(x.size / TARGET_SR), 2),
            "sha256": digest,
        }
        print(f"  wrote {name}  ({label})")

    # Honest expectations: only what the current system can actually be held to.
    files["demo_01_genuine.wav"]["expected_verdict"] = "AUTHENTIC_TARGET"
    # The voice in demo_02 *is* the target's — tanh saturation and a resample
    # distort the signal but do not change who is speaking. AUTHENTIC_TARGET is
    # therefore the honest ground truth. What this file cannot do is test clone
    # detection, so say that out loud rather than letting the label imply it.
    files["demo_02_degraded_copy.wav"]["expected_verdict"] = "AUTHENTIC_TARGET"
    files["demo_02_degraded_copy.wav"]["expectation_note"] = (
        "Signal-degraded copy of the target voice, NOT a neural voice clone. "
        "AUTHENTIC_TARGET is correct because the speaker is still the target. "
        "This file does not exercise the countermeasure (CM) path: the shipped CM "
        "detector is a stub and has never seen synthetic audio."
    )
    files["demo_03_partial_splice.wav"]["expected_verdict"] = "SUSPICIOUS_SPLICED_AUDIO"
    files["demo_03_partial_splice.wav"]["splice_regions_sec"] = [[4.0, 8.0]]
    files["demo_04_impostor.wav"]["expected_verdict"] = "IMPOSTOR_SPEAKER"

    gt = {
        "dataset_name": "VaniRakshak Demo Benchmark Suite",
        "sample_rate": TARGET_SR,
        "enrolled_target": USER_ID,
        "rebuilt_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance_note": (
            "Rebuilt by scripts/build_demo_suite.py from real recordings. The target "
            "clips are held out of anchor enrolment. demo_02 is a signal-degraded "
            "copy, NOT a neural voice clone."
        ),
        "files": files,
    }
    for d in OUT_DIRS:
        (d / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")

    # The old filename is gone; keep data/demo and public/demo identical.
    stale = REPO_ROOT / "frontend" / "public" / "demo" / "demo_02_full_clone.wav"
    if stale.is_file():
        for d in OUT_DIRS:
            p = d / "demo_02_full_clone.wav"
            if p.is_file():
                p.unlink()
        print("  removed stale demo_02_full_clone.wav")

    print(f"\nWrote ground_truth.json to {len(OUT_DIRS)} locations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
