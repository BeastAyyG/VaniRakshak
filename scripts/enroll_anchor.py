#!/usr/bin/env python
"""Enrol the target speaker (anchor) into the serving backend's ASV model.

Until this runs, ``metrics.asv_consistency`` is a hard-coded 0.5 and the
"target-conditioned" half of the product does nothing — the claimed identity is
never enrolled, so `DetectionPipeline.analyse()` short-circuits.

What this does:
  1. Reads the reference recordings in ``data/anchor_speaker/`` (MP3/WAV/FLAC).
  2. Resamples to 16 kHz, runs an energy VAD, keeps voiced frames only.
  3. Embeds them with the *serving* ASV model (``vanirakshak.detectors.asv``),
     so the vector lives in the same space the live pipeline scores against.
  4. Writes ``data/anchor_asv_embedding.json`` — a 192-float vector plus provenance.
     No audio is written; the raw recordings are never retained (DPDP §8(7)).
  5. With ``--evaluate``, scores held-out anchor clips against other speakers
     and prints the separation actually achieved.

The committed JSON means a fresh clone (which has no audio — ``*.mp3`` /
``*.wav`` are gitignored) still boots with a working anchor.

Usage:
    python scripts/enroll_anchor.py
    python scripts/enroll_anchor.py --evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "vanirakshak-backend"
sys.path.insert(0, str(BACKEND_ROOT))

ANCHOR_DIR = REPO_ROOT / "data" / "anchor_speaker"
OTHER_DIR = REPO_ROOT / "data" / "other_speakers"
# Each ASV backend embeds into a different vector space, so each gets its own
# anchor file — anchor.py's load_anchor() picks the matching one for
# whichever backend is actually running (VR_ASV_BACKEND) and refuses to load
# a mismatched one. Loading an ecapa-space anchor into the stub (or vice
# versa) produces near-random cosine similarities that silently misfire the
# impostor-block override on every caller, real or not.
OUT_PATHS = {
    "stub": REPO_ROOT / "data" / "anchor_asv_embedding.json",
    "ecapa": REPO_ROOT / "data" / "anchor_asv_embedding_ecapa.json",
}

USER_ID = "narendra_modi"
TARGET_SR = 16000
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")


def _read_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Decode any libsndfile-supported file to mono float32 at its native rate."""
    import soundfile as sf

    x, sr = sf.read(str(path), always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, int(sr)


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, TARGET_SR)
    return resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)


def _vad(x: np.ndarray, sr: int, frame_ms: int = 25) -> np.ndarray:
    """Crude energy VAD: drop frames quieter than 10% of the loudest frame.

    Enrolment quality depends on voiced-only audio; leading silence and music
    beds otherwise dominate the spectral average.
    """
    n = int(sr * frame_ms / 1000)
    if x.size < n:
        return x
    nframes = x.size // n
    frames = x[: nframes * n].reshape(nframes, n)
    energy = np.sqrt((frames**2).mean(axis=1))
    if energy.max() <= 0:
        return x
    keep = energy > 0.10 * energy.max()
    if not keep.any():
        return x
    return frames[keep].ravel()


def load_dir(directory: Path, limit: int | None = None) -> List[Tuple[str, np.ndarray]]:
    if not directory.is_dir():
        return []
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if limit:
        files = files[:limit]
    out = []
    for p in files:
        try:
            x, sr = _read_audio(p)
            x = _resample(x, sr)
            x = _vad(x, TARGET_SR)
            out.append((p.name, x))
        except Exception as e:  # a single unreadable clip must not abort enrolment
            print(f"  ! skipping {p.name}: {type(e).__name__}: {e}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluate", action="store_true", help="score anchor vs other speakers after enrolling")
    ap.add_argument("--anchor-dir", default=str(ANCHOR_DIR))
    ap.add_argument(
        "--out",
        default=None,
        help="defaults to the backend-specific path in OUT_PATHS (data/anchor_asv_embedding.json "
        "for --backend stub, data/anchor_asv_embedding_ecapa.json for --backend ecapa)",
    )
    ap.add_argument(
        "--holdout",
        type=int,
        default=3,
        help="keep the last N clips out of enrolment (default 3) so the demo suite "
        "can be built from audio the anchor never saw",
    )
    ap.add_argument(
        "--backend",
        choices=["stub", "ecapa"],
        default="stub",
        help="'stub' (default): zero-dependency numpy/scipy embedding. "
        "'ecapa': real speechbrain ECAPA-TDNN embedding (requires "
        "requirements-asv.txt) — produces an anchor that actually carries "
        "voice biometrics, needed for real multi-speaker enrollment.",
    )
    args = ap.parse_args()

    if args.backend == "ecapa":
        from vanirakshak.detectors.asv_ecapa import ASVModelECAPA  # type: ignore

        def ASVModel():  # noqa: N802 - shadows the stub factory used below
            inst = ASVModelECAPA.try_create()
            if inst is None:
                print("!! --backend ecapa requested but the real model failed to load")
                print("   (missing `speechbrain`? run: pip install -r requirements-asv.txt)")
                raise SystemExit(1)
            return inst
    else:
        from vanirakshak.detectors.asv import ASVModel  # type: ignore

    anchor_dir = Path(args.anchor_dir)
    all_clips = load_dir(anchor_dir)
    if not all_clips:
        print(f"No readable audio in {anchor_dir} — cannot enrol.")
        print("Install `soundfile` and add reference recordings, then re-run.")
        return 1
    # Hold out the tail so `scripts/build_demo_suite.py` can build its "genuine"
    # clip from audio this embedding never saw. Enrolling and testing on the
    # same audio is how the previous benchmark suite ended up meaningless.
    n_hold = max(0, min(args.holdout, len(all_clips) - 2))
    clips, held_out = all_clips[: len(all_clips) - n_hold], all_clips[len(all_clips) - n_hold :]
    if held_out:
        print(f"Held out {len(held_out)} clip(s) for evaluation: {[n for n, _ in held_out]}")

    print(f"Enrolling '{USER_ID}' from {len(clips)} reference recording(s) in {anchor_dir}:")
    for name, x in clips:
        print(f"  - {name}: {x.size / TARGET_SR:.1f}s voiced @ {TARGET_SR} Hz")

    # Enrol on the concatenation so the embedding summarises the whole voice.
    joined = np.concatenate([x for _, x in clips])
    model = ASVModel()
    emb = model.enrol(USER_ID, joined, TARGET_SR)
    digest = hashlib.sha256(joined.tobytes()).hexdigest()

    # Sanity: the enrolled vector must reproduce on re-embedding the same audio.
    check = model.similarity(USER_ID, joined, TARGET_SR)
    print(f"\nSelf-similarity check: {check:.4f} (expect ~1.0)")

    # Calibration: raw cosine over these features only spans ~0.96-0.99, so the
    # default (cos+1)/2 mapping returns ~0.99 for target AND impostor. Fit a
    # logistic to the measured gap instead. Target scores are leave-one-out so
    # a clip is never scored against an embedding it helped build.
    calibration = None
    others = load_dir(OTHER_DIR, limit=20)
    if others and len(clips) >= 2:
        loo = []
        for i, (_name, x) in enumerate(clips):
            held_out = [c for j, c in enumerate(clips) if j != i]  # (name, audio) pairs
            m2 = ASVModel()
            m2.enrol(USER_ID, np.concatenate([h[1] for h in held_out]), TARGET_SR)
            loo.append(m2.similarity(USER_ID, x, TARGET_SR))
        other_scores = [model.similarity(USER_ID, x, TARGET_SR) for _, x in others]
        mid = (float(np.min(loo)) + float(np.max(other_scores))) / 2.0
        scale = max((float(np.mean(loo)) - float(np.mean(other_scores))) / 4.0, 1e-6)
        calibration = {
            "mid": round(mid, 6),
            "scale": round(scale, 6),
            "target_scores_loo": {"mean": round(float(np.mean(loo)), 6), "min": round(float(np.min(loo)), 6)},
            "nontarget_scores": {
                "n": len(other_scores),
                "mean": round(float(np.mean(other_scores)), 6),
                "max": round(float(np.max(other_scores)), 6),
            },
            "method": "logistic fitted between leave-one-out target scores and non-target scores",
        }
        print(f"\nCalibration fitted: mid={mid:.6f} scale={scale:.6f}")
        print(f"  target (leave-one-out) mean={np.mean(loo):.6f} min={np.min(loo):.6f}")
        print(f"  non-target           n={len(other_scores)} mean={np.mean(other_scores):.6f} "
              f"max={np.max(other_scores):.6f}")
    else:
        print("\nNo calibration fitted (need >=2 anchor clips and some non-target clips).")

    payload = {
        "user_id": USER_ID,
        "model": model.name,
        "embedding_dim": int(emb.shape[0]),
        "sample_rate": TARGET_SR,
        "source_clips": [n for n, _ in clips],
        "voiced_seconds": round(float(joined.size / TARGET_SR), 2),
        "audio_sha256": digest,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration": calibration,
        "embedding": [round(float(v), 6) for v in emb],
        "note": (
            "Real speechbrain/spkrec-ecapa-voxceleb ECAPA-TDNN embedding — "
            "genuine voice biometrics, actually discriminates speakers."
            if args.backend == "ecapa"
            else "Embedding from the serving ASV stub (Welch-PSD features, seeded random "
            "projection) — NOT ECAPA-TDNN. Signal-dependent and target-conditioned, "
            "but far weaker than a real speaker model."
        ),
    }
    out = Path(args.out) if args.out else OUT_PATHS[args.backend]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO_ROOT)}  ({out.stat().st_size} bytes, no audio retained)")

    if not args.evaluate:
        return 0

    print("\nSeparation check (higher = more like the enrolled voice)")
    anchor_scores = [model.similarity(USER_ID, x, TARGET_SR) for _, x in clips]
    print(f"  anchor clips     n={len(anchor_scores)}  mean={np.mean(anchor_scores):.4f}  "
          f"min={np.min(anchor_scores):.4f}  max={np.max(anchor_scores):.4f}")

    others = others or load_dir(OTHER_DIR, limit=20)
    if others:
        other_scores = [model.similarity(USER_ID, x, TARGET_SR) for _, x in others]
        print(f"  other speakers   n={len(other_scores)}  mean={np.mean(other_scores):.4f}  "
              f"min={np.min(other_scores):.4f}  max={np.max(other_scores):.4f}")
        gap = float(np.mean(anchor_scores) - np.mean(other_scores))
        print(f"  separation (mean anchor - mean other): {gap:+.4f}")
        if gap <= 0:
            print("  !! No separation — this embedding cannot distinguish the target.")
        else:
            lo, hi = float(np.min(anchor_scores)), float(np.max(other_scores))
            print(f"  overlap: {'YES' if lo < hi else 'no'} "
                  f"(worst anchor {lo:.4f} vs best impostor {hi:.4f})")
    else:
        print(f"  (no comparison clips in {OTHER_DIR})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
