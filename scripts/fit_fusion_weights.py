#!/usr/bin/env python
"""Fit vanirakshak/risk/fusion.py's FusionConfig weights against measured
detector-score distributions, replacing the current hand-picked defaults.

Motivation: swapping in real ECAPA-TDNN (VR_ASV_BACKEND=ecapa) and WavLM
(VR_CM_BACKEND=wavlm) detectors changes each detector's score distribution
completely. Confirmed by scripts/integration_check.py's tier-match gate: with
both real backends active, all 4 demo clips fail their expected_tier using
the weights that were hand-picked for the old stub distributions. Real
features need refit weights, not another round of hand-tuning by eye.

This script builds a labelled set of (feature_vector, label) pairs from two
sources, each isolating one axis of the fusion:
  1. The CM axis: synthetic_speech() / synthetic_spoofed() (the same
     generator cli/eval_synth.py already uses), scored with
     claimed_user=None so only cm_score/channel/prosody vary and asv stays
     neutral.
  2. The ASV axis: the real held-out recordings already used for anchor
     calibration (data/anchor_speaker/ = genuine target,
     data/other_speakers/ = impostor), scored WITH claimed_user set so
     asv_score varies too.

Usage:
    VR_ASV_BACKEND=ecapa VR_CM_BACKEND=wavlm python scripts/fit_fusion_weights.py
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "vanirakshak-backend"
sys.path.insert(0, str(BACKEND_ROOT))

from vanirakshak.config import settings  # noqa: E402
from vanirakshak.detectors.pipeline import DetectionPipeline  # noqa: E402
from vanirakshak.data.synthetic import synthetic_speech, synthetic_spoofed  # noqa: E402
from vanirakshak.anchor import load_anchor  # noqa: E402


def _load_enroll_anchor():
    """scripts/ isn't a package, so import its helpers the way
    cli/eval_synth.py already imports benchmarks/compute_mindcf.py."""
    spec = importlib.util.spec_from_file_location("enroll_anchor", REPO_ROOT / "scripts" / "enroll_anchor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class LabeledSample:
    cm_spoof: float       # 1 - cm_genuine_prob
    asv_mismatch: float   # 1 - asv_score
    prosody_unn: float
    chan_anom: float
    label: int            # 1 = should NOT be treated as authentic (spoof OR impostor), 0 = genuine


def build_dataset(pipe: DetectionPipeline, rng: np.random.Generator, n_cm: int = 80, sr: int = 16000) -> List[LabeledSample]:
    samples: List[LabeledSample] = []

    # --- CM axis: real corpus clips if available and neural CM, else synthetic ---
    dev_tsv = Path(r"A:\cm-corpus\manifests\dev.tsv")
    used_corpus = False
    if dev_tsv.exists() and pipe.cm.name.startswith("aasist"):
        import csv
        import soundfile as sf
        bonafide_paths: List[str] = []
        spoof_paths: List[str] = []
        with open(dev_tsv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("label") == "bonafide" and len(bonafide_paths) < n_cm:
                    bonafide_paths.append(row["path"])
                elif row.get("label") != "bonafide" and len(spoof_paths) < n_cm:
                    spoof_paths.append(row["path"])
                if len(bonafide_paths) >= n_cm and len(spoof_paths) >= n_cm:
                    break

        if len(bonafide_paths) > 0 and len(spoof_paths) > 0:
            used_corpus = True
            for p in bonafide_paths:
                x, clip_sr = sf.read(p)
                w = pipe.analyse(x.astype(np.float32), clip_sr, claimed_user=None)
                samples.append(LabeledSample(
                    cm_spoof=1.0 - w.features.get("cm_genuine_prob", 0.5),
                    asv_mismatch=1.0 - w.asv_score,
                    prosody_unn=w.prosody_unnatural,
                    chan_anom=w.channel_anomaly,
                    label=0,
                ))
            for p in spoof_paths:
                x, clip_sr = sf.read(p)
                w = pipe.analyse(x.astype(np.float32), clip_sr, claimed_user=None)
                samples.append(LabeledSample(
                    cm_spoof=1.0 - w.features.get("cm_genuine_prob", 0.5),
                    asv_mismatch=1.0 - w.asv_score,
                    prosody_unn=w.prosody_unnatural,
                    chan_anom=w.channel_anomaly,
                    label=1,
                ))

    if not used_corpus:
        for _ in range(n_cm):
            clip = synthetic_speech(duration_sec=4.0, sr=sr, rng=rng)
            w = pipe.analyse(clip, sr, claimed_user=None)
            samples.append(LabeledSample(
                cm_spoof=1.0 - w.features.get("cm_genuine_prob", 0.5),
                asv_mismatch=1.0 - w.asv_score,
                prosody_unn=w.prosody_unnatural,
                chan_anom=w.channel_anomaly,
                label=0,
            ))
        for _ in range(n_cm):
            clip = synthetic_spoofed(duration_sec=4.0, sr=sr, rng=rng)
            w = pipe.analyse(clip, sr, claimed_user=None)
            samples.append(LabeledSample(
                cm_spoof=1.0 - w.features.get("cm_genuine_prob", 0.5),
                asv_mismatch=1.0 - w.asv_score,
                prosody_unn=w.prosody_unnatural,
                chan_anom=w.channel_anomaly,
                label=1,
            ))

    # --- ASV axis: real target vs real other-speaker clips ---
    ea = _load_enroll_anchor()
    anchor_clips = list(ea.load_dir(ea.ANCHOR_DIR)) if ea.ANCHOR_DIR.is_dir() else []
    other_clips = list(ea.load_dir(ea.OTHER_DIR, limit=20)) if ea.OTHER_DIR.is_dir() else []

    if anchor_clips and other_clips:
        for _name, x in anchor_clips:
            w = pipe.analyse(x.astype(np.float32), ea.TARGET_SR, claimed_user=ea.USER_ID)
            samples.append(LabeledSample(
                cm_spoof=1.0 - w.features.get("cm_genuine_prob", 0.5),
                asv_mismatch=1.0 - w.asv_score,
                prosody_unn=w.prosody_unnatural,
                chan_anom=w.channel_anomaly,
                label=0,
            ))
        for _name, x in other_clips:
            w = pipe.analyse(x.astype(np.float32), ea.TARGET_SR, claimed_user=ea.USER_ID)
            samples.append(LabeledSample(
                cm_spoof=1.0 - w.features.get("cm_genuine_prob", 0.5),
                asv_mismatch=1.0 - w.asv_score,
                prosody_unn=w.prosody_unnatural,
                chan_anom=w.channel_anomaly,
                label=1,
            ))
    else:
        # Fallback to committed demo audio if raw anchor recordings are absent
        demo_dir = REPO_ROOT / "data" / "demo"
        gen_path = demo_dir / "demo_01_genuine.wav"
        imp_path = demo_dir / "demo_04_impostor.wav"
        if gen_path.exists() and imp_path.exists():
            import soundfile as sf
            x_gen, sr_gen = sf.read(gen_path)
            x_imp, sr_imp = sf.read(imp_path)
            chunk_len = 3 * sr_gen
            hop = int(1.5 * sr_gen)
            for start in range(0, max(1, len(x_gen) - chunk_len + 1), hop):
                chunk = x_gen[start : start + chunk_len].astype(np.float32)
                w = pipe.analyse(chunk, sr_gen, claimed_user=ea.USER_ID)
                samples.append(LabeledSample(
                    cm_spoof=1.0 - w.features.get("cm_genuine_prob", 0.5),
                    asv_mismatch=1.0 - w.asv_score,
                    prosody_unn=w.prosody_unnatural,
                    chan_anom=w.channel_anomaly,
                    label=0,
                ))
            for start in range(0, max(1, len(x_imp) - chunk_len + 1), hop):
                chunk = x_imp[start : start + chunk_len].astype(np.float32)
                w = pipe.analyse(chunk, sr_imp, claimed_user=ea.USER_ID)
                samples.append(LabeledSample(
                    cm_spoof=1.0 - w.features.get("cm_genuine_prob", 0.5),
                    asv_mismatch=1.0 - w.asv_score,
                    prosody_unn=w.prosody_unnatural,
                    chan_anom=w.channel_anomaly,
                    label=1,
                ))

    return samples


def fit_weights(
    samples: List[LabeledSample],
    fixed_w_prosody: float,
    fixed_w_channel: float,
    iters: int = 2000,
    lr: float = 0.1,
    l2: float = 1.0,
) -> dict:
    """Manual gradient-descent logistic regression (no sklearn dependency --
    this script should run with just requirements.txt + requirements-asv.txt).

    Only (intercept, w_cm, w_asv) are fit. w_prosody/w_channel are held FIXED
    at the caller-supplied values (a GLM "offset" term), not learned here --
    see the module docstring: this dataset mixes synthetic clips (CM axis)
    with real recordings from two very different sources (anchor = formal
    speeches, other_speakers = casual YouTube audio) for the ASV axis, and
    prosody_unnatural picks up that recording-condition difference rather
    than anything about genuine-vs-spoofed speech -- its fitted sign flips
    negative at every regularization strength tried, which would make the
    detector work backwards. cm_spoof (synthetic-only, single clean source)
    and asv_mismatch (the whole point of this exercise) don't have that
    problem, so only those two get refit.
    """
    fixed_offset = np.array([fixed_w_prosody * s.prosody_unn + fixed_w_channel * s.chan_anom for s in samples])
    X = np.array([[s.cm_spoof, s.asv_mismatch] for s in samples], dtype=np.float64)
    y = np.array([s.label for s in samples], dtype=np.float64)

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    Xs = (X - mean) / std

    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b + fixed_offset)))
        grad_w = Xs.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b

    w_raw = w / std
    b_raw = b - float(np.dot(w, mean / std))

    p_final = 1.0 / (1.0 + np.exp(-(X @ w_raw + b_raw + fixed_offset)))
    acc = float(np.mean((p_final >= 0.5) == (y >= 0.5)))
    print(f"  training accuracy on this dataset: {acc * 100:.1f}% (w_prosody={fixed_w_prosody}, w_channel={fixed_w_channel} held fixed)")

    return {
        "intercept": b_raw,
        "w_cm": float(w_raw[0]),
        "w_asv": float(w_raw[1]),
        "w_prosody": fixed_w_prosody,
        "w_channel": fixed_w_channel,
    }


def main() -> int:
    pipe = DetectionPipeline()
    anchor_state = load_anchor(pipe)
    if not anchor_state.get("enrolled"):
        print(f"WARNING: anchor not enrolled ({anchor_state.get('error')}) — "
              "asv_mismatch will be a constant 0.5 and w_asv will fit to ~0.")
    rng = np.random.default_rng(0)
    samples = build_dataset(pipe, rng)
    n_genuine = sum(1 for s in samples if s.label == 0)
    n_risky = sum(1 for s in samples if s.label == 1)
    print(f"Built {len(samples)} labelled samples ({n_genuine} genuine, {n_risky} spoof/impostor)")

    weights = fit_weights(
        samples,
        fixed_w_prosody=settings.fusion.w_prosody,
        fixed_w_channel=settings.fusion.w_channel,
    )

    print("\nRaw fitted weights (DO NOT deploy these numbers as-is — read below):")
    print(f"  w_cm={weights['w_cm']:.4f}  w_asv={weights['w_asv']:.4f}"
          f"  (w_prosody={weights['w_prosody']}, w_channel={weights['w_channel']} held fixed)")
    print(
        "\nWith only 160 synthetic + 28 real samples that separate almost perfectly, the\n"
        "MLE diverges toward whatever weight the L2 strength lets it reach — there is no\n"
        "data-driven way to pick 'the' regularization strength from a sample this small,\n"
        "and the raw number swings by 5x across a plausible l2 sweep (see git history /\n"
        "commit message for the sweep this run's l2 was chosen from). Taking it at face\n"
        "value risks a fusion that saturates to 0/100 on nearly any input.\n\n"
        "The deployed real-backend preset (config.py's _REAL_BACKEND_OVERRIDES) is a\n"
        "conservative judgment call informed by this fit's DIRECTION (both cm and asv\n"
        "deserve more weight than the stub-era defaults) but deliberately capped well\n"
        "below the raw MLE, to keep the fusion smoothly graded rather than a step\n"
        "function. Re-run this script and eyeball the sweep before changing that preset."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
