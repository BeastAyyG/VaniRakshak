"""Telephone-channel evaluation harness for the CM detector.

Scores a manifest under explicit, deterministic call-channel conditions and
reports pooled EER/minDCF plus **per-attack** EER.

Why this exists: the G4 criterion is per-attack (A19 and A20 below 12 %), but
the training log only prints *pooled* dev EER. Pooled EER hides exactly the
per-attack failures G4 is meant to fix. This harness makes the criterion
directly measurable, and adds the channel dimension -- whether the verdict
survives the telephone path.

Channel caveat that matters for interpretation
----------------------------------------------
ASVspoof 5 was **already recorded over real telephone networks**, so for that
corpus `clean` *is* the call condition; adding a codec on top would be
unrealistic double-transcoding. The codec conditions exist for corpora that
were NOT captured over a phone (In-the-Wild, IndicVoices, your own studio
recordings), where you must simulate the call path to say anything about
call performance.

Usage:
    python scripts/eval_call_channel.py \
        --checkpoint data/cm_g4_rawboost_s42.pt \
        --manifest data/eval_benchmark/eval_slice_20k.tsv \
        --conditions clean --limit 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.compute_mindcf import compute_eer, compute_mindcf  # noqa: E402
from cm_frontend import load_frontend_for  # noqa: E402
from ml.aasist_model import AASISTLHead  # noqa: E402
from ml.augment import (  # noqa: E402
    add_babble_simple,
    add_noise,
    alaw_encode_decode,
    amr_wb_simulate,
    apply_rir,
    gsm_efr_simulate,
    prepare_audio_window,
    random_gain,
    synthetic_rir,
    telephony_bandpass,
    ulaw_encode_decode,
)
from ml.train_cm import select_wavlm_features  # noqa: E402

SR = 16000


def build_channel(name: str, sr: int = SR):
    """Return a deterministic (audio -> audio) call-channel transform."""
    rng = np.random.default_rng(1234)

    def _identity(x):
        return x

    def _bandpass(x):
        return telephony_bandpass(x, sr=sr, lowcut=300.0, highcut=3400.0)

    def _noise(x):
        return add_noise(x, snr_db=15.0, rng=rng)

    def _babble(x):
        return add_babble_simple(x, snr_db=15.0, sr=sr, rng=rng)

    def _reverb(x):
        rir = synthetic_rir(sr, t60_ms=200.0, rng=rng)
        return apply_rir(x, rir)

    def _full_call(x):
        # Realistic PSTN/mobile chain: handset bandpass -> G.711 A-law ->
        # line noise -> level jitter. Order matters; this matches the order
        # CodecTelephonyAugmenter applies during training.
        x = telephony_bandpass(x, sr=sr, lowcut=300.0, highcut=3400.0)
        x = alaw_encode_decode(x)
        x = add_noise(x, snr_db=20.0, rng=rng)
        x = random_gain(x, low_db=-3.0, high_db=3.0, rng=rng)
        return x

    table = {
        "clean": _identity,
        "alaw": alaw_encode_decode,          # G.711 A-law (Europe/India PSTN)
        "ulaw": ulaw_encode_decode,          # G.711 mu-law (US/Japan PSTN)
        "amr_wb": lambda x: amr_wb_simulate(x, sr),
        "gsm_efr": lambda x: gsm_efr_simulate(x, sr),
        "bandpass": _bandpass,
        "noise": _noise,
        "babble": _babble,
        "reverb": _reverb,
        "full_call": _full_call,
    }
    if name not in table:
        raise ValueError(f"unknown condition {name!r}; choose from {sorted(table)}")
    return table[name]


def load_manifest(path: Path, limit: int | None):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            rows.append((row[0], row[1], row[2]))
            if limit and len(rows) >= limit:
                break
    return header, rows


@torch.no_grad()
def score_all(model, wavlm, rows, channel, device, merge_layers, layer_index, batch=16, dtype=torch.float32):
    bon, spf, per_atk = [], [], {}
    wins, labels, atks = [], [], []

    for path, label, atk in rows:
        try:
            x, sr = sf.read(path, dtype="float32")
        except Exception:
            continue
        w = prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False)
        w = channel(np.asarray(w, dtype=np.float32))
        wins.append(np.asarray(w, dtype=np.float32))
        labels.append(1 if label == "bonafide" else 0)
        atks.append(atk if label != "bonafide" else "-")

    idx = 0
    while idx < len(wins):
        chunk = wins[idx : idx + batch]
        tens = torch.from_numpy(np.stack(chunk)).to(device).to(dtype)
        out = wavlm(tens, output_hidden_states=True)
        feats = select_wavlm_features(out, layer_index, merge_layers)
        out_h = model(feats)
        # AASISTLHead returns (embeddings, logits) -- the trainer unpacks it as
        # `_, logits = model(feats)`. Element 0 is NOT the logits.
        logits = out_h[1] if isinstance(out_h, (tuple, list)) else out_h
        # Score convention must match ml/train_cm.py:920 exactly --
        # llr = logits[:,1] - logits[:,0], with bonafide scoring higher.
        s = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
        for j, val in enumerate(s):
            k = idx + j
            if labels[k] == 1:
                bon.append(float(val))
            else:
                spf.append(float(val))
                per_atk.setdefault(atks[k], []).append(float(val))
        idx += batch

    return np.array(bon), np.array(spf), {k: np.array(v) for k, v in per_atk.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=r"data\eval_benchmark\eval_slice_20k.tsv")
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    merge_layers = ckpt.get("merge_layers")
    layer_index = int(ckpt.get("layer_index", 5))
    freeze_merge = bool(ckpt.get("freeze_merge", False))
    print(f"checkpoint: {ckpt_path.name}")
    print(f"  merge_layers={merge_layers} layer_index={layer_index} seed={ckpt.get('seed')}")
    print(f"  trained dev_eer={ckpt.get('dev_eer')} dev_min_dcf={ckpt.get('dev_min_dcf')}")

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    wavlm, dtype = load_frontend_for(ckpt, device)

    model = AASISTLHead(in_dim=768, merge_layers=merge_layers, freeze_merge=freeze_merge)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device).eval()

    _, rows = load_manifest(Path(args.manifest), args.limit)
    print(f"manifest: {args.manifest} -> {len(rows)} rows")

    results = {}
    for cond in args.conditions:
        t0 = time.time()
        channel = build_channel(cond)
        bon, spf, per_atk = score_all(
            model, wavlm, rows, channel, device, merge_layers, layer_index, args.batch, dtype
        )
        eer, thr = compute_eer(bon, spf)
        dcf, _ = compute_mindcf(bon, spf)
        atk_eer = {}
        for k in sorted(per_atk):
            if len(per_atk[k]) >= 20:
                a_eer, _ = compute_eer(bon, per_atk[k])
                atk_eer[k] = round(float(a_eer) * 100, 3)
        results[cond] = {
            "pooled_eer_pct": round(float(eer) * 100, 3),
            "pooled_minDCF": round(float(dcf), 4),
            "threshold": round(float(thr), 4),
            "n_bonafide": int(bon.size),
            "n_spoof": int(spf.size),
            "per_attack_eer_pct": atk_eer,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"\n[{cond}] pooled EER {float(eer)*100:.3f}%  minDCF {float(dcf):.4f}  "
              f"({results[cond]['seconds']}s)")
        for k, v in sorted(atk_eer.items()):
            flag = "  <-- G4 criterion attack" if k in ("A19", "A20") else ""
            print(f"    {k}: {v:6.2f}%{flag}")

    out = Path(args.out) if args.out else REPO_ROOT / "reports" / (
        f"call_channel_eval_{Path(args.checkpoint).stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "manifest": str(args.manifest),
                "n_rows": len(rows),
                "conditions": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
