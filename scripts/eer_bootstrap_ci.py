"""Bootstrap confidence intervals for pooled and per-attack EER.

Why this exists: the docs/10 G4 criterion is written on A19 and A20, and those
two attacks have only ~935 and ~921 spoof samples in the 20k eval slice. A
per-attack EER quoted to three decimals (16.257 %) invites reading differences
of 1-2 pp as real. They usually are not.

This scores a checkpoint once, caches the raw scores, then bootstraps:
resample bonafide and spoof scores with replacement, recompute EER, repeat.
The percentile interval is the answer. No normality assumption, no analytic
approximation that I would then have to defend.

Usage:
    python scripts/eer_bootstrap_ci.py --checkpoint data/cm_g4_rawboost_s42.pt \
        [--reps 2000] [--attacks A19 A20] [--reuse]

--reuse skips scoring and reloads the cached .npz (free re-analysis).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.compute_mindcf import compute_eer  # noqa: E402
from cm_frontend import load_frontend_for  # noqa: E402
from eval_call_channel import build_channel, load_manifest, score_all  # noqa: E402
from ml.aasist_model import AASISTLHead  # noqa: E402


def score_and_cache(ckpt: Path, manifest: Path, batch: int, device: str,
                    cache: Path, layer_override=None) -> None:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    merge_layers = ck.get("merge_layers")
    raw_li = ck.get("layer_index")
    layer_index = int(raw_li) if raw_li is not None else 5
    if layer_override is not None:
        layer_index = int(layer_override)

    wavlm, dtype = load_frontend_for(ck, device)
    model = AASISTLHead(in_dim=768, merge_layers=merge_layers,
                        freeze_merge=bool(ck.get("freeze_merge", False)))
    model.load_state_dict(ck["state_dict"], strict=False)
    model.to(device).eval()

    _, rows = load_manifest(manifest, None)
    channel = build_channel("clean")
    bon, spf, per_atk = score_all(model, wavlm, rows, channel, device,
                                  merge_layers, layer_index, batch, dtype)
    np.savez_compressed(
        cache,
        bon=np.asarray(bon, dtype=np.float64),
        spf=np.asarray(spf, dtype=np.float64),
        **{f"atk_{k}": np.asarray(v, dtype=np.float64) for k, v in per_atk.items()},
    )
    print(f"cached {len(bon)} bonafide / {len(spf)} spoof scores -> {cache}")


def bootstrap_ci(bon: np.ndarray, spf: np.ndarray, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    nb, ns = len(bon), len(spf)
    point, _ = compute_eer(bon, spf)
    draws = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        b = bon[rng.integers(0, nb, nb)]
        s = spf[rng.integers(0, ns, ns)]
        draws[i], _ = compute_eer(b, s)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi), float(np.std(draws))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=r"data\eval_benchmark\eval_slice_20k.tsv")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attacks", nargs="*", default=["A19", "A20"])
    ap.add_argument("--layer-index", type=int, default=None)
    ap.add_argument("--reuse", action="store_true", help="rescore=False; reload cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    # Key the cache on BOTH checkpoint and manifest. Keying on the checkpoint
    # alone means scoring the same model on a second corpus (e.g. In-the-Wild)
    # silently overwrites the first, and --reuse then re-analyses the wrong
    # scores with no warning.
    mstem = Path(args.manifest).stem
    cache = REPO_ROOT / "reports" / f"scores_{ckpt.stem}__{mstem}.npz"

    if not args.reuse or not cache.exists():
        t = time.time()
        score_and_cache(ckpt, Path(args.manifest), args.batch, args.device,
                        cache, args.layer_index)
        print(f"  scoring took {time.time() - t:.0f}s")

    z = np.load(cache)
    bon = z["bon"]
    spf = z["spf"]

    print(f"\nbootstrapping with {args.reps} reps (seed {args.seed})\n")
    rows = [("pooled", bon, spf)]
    for a in args.attacks:
        key = f"atk_{a}"
        if key in z:
            rows.append((a, bon, z[key]))

    print(f"{'attack':8s} {'n_spoof':>8s} {'EER %':>8s} {'95% CI':>18s} {'+/-':>7s}")
    out = {
        "checkpoint": str(ckpt),
        "reps": args.reps,
        "seed": args.seed,
        "n_bonafide": int(len(bon)),
        "pooled": None,
        "attacks": {},
    }
    for name, b, s in rows:
        pt, lo, hi, sd = bootstrap_ci(b, s, args.reps, args.seed)
        print(f"{name:8s} {len(s):8d} {pt*100:8.3f} "
              f"{f'[{lo*100:.2f}, {hi*100:.2f}]':>18s} {sd*100*1.96:7.2f}")
        rec = {
            "n_spoof": int(len(s)),
            "eer_pct": round(pt * 100, 3),
            "ci95_pct": [round(lo * 100, 3), round(hi * 100, 3)],
            "half_width_pp": round(sd * 100 * 1.96, 3),
        }
        if name == "pooled":
            out["pooled"] = rec
        else:
            out["attacks"][name] = rec

    if args.out:
        import json
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
