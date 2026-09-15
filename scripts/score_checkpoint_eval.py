"""Score CM checkpoint(s) on an eval manifest and write a NEW reports JSON.

Modelled on ``scripts/score_pre_aug_eval.py`` but reusable: several checkpoints
in one pass, explicit output path, and no hardcoded ``A:\\...`` manifest.

Two deliberate behaviours:

* **Never overwrites an existing report** (docs/10 ground rule 6).
* **Refuses to stay quiet about unreadable audio.** The original harness
  substitutes a zeros window on read failure, which silently turns missing
  utterances into a confident bonafide/spoof score. Here failures are counted
  and reported, and the run aborts above ``--max-missing-frac``.

Scoring is uncalibrated: ``ml/train_cm.py`` does not fit a Platt scaler, so
raw LLRs are used. The benchmark baselines in ``reports/`` are raw too (their
metrics carry ``"calibrated": false``), so CLLR here is comparable against
them. It is *not* comparable against a calibrated figure such as §12's 0.7204.

Streaming, not pre-loading
--------------------------
The same utterances are scored by every checkpoint, so the obvious
optimisation is to decode them once and hold them all. That works for the
20 k eval slice and is **impossible for the full partition**:

    4 s window @ 16 kHz = 64,000 samples = 256 KB as float32
    20,000 utterances   x 256 KB =  4.8 GB   <- fits
    628,848 utterances  x 256 KB = 149.9 GB  <- does not

A first attempt at the full-partition re-score (2026-09-15) was killed by the
OOM killer after 1 h 18 m for exactly this reason, having written no report
and — because Python block-buffers stdout into a pipe — no log either.

So audio is now decoded in ``--audio-chunk`` windows and discarded once
scored. Only the scores are retained (two float64 per utterance, ~10 MB for
the whole partition), which is all the metrics need: EER, minDCF, actDCF and
CLLR are all computed from the pooled score arrays, so chunking is
metric-preserving. ``--audio-chunk`` is rounded up to a multiple of
``--batch-size`` so batch composition is identical to a whole-array pass and
the numbers are bit-reproducible.

Peak memory is ``--audio-chunk x 256 KB`` (default 4,096 -> 1.0 GB), plus one
frontend and head per checkpoint.

Usage:
    python scripts/score_checkpoint_eval.py \
        --checkpoint S42=data/cm_scratch_S42.pt \
        --checkpoint B42=data/cm_scratch_B42.pt \
        --out reports/cm_from_scratch_benchmark.json

    # full partition (628,848 utterances)
    python scripts/score_checkpoint_eval.py \
        --checkpoint S42=data/cm_scratch_S42.pt \
        --manifest A:/cm-corpus/manifests/eval_available.tsv \
        --out reports/cm_from_scratch_full_partition.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.compute_mindcf import compute_eer, evaluate_system  # noqa: E402
from cm_frontend import assert_frozen_checkpoint, load_frontend_for  # noqa: E402
from ml.aasist_model import AASISTLHead  # noqa: E402
from ml.augment import prepare_audio_window  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "data" / "eval_benchmark" / "eval_slice_20k.tsv"

# Samples per decoded window: 4.0 s at 16 kHz, float32. Used only for the
# memory note printed at startup.
_WINDOW_SAMPLES = 64000
_LOAD_THREADS = 16


def load_manifest(path: Path, limit: int | None) -> List[Tuple[str, int, str]]:
    records: List[Tuple[str, int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if len(header) < 3:
            raise SystemExit(f"{path}: expected >=3 TSV columns, got {header!r}")
        for row in reader:
            if len(row) < 3:
                continue
            records.append((row[0], 1 if row[1] == "bonafide" else 0, row[2]))
            if limit and len(records) >= limit:
                break
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH",
                    help="Repeatable. Label is the key used in the output JSON.")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON inside reports/. Must not already exist.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--audio-chunk", type=int, default=4096,
                    help="Utterances decoded per streaming window (default 4096 "
                         "= ~1.0 GB). Rounded up to a multiple of --batch-size so "
                         "batch composition matches a whole-array pass.")
    ap.add_argument("--max-missing-frac", type=float, default=0.05,
                    help="Abort if more than this fraction of utterances fail to read.")
    args = ap.parse_args()

    models: Dict[str, Path] = {}
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint expects LABEL=PATH, got {spec!r}")
        label, _, p = spec.partition("=")
        models[label] = Path(p)

    if args.out.exists():
        raise SystemExit(
            f"{args.out} already exists. Ground rule 6: reports are the record. "
            "Choose a new filename."
        )

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    records = load_manifest(args.manifest, args.limit)
    print(f"Scoring {len(records)} utterances from {args.manifest} on {device}")

    # Align the streaming window to the batch size so batches are formed
    # identically to a single whole-array pass -- this is what keeps a chunked
    # run bit-reproducible against an unchunked one.
    batch_size = max(1, args.batch_size)
    chunk = max(batch_size, args.audio_chunk)
    if chunk % batch_size:
        chunk += batch_size - (chunk % batch_size)
    peak_gb = chunk * _WINDOW_SAMPLES * 4 / 1024 ** 3
    print(f"  streaming in windows of {chunk} utterances "
          f"(~{peak_gb:.2f} GB of audio resident; {len(records)} total)")

    read_failures = 0

    def load_audio(item: Tuple[str, int, str]):
        nonlocal read_failures
        p, label, atk = item
        try:
            x, sr = sf.read(p, dtype="float32")
            w = prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False)
            return w, label, atk
        except Exception:
            read_failures += 1
            return np.zeros(_WINDOW_SAMPLES, dtype=np.float32), label, atk

    labels_all = np.array([y for _, y, _ in records])
    n_bon = int((labels_all == 1).sum())
    print(f"  bonafide={n_bon}  spoof={len(labels_all) - n_bon}")

    results: Dict[str, object] = {
        "manifest": str(args.manifest),
        "n_records": len(records),
        "n_bonafide": n_bon,
        "n_spoof": len(labels_all) - n_bon,
        "unreadable": read_failures,
        "device": device,
        "calibration": "none (raw LLR; ml/train_cm.py fits no Platt scaler)",
        "audio_chunk": chunk,
        "models": {},
    }

    # ------------------------------------------------------------------
    # Load every checkpoint ONCE, up front. Heads are small and the frozen
    # frontend is shared per checkpoint, so holding all of them costs a few
    # hundred MB -- against the 149.9 GB that holding the *audio* would.
    # ------------------------------------------------------------------
    loaded: Dict[str, Dict[str, object]] = {}
    accum: Dict[str, Dict[str, object]] = {}

    for label, ckpt_path in models.items():
        if not ckpt_path.is_file():
            print(f"  {label}: SKIPPED, {ckpt_path} not found")
            continue
        print(f"\n=== {label} :: {ckpt_path} ===")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert_frozen_checkpoint(ckpt, str(Path(__file__).name))

        if ckpt.get("merge_layers"):
            raise SystemExit(f"{ckpt_path}: merge_layers checkpoints are not supported here")

        wavlm, dtype = load_frontend_for(ckpt, device)
        in_dim = 768 if ckpt.get("frontend") == "microsoft/wavlm-base-plus" else 1024
        model = AASISTLHead(in_dim=in_dim).to(device)
        # strict=True: a silent partial load is exactly the defect PR #6 removed
        # from the trainer. A from-scratch head of this architecture has every
        # key, so anything missing or unexpected is a wrong checkpoint.
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.to(device)
        model.eval()

        # Mirror ml/train_cm.py::select_wavlm_features: the layer the head was
        # actually trained on. Checkpoints older than the field default to the
        # last layer, which is what data/cm_aasist.pt (post-aug L12) used.
        # hidden_states[0] is the CNN embedding output, a legitimate layer, so
        # only a missing field falls back to the last layer.
        layer_index = ckpt.get("layer_index")
        layer_index = -1 if layer_index is None else int(layer_index)
        print(f"  frontend layer: {layer_index}")

        loaded[label] = {
            "ckpt": ckpt,
            "path": ckpt_path,
            "wavlm": wavlm,
            "dtype": dtype,
            "model": model,
            "layer_index": layer_index,
        }
        accum[label] = {"bon": [], "spf": [], "atk": {}}

    if not loaded:
        raise SystemExit("no checkpoints to score")

    # ------------------------------------------------------------------
    # Stream the manifest: decode a window, score it through EVERY model,
    # keep the scores, drop the audio.
    # ------------------------------------------------------------------
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=_LOAD_THREADS) as ex, torch.no_grad():
        for start in range(0, len(records), chunk):
            stop = min(start + chunk, len(records))
            window = records[start:stop]

            audio: List[np.ndarray] = []
            for w, _, _ in ex.map(load_audio, window):
                audio.append(w)

            labels_w = labels_all[start:stop]
            attacks_w = [r[2] for r in window]

            for label, st in loaded.items():
                wavlm = st["wavlm"]
                model = st["model"]
                dtype = st["dtype"]
                layer_index = st["layer_index"]
                acc = accum[label]
                for i in range(0, len(audio), batch_size):
                    batch = np.stack(audio[i:i + batch_size])
                    y_b = labels_w[i:i + batch_size]
                    a_b = attacks_w[i:i + batch_size]
                    audio_t = torch.from_numpy(batch).to(dtype).to(device)
                    hidden = wavlm(audio_t, output_hidden_states=True).hidden_states[layer_index].float()
                    _, logits = model(hidden)
                    llrs = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
                    for s, y, a in zip(llrs, y_b, a_b):
                        score = float(s)
                        if y == 1:
                            acc["bon"].append(score)
                        else:
                            acc["spf"].append(score)
                            acc["atk"].setdefault(a, []).append(score)

            del audio
            elapsed = time.time() - t0
            print(f"  scored {stop}/{len(records)} "
                  f"({stop / elapsed:.1f} utts/s overall, {elapsed / 60:.1f} min elapsed)")

            # Checked per window rather than once at the end: a corpus that is
            # not mounted should fail in seconds, not after an hour of scoring.
            missing_frac = read_failures / max(1, stop)
            if missing_frac > args.max_missing_frac:
                raise SystemExit(
                    f"Aborting: {missing_frac:.2%} of the first {stop} utterances is "
                    "unreadable. The eval corpus is not mounted here, or the "
                    "manifest paths are stale."
                )

    if read_failures:
        print(f"WARNING: {read_failures} utterances could not be read "
              f"({read_failures / max(1, len(records)):.2%}); they were scored as silence.")
    results["unreadable"] = read_failures
    print(f"\nDecoded and scored {len(records)} utterances in "
          f"{(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------------------------
    # Metrics, from the pooled scores only.
    # ------------------------------------------------------------------
    for label, st in loaded.items():
        ckpt = st["ckpt"]
        acc = accum[label]
        bon = np.array(acc["bon"])
        spf = np.array(acc["spf"])
        res = evaluate_system(bon, spf)
        print(f"\n=== {label} ===")
        print(f"  pooled EER : {res['eer'] * 100.0:.3f}%")
        print(f"  minDCF     : {res['min_dcf']:.4f}")
        print(f"  CLLR       : {res['cllr']:.4f}")

        per_atk: Dict[str, float] = {}
        for a, scores in sorted(acc["atk"].items()):
            s_arr = np.array(scores)
            if len(s_arr) > 0 and len(bon) > 0:
                eer, _ = compute_eer(bon, s_arr)
                per_atk[a] = round(eer * 100.0, 3)
                print(f"    {a:<6}: {per_atk[a]:>7.3f}%")

        results["models"][label] = {  # type: ignore[index]
            "checkpoint": str(st["path"]),
            "layer_index": ckpt.get("layer_index"),
            "init_weights": ckpt.get("init_weights"),
            "seed": ckpt.get("seed"),
            "augmentation": ckpt.get("augmentation"),
            "dev_eer": ckpt.get("dev_eer"),
            "dev_min_dcf": ckpt.get("dev_min_dcf"),
            "metrics": res,
            "per_attack_eer": per_atk,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
