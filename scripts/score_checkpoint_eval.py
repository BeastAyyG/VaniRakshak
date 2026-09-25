"""Score CM checkpoint(s) on the 20k eval slice and write a NEW reports JSON.

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

Usage:
    python scripts/score_checkpoint_eval.py \
        --checkpoint S42=data/cm_scratch_S42.pt \
        --checkpoint B42=data/cm_scratch_B42.pt \
        --out reports/cm_from_scratch_benchmark.json \
        --dump-scores C:/vr-out/p2/dev

``--dump-scores DIR`` also writes ``DIR/<label>.npz`` (raw_llr, label,
attack, row_index, manifest_sha256) — the input to
``scripts/calibrate_decision_layer.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    ap.add_argument("--max-missing-frac", type=float, default=0.05,
                    help="Abort if more than this fraction of utterances fail to read.")
    ap.add_argument("--dump-scores", type=Path, default=None,
                    help="Directory for per-utterance <label>.npz score dumps (must not exist per label).")
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
            return np.zeros(64000, dtype=np.float32), label, atk

    # Pre-load everything once: the same utterances are scored by every model,
    # so re-decoding them per checkpoint would quadruple the I/O for nothing.
    t0 = time.time()
    audio: List[np.ndarray] = []
    labels: List[int] = []
    attacks: List[str] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, (w, y, a) in enumerate(ex.map(load_audio, records)):
            audio.append(w)
            labels.append(y)
            attacks.append(a)
            if i and i % 5000 == 0:
                print(f"  loaded {i}/{len(records)}")
    print(f"Loaded audio in {time.time() - t0:.1f}s")

    missing_frac = read_failures / max(1, len(records))
    if read_failures:
        print(f"WARNING: {read_failures} utterances could not be read "
              f"({missing_frac:.2%}); they were scored as silence.")
    if missing_frac > args.max_missing_frac:
        raise SystemExit(
            f"Aborting: {missing_frac:.2%} of the slice is unreadable. The eval "
            "corpus is not mounted here, or the manifest paths are stale."
        )

    labels_arr = np.array(labels)
    bonafide_mask = labels_arr == 1
    n_bon = int(bonafide_mask.sum())
    print(f"  bonafide={n_bon}  spoof={len(labels_arr) - n_bon}")

    results: Dict[str, object] = {
        "manifest": str(args.manifest),
        "n_records": len(records),
        "n_bonafide": n_bon,
        "n_spoof": len(labels_arr) - n_bon,
        "unreadable": read_failures,
        "device": device,
        "calibration": "none (raw LLR; ml/train_cm.py fits no Platt scaler)",
        "models": {},
    }

    for label, ckpt_path in models.items():
        if not ckpt_path.is_file():
            print(f"  {label}: SKIPPED, {ckpt_path} not found")
            continue
        print(f"\n=== {label} :: {ckpt_path} ===")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert_frozen_checkpoint(ckpt, str(Path(__file__).name))

        wavlm, dtype = load_frontend_for(ckpt, device)
        in_dim = 768 if ckpt.get("frontend") == "microsoft/wavlm-base-plus" else 1024
        model = AASISTLHead(in_dim=in_dim).to(device)
        # strict=True: a silent partial load is exactly the defect PR #6 removed
        # from the trainer. A from-scratch head of this architecture has every
        # key, so anything missing or unexpected is a wrong checkpoint.
        if ckpt.get("merge_layers"):
            raise SystemExit(f"{ckpt_path}: merge_layers checkpoints are not supported here")
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

        all_scores: List[float] = []
        bon_scores: List[float] = []
        spf_scores: List[float] = []
        atk_scores: Dict[str, List[float]] = {}
        t1 = time.time()
        with torch.no_grad():
            for i in range(0, len(audio), args.batch_size):
                batch = np.stack(audio[i:i + args.batch_size])
                y_b = labels_arr[i:i + args.batch_size]
                a_b = attacks[i:i + args.batch_size]
                audio_t = torch.from_numpy(batch).to(dtype).to(device)
                hidden = wavlm(audio_t, output_hidden_states=True).hidden_states[layer_index].float()
                _, logits = model(hidden)
                llrs = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
                for s, y, a in zip(llrs, y_b, a_b):
                    score = float(s)
                    all_scores.append(score)
                    if y == 1:
                        bon_scores.append(score)
                    else:
                        spf_scores.append(score)
                        atk_scores.setdefault(a, []).append(score)
                done = i + len(batch)
                if done % 5000 < args.batch_size:
                    print(f"    scored {done}/{len(audio)} "
                          f"({done / (time.time() - t1):.1f} utts/s)")

        bon = np.array(bon_scores)
        spf = np.array(spf_scores)
        res = evaluate_system(bon, spf)
        print(f"  pooled EER : {res['eer'] * 100.0:.3f}%")
        print(f"  minDCF     : {res['min_dcf']:.4f}")
        print(f"  CLLR       : {res['cllr']:.4f}")

        per_atk: Dict[str, float] = {}
        for a, scores in sorted(atk_scores.items()):
            s_arr = np.array(scores)
            if len(s_arr) > 0 and len(bon) > 0:
                eer, _ = compute_eer(bon, s_arr)
                per_atk[a] = round(eer * 100.0, 3)
                print(f"    {a:<6}: {per_atk[a]:>7.3f}%")

        results["models"][label] = {  # type: ignore[index]
            "checkpoint": str(ckpt_path),
            "layer_index": ckpt.get("layer_index"),
            "init_weights": ckpt.get("init_weights"),
            "seed": ckpt.get("seed"),
            "augmentation": ckpt.get("augmentation"),
            "dev_eer": ckpt.get("dev_eer"),
            "dev_min_dcf": ckpt.get("dev_min_dcf"),
            "metrics": res,
            "per_attack_eer": per_atk,
        }
        if args.dump_scores is not None:
            dump = args.dump_scores / f"{label}.npz"
            if dump.exists():
                raise SystemExit(f"{dump} exists; refusing to overwrite a score dump")
            args.dump_scores.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dump,
                raw_llr=np.asarray(all_scores, dtype=np.float32),
                label=labels_arr.astype(np.int8),
                attack=np.asarray(attacks),
                row_index=np.arange(len(all_scores), dtype=np.int64),
                manifest_sha256=np.asarray(hashlib.sha256(args.manifest.read_bytes()).hexdigest()),
            )
            print(f"  dumped per-utterance scores -> {dump}")
            results["models"][label]["score_dump"] = str(dump)  # type: ignore[index]

        del model, wavlm
        if device == "cuda":
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
