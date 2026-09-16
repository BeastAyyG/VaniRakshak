"""Score fine-tuned (--unfreeze-top-n) checkpoints on the 20k eval slice.

The other harnesses build a stock fp16 WavLM unconditionally, which is correct
for the frozen-encoder checkpoints they were written for and silently wrong for
a fine-tuned one. This one loads the frontend each checkpoint was actually
trained against, via scripts/cm_frontend.py, and refuses if the weights don't
apply.

Baselines are NOT recomputed -- they are read from reports/, which are the
record (docs/10 ground rule 6). Only the new columns are scored.

    python scripts/evaluate_finetune_benchmark.py data/cm_aasist_ft4_s42.pt [more.pt ...]
"""

from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.compute_mindcf import compute_eer, evaluate_system  # noqa: E402
from cm_frontend import load_frontend_for  # noqa: E402
from ml.aasist_model import AASISTLHead  # noqa: E402
from ml.augment import prepare_audio_window  # noqa: E402

EVAL_SLICE = REPO_ROOT / "data" / "eval_benchmark" / "eval_slice_20k.tsv"
BASELINE_REPORT = REPO_ROOT / "reports" / "cm_uniform_ablation_benchmark.json"
OUT_REPORT = REPO_ROOT / "reports" / "cm_finetune_benchmark.json"


def load_records():
    rows = []
    with open(EVAL_SLICE, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for x in r:
            rows.append((x[0], 1 if x[1] == "bonafide" else 0, x[2]))
    return rows


def load_audio(item):
    path, label, atk = item
    try:
        x, sr = sf.read(path, dtype="float32")
        # random_crop=False, matching every other harness -- a random crop here
        # would make the columns incomparable.
        return prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False), label, atk
    except Exception:
        return np.zeros(64000, dtype=np.float32), label, atk


def score_checkpoint(ckpt_path: Path, records, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    merge_layers = ckpt.get("merge_layers") or None
    layer_index = int(ckpt.get("layer_index", -1))

    print(f"\n=== {ckpt_path.name} ===")
    print(f"  unfreeze_top_n={ckpt.get('unfreeze_top_n', 0)} merge_layers={merge_layers} "
          f"seed={ckpt.get('seed')}")

    wavlm, dtype = load_frontend_for(ckpt, device)
    head = AASISTLHead(in_dim=768, merge_layers=merge_layers).to(device)
    head.load_state_dict(ckpt["state_dict"], strict=True)
    head.eval()

    bon, spf, per_attack = [], [], {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i in range(0, len(records), 64):
            batch = records[i:i + 64]
            proc = list(ex.map(load_audio, batch))
            at = torch.from_numpy(np.stack([p[0] for p in proc])).to(device, dtype=dtype)
            with torch.no_grad():
                out = wavlm(at, output_hidden_states=True)
                if merge_layers:
                    feats = [out.hidden_states[j].float() for j in range(merge_layers)]
                else:
                    feats = out.hidden_states[layer_index].float()
                _, logits = head(feats)
                llr = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
            for s, (_, y, atk) in zip(llr, batch):
                if y == 1:
                    bon.append(float(s))
                else:
                    spf.append(float(s))
                    per_attack.setdefault(atk, []).append(float(s))
            if i and (i // 64) % 50 == 0:
                print(f"  {i}/{len(records)} ({i / (time.time() - t0):.0f} utts/s)")
                sys.stdout.flush()

    bon_a = np.array(bon)
    metrics = evaluate_system(bon_a, np.array(spf))
    atk_eer = {
        k: round(float(compute_eer(bon_a, np.array(v))[0]) * 100.0, 3)
        for k, v in sorted(per_attack.items())
    }
    return {
        "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "unfreeze_top_n": int(ckpt.get("unfreeze_top_n", 0) or 0),
        "encoder_lr": ckpt.get("encoder_lr"),
        "frontend_layer_indices": ckpt.get("frontend_layer_indices"),
        "merge_layers": merge_layers,
        "seed": ckpt.get("seed"),
        "dev_eer": ckpt.get("dev_eer"),
        "dev_min_dcf": ckpt.get("dev_min_dcf"),
        "metrics": metrics,
        "per_attack_eer": atk_eer,
    }, len(bon), len(spf)


def main():
    paths = [Path(p).resolve() for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        sys.exit(2)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        sys.exit(f"checkpoint(s) not found: {missing}")

    records = load_records()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Scoring {len(paths)} checkpoint(s) on {len(records)} records ({device})")

    models, n_bon, n_spf = {}, 0, 0
    for p in paths:
        entry, n_bon, n_spf = score_checkpoint(p, records, device)
        models[p.stem] = entry
        m = entry["metrics"]
        print(f"  EER={m['eer'] * 100:.3f}%  minDCF={m['min_dcf']:.4f}  "
              f"actDCF={m['act_dcf']:.4f}  CLLR={m['cllr']:.4f}")

    # Baselines are the record -- read, never recomputed, never rewritten.
    baselines = {}
    if BASELINE_REPORT.is_file():
        baselines = json.load(open(BASELINE_REPORT, encoding="utf-8"))["models"]

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "evaluation_partition": "data/eval_benchmark/eval_slice_20k.tsv",
            "total_utterances": len(records),
            "bonafide_count": n_bon,
            "spoof_count": n_spf,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "description": "G3 encoder fine-tuning vs the frozen-encoder baselines",
            "baselines_copied_from": BASELINE_REPORT.name,
            "models": {**baselines, **models},
        },
        open(OUT_REPORT, "w", encoding="utf-8"),
        indent=2,
    )
    print(f"\nWrote {OUT_REPORT.relative_to(REPO_ROOT)}")

    print("\n--- G3 criteria (docs/10) ---")
    for name, e in models.items():
        m = e["metrics"]
        print(f"{name}: EER {m['eer'] * 100:.3f}% (bar <11.0, "
              f"{'PASS' if m['eer'] < 0.11 else 'MISS'})  "
              f"minDCF {m['min_dcf']:.4f} (bar <0.75, "
              f"{'PASS' if m['min_dcf'] < 0.75 else 'MISS'})")
    print("Best frozen baseline for reference: EER 12.666%, minDCF 0.8731.")
    print("Noise floor is 0.8 pp pooled EER -- a difference under that is not a result.")


if __name__ == "__main__":
    main()
