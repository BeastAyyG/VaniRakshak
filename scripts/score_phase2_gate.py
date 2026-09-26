"""Phase 2 gate-split scorer (docs/13 §2.1 / docs/14 §6 step 5).

Scorer v2: one WavLM forward per batch feeds every frozen head simultaneously.
Streams audio on-demand — never pre-loads the full gate split (~156 GB float32).

Heads scored (docs/13 §2.4):
  - incumbent:  data/cm_aasist.pt
  - layer5:     data/cm_aasist_layer5.pt
  - S42:        data/cm_docs20_armb_s42.pt
  - S43:        data/cm_docs20_armb_s43.pt
  - B42:        data/cm_g4_control_s42.pt
  - B43:        data/cm_g4_control_s43.pt

All heads share the same WavLM-base-plus frontend; each uses the layer stored in
its checkpoint (layer_index key; defaults to -1 = last layer).

Outputs:
  C:\\vr-out\\p2\\gate_scores.json    — report (never overwrites)
  C:\\vr-out\\p2\\gate\\<label>.npz   — per-utterance score dumps

Gate split is disjointness-asserted against the screen split before scoring.

Usage:
    python scripts/score_phase2_gate.py [--batch-size 32] [--device cuda]

Pre-registered criteria evaluated afterwards by compute_phase2_criteria.py:
  C-2a (complementarity): S42+layer5 fusion vs S42+S43 control on gate split.
  C-2b (noise floor):     seed SD from {S42,S43} and {B42,B43}.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from ml.train_cm import select_wavlm_features  # noqa: E402

GATE_MANIFEST = Path("A:/cm-corpus/manifests/gate_split_608k.tsv")
SCREEN_MANIFEST = Path("A:/cm-corpus/manifests/screen_split_20k.tsv")
OUT_DIR = Path("C:/vr-out/p2")
OUT_REPORT = OUT_DIR / "gate_scores.json"
OUT_DUMP_DIR = OUT_DIR / "gate"

HEADS: List[Tuple[str, str]] = [
    ("incumbent", "data/cm_aasist.pt"),
    ("layer5",    "data/cm_aasist_layer5.pt"),
    ("S42",       "data/cm_docs20_armb_s42.pt"),
    ("S43",       "data/cm_docs20_armb_s43.pt"),
    ("B42",       "data/cm_g4_control_s42.pt"),
    ("B43",       "data/cm_g4_control_s43.pt"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest_paths(path: Path) -> set:
    """Load just the file paths from a TSV manifest (col 0)."""
    paths = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if row:
                paths.add(row[0])
    return paths


def stream_manifest(path: Path, limit: Optional[int] = None):
    """Yield (path, label_int, attack_id) rows from a TSV manifest."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if len(header) < 3:
            raise SystemExit(f"{path}: expected >=3 TSV columns, got {header!r}")
        count = 0
        for row in reader:
            if len(row) < 3:
                continue
            yield row[0], (1 if row[1] == "bonafide" else 0), row[2]
            count += 1
            if limit and count >= limit:
                break


def assert_disjoint(gate: Path, screen: Path) -> None:
    """Abort if gate and screen manifests share any file paths."""
    print("Asserting gate/screen disjointness...", end="", flush=True)
    gate_paths = load_manifest_paths(gate)
    screen_paths = load_manifest_paths(screen)
    overlap = gate_paths & screen_paths
    if overlap:
        raise SystemExit(
            f"FATAL: {len(overlap)} paths appear in both gate and screen splits. "
            "Pre-registered disjointness violated."
        )
    print(f" OK ({len(gate_paths)} gate / {len(screen_paths)} screen, 0 overlap)")


def count_manifest(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row:
                n += 1
    return n


def load_audio(file_path: str) -> Optional[np.ndarray]:
    try:
        x, sr = sf.read(file_path, dtype="float32")
        w = prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False)
        return w
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Utterances per GPU batch (default 32 for gate split memory safety)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only first N rows (dev/debug only)")
    ap.add_argument("--max-missing-frac", type=float, default=0.05)
    ap.add_argument("--gate-manifest", type=Path, default=GATE_MANIFEST)
    ap.add_argument("--screen-manifest", type=Path, default=SCREEN_MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT_REPORT)
    ap.add_argument("--dump-dir", type=Path, default=OUT_DUMP_DIR)
    args = ap.parse_args()

    # Pre-flight checks
    if args.out.exists():
        raise SystemExit(
            f"{args.out} already exists. Ground rule 6: reports are the record. "
            "Choose a new filename or delete the old report explicitly."
        )
    if not args.gate_manifest.exists():
        raise SystemExit(f"Gate manifest not found: {args.gate_manifest}")
    if not args.screen_manifest.exists():
        raise SystemExit(f"Screen manifest not found: {args.screen_manifest}")

    assert_disjoint(args.gate_manifest, args.screen_manifest)

    device = (
        args.device
        if (args.device.startswith("cuda") and torch.cuda.is_available())
        else "cpu"
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU")
    print(f"Device: {device}")

    # Verify all checkpoint files exist before starting
    missing_ckpts = []
    for label, ckpt_path in HEADS:
        p = REPO_ROOT / ckpt_path
        if not p.exists():
            missing_ckpts.append(f"  {label}: {p}")
    if missing_ckpts:
        raise SystemExit("Missing checkpoints:\n" + "\n".join(missing_ckpts))

    # Manifest stats
    print("Counting manifest rows...", end="", flush=True)
    n_total = count_manifest(args.gate_manifest)
    if args.limit:
        n_total = min(n_total, args.limit)
    print(f" {n_total} rows")

    gate_sha = sha256_file(args.gate_manifest)
    print(f"Gate manifest SHA-256: {gate_sha[:16]}...")

    # Dump dir setup
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    for label, _ in HEADS:
        dump_path = args.dump_dir / f"{label}.npz"
        if dump_path.exists():
            raise SystemExit(
                f"{dump_path} exists. Refusing to overwrite a score dump. "
                "Delete it manually to re-run."
            )

    # Load all checkpoints and verify they're frozen
    print("\nLoading checkpoints...")
    loaded: List[Tuple[str, dict, object, object, int, object]] = []  # label, ckpt, wavlm, head, layer_idx, merge_layers

    # All heads share the same WavLM base-plus frontend — load it once
    first_ckpt_path = REPO_ROOT / HEADS[0][1]
    first_ckpt = torch.load(str(first_ckpt_path), map_location="cpu", weights_only=False)
    assert_frozen_checkpoint(first_ckpt, "score_phase2_gate.py")
    print(f"  Loading shared WavLM frontend from {HEADS[0][0]}...")
    shared_wavlm, shared_dtype = load_frontend_for(first_ckpt, device)
    shared_wavlm.eval()

    for label, ckpt_path_str in HEADS:
        ckpt_path = REPO_ROOT / ckpt_path_str
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert_frozen_checkpoint(ckpt, "score_phase2_gate.py")
        merge_layers = ckpt.get("merge_layers") or None  # None means single-layer mode
        in_dim = 768 if ckpt.get("frontend") == "microsoft/wavlm-base-plus" else 1024
        head = AASISTLHead(in_dim=in_dim, merge_layers=merge_layers).to(device)
        head.load_state_dict(ckpt["state_dict"], strict=True)
        head.eval()
        layer_idx = ckpt.get("layer_index")
        layer_idx = -1 if layer_idx is None else int(layer_idx)
        print(f"  {label}: layer={layer_idx}, merge_layers={merge_layers}, dev_eer={ckpt.get('dev_eer', '?')}, seed={ckpt.get('seed', '?')}")
        loaded.append((label, ckpt, shared_wavlm, head, layer_idx, merge_layers))


    # Per-head accumulators
    all_scores: Dict[str, List[float]] = {lbl: [] for lbl, *_ in HEADS}
    all_labels: List[int] = []
    all_attacks: List[str] = []
    read_failures = 0

    print(f"\nStreaming {n_total} utterances from gate split...")
    t0 = time.time()
    batch_paths: List[str] = []
    batch_labels: List[int] = []
    batch_attacks: List[str] = []

    def flush_batch():
        nonlocal read_failures
        if not batch_paths:
            return
        # Load audio for this batch
        audio_batch: List[np.ndarray] = []
        valid_mask: List[bool] = []
        for bp in batch_paths:
            w = load_audio(bp)
            if w is None:
                read_failures += 1
                audio_batch.append(np.zeros(64000, dtype=np.float32))
                valid_mask.append(False)
            else:
                audio_batch.append(w)
                valid_mask.append(True)

        batch_np = np.stack(audio_batch)
        audio_t = torch.from_numpy(batch_np).to(shared_dtype).to(device)

        with torch.no_grad():
            # One WavLM forward — all hidden states available; each head picks its own
            all_hidden = shared_wavlm(audio_t, output_hidden_states=True)

            for i, (label, ckpt, wavlm, head_model, layer_idx, merge_layers) in enumerate(loaded):
                # select_wavlm_features returns a list for merge_layers, tensor otherwise
                feats = select_wavlm_features(all_hidden, layer_idx, merge_layers)
                _, logits = head_model(feats)
                llrs = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
                for s in llrs:
                    all_scores[label].append(float(s))

        all_labels.extend(batch_labels)
        all_attacks.extend(batch_attacks)
        batch_paths.clear()
        batch_labels.clear()
        batch_attacks.clear()

    row_i = 0
    for file_path, label_int, attack_id in stream_manifest(args.gate_manifest, args.limit):
        batch_paths.append(file_path)
        batch_labels.append(label_int)
        batch_attacks.append(attack_id)
        if len(batch_paths) >= args.batch_size:
            flush_batch()
            row_i += args.batch_size
            if row_i % 5000 < args.batch_size:
                elapsed = time.time() - t0
                rate = row_i / elapsed if elapsed > 0 else 0
                eta_s = (n_total - row_i) / rate if rate > 0 else float("inf")
                eta_min = eta_s / 60
                print(f"  {row_i}/{n_total} scored ({rate:.1f} utts/s, ETA {eta_min:.1f} min)")

    flush_batch()  # remainder

    elapsed = time.time() - t0
    print(f"\nScoring complete in {elapsed/60:.1f} min")

    # Check missing fraction
    n_scored = len(all_labels)
    missing_frac = read_failures / max(1, n_scored)
    if missing_frac > args.max_missing_frac:
        raise SystemExit(
            f"Aborting: {missing_frac:.2%} of gate split unreadable. "
            "Check that A: is mounted and manifest paths are correct."
        )
    if read_failures:
        print(f"WARNING: {read_failures} files unreadable ({missing_frac:.2%}), scored as silence")

    labels_arr = np.array(all_labels, dtype=np.int8)
    attacks_arr = np.array(all_attacks)
    bonafide_mask = labels_arr == 1
    n_bon = int(bonafide_mask.sum())
    n_spoof = n_scored - n_bon
    print(f"bonafide={n_bon}  spoof={n_spoof}  total={n_scored}")

    # Build report
    results: dict = {
        "manifest": str(args.gate_manifest),
        "manifest_sha256": gate_sha,
        "n_records": n_scored,
        "n_bonafide": n_bon,
        "n_spoof": n_spoof,
        "unreadable": read_failures,
        "device": device,
        "calibration": "none (raw LLR)",
        "phase": "2",
        "models": {},
    }

    # Compute metrics per head + save score dumps
    print("\nComputing metrics and saving dumps...")
    for label, ckpt_path_str in HEADS:
        ckpt_path = REPO_ROOT / ckpt_path_str
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        scores = np.asarray(all_scores[label], dtype=np.float32)
        bon = scores[bonafide_mask]
        spf = scores[~bonafide_mask]

        res = evaluate_system(bon, spf)
        print(f"  {label}: EER={res['eer']*100:.3f}%  minDCF={res['min_dcf']:.4f}  CLLR={res['cllr']:.4f}")

        # Per-attack EER
        per_atk: Dict[str, float] = {}
        for atk in np.unique(attacks_arr[~bonafide_mask]):
            mask = (attacks_arr == atk) & ~bonafide_mask
            spf_atk = scores[mask]
            if len(spf_atk) > 0 and len(bon) > 0:
                eer, _ = compute_eer(bon, spf_atk)
                per_atk[str(atk)] = round(eer * 100.0, 3)
                print(f"    {atk:<6}: {per_atk[str(atk)]:>7.3f}%")

        # Save dump
        dump_path = args.dump_dir / f"{label}.npz"
        np.savez_compressed(
            dump_path,
            raw_llr=scores,
            label=labels_arr,
            attack=attacks_arr,
            row_index=np.arange(n_scored, dtype=np.int64),
            manifest_sha256=np.asarray(gate_sha),
        )
        print(f"  Dumped -> {dump_path}")

        results["models"][label] = {
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": sha256_file(ckpt_path)[:16],
            "layer_index": ckpt.get("layer_index"),
            "seed": ckpt.get("seed"),
            "g4_rawboost": ckpt.get("g4_rawboost"),
            "codec_augment": ckpt.get("codec_augment"),
            "dev_eer": ckpt.get("dev_eer"),
            "dev_min_dcf": ckpt.get("dev_min_dcf"),
            "metrics": res,
            "per_attack_eer": per_atk,
            "score_dump": str(dump_path),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved report -> {args.out}")
    print("\nNext step: run scripts/compute_phase2_criteria.py to evaluate C-2a and C-2b.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
