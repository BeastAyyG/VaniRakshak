"""VaniRakshak — ASV & Anti-Spoofing Dataset Split Generator.

Partitions enrolled target audio and recorded speech into standardized
train and test / evaluation splits following forensic biometric conventions:
  - Manifest CSV Mode: Reads manifest.csv (filepath, speaker_id) and produces
    stratified train.csv and test.csv in data/splits/
  - Protocol Directory Mode: Partitions WAV audio into train/dev/eval JSON splits
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "splits"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vanirakshak.make_splits")


def resolve_existing_path(path_str: str) -> Path:
    """Resolve a path relative to CWD, script directory, data dir, or repo root."""
    p = Path(path_str)
    if p.exists():
        return p.resolve()

    script_dir = Path(__file__).resolve().parent
    data_dir = REPO_ROOT / "data"
    search_dirs = [Path.cwd(), script_dir, data_dir, REPO_ROOT]

    for base in search_dirs:
        candidate = (base / path_str).resolve()
        if candidate.exists():
            return candidate

    # Return standard resolved path even if pending creation
    return (Path.cwd() / path_str).resolve()


def resolve_output_dir(path_str: Optional[str]) -> Path:
    """Resolve destination directory safely."""
    if not path_str:
        return DEFAULT_OUT_DIR
    p = Path(path_str)
    if p.is_absolute():
        return p
    # Try relative to CWD first
    candidate = (Path.cwd() / p).resolve()
    return candidate


_CHUNK_RE = re.compile(r"[_-](?:part|seg|segment|chunk|clip)[_-]?\d+$", re.IGNORECASE)
_CHUNK_INDEX_RE = re.compile(r"(?:part|seg|segment|chunk|clip)[_-]?(\d+)$", re.IGNORECASE)


def session_id_for(filepath: str) -> str:
    """Return the source-recording id a clip was cut from.

    Chunked exports such as ``<source>_part03.wav`` and ``<source>_part04.wav``
    come from the same recording session, so the chunk suffix is collapsed. This
    lets the splitter keep a whole session on one side of the train/test boundary.
    """
    stem = Path(str(filepath).replace("\\", "/")).stem
    return _CHUNK_RE.sub("", stem).lower() or stem.lower()


def chunk_sort_key(row: Dict[str, str]) -> Tuple[str, int, str]:
    """Order clips by source recording, then chunk index, then name."""
    fp = str(row.get("filepath") or row.get("path") or "")
    stem = Path(fp.replace("\\", "/")).stem
    m = _CHUNK_INDEX_RE.search(stem)
    return (session_id_for(fp), int(m.group(1)) if m else -1, stem.lower())


def split_manifest_csv(
    manifest_path: Path,
    out_dir: Path,
    train_ratio: float = 0.80,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[Path, Path]:
    """Split a manifest CSV (filepath, speaker_id) into train.csv and test.csv."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []

    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            fieldnames = [fn.strip() for fn in reader.fieldnames]
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items() if k is not None}
            if cleaned and any(cleaned.values()):
                rows.append(cleaned)

    if not rows:
        raise ValueError(f"Manifest CSV is empty: {manifest_path}")

    # Detect speaker_id and filepath columns
    speaker_key = next((k for k in fieldnames if k.lower() in ("speaker_id", "speaker", "spk_id", "label")), None)
    if not speaker_key:
        speaker_key = fieldnames[1] if len(fieldnames) > 1 else fieldnames[0]

    filepath_key = next((k for k in fieldnames if k.lower() in ("filepath", "file_path", "path", "file", "audio")), None)
    if not filepath_key:
        filepath_key = fieldnames[0]

    logger.info(
        f"Loaded {len(rows)} entries from {manifest_path} "
        f"(filepath col: '{filepath_key}', speaker col: '{speaker_key}')"
    )

    train_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    if stratify and len(rows) > 1:
        # Group by speaker, then by recording session, so that clips cut from the
        # same source never land on both sides of the train/test boundary.
        by_speaker: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
        for r in rows:
            spk = r.get(speaker_key, "unknown")
            sess = session_id_for(str(r.get(filepath_key, "")))
            by_speaker.setdefault(spk, {}).setdefault(sess, []).append(r)

        for spk, sessions in by_speaker.items():
            spk_items = [r for sess in sessions.values() for r in sess]
            n_items = len(spk_items)
            if n_items == 1:
                # Single sample: put in train unless train_rows is already much larger
                train_rows.append(spk_items[0])
                continue

            n_train = max(1, min(n_items - 1, int(round(n_items * train_ratio))))

            if len(sessions) > 1:
                # Hold out whole recording sessions. The last session in the shuffled
                # order is reserved for training so the split can never come out empty.
                sess_ids = sorted(sessions)
                random.shuffle(sess_ids)
                held: List[str] = []
                n_held = 0
                for sid in sess_ids[:-1]:
                    if n_held >= n_items - n_train:
                        break
                    held.append(sid)
                    n_held += len(sessions[sid])
                if not held:
                    held = [sess_ids[0]]
                spk_train = [r for sid in sess_ids if sid not in held for r in sessions[sid]]
                spk_test = [r for sid in held for r in sessions[sid]]
            else:
                # One recording session for this speaker: a session-disjoint split is
                # impossible. Hold out a contiguous block instead so test clips are not
                # interleaved with their immediate neighbours in the training set.
                logger.warning(
                    "Speaker '%s' has only one recording session (%d clips): a "
                    "session-disjoint split is impossible. Holding out a contiguous "
                    "block instead - metrics involving this speaker remain optimistic.",
                    spk,
                    n_items,
                )
                ordered = sorted(spk_items, key=chunk_sort_key)
                spk_train = ordered[:n_train]
                spk_test = ordered[n_train:]

            train_rows.extend(spk_train)
            test_rows.extend(spk_test)
    else:
        shuffled = list(rows)
        random.shuffle(shuffled)
        n_train = max(1, min(len(rows) - 1, int(round(len(rows) * train_ratio))))
        train_rows = shuffled[:n_train]
        test_rows = shuffled[n_train:]

    # Fallback to ensure test set is not empty if total rows >= 2
    if len(rows) >= 2 and not test_rows:
        test_rows.append(train_rows.pop())

    # Shuffle both splits with seed
    random.shuffle(train_rows)
    random.shuffle(test_rows)

    train_csv_path = out_dir / "train.csv"
    test_csv_path = out_dir / "test.csv"

    # Write CSV splits
    for path, data_subset in [(train_csv_path, train_rows), (test_csv_path, test_rows)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_subset)

    # Also emit JSON splits for downstream compatibility
    train_json_path = out_dir / "train.json"
    eval_json_path = out_dir / "eval.json"
    train_json_path.write_text(
        json.dumps({"split": "train", "count": len(train_rows), "files": train_rows}, indent=2),
        encoding="utf-8",
    )
    eval_json_path.write_text(
        json.dumps({"split": "test", "count": len(test_rows), "files": test_rows}, indent=2),
        encoding="utf-8",
    )

    logger.info(f"Split completed: Train={len(train_rows)} samples -> {train_csv_path}")
    logger.info(f"Split completed: Test={len(test_rows)} samples -> {test_csv_path}")

    return train_csv_path, test_csv_path


def create_standard_splits(
    data_dir: Path,
    out_dir: Path,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Scan audio directory and generate reproducible train, dev, and eval JSON splits."""
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(data_dir.rglob("*.wav")))
    if not wav_files:
        logger.warning(f"No WAV files found in {data_dir}. Generating synthetic split protocol.")
        records = [
            {"id": f"sample_{i:04d}", "label": "genuine" if i % 2 == 0 else "spoof", "codec": "pcm"}
            for i in range(100)
        ]
    else:
        records = []
        for p in wav_files:
            name = p.stem.lower()
            label = "genuine" if "genuine" in name else "spoof"
            records.append({
                "id": p.name,
                "path": str(p.relative_to(REPO_ROOT)),
                "label": label,
                "is_clone": "clone" in name or "spoof" in name,
                "is_spliced": "splice" in name,
            })

    random.shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train_set = records[:n_train]
    dev_set = records[n_train : n_train + n_dev]
    eval_set = records[n_train + n_dev :]

    for split_name, split_data in [("train", train_set), ("dev", dev_set), ("eval", eval_set)]:
        out_file = out_dir / f"{split_name}.json"
        manifest = {
            "split": split_name,
            "count": len(split_data),
            "files": split_data,
        }
        out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Wrote {split_name} split ({len(split_data)} records) to {out_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VaniRakshak — Create Reproducible Train/Test Dataset Splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest CSV file (e.g., ../manifest.csv or data/manifest.csv)",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        "--out_dir",
        "--out-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Destination directory for splits (default: data/splits or ../splits)",
    )
    parser.add_argument(
        "--train_ratio",
        "--train-ratio",
        dest="train_ratio",
        type=float,
        default=0.80,
        help="Ratio of data allocated to training split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible partitioning",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Legacy mode: Directory containing WAV files to scan",
    )

    args = parser.parse_args()

    if args.manifest:
        manifest_path = resolve_existing_path(args.manifest)
        out_dir = resolve_output_dir(args.output_dir)

        print("\n" + "=" * 65)
        print("  [VANIRAKSHAK DATASET SPLIT GENERATOR]")
        print("=" * 65)
        print(f" Manifest Source : {manifest_path}")
        print(f" Output Directory: {out_dir}")
        print(f" Train Ratio     : {args.train_ratio:.0%}")
        print(f" Random Seed     : {args.seed}")
        print("-" * 65)

        train_p, test_p = split_manifest_csv(
            manifest_path=manifest_path,
            out_dir=out_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )

        print("-" * 65)
        print(" Split successfully generated:")
        print(f"  * Train split : {train_p}")
        print(f"  * Test split  : {test_p}")
        print("=" * 65 + "\n")
        return 0

    # Legacy Directory Scan Mode
    data_dir_str = args.data_dir or "data/demo"
    data_dir = resolve_existing_path(data_dir_str)
    out_dir = resolve_output_dir(args.output_dir)

    logger.info(f"Scanning audio directory: {data_dir}")
    create_standard_splits(data_dir, out_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
