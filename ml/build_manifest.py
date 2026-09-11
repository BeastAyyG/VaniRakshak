"""Manifest builder for ASVspoof 2019 LA and ASVspoof 5 datasets.

Parses official protocol files and generates canonical 3-column TSV manifests:
path<TAB>label<TAB>attack_id
as specified in docs/08-cm-training-plan.md §3.3.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vanirakshak.build_manifest")


def parse_protocol_and_build_manifest(
    protocol_file: Path,
    audio_dir: Path,
    output_tsv: Path,
    audio_ext: str = ".flac",
    limit: Optional[int] = None,
    only_existing: bool = False,
) -> int:
    """Parse ASVspoof protocol file and generate canonical manifest TSV."""
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    n_bonafide = 0
    n_spoof = 0
    missing = 0

    with open(protocol_file, "r", encoding="utf-8") as in_f, open(
        output_tsv, "w", encoding="utf-8", newline=""
    ) as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["path", "label", "attack_id"])

        for line in in_f:
            parts = line.strip().split()
            if not parts:
                continue

            # ASVspoof 5 format (10 columns):
            # [SPEAKER, FILENAME, GENDER, -, -, -, CODEC, ATTACK_ID, KEY, -]
            # e.g.: T_4850 T_0000000000 F - - - AC3 A05 spoof -
            #       T_3734 T_0000000011 F - - - - bonafide bonafide -
            if len(parts) >= 9 and parts[8].lower() in ("bonafide", "spoof"):
                filename = parts[1]
                key = parts[8].lower()
                attack_id = parts[7] if key == "spoof" else "-"
            # Standard ASVspoof 2019 format (5 columns):
            # [SPEAKER_ID, FILENAME, SYSTEM_ID/ENV, ATTACK_ID, KEY]
            # e.g.: LA_0079 LA_T_1138215 - - bonafide
            #       LA_0079 LA_T_1271820 - A01 spoof
            elif len(parts) >= 5 and parts[-1].lower() in ("bonafide", "spoof"):
                filename = parts[1]
                attack_id = parts[3]
                key = parts[-1].lower()
            elif len(parts) >= 4 and parts[3].lower() in ("bonafide", "spoof"):
                filename = parts[1]
                attack_id = parts[2]
                key = parts[3].lower()
            else:
                # Fallback search for key in line
                key = None
                for p in parts:
                    if p.lower() in ("bonafide", "spoof"):
                        key = p.lower()
                        break
                if not key:
                    continue
                filename = parts[1] if len(parts) > 1 else parts[0]
                attack_id = "-"

            if attack_id.lower() in ("-", "none", "null", "bonafide"):
                attack_id = "-"

            # Resolve audio path
            if not filename.endswith(audio_ext):
                audio_file = audio_dir / f"{filename}{audio_ext}"
            else:
                audio_file = audio_dir / filename

            # Note: We check if file exists or record absolute path
            abs_path = audio_file.resolve()
            if not abs_path.is_file():
                missing += 1
                if only_existing:
                    continue

            # as_posix(), not str(): a manifest built on the Windows training
            # box must stay readable if the corpus is later mounted on Linux
            # (or scored in CI). str() emits backslashes on Windows, which a
            # POSIX reader treats as literal filename characters rather than
            # separators. Forward slashes are accepted by Path() on both.
            writer.writerow([abs_path.as_posix(), key, attack_id])
            count += 1
            if key == "bonafide":
                n_bonafide += 1
            else:
                n_spoof += 1

            if limit and count >= limit:
                break

    logger.info(
        "Manifest written to %s (%d records: %d bonafide, %d spoof, %d files missing on disk)",
        output_tsv,
        count,
        n_bonafide,
        n_spoof,
        missing,
    )
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical TSV manifest for ASVspoof")
    parser.add_argument("--protocol", type=Path, required=True, help="Official protocol text file")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing audio FLAC/WAV files")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest TSV path")
    parser.add_argument("--ext", type=str, default=".flac", help="Audio file extension")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit")
    parser.add_argument(
        "--only-existing",
        action="store_true",
        help="Only include audio files that currently exist on disk (skips missing archives)",
    )
    args = parser.parse_args()

    parse_protocol_and_build_manifest(
        protocol_file=args.protocol,
        audio_dir=args.audio_dir,
        output_tsv=args.output,
        audio_ext=args.ext,
        limit=args.limit,
        only_existing=args.only_existing,
    )


if __name__ == "__main__":
    main()
