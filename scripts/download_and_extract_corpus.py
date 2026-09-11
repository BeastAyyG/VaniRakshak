"""Automated ASVspoof 5 corpus downloader and extractor.

Manages sequential download, extraction, and disk cleanup of ASVspoof 5
partitions (train and dev) on Drive A to stay within disk budget constraints.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.parallel_downloader import get_remote_file_size, parallel_download
except ImportError:
    from parallel_downloader import get_remote_file_size, parallel_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("corpus_loader")

HF_BASE_URL = "https://huggingface.co/datasets/jungjee/asvspoof5/resolve/main"

DEV_ARCHIVES = [
    "flac_D_aa.tar",
    "flac_D_ab.tar",
    "flac_D_ac.tar",
]

TRAIN_ARCHIVES = [
    "flac_T_aa.tar",
    "flac_T_ab.tar",
    "flac_T_ac.tar",
    "flac_T_ad.tar",
    "flac_T_ae.tar",
]

ARCHIVE_MD5 = {
    "flac_D_aa.tar": "df0be44957623991028cce59792beb17",
    "flac_D_ab.tar": "1e8cd685d89b64502692f1bcf1a13db3",
    "flac_D_ac.tar": "5e0031f08c30e4bdbf0c59f91b2d662b",
    "flac_T_aa.tar": "b0cc86b14826a7701b52aad4f53daf9c",
    "flac_T_ab.tar": "d05be3f4be7a343fdbdd0ed29fdff2e1",
    "flac_T_ac.tar": "70d1ba4ad75a20aef3dae541fbd321e3",
    "flac_T_ad.tar": "c9bb56af2cc410d98338d74babcb95c9",
    "flac_T_ae.tar": "a3969816982f3e52300d6147ad796df7",
}


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    """Compute and verify MD5 hash of an archive file."""
    logger.info("Computing MD5 for %s...", file_path.name)
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != expected_md5:
        logger.warning("MD5 mismatch for %s: expected %s, got %s", file_path.name, expected_md5, actual)
        return False
    logger.info("MD5 verified OK for %s", file_path.name)
    return True


def extract_tar(tar_path: Path, dest_dir: Path) -> bool:
    """Extract tar archive using system bsdtar/tar."""
    logger.info("Extracting %s into %s...", tar_path.name, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cmd = ["tar", "-xf", str(tar_path), "-C", str(dest_dir)]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        logger.error("Failed to extract %s (exit code %d)", tar_path.name, res.returncode)
        return False
    elapsed = time.perf_counter() - t0
    logger.info("Extracted %s in %.1fs", tar_path.name, elapsed)
    return True


def process_archive(
    archive_name: str,
    target_dir: Path,
    concurrency: int = 8,
    verify_checksum: bool = False,
    keep_tar: bool = False,
) -> bool:
    """Download, extract, and optionally delete an individual archive."""
    marker_file = target_dir / f".{archive_name}.done"
    if marker_file.is_file():
        logger.info("Archive %s already processed and extracted (marker exists). Skipping.", archive_name)
        return True

    url = f"{HF_BASE_URL}/{archive_name}"
    tar_path = target_dir / archive_name

    # Download if needed
    needs_download = False
    if not tar_path.is_file() or (tar_path.with_suffix(tar_path.suffix + ".chunks.json").is_file()):
        needs_download = True
    else:
        try:
            expected_size = get_remote_file_size(url)
            if tar_path.stat().st_size != expected_size:
                logger.warning(
                    "Existing %s size mismatch (%d != %d). Re-downloading.",
                    archive_name,
                    tar_path.stat().st_size,
                    expected_size,
                )
                needs_download = True
        except Exception as e:
            logger.warning("Could not check remote size for %s: %s", archive_name, e)

    if needs_download:
        logger.info("Starting download of %s...", archive_name)
        ok = parallel_download(url, tar_path, concurrency=concurrency)
        if not ok:
            logger.error("Download failed for %s", archive_name)
            return False

    # Optional checksum
    if verify_checksum:
        expected = ARCHIVE_MD5.get(archive_name)
        if expected and not verify_md5(tar_path, expected):
            logger.error("Checksum verification failed for %s", archive_name)
            return False

    # Extract
    if not extract_tar(tar_path, target_dir):
        return False

    # Mark completion
    marker_file.write_text(f"extracted {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Reclaim disk space
    if not keep_tar:
        try:
            tar_path.unlink()
            logger.info("Removed archive %s to reclaim disk space.", archive_name)
        except OSError as e:
            logger.warning("Could not delete %s: %s", tar_path, e)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="ASVspoof 5 corpus automated manager")
    parser.add_argument(
        "--partition",
        choices=["dev", "train", "all"],
        default="dev",
        help="Partition to download and extract (default: dev)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(r"A:\cm-corpus\asvspoof5"),
        help="Base corpus directory on Drive A",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel chunk download workers",
    )
    parser.add_argument(
        "--verify-md5",
        action="store_true",
        help="Verify MD5 checksum before extraction",
    )
    parser.add_argument(
        "--keep-tar",
        action="store_true",
        help="Keep tar files after extraction",
    )
    args = parser.parse_args()

    archives_to_run = []
    if args.partition in ("dev", "all"):
        archives_to_run.extend(DEV_ARCHIVES)
    if args.partition in ("train", "all"):
        archives_to_run.extend(TRAIN_ARCHIVES)

    logger.info("Target directory: %s", args.corpus_dir)
    logger.info("Processing %d archives for partition '%s'...", len(archives_to_run), args.partition)

    for arch in archives_to_run:
        success = process_archive(
            arch,
            args.corpus_dir,
            concurrency=args.concurrency,
            verify_checksum=args.verify_md5,
            keep_tar=args.keep_tar,
        )
        if not success:
            logger.error("Pipeline failed on %s", arch)
            return 1

    logger.info("All requested archives processed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
