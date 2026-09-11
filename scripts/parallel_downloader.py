"""Robust multi-threaded segmented chunk downloader with auto-resume and IPv4 routing.

Bypasses CDN single-stream resets by fetching concurrent 50MB byte-range chunks.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import socket
import sys
import time
import urllib.request

# Force IPv4 to prevent Windows getaddrinfo IPv6 hang/reset
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(*args, **kwargs):
    responses = _orig_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET] or responses


socket.getaddrinfo = _ipv4_getaddrinfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("parallel_downloader")

CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB chunks


def get_remote_file_size(url: str) -> int:
    """Send HEAD request to determine file size and range support."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        length = resp.headers.get("Content-Length")
        if not length:
            raise ValueError(f"Content-Length not provided by {url}")
        return int(length)


def download_chunk(
    url: str,
    output_path: Path,
    chunk_index: int,
    start: int,
    end: int,
    max_retries: int = 10,
) -> bool:
    """Download a single byte range with automatic retry."""
    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "Mozilla/5.0",
    }
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                expected_len = end - start + 1
                if len(data) != expected_len:
                    raise IOError(f"Truncated chunk: got {len(data)}, expected {expected_len}")

                with open(output_path, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return True
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
            else:
                logger.error("Chunk %d failed after %d retries: %s", chunk_index, max_retries, e)
                return False
    return False


def parallel_download(
    url: str,
    output_path: Path,
    concurrency: int = 8,
    chunk_size: int = CHUNK_SIZE,
) -> bool:
    """Download a large file using concurrent range requests with state tracking."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_file = output_path.with_suffix(output_path.suffix + ".chunks.json")

    logger.info("Connecting to %s...", url)
    total_size = get_remote_file_size(url)
    logger.info("Target: %s (Total size: %.2f GB)", output_path.name, total_size / (1024**3))

    # Pre-allocate sparse file if not already existing
    if not output_path.is_file() or output_path.stat().st_size != total_size:
        with open(output_path, "wb") as f:
            f.truncate(total_size)
        if progress_file.is_file():
            progress_file.unlink()

    # Calculate chunks
    chunks = []
    offset = 0
    idx = 0
    while offset < total_size:
        end = min(offset + chunk_size - 1, total_size - 1)
        chunks.append({"idx": idx, "start": offset, "end": end})
        offset = end + 1
        idx += 1

    completed_indices = set()
    if progress_file.is_file():
        try:
            with open(progress_file, "r") as pf:
                completed_indices = set(json.load(pf))
            logger.info("Resuming: %d / %d chunks already completed", len(completed_indices), len(chunks))
        except Exception:
            completed_indices = set()

    remaining_chunks = [c for c in chunks if c["idx"] not in completed_indices]
    if not remaining_chunks:
        logger.info("File already complete!")
        if progress_file.is_file():
            progress_file.unlink()
        return True

    logger.info(
        "Downloading %d remaining chunks using %d threads...", len(remaining_chunks), concurrency
    )
    t0 = time.perf_counter()
    bytes_downloaded = sum(c["end"] - c["start"] + 1 for c in chunks if c["idx"] in completed_indices)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(
                download_chunk,
                url,
                output_path,
                c["idx"],
                c["start"],
                c["end"],
            ): c
            for c in remaining_chunks
        }

        for future in as_completed(future_map):
            chunk_info = future_map[future]
            success = future.result()
            if success:
                completed_indices.add(chunk_info["idx"])
                bytes_downloaded += chunk_info["end"] - chunk_info["start"] + 1
                # Save progress checkpoint
                with open(progress_file, "w") as pf:
                    json.dump(list(completed_indices), pf)

                elapsed = max(0.1, time.perf_counter() - t0)
                speed_mb = (bytes_downloaded / (1024 * 1024)) / elapsed
                percent = (len(completed_indices) / len(chunks)) * 100.0
                logger.info(
                    "Progress: %.1f%% (%d/%d chunks, %.1f MB/s)",
                    percent,
                    len(completed_indices),
                    len(chunks),
                    speed_mb,
                )
            else:
                logger.error("Failed chunk %d", chunk_info["idx"])
                return False

    if progress_file.is_file():
        progress_file.unlink()

    logger.info("Download completed in %.1fs: %s", time.perf_counter() - t0, output_path)
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parallel_downloader.py <URL> <OUTPUT_PATH> [CONCURRENCY]")
        sys.exit(1)
    url_arg = sys.argv[1]
    out_arg = Path(sys.argv[2])
    concurrency_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    ok = parallel_download(url_arg, out_arg, concurrency=concurrency_arg)
    sys.exit(0 if ok else 1)
