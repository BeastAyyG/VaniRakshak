"""Resumable downloader for external deepfake-evaluation corpora.

Curl is blocked in this sandbox, so this uses urllib with HTTP Range resume.
Downloads land on A:\\ (252 GB free) -- C:\\ has only ~37 GB.

Usage:
    python scripts/download_call_testset.py --list
    python scripts/download_call_testset.py --name itw
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"A:\cm-corpus\external")

# name -> (url, relative destination filename, what it tests that nothing else does)
SOURCES: dict[str, tuple[str, str, str]] = {
    "itw": (
        "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip",
        "itw/release_in_the_wild.zip",
        "37.9 h of REAL deepfakes scraped from online video (58 public figures). "
        "Not lab-generated -- the only widely used set of genuinely in-the-wild fakes.",
    ),
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length", 0))


def download(name: str) -> int:
    if name not in SOURCES:
        print(f"unknown source {name!r}; known: {', '.join(SOURCES)}")
        return 2

    url, rel, _ = SOURCES[name]
    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    total = remote_size(url)
    print(f"[{name}] {url}")
    print(f"[{name}] -> {dest}  (remote {human(total)})", flush=True)

    have = dest.stat().st_size if dest.exists() else 0
    mode = "ab" if have else "wb"
    headers = {"Range": f"bytes={have}-"} if have else {}

    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, mode) as fh:
        if have:
            print(f"[{name}] resuming from {human(have)}", flush=True)
        got = have
        last = t0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            now = time.time()
            if now - last >= 10:
                last = now
                pct = (got / total * 100) if total else 0
                rate = got / max(now - t0, 1e-9)
                print(
                    f"[{name}] {human(got)} / {human(total)} ({pct:.1f}%) "
                    f"{human(rate)}/s",
                    flush=True,
                )
    print(f"[{name}] done: {human(got)} in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0 if not total or got >= total else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--name", type=str)
    args = ap.parse_args()

    if args.list:
        for k, (url, rel, why) in SOURCES.items():
            print(f"{k}: {rel}\n    {why}\n    {url}\n")
        return 0
    if not args.name:
        ap.error("--name required (or --list)")
    return download(args.name)


if __name__ == "__main__":
    sys.exit(main())
