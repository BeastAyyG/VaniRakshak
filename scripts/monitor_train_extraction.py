"""Monitor Train WavLM feature extraction process, log throughput/ETA/GPU, and verify completion."""

from __future__ import annotations

import csv
import os
import pathlib
import subprocess
import sys
import time
import numpy as np
import psutil

TRAIN_MANIFEST = pathlib.Path(r"A:\cm-corpus\manifests\train.tsv")
FEATURE_DIR = pathlib.Path(r"C:\Users\singu\cm-features\train")
TENSORS_DIR = FEATURE_DIR / "tensors"
OUT_MANIFEST = FEATURE_DIR / "train_features.tsv"
TOTAL_RECORDS = 182357
EXPECTED_TENSOR_SIZE = 305792
EXPECTED_SHAPE = (199, 768)


def get_feature_count() -> int:
    if not OUT_MANIFEST.exists():
        return 0
    with open(OUT_MANIFEST, "r", encoding="utf-8", errors="ignore") as f:
        lines = sum(1 for _ in f)
        return max(0, lines - 1)


def get_gpu_metrics() -> str:
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 6:
                return (
                    f"GPU Util: {parts[0]}% | Mem Util: {parts[1]}% | "
                    f"VRAM: {parts[2]}/{parts[3]} MiB | Temp: {parts[4]}C | Power: {parts[5]}W"
                )
    except Exception as e:
        return f"GPU query error: {e}"
    return "GPU metrics unavailable"


def find_extraction_process(target_pid: int = 16528) -> psutil.Process | None:
    if psutil.pid_exists(target_pid):
        try:
            p = psutil.Process(target_pid)
            if "python" in p.name().lower():
                return p
        except Exception:
            pass
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if "extract_cm_features.py" in cmd and "train.tsv" in cmd:
                return p
        except Exception:
            pass
    return None


def verify_tensors() -> bool:
    print("\n" + "=" * 70)
    print("STARTING POST-EXTRACTION VERIFICATION OF TRAIN TENSORS")
    print("=" * 70)

    print(f"Scanning directory: {TENSORS_DIR}")
    npy_files = list(TENSORS_DIR.rglob("*.npy"))
    total_npy = len(npy_files)
    print(f"Total .npy files found on disk: {total_npy} / {TOTAL_RECORDS}")

    manifest_rows = 0
    if OUT_MANIFEST.exists():
        with open(OUT_MANIFEST, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            manifest_rows = sum(1 for _ in reader)
        print(f"Header in {OUT_MANIFEST.name}: {header}")
        print(f"Records in {OUT_MANIFEST.name}: {manifest_rows} / {TOTAL_RECORDS}")

    print("Verifying file sizes (target: 305,792 bytes)...")
    bad_sizes = [f for f in npy_files if f.stat().st_size != EXPECTED_TENSOR_SIZE]
    if bad_sizes:
        print(f"ERROR: Found {len(bad_sizes)} files with unexpected size!")
        for b in bad_sizes[:5]:
            print(f"  {b.name}: {b.stat().st_size} bytes")
    else:
        print(f"SUCCESS: All {total_npy} files have exact size {EXPECTED_TENSOR_SIZE} bytes.")

    print("Checking sample tensor contents across shards...")
    sample_indices = [
        0,
        total_npy // 4,
        total_npy // 2,
        3 * total_npy // 4,
        total_npy - 1,
    ]
    all_valid = True
    for idx in sample_indices:
        if 0 <= idx < total_npy:
            f = npy_files[idx]
            try:
                arr = np.load(f)
                valid_shape = arr.shape == EXPECTED_SHAPE
                valid_dtype = arr.dtype == np.float16
                is_finite = bool(np.isfinite(arr).all())
                print(
                    f"Sample [{idx}] {f.name}: shape={arr.shape} (ok={valid_shape}), "
                    f"dtype={arr.dtype} (ok={valid_dtype}), finite={is_finite}"
                )
                if not (valid_shape and valid_dtype and is_finite):
                    all_valid = False
            except Exception as e:
                print(f"ERROR loading {f.name}: {e}")
                all_valid = False

    success = (
        total_npy == TOTAL_RECORDS
        and manifest_rows == TOTAL_RECORDS
        and len(bad_sizes) == 0
        and all_valid
    )
    print("=" * 70)
    print(f"FINAL VERIFICATION RESULT: {'PASSED' if success else 'FAILED'}")
    print("=" * 70)
    return success


def main() -> None:
    print(f"Starting Train Extraction Monitor at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target utterances: {TOTAL_RECORDS}")

    proc = find_extraction_process()
    if proc is None:
        print("Warning: Train extraction process not found immediately. Checking if already done...")
        count = get_feature_count()
        if count >= TOTAL_RECORDS:
            print("Feature count matches target. Running verification...")
            if verify_tensors():
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            print(f"Process not running and count is {count}/{TOTAL_RECORDS}. Exiting with failure.")
            sys.exit(1)

    pid = proc.pid
    print(f"Attached to Train Extraction Process PID: {pid}")

    prev_count = get_feature_count()
    prev_time = time.time()
    start_time = prev_time

    while True:
        time.sleep(30)
        curr_time = time.time()
        curr_count = get_feature_count()

        interval = curr_time - prev_time
        interval_utts = curr_count - prev_count
        instant_rate = interval_utts / max(interval, 1e-6)

        total_elapsed = curr_time - start_time
        total_rate = curr_count / max(total_elapsed, 1e-6)

        pct = (curr_count / TOTAL_RECORDS) * 100
        remaining = TOTAL_RECORDS - curr_count
        effective_rate = instant_rate if instant_rate > 10 else total_rate
        eta_s = remaining / max(effective_rate, 1e-6)
        eta_m = eta_s / 60

        gpu_info = get_gpu_metrics()

        print(
            f"[{time.strftime('%H:%M:%S')}] {curr_count:,}/{TOTAL_RECORDS:,} ({pct:5.2f}%) | "
            f"Rate: {instant_rate:5.1f} utt/s (avg {total_rate:5.1f}) | "
            f"ETA: {eta_m:4.1f}m | {gpu_info}"
        )
        sys.stdout.flush()

        prev_count = curr_count
        prev_time = curr_time

        if not psutil.pid_exists(pid):
            print(f"\nProcess {pid} has exited.")
            time.sleep(5)
            break

    if verify_tensors():
        print("Train feature extraction and verification successfully completed.")
        sys.exit(0)
    else:
        print("Train feature extraction verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
