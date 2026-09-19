#!/usr/bin/env python
"""Verify the deployed countermeasure is the real model, not the stub.

`vanirakshak/detectors/cm_aasist.py` is an intentional landing zone:
`try_create()` returns ``None`` until weights are pointed at, and the
pipeline then silently falls back to the stub while reporting
``aasist_missing_weights``. A deployment can therefore look healthy and be
running the heuristic. This script is the guard against exactly that.

It checks, in order, and exits non-zero on the first failure:

  1. the resolved weights file exists;
  2. the checkpoint satisfies the backend's own schema (it calls the
     backend's `_validate_checkpoint` rather than re-implementing the rules,
     which would drift);
  3. the constructed detector is NOT the stub (`name !=
     "aasist_missing_weights"`);
  4. the checkpoint carries a Platt calibration block, so scores are
     calibrated rather than raw;
  5. with `--audio`: the detector actually scores an utterance and the
     genuine probability is finite and in [0, 1].

Usage:
    python scripts/verify_cm_deployment.py
    python scripts/verify_cm_deployment.py --audio test_audio/hindi_sample.wav
    python scripts/verify_cm_deployment.py --json

Exit codes: 0 = all checks pass, 1 = a check failed, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "vanirakshak-backend"
sys.path.insert(0, str(BACKEND_ROOT))

STUB_NAME = "aasist_missing_weights"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, default=None,
                    help="also score this wav end to end")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of prose")
    args = ap.parse_args()

    try:
        import numpy as np
        from vanirakshak.detectors import cm_aasist as cm
        from vanirakshak.detectors.pipeline import DetectionPipeline
    except Exception as exc:  # noqa: BLE001 - report, never silently pass
        print(f"FAIL: cannot import the backend ({type(exc).__name__}: {exc})")
        return 2

    results: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})
        if not args.json:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    if not args.json:
        print("Deployed countermeasure check")
        print("-" * 70)

    # 1 -- resolution
    path = cm.weights_path()
    check("weights_resolved", True, str(path))
    exists = path.exists()
    check("weights_exist", exists,
          f"{path.stat().st_size:,} bytes" if exists else "FILE MISSING")
    if not exists:
        if args.json:
            print(json.dumps({"ok": False, "checks": results}, indent=2))
        return 1

    # 2 -- schema, judged by the backend's own rules
    try:
        import torch

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        cm._validate_checkpoint(ckpt, path)
        check("schema_valid", True, "backend _validate_checkpoint accepted it")
    except Exception as exc:  # noqa: BLE001
        check("schema_valid", False, f"{type(exc).__name__}: {exc}")
        if args.json:
            print(json.dumps({"ok": False, "checks": results}, indent=2))
        return 1

    cal = ckpt.get("calibration")
    check("calibration_present", bool(cal),
          json.dumps(cal) if cal else "ABSENT -- scores are raw LLRs")

    # 3 + 4 -- the real detector
    try:
        pipe = DetectionPipeline()
    except Exception as exc:  # noqa: BLE001
        check("detector_built", False, f"{type(exc).__name__}: {exc}")
        if args.json:
            print(json.dumps({"ok": False, "checks": results}, indent=2))
        return 1

    # Checking only "!= aasist_missing_weights" is NOT enough. With
    # VR_CM_BACKEND unset the pipeline builds the numpy/scipy stub, whose
    # name is "synthetic_cm_v1" -- a different string, so that comparison
    # passes while the heuristic is doing the deciding. Test what IS
    # running, not just that one specific sentinel is absent.
    from vanirakshak.detectors import backend_selection as bs

    resolved = bs.resolve_cm_backend()
    check("cm_backend_real", resolved in bs.REAL_CM_BACKENDS,
          f"CM backend resolves to '{resolved}'"
          + ("" if resolved in bs.REAL_CM_BACKENDS else
             " -- set VR_CM_BACKEND=aasist, or put a checkpoint where "
             "weights_path() looks"))

    name = getattr(pipe.cm, "name", "<unknown>")
    is_real = name.startswith(bs.REAL_CM_BACKENDS)  # tuple: ("wavlm", "aasist")
    check("detector_not_stub", is_real,
          f"name={name}" if is_real else
          f"name={name} -- that is a STUB, not the trained countermeasure. "
          f"Scores are coming from the heuristic.")

    # 5 -- optional live scoring
    if args.audio:
        if not args.audio.exists():
            check("live_score", False, f"audio missing: {args.audio}")
        else:
            try:
                import soundfile as sf

                x, sr = sf.read(str(args.audio))
                w = pipe.analyse(np.asarray(x, dtype=np.float32), sr,
                                 claimed_user=None)
                p = float(w.features.get("cm_genuine_prob", float("nan")))
                check("live_score", p == p and 0.0 <= p <= 1.0,
                      f"cm_genuine_prob={p:.6f} on {args.audio.name}")
            except Exception as exc:  # noqa: BLE001
                check("live_score", False, f"{type(exc).__name__}: {exc}")

    ok = all(r["ok"] for r in results)

    if args.json:
        print(json.dumps({
            "ok": ok,
            "weights_path": str(path),
            "detector_name": name,
            "checks": results,
        }, indent=2))
    else:
        print("-" * 70)
        print(f"  {'ALL CHECKS PASS' if ok else 'DEPLOYMENT CHECK FAILED'}")
        if not cal:
            print("  (calibration cannot move EER or minDCF -- it fixes the "
                  "threshold. Run scripts/fit_cm_calibration.py.)")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
