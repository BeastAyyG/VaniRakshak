"""VaniRakshak — Unified Forensic Benchmark Orchestrator.

Runs the complete benchmark battery:
  1. ASVspoof 5 Metric Suite (minDCF, actDCF, CLLR, EER)
  2. End-to-End Single-Thread CPU Latency & SLA Verification (p50, p95, p99)
  3. Telephony Codec & Network Channel Robustness
  4. Standardized Demo Audio Ground-Truth Validation

Execute via:
    python benchmarks/run_benchmarks.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repository root is on sys.path.
#
# ORDER MATTERS. There are two importable packages named `benchmarks`:
#   benchmarks/                        <- the live one (docs point here)
#   vanirakshak-backend/benchmarks/    <- a dead byte-identical duplicate
# Whichever parent directory lands at sys.path[0] wins, so inserting
# BACKEND_DIR last silently made the benchmark import the DEAD COPY -- edits to
# benchmarks/compute_mindcf.py had no effect at all. Insert BACKEND_DIR first
# and REPO_ROOT second so REPO_ROOT ends up at sys.path[0] and `benchmarks`
# unambiguously resolves to the live module.
#
# (The dead vanirakshak-backend/benchmarks/ duplicate has since been deleted --
# 0 importers. Only the live benchmarks/ module remains.)
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "vanirakshak-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compute_mindcf import evaluate_system
from vanirakshak.cli.bench import run_bench
from vanirakshak.data.augment import (
    alaw_encode_decode,
    ulaw_encode_decode,
    amr_wb_simulate,
    gsm_efr_simulate,
    add_noise,
)
from vanirakshak.data.synthetic import synthetic_speech, synthetic_spoofed
from vanirakshak.detectors.pipeline import DetectionPipeline
from vanirakshak.session import CallSession, SessionContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vanirakshak.benchmarks")

# ---------------------------------------------------------------------------
# Ground-truth verdict -> the risk tier the system is REQUIRED to reach.
#
# Tier ordering and cutoffs come from vanirakshak/config.py:
#   ALLOW < 35 <= CHALLENGE < 70 <= BLOCK
#
# POLICY LIVES IN THE DATA, NOT HERE. Each demo file declares its `expected_tier`
# in data/demo/ground_truth.json; the benchmark reads that field first and falls
# back to EXPECTED_TIER only if it is absent. Edit the JSON to change policy --
# this dict is just the fallback so the benchmark still runs on older data.
#
# AUTHENTIC_TARGET is held to an exact ALLOW: challenging a genuine caller is a
# false positive that blocks a legitimate transaction, so "at least ALLOW" would
# be meaningless (ALLOW is the floor -- everything trivially satisfies it).
#
# The two CONFIRMED-fraud classes require BLOCK, not merely CHALLENGE. This is
# the important part: the system currently emits CHALLENGE for every input, so a
# "minimum tier of CHALLENGE" policy would let it pass 3 of 4 by accident. Per
# the README, BLOCK is the tier that "engages the transaction interlock", which
# is the correct response to a confirmed clone or a confirmed impostor.
# SUSPICIOUS_SPLICED_AUDIO is held to CHALLENGE because it is suspicious rather
# than confirmed.
#
# OWNER DECISION: requiring BLOCK (not CHALLENGE) for clone/impostor is a
# deliberate call -- reasonable people could argue CHALLENGE first, then block.
# It is encoded in data/demo/ground_truth.json#expected_tier; change it there.
# ---------------------------------------------------------------------------
TIER_ORDER = {"ALLOW": 0, "CHALLENGE": 1, "BLOCK": 2}
EXPECTED_TIER = {
    "AUTHENTIC_TARGET": "ALLOW",
    "SYNTHETIC_VOICE_CLONE": "BLOCK",
    "SUSPICIOUS_SPLICED_AUDIO": "CHALLENGE",
    "IMPOSTOR_SPEAKER": "BLOCK",
}

# docs/04-metrics-and-honest-evaluation.md: "An uninformative system scores
# minDCF = 1.0." Anything at or above that carries no information, so a codec
# condition that degrades the detector to that point is a FAIL, not a PASS.
MINDCF_UNINFORMATIVE = 1.0


def verdict_matches(tier: str, required_tier: str, is_authentic: bool = False) -> bool:
    """True when the emitted tier satisfies the required (policy) tier.

    `required_tier` is ALLOW/CHALLENGE/BLOCK, sourced from
    data/demo/ground_truth.json#expected_tier (with EXPECTED_TIER as fallback).
    `is_authentic=True` enforces an exact ALLOW: a genuine caller must never be
    challenged, since that is a false positive that blocks a real transaction.
    An unknown required tier cannot be asserted, so it fails.
    """
    if required_tier not in TIER_ORDER:
        return False
    if is_authentic:
        return tier == "ALLOW"
    return TIER_ORDER.get(tier, -1) >= TIER_ORDER[required_tier]


def run_all_benchmarks() -> dict:
    t_global_start = time.perf_counter()
    sr = 16000
    pipe = DetectionPipeline()
    rng = np.random.default_rng(42)

    print("\n" + "=" * 72)
    print("           VANIRAKSHAK UNIFIED FORENSIC BENCHMARK SUITE")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Benchmark 1: ASVspoof 5 Challenge Metrics
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK 1/4] Evaluating ASVspoof 5 Metric Suite (100 pairs)...")
    bon_scores, spf_scores = [], []
    for _ in range(100):
        bon = synthetic_speech(4.0, sr=sr, rng=rng).astype(np.float32) / 32768.0
        spf = synthetic_spoofed(4.0, sr=sr, rng=rng).astype(np.float32) / 32768.0
        bon_scores.append(pipe.analyse(bon, sr).cm_score)
        spf_scores.append(pipe.analyse(spf, sr).cm_score)

    m1 = evaluate_system(np.array(bon_scores), np.array(spf_scores))
    print("  --------------------------------------------------------")
    print(f"  * Equal Error Rate (EER)          : {m1['EER']:.2f}%")
    print(f"  * Minimum Detection Cost (minDCF) : {m1['minDCF']:.4f}")
    print(f"  * Actual Detection Cost (actDCF)  : {m1['actDCF']:.4f}"
          f"   @ tau={m1['threshold_actdcf']:.3f} ({m1['threshold_source']})")
    print(f"  * actDCF at tau_Bayes             : {m1['actDCF_bayes']:.4f}"
          f"   @ tau={m1['threshold_bayes']:.3f}")
    print(f"  * Log-Likelihood Ratio Cost (CLLR): {m1['CLLR']:.4f}")
    print(f"  * Score calibration               : "
          f"{'OK' if m1['calibrated'] else 'MISCALIBRATED'}")
    print("  --------------------------------------------------------")
    if not m1["calibrated"]:
        print("  ! Scores are not calibrated LLRs. EER/minDCF measure separation only;")
        print("    actDCF at the Bayes threshold is the number that reflects deployment.")
        print("    See docs/04-metrics-and-honest-evaluation.md section 1.2.")
    if m1["CLLR"] > 1.0:
        print(f"  ! CLLR {m1['CLLR']:.4f} > 1.0 -- per docs/04 section 1.3 this is")
        print("    \"misleading calibration (confident false information)\".")

    # -------------------------------------------------------------------------
    # Benchmark 2: CPU Streaming Latency SLA
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK 2/4] Measuring End-to-End Latency Profile (200 windows)...")
    sess = CallSession("bench-latency", SessionContext(caller_id="+91-BENCH", claimed_identity="bench_anchor"))
    sess.enrol("bench_anchor", synthetic_speech(4.0, sr=sr), sr)
    hop = int(sr * 1.0)
    latencies = []
    # Warmup
    for _ in range(5):
        sess.push_chunk(np.zeros(int(sr * 0.5), dtype=np.int16))

    for _ in range(200):
        chunk = np.random.randint(-2000, 2000, size=hop, dtype=np.int16)
        t0 = time.perf_counter()
        sess.push_chunk(chunk)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    mean_lat = float(lat_arr.mean())
    fps = 1000.0 / mean_lat

    print("  --------------------------------------------------------")
    print(f"  * Median Latency (p50)            : {p50:.2f} ms")
    print(f"  * 95th Percentile Latency (p95)   : {p95:.2f} ms  (SLA < 200 ms: {'COMPLIANT' if p95 < 200 else 'VIOLATED'})")
    print(f"  * 99th Percentile Latency (p99)   : {p99:.2f} ms")
    print(f"  * Processing Throughput           : {fps:.1f} hops/sec (~{fps:.0f}x real-time)")
    print("  --------------------------------------------------------")

    # -------------------------------------------------------------------------
    # Benchmark 3: Telephony Robustness
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK 3/4] Testing Telephony Codec & Network Channel Robustness...")
    codecs = {
        "Clean (Broadband)": lambda x: x,
        "G.711 A-law": alaw_encode_decode,
        "G.711 mu-law": ulaw_encode_decode,
        "AMR-WB (VoLTE)": lambda x: amr_wb_simulate(x, sr),
        "GSM-EFR (2G)": lambda x: gsm_efr_simulate(x, sr),
        "Noise (15dB SNR)": lambda x: add_noise(x, snr_db=15.0),
    }

    m3 = {}
    n_codec_pass = 0
    print(f"  {'Codec / Channel Condition':<26} | {'EER':<8} | {'minDCF':<8} | Status")
    print("  " + "-" * 26 + " | " + "-" * 8 + " | " + "-" * 8 + " | ------")
    for cname, fn in codecs.items():
        b_list, s_list = [], []
        for _ in range(40):
            b = synthetic_speech(4.0, sr=sr, rng=rng).astype(np.float32) / 32768.0
            s = synthetic_spoofed(4.0, sr=sr, rng=rng).astype(np.float32) / 32768.0
            b_list.append(pipe.analyse(fn(b), sr).cm_score)
            s_list.append(pipe.analyse(fn(s), sr).cm_score)
        res = evaluate_system(np.array(b_list), np.array(s_list))
        # minDCF >= 1.0 means the detector is no better than uninformative under
        # this channel condition (docs/04 section 1.1). That is a FAIL.
        passed = bool(res["minDCF"] < MINDCF_UNINFORMATIVE)
        n_codec_pass += int(passed)
        res["passed"] = passed
        m3[cname] = res
        print(f"  {cname:<26} | {res['EER']:>6.2f}% | {res['minDCF']:>8.4f} | "
              f"{'PASS' if passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Benchmark 4: Standardized Demo Suite
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK 4/4] Validating Standardized Demo Benchmark Files...")
    demo_dir = REPO_ROOT / "data" / "demo"
    gt_file = demo_dir / "ground_truth.json"
    demo_results = {}
    n_demo_pass = 0
    n_demo_total = 0
    if gt_file.is_file():
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        import wave

        print(f"  {'File':<26} | {'Risk':<6} | {'Tier':<10} | {'Expected':<24} | {'Required':<10} | Verdict")
        print("  " + "-" * 26 + " | " + "-" * 6 + " | " + "-" * 10 + " | " + "-" * 24
              + " | " + "-" * 10 + " | -------")
        for fname, meta in gt.get("files", {}).items():
            fpath = demo_dir / fname
            if not fpath.is_file():
                continue
            with wave.open(str(fpath), "rb") as w:
                raw = w.readframes(w.getnframes())
                pcm = np.frombuffer(raw, dtype=np.int16)

            dsess = CallSession(f"bench-{fname}", SessionContext(caller_id="+91-DEMO"))
            hop_samples = dsess.buffer.hop_samples
            dsess.push_chunk(pcm[:dsess.buffer.window_samples])
            idx = dsess.buffer.window_samples
            while idx + hop_samples <= pcm.size:
                dsess.push_chunk(pcm[idx : idx + hop_samples])
                idx += hop_samples

            last = dsess.history[-1] if dsess.history else None
            risk = last.risk if last else 0.0
            tier = last.tier if last else "UNKNOWN"
            expected = meta.get("expected_verdict", "UNKNOWN")

            # Policy tier lives in the data file (data/demo/ground_truth.json);
            # fall back to the built-in map only if the field is absent.
            required = meta.get("expected_tier") or EXPECTED_TIER.get(expected, "?")
            passed = verdict_matches(
                tier, required, is_authentic=(expected == "AUTHENTIC_TARGET")
            )
            n_demo_total += 1
            n_demo_pass += int(passed)

            demo_results[fname] = {
                "risk": round(risk, 1),
                "tier": tier,
                "expected": expected,
                "required_tier": required,
                "windows": len(dsess.history),
                "passed": passed,
            }
            print(f"  {fname:<26} | {risk:>6.1f} | {tier:<10} | {expected:<24} | "
                  f"{required:<10} | {'PASS' if passed else 'FAIL'}")

    total_elapsed = time.perf_counter() - t_global_start

    # -------------------------------------------------------------------------
    # Honest summary. A benchmark whose only possible output is "SUCCESS" is
    # decoration, not a test -- so count the failures and say so.
    # -------------------------------------------------------------------------
    checks = []
    if m1["calibrated"]:
        checks.append(("ASVspoof 5 score calibration", True))
    else:
        msg = (
            f"ASVspoof 5 score calibration (actDCF@tau_Bayes={m1['actDCF_bayes']:.2f} "
            f"vs minDCF={m1['minDCF']:.2f})"
        )
        checks.append((msg, False))
    checks.append((f"Latency SLA p95 < 200 ms (p95={p95:.2f})", p95 < 200.0))
    failed_codecs = [cname for cname, r in m3.items() if not r["passed"]]
    msg = "Telephony robustness ("
    msg += (
        "all conditions informative"
        if not failed_codecs
        else f"{len(failed_codecs)}/{len(m3)} conditions uninformative: {', '.join(failed_codecs)}"
    )
    msg += ")"
    checks.append((msg, n_codec_pass == len(m3)))
    if n_demo_total:
        msg = (
            f"Demo ground-truth verification ({n_demo_pass}/{n_demo_total} files "
            f"reach their required tier)"
        )
        checks.append((msg, n_demo_pass == n_demo_total))
    else:
        checks.append(("Demo ground-truth verification (no files scored)", False))

    all_passed = all(ok for _, ok in checks)

    print("\n" + "=" * 72)
    print("  CHECK SUMMARY")
    print("=" * 72)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 72)
    if all_passed:
        print(f"  ALL BENCHMARKS PASSED IN {total_elapsed:.2f} SECONDS")
    else:
        failures = sum(1 for _, ok in checks if not ok)
        print(f"  {failures} OF {len(checks)} CHECKS FAILED "
              f"({total_elapsed:.2f}s) -- see above for detail.")
        print("  These failures are the real result. Do not silence them without")
        print("  fixing the underlying cause.")
    print("=" * 72)

    report = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_benchmark_duration_sec": round(total_elapsed, 2),
        "all_passed": all_passed,
        "checks": [{"check": name, "passed": ok} for name, ok in checks],
        "asvspoof5": m1,
        "latency_sla": {
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "mean_ms": mean_lat,
            "sla_threshold_p95_ms": 200.0,
            "compliant": p95 < 200.0,
        },
        "telephony_robustness": {
            k: {"EER_pct": v["EER"], "minDCF": v["minDCF"], "passed": v.get("passed")}
            for k, v in m3.items()
        },
        "demo_verification": demo_results,
        "demo_verification_summary": {
            "passed": n_demo_pass,
            "total": n_demo_total,
        },
    }

    out_json = REPO_ROOT / "benchmarks" / "latest_benchmark_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[Artifact] Benchmark report written to: {out_json}")
    return report


if __name__ == "__main__":
    # Non-zero exit so CI cannot treat a failing benchmark as a green build.
    sys.exit(0 if run_all_benchmarks()["all_passed"] else 1)
