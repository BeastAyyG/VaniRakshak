#!/usr/bin/env python
"""VaniRakshak end-to-end integration check.

Verifies the whole vertical slice the dashboard depends on:

  1. GET  /api/v1/health            -> service is up
  2. POST /api/v1/analyze/audio     -> ForensicReportData contract
  3. POST /api/v1/verify-speaker    -> trained ml/ identity model (or honest
                                       "unavailable" — must never be a 500)
  4. GET  /api/v1/audit             -> DPDP hash chain intact
  5. WS   /ws/stream                -> JSON handshake then int16 PCM over a
                                       real socket, exactly as the browser
                                       does it (docs/PROTOCOL.md)

By default the app is booted **in-process** on an ephemeral port, so this needs
no running server. Pass ``--live`` to exercise an already-running server
(``python run.py`` in another terminal) instead.

Usage:
    python scripts/integration_check.py
    python scripts/integration_check.py --live --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "vanirakshak-backend"
DEMO_DIR = REPO_ROOT / "frontend" / "public" / "demo"
GROUND_TRUTH_PATH = REPO_ROOT / "data" / "demo" / "ground_truth.json"

# (filename, expected verdict). Expected comes from *provenance*, not from a
# wish: demo_02 is a signal-degraded copy of the target voice, so the honest
# ground truth is AUTHENTIC_TARGET, not SYNTHETIC_VOICE_CLONE. It is not a
# neural clone and does not exercise the countermeasure path.
DEMOS = [
    ("demo_01_genuine.wav", "AUTHENTIC_TARGET"),
    ("demo_02_degraded_copy.wav", "AUTHENTIC_TARGET"),
    ("demo_03_partial_splice.wav", "SUSPICIOUS_SPLICED_AUDIO"),
    ("demo_04_impostor.wav", "IMPOSTOR_SPEAKER"),
]

# Every key the dashboard reads out of the /analyze/audio response. If one of
# these disappears the console silently renders "N/A" — so assert on them.
REPORT_KEYS = [
    "session_id",
    "audio_sha256",
    "duration_sec",
    "device_used",
    "windows_count",
    "overall_risk_score",
    "verdict",
    "mean_acoustic_anomaly",
    "mean_speaker_sim",
    "dpdp_compliance",
    "windows",
]

WS_VERDICT_KEYS = [
    "risk_score",
    "p_spoof",
    "tier",
    "metrics",
    "interlock_active",
    "receipt_hash",
]

# Every key docs/PROTOCOL.md documents on a live challenge object
# (challenge/engine.py::Challenge.to_dict). expected_digits is a list of
# single-digit strings; expected_full is the full spoken phrase (a regression
# guard for the bug where it used to be just the digits, e.g. "8 4" instead
# of "mango 8 nadi 4").
CHALLENGE_KEYS = ["phrase", "expected_digits", "expected_full", "instructions", "issued_at_ms", "seed"]

# The `semantic` block's key set must be identical whether it's idle
# (session.py::_SEMANTIC_IDLE), degraded (semantic/signal.py::_UNAVAILABLE_BASE),
# or a populated transcript (semantic/signal.py::analyze_window's success
# return) — a regression guard for the schema-drift bug where the idle shape
# was missing keyword_risk/claim_risk/claim_findings that only appeared once
# ASR actually ran, silently changing the object's shape mid-session.
REQUIRED_SEMANTIC_KEYS = {
    "available", "transcript", "hinglish_text", "language", "matches",
    "categories_hit", "semantic_risk", "keyword_risk", "claim_risk",
    "claim_findings", "provider", "reason",
}


class Checks:
    """Tiny PASS/FAIL accumulator."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return bool(ok)

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.rows if not ok)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _client_factory(live: bool, base_url: str):
    """Return (client, base_url) for REST calls."""
    if live:
        import httpx

        class LiveClient:
            def __init__(self, base: str) -> None:
                self._c = httpx.Client(base_url=base, timeout=60.0)

            def get(self, url: str, **kw):
                return self._c.get(url, **kw)

            def post(self, url: str, **kw):
                return self._c.post(url, **kw)

        return LiveClient(base_url), base_url

    sys.path.insert(0, str(BACKEND_ROOT))
    from fastapi.testclient import TestClient  # type: ignore

    from vanirakshak.server.app import app  # type: ignore

    return TestClient(app), ""


def check_health(client, checks: Checks) -> bool:
    print("\n  Service liveness")
    r = client.get("/api/v1/health")
    ok = r.status_code == 200
    body: Dict[str, Any] = r.json() if ok else {}
    checks.record(
        "GET /api/v1/health -> 200",
        ok,
        f"status={body.get('status')} device={body.get('hardware', {}).get('device_name')}",
    )
    # The dashboard reads both of these; missing keys render empty pills.
    return checks.record(
        "health carries hardware.device_name + models.wavlm_model",
        bool(body.get("hardware", {}).get("device_name")) and bool(body.get("models", {}).get("wavlm_model")),
    )


def check_analyze(client, checks: Checks) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    print("\n  REST /api/v1/analyze/audio (dashboard file + benchmark path)")
    for fname, _expected in DEMOS:
        path = DEMO_DIR / fname
        if not path.is_file():
            checks.record(f"demo asset {fname}", False, "missing")
            continue
        files = {"file": (fname, path.read_bytes(), "audio/wav")}
        data = {"claimed_identity": "narendra_modi"}
        r = client.post("/api/v1/analyze/audio", files=files, data=data)
        if r.status_code != 200:
            checks.record(f"{fname} -> 200", False, f"HTTP {r.status_code}: {r.text[:160]}")
            continue
        body = r.json()
        results[fname] = body
        missing = [k for k in REPORT_KEYS if k not in body]
        checks.record(
            f"{fname} -> ForensicReportData contract",
            not missing,
            f"risk={body.get('overall_risk_score')} verdict={body.get('verdict')} "
            f"windows={body.get('windows_count')}" + (f" MISSING={missing}" if missing else ""),
        )
    return results


def _real_backends_active() -> Dict[str, str]:
    """Which detector backends this run's server was configured for.

    Read from this process's own environment — the caller is expected to set
    these before booting the in-process app (or before starting `run.py` for
    `--live`), the same env vars `pipeline.py`'s factory reads.
    """
    return {
        "asv": os.environ.get("VR_ASV_BACKEND", "stub").lower(),
        "cm": os.environ.get("VR_CM_BACKEND", "stub").lower(),
    }


def check_discrimination(results: Dict[str, Dict[str, Any]], checks: Checks) -> None:
    """Do the four benchmark clips actually score differently?

    With the stub detectors (default, zero-dependency) this stays purely
    informational — docs/archive/AUDIT_REPORT.md §5 documents why the stubs don't
    separate these clips, and asserting on them would just be a flaky test
    dressed up as an accuracy claim.

    When real detector backends are active (VR_ASV_BACKEND=ecapa and/or
    VR_CM_BACKEND=wavlm), this additionally asserts each clip's risk *tier*
    against `data/demo/ground_truth.json`'s `expected_tier` — an accuracy
    claim is only asserted when it's backed by real features.
    """
    if len(results) < 4:
        return
    print("\n  Benchmark discrimination (informational)")
    risks = [results[d]["overall_risk_score"] for d, _ in DEMOS if d in results]
    spread = max(risks) - min(risks)
    print(f"    risk scores: {risks}")
    print(f"    spread: {spread:.2f} points")
    if spread < 5.0:
        print(
            "    NOTE: spread < 5 pts — the stub detectors do not separate these\n"
            "          clips. Treat the benchmark table as a UI/API smoke test,\n"
            "          NOT as measured detection accuracy. See docs/archive/AUDIT_REPORT.md §5."
        )

    backends = _real_backends_active()
    real_active = backends["asv"] == "ecapa" or backends["cm"] == "wavlm"
    if not real_active:
        print(f"    (tier-match gate skipped — stub backends active: {backends})")
        return

    if not GROUND_TRUTH_PATH.is_file():
        checks.record("ground_truth.json present for tier gate", False, str(GROUND_TRUTH_PATH))
        return
    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))["files"]

    print(f"\n  Tier-match gate (real backends active: {backends})")
    for fname, _expected_verdict in DEMOS:
        if fname not in results or fname not in gt:
            continue
        expected_tier = gt[fname].get("expected_tier")
        actual_tier = results[fname].get("tier")
        if not expected_tier:
            continue
        checks.record(
            f"{fname} tier == {expected_tier}",
            actual_tier == expected_tier,
            f"got tier={actual_tier} risk={results[fname].get('overall_risk_score')}",
        )


def check_verify_speaker(client, checks: Checks) -> None:
    print("\n  POST /api/v1/verify-speaker (trained ml/ identity model)")
    path = DEMO_DIR / "demo_01_genuine.wav"
    files = {"file": (path.name, path.read_bytes(), "audio/wav")}
    r = client.post("/api/v1/verify-speaker", files=files)
    # A 500 here is the bug this check exists to catch: a missing torch/librosa
    # used to escape as an unhandled ASGI exception.
    ok_status = r.status_code == 200
    checks.record("verify-speaker never 500s", ok_status, f"HTTP {r.status_code}")
    if not ok_status:
        print(f"    body: {r.text[:200]}")
        return
    body = r.json()
    if body.get("available"):
        checks.record(
            "identity model loaded and scored",
            True,
            f"label={body.get('label')} p_target={body.get('prob_target')} llr={body.get('llr')}",
        )
    else:
        checks.record(
            "identity model degrades honestly (available=false + reason)",
            body.get("label") == "unavailable" and bool(body.get("reason")),
            f"reason={body.get('reason')}",
        )
        print(
            "    NOTE: identity model not loadable in this environment.\n"
            "          `pip install -r vanirakshak-backend/requirements-ml.txt` enables it."
        )


def check_audit(client, checks: Checks) -> None:
    print("\n  GET /api/v1/audit (DPDP hash chain)")
    r = client.get("/api/v1/audit?n=10")
    if r.status_code != 200:
        checks.record("audit endpoint", False, f"HTTP {r.status_code}")
        return
    body = r.json()
    checks.record(
        "audit chain verifies",
        bool(body.get("chain_valid")),
        f"events={len(body.get('events', []))}",
    )


def check_ws_stream(live: bool, base_url: str, checks: Checks) -> None:
    """Replay exactly what the browser does: JSON handshake, then int16 PCM.

    Runs over a real socket (uvicorn on an ephemeral port when not --live), so
    it exercises the same framing the AudioWorklet produces.
    """
    print("\n  WS /ws/stream (live microphone protocol)")

    server = None
    if live:
        url = base_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws/stream"
    else:
        sys.path.insert(0, str(BACKEND_ROOT))
        import uvicorn  # type: ignore

        from vanirakshak.server.app import app  # type: ignore

        port = _free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        url = f"ws://127.0.0.1:{port}/ws/stream"

    sr = 16000
    # Analysis window is 4.04 s with a 1.0 s hop — stream 6 s so at least two
    # verdicts must come back.
    #
    # Stream the *impostor* clip rather than noise: the point of this check is to
    # prove the live path is target-conditioned, and noise cannot exercise the
    # ASV. The anchor identity is read from the same file the server loads, so a
    # missing anchor shows up here instead of silently degrading to asv=0.5.
    import wave as _wave

    anchor_id = ""
    anchor_path = REPO_ROOT / "data" / "anchor_asv_embedding.json"
    if anchor_path.is_file():
        anchor_id = str(json.loads(anchor_path.read_text(encoding="utf-8")).get("user_id", ""))

    src = DEMO_DIR / "demo_04_impostor.wav"
    if src.is_file():
        with _wave.open(str(src), "rb") as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    else:
        rng = np.random.default_rng(7)
        audio = (rng.normal(0, 0.02, sr * 6) * 32767).astype(np.int16)
    frame = 320  # 20 ms, like the AudioWorklet

    try:
        from websockets.sync.client import connect

        verdicts: List[Dict[str, Any]] = []
        ack: Dict[str, Any] = {}
        with connect(url, open_timeout=10) as ws:
            ws.send(
                json.dumps(
                    {
                        "session_id": "integration-check",
                        "sample_rate": sr,
                        "caller_id": "integration-check",
                        "claimed_identity": anchor_id,
                        "transaction_value_inr": 0,
                        "origin_country": "IN",
                        "prior_fraud_score": 0,
                    }
                )
            )
            # Handshake ack arrives before any verdict (PROTOCOL.md §1) —
            # consume it here so it never lands in the verdict list.
            try:
                first = ws.recv(timeout=2.0)
                if isinstance(first, str):
                    parsed = json.loads(first)
                    if parsed.get("type") == "session_ack":
                        ack = parsed
                    else:
                        verdicts.append(parsed)
            except TimeoutError:
                pass
            for i in range(0, audio.size, frame):
                ws.send(audio[i : i + frame].tobytes())
                try:
                    msg = ws.recv(timeout=0.05)
                    if isinstance(msg, str):
                        verdicts.append(json.loads(msg))
                except TimeoutError:
                    pass
            # Drain anything still in flight.
            deadline = time.time() + 3.0
            while time.time() < deadline and len(verdicts) < 2:
                try:
                    msg = ws.recv(timeout=0.5)
                    if isinstance(msg, str):
                        verdicts.append(json.loads(msg))
                except TimeoutError:
                    break
    except Exception as e:
        checks.record("websocket stream", False, f"{type(e).__name__}: {e}")
        return
    finally:
        if server is not None:
            server.should_exit = True

    checks.record(
        "handshake accepted (no close 1003) + verdicts returned",
        len(verdicts) > 0,
        f"verdicts={len(verdicts)} over 6.0s stream",
    )
    checks.record(
        "session_ack echoes resolved session_id before verdicts",
        ack.get("type") == "session_ack" and ack.get("session_id") == "integration-check",
        f"ack={ack}",
    )
    if not verdicts:
        return
    v = verdicts[-1]
    missing = [k for k in WS_VERDICT_KEYS if k not in v]
    checks.record(
        "verdict carries full protocol payload",
        not missing,
        f"tier={v.get('tier')} risk={round(v.get('risk_score', 0), 1)}"
        + (f" MISSING={missing}" if missing else ""),
    )
    metrics = v.get("metrics", {})
    checks.record(
        "metrics.asv_consistency + metrics.snr_db present",
        "asv_consistency" in metrics and "snr_db" in metrics,
        f"asv={metrics.get('asv_consistency')} snr={metrics.get('snr_db')}",
    )
    checks.record(
        "receipt_hash emitted (DPDP audit anchor)",
        bool(v.get("receipt_hash")),
        str(v.get("receipt_hash"))[:16] + "...",
    )

    # metrics.asv_evaluated must appear on every verdict — a regression guard
    # for the bug where claimed_identity defaulted to "" everywhere, so ASV
    # never ran and asv_consistency was a silent hardcoded 0.5.
    missing_asv_eval = [i for i, vv in enumerate(verdicts) if "asv_evaluated" not in vv.get("metrics", {})]
    checks.record(
        "metrics.asv_evaluated present on every verdict",
        not missing_asv_eval,
        f"asv_evaluated={v.get('metrics', {}).get('asv_evaluated')}"
        + (f" MISSING on verdicts {missing_asv_eval}" if missing_asv_eval else ""),
    )

    # semantic block: same key set whether idle, degraded, or populated.
    semantic_key_mismatches = {
        i: sorted(REQUIRED_SEMANTIC_KEYS - set(vv.get("semantic", {}).keys()))
        for i, vv in enumerate(verdicts)
        if REQUIRED_SEMANTIC_KEYS - set(vv.get("semantic", {}).keys())
    }
    checks.record(
        "semantic block has the full key set on every verdict (idle/degraded/populated)",
        not semantic_key_mismatches,
        f"sample keys={sorted(v.get('semantic', {}).keys())}"
        + (f" MISSING={semantic_key_mismatches}" if semantic_key_mismatches else ""),
    )

    # challenge object: this fixture (demo_04_impostor.wav) reliably fires a
    # CHALLENGE before escalating to BLOCK, so this is not conditional.
    challenge_verdicts = [vv for vv in verdicts if vv.get("trigger_challenge") and vv.get("challenge")]
    if challenge_verdicts:
        ch = challenge_verdicts[0]["challenge"]
        missing_ch = [k for k in CHALLENGE_KEYS if k not in ch]
        digits_ok = isinstance(ch.get("expected_digits"), list) and all(
            isinstance(d, str) and len(d) == 1 for d in ch.get("expected_digits", [])
        )
        # expected_full must be the full spoken phrase, not just the digits —
        # i.e. it must contain every word token, not only the digit tokens.
        full_has_words = bool(ch.get("expected_full")) and any(
            w.isalpha() for w in str(ch.get("expected_full", "")).split()
        )
        checks.record(
            "challenge object has full protocol shape",
            not missing_ch and digits_ok and full_has_words,
            f"phrase={ch.get('phrase')!r} expected_full={ch.get('expected_full')!r}"
            + (f" MISSING={missing_ch}" if missing_ch else "")
            + ("" if digits_ok else " BAD_DIGITS_TYPE")
            + ("" if full_has_words else " expected_full missing word tokens"),
        )
    else:
        print("    NOTE: no challenge fired this run — demo_04_impostor.wav usually triggers one on its first window.")

    # VR_IMPOSTOR_BLOCK_BELOW override: a deterministic tier-forcing rule that
    # docs/PROTOCOL.md documents but a client deriving its own tier from
    # risk_score alone would miss. Only checkable in-process, where this
    # process's own config actually governs the server that produced `verdicts`.
    if not live:
        sys.path.insert(0, str(BACKEND_ROOT))
        from vanirakshak.config import settings  # type: ignore

        threshold = settings.risk.impostor_block_below
        override_violations = [
            (i, vv.get("metrics", {}).get("asv_consistency"), vv.get("tier"))
            for i, vv in enumerate(verdicts)
            if vv.get("metrics", {}).get("asv_consistency") is not None
            and vv["metrics"]["asv_consistency"] < threshold
            and vv.get("tier") != "BLOCK"
        ]
        checks.record(
            f"VR_IMPOSTOR_BLOCK_BELOW override holds (asv_consistency < {threshold} => BLOCK)",
            not override_violations,
            f"checked {len(verdicts)} verdicts" + (f" VIOLATIONS={override_violations}" if override_violations else ""),
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="VaniRakshak end-to-end integration check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--live", action="store_true", help="exercise a running server instead of booting in-process")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    print("=" * 78)
    print("VaniRakshak integration check" + ("  (LIVE)" if args.live else "  (in-process)"))
    print("=" * 78)

    checks = Checks()
    client, base = _client_factory(args.live, args.base_url)

    check_health(client, checks)
    results = check_analyze(client, checks)
    check_discrimination(results, checks)
    check_verify_speaker(client, checks)
    check_audit(client, checks)
    check_ws_stream(args.live, args.base_url, checks)

    print("\n" + "=" * 78)
    total = len(checks.rows)
    failed = checks.failed
    print(f"{total - failed}/{total} checks passed")
    if failed:
        print("\nFailures:")
        for name, ok, detail in checks.rows:
            if not ok:
                print(f"  - {name}: {detail}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
