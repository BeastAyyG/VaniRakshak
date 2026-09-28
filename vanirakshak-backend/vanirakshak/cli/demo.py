"""Offline end-to-end demo (no server, no GPU, no audio file needed).

This is the "press one button and see the whole system" path used
during the live demo. It:

  1. generates 4 s of synthetic genuine speech + 4 s of synthetic
     spoofed speech (deterministic seed),
  2. pushes both through the full CallSession pipeline,
  3. prints each window's risk + tier,
  4. issues a dynamic challenge when risk crosses 35,
  5. grades a simulated human response,
  6. writes a DPDP-compliant audit receipt.

Run::

    python -m vanirakshak demo
"""

from __future__ import annotations

import sys
import time
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

import dataclasses

from ..config import settings, FusionConfig, RiskConfig
from ..data.synthetic import synthetic_speech, synthetic_spoofed
from ..data.augment import apply_random_pipeline
from ..session import CallSession, SessionContext
from ..risk.fusion import RiskFusionEngine


def _push_full_clip(sess: CallSession, clip: np.ndarray) -> List:
    """Push a clip in 1.0 s hops and collect per-window results."""
    sr = sess.sr
    hop = int(sr * 1.0)
    out = []
    i = 0
    while i + hop <= clip.size:
        r = sess.push_chunk(clip[i : i + hop])
        if r is not None:
            out.append(r)
        i += hop
    return out


def _print_row(label: str, t: float, risk: float, tier: str, metrics: dict) -> None:
    cm = metrics.get("cm_score", 0.0)
    asv = metrics.get("asv_consistency", 0.5)
    chan = metrics.get("channel_anomaly", 0.0)
    pros = metrics.get("prosody_unnatural", 0.0)
    print(
        f"  t={t:5.1f}s  {label:>9}  risk={risk:5.1f}  tier={tier:<9}  "
        f"cm={cm:+.2f}  asv={asv:.2f}  chan={chan:.2f}  pros={pros:.2f}"
    )


def run_demo() -> int:
    sr = settings.audio.sample_rate
    rng = np.random.default_rng(0)
    print("=" * 72)
    print("  VaniRakshak — offline end-to-end demo (no server, no GPU, no audio)")
    print("=" * 72)
    print(f"  sample rate         : {sr} Hz")
    print(f"  window / hop        : {settings.audio.window_seconds:.2f}s / {settings.audio.hop_seconds:.2f}s")
    print(f"  risk thresholds     : ALLOW < {settings.risk.allow_threshold} ≤ CHALLENGE < {settings.risk.challenge_threshold} ≤ BLOCK")
    print()

    # 1. enrol a "genuine user" with 4s of clean synthetic speech
    print("[1] Enrolling genuine user 'demo_user' (4.0 s) ...")
    # Use the aggressive demo preset so the BLOCK transition is visible
    # on stage. Real production re-fits these on ASVspoof 5.
    # The demo's audio is synthetic, so no ASR engine could hear digits in
    # it: grade the scripted text instead (drill mode, docs/15 N2).
    DRILL = dataclasses.replace(settings.challenge, client_text_drill=True)
    fusion = RiskFusionEngine(fusion_cfg=FusionConfig(
        intercept=-2.4, w_cm=3.5, w_asv=2.5, w_prosody=1.2, w_channel=1.5, w_context=1.0,
    ))
    sess = CallSession(
        "demo-session",
        SessionContext(
            caller_id="+91-9000000000",
            claimed_identity="demo_user",
            transaction_value_inr=4_00_000,  # ₹4 lakh — high value
            origin_country="IN",
            prior_fraud_score=0.0,
        ),
        fusion_engine=fusion,
        challenge_cfg=DRILL,
    )
    genuine = synthetic_speech(duration_sec=4.0, sr=sr, rng=rng)
    sess.enrol("demo_user", genuine, sr)
    print(f"    ✔ enrolled (embedding dim=192, audio NOT retained)")
    print()

    # 2. simulate a GENUINE call (telephony codec, mild noise)
    print("[2] Stream: GENUINE call (with codec degradation) ...")
    genuine_long = synthetic_speech(duration_sec=5.5, sr=sr, rng=np.random.default_rng(11))
    genuine_long = apply_random_pipeline(genuine_long, sr, seed=1)
    rows = _push_full_clip(sess, genuine_long)
    for r in rows:
        _print_row("GENUINE", r.timestamp_ms / 1000.0, r.risk, r.tier, r.metrics)
    last_tier = rows[-1].tier if rows else "ALLOW"
    print()

    # 3. now a SPOOFED call (impersonating a different speaker to trigger ASV mismatch)
    print("[3] Stream: SPOOFED call (TTS-style synthetic, mismatched identity) ...")
    sess.context.claimed_identity = "imposter_claiming_demo_user"
    sess.fusion.reset()
    spoof_long = synthetic_spoofed(duration_sec=5.5, sr=sr, rng=np.random.default_rng(12))
    spoof_long = apply_random_pipeline(spoof_long, sr, seed=2)
    rows2 = _push_full_clip(sess, spoof_long)
    challenge = None
    for r in rows2:
        _print_row("SPOOF", r.timestamp_ms / 1000.0, r.risk, r.tier, r.metrics)
        if r.trigger_challenge and challenge is None:
            challenge = r.challenge
    print()

    # 4. challenge flow
    if challenge is None:
        print("  (no challenge triggered — risk did not cross 35; issuing one manually for demo)")
        ch = sess.challenge.issue(lang="mixed")
        sess._active_challenge = ch
        sess._triggered_challenge = True
        challenge = ch.to_dict()
    print("[4] Challenge engine issued:")
    print(f"    phrase        : \"{challenge['phrase']}\"")
    # expected_digits is kept off the wire (W2); the demo reads the session's copy.
    expected = sess._active_challenge.expected_digits
    print(f"    expected digits: {''.join(expected)}")
    print(f"    instructions  : {challenge['instructions']}")
    print()

    # 5. grade a simulated HUMAN response
    print("[5] Grading a simulated HUMAN response (correct digits, ~900 ms latency) ...")
    response_text = " ".join(expected) + " uh I think that is it"
    grade = sess.grade_challenge(response_text, response_latency_ms=900)
    print(f"    result        : {grade}")
    print()

    # 6. now grade a simulated REPLAY/TTS response (too fast)
    print("[6] Grading a simulated REPLAY/TTS response (too fast, 180 ms) ...")
    sess2 = CallSession("demo-replay", SessionContext(caller_id="+91-1111111111", claimed_identity="imposter", transaction_value_inr=4_00_000), fusion_engine=fusion, challenge_cfg=DRILL)
    sess2.enrol("demo_user", genuine, sr)
    # push enough spoofed to force the challenge to fire
    sp_short = synthetic_spoofed(duration_sec=5.5, sr=sr, rng=np.random.default_rng(13))
    _push_full_clip(sess2, sp_short)
    if sess2._active_challenge is None:
        ch2 = sess2.challenge.issue(lang="mixed")
        sess2._active_challenge = ch2
        sess2._triggered_challenge = True
    grade2 = sess2.grade_challenge(" ".join(sess2._active_challenge.expected_digits), response_latency_ms=180)
    print(f"    result        : {grade2}")
    print(f"    session.blocked = {sess2.blocked}")
    print()

    # 7. show audit chain
    print("[7] Last 3 audit events (DPDP-compliant, SHA-256 hash-chained) ...")
    for ev in sess.audit.tail(3):
        d = ev.to_public_dict()
        print(f"    risk={d['risk_score']:5.1f} tier={d['tier']:<9} receipt={d['receipt_hash'][:16]}…")
    print(f"    chain_valid    : {sess.audit.verify_chain()}")
    print()
    print("=" * 72)
    print("  Demo complete. Next: `python -m vanirakshak serve` to expose the API.")
    print("=" * 72)
    return 0
