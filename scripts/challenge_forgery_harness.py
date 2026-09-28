"""P1-C harness: can a scripted client pass the challenge on text alone?

docs/15 N2 moved digit grading from the client's ``response_text`` to the
server's own transcript of the response audio. This script attacks that with
the real local ASR engine (``VR_CHALLENGE_ASR_PROVIDER``, default ``local`` =
faster-whisper) through the real session code.

Every attack trial sends the **correct** digits as ``response_text`` with a
human latency, over response audio that does not say them:

* ``silence``    -- near-silent audio (low dither)
* ``noise``      -- white noise at speech level
* ``vocoder``    -- the repo's synthetic TTS-like signal (no words)
* ``tts_wrong``  -- TTS speaking the wrong digits  (``--tts``, Windows SAPI)

P1-C passes when each attack fails in >= 90 % of trials.

A **positive control** guards against a transcriber that fails everything:

* ``tts_correct`` -- TTS speaking the right digits (``--tts``). It must be
  graded correct, so this measures that the engine hears digits at all.
  Note what it also shows: a TTS that says the right digits passes the
  *transcript*; only the response-audio CM check (``VR_CHALLENGE_CM_LLR_MAX``,
  set by docs/14 §6 step 2 after T1) can stop it. The CM check is off here so
  the transcript gate is measured alone.

``--genuine-proxy`` (docs/14 W2.c) measures the other side: how often a
*correct* spoken answer passes. It is a proxy, not a friction measurement:
the speakers are the installed SAPI voices, not people. Digits are spoken as
English words and as romanised Hindi words ("teen, paanch"), and heard clean,
through G.711 mu-law at 8 kHz, and through G.711 with noise at 15 dB SNR.
Real Hindi answers from people (docs/15 D1) are still needed.

Usage:
    python scripts/challenge_forgery_harness.py --trials 30 --tts \\
        --out reports/challenge_p1c_harness.json
    python scripts/challenge_forgery_harness.py --trials 15 --genuine-proxy \\
        --out reports/challenge_genuine_proxy.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "vanirakshak-backend"))

from vanirakshak.challenge.transcribe import ChallengeTranscriber  # noqa: E402
from vanirakshak.config import ChallengeConfig, RiskConfig, settings  # noqa: E402
from vanirakshak.data.synthetic import synthetic_spoofed  # noqa: E402
from vanirakshak.risk.fusion import RiskFusionEngine  # noqa: E402
from vanirakshak.risk.simulate import WindowScore, _ScriptedPipeline  # noqa: E402
from vanirakshak.session import CallSession, SessionContext  # noqa: E402

SR = 16000
# Tier CHALLENGE on the first window, then clean windows so only the
# transcript decides (the CM check is disabled below).
BORDERLINE = WindowScore(cm_llr=-1.0, asv=0.9, asv_enrolled=True)
CLEAN = WindowScore(cm_llr=6.0, asv=0.95, asv_enrolled=True)
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


HINDI_WORDS = ["shunya", "ek", "do", "teen", "chaar", "paanch", "chhe", "saat", "aath", "nau"]
_TTS_CACHE: Dict[tuple, Optional[np.ndarray]] = {}


def sapi_voices() -> List[str]:
    if platform.system() != "Windows":
        return []
    ps = ("Add-Type -AssemblyName System.Speech; "
          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             check=True, capture_output=True, text=True, timeout=60)
        return [v.strip() for v in out.stdout.splitlines() if v.strip()]
    except Exception:
        return []


def g711(x: np.ndarray, noise_snr_db: Optional[float] = None, seed: int = 0) -> np.ndarray:
    """16 kHz int16 -> 8 kHz mu-law (8-bit) -> 16 kHz int16, optionally with
    white noise added first at ``noise_snr_db``."""
    from scipy.signal import resample_poly

    y = x.astype(np.float32) / 32768.0
    if noise_snr_db is not None:
        p = float(np.mean(y ** 2)) or 1e-9
        y = y + np.random.default_rng(seed).normal(0, np.sqrt(p / 10 ** (noise_snr_db / 10)), y.size)
    y = np.clip(resample_poly(y, 1, 2), -1, 1)
    mu = 255.0
    enc = np.round((np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu) + 1) / 2 * 255)
    dec = enc / 255 * 2 - 1
    y = np.sign(dec) * ((1 + mu) ** np.abs(dec) - 1) / mu
    return (np.clip(resample_poly(y, 2, 1), -1, 1) * 32767).astype(np.int16)


def sapi_tts(text: str, voice: Optional[str] = None) -> Optional[np.ndarray]:
    """16 kHz int16 speech from Windows SAPI, or None where unavailable."""
    key = (text, voice)
    if key not in _TTS_CACHE:
        _TTS_CACHE[key] = _sapi_tts(text, voice)
    return _TTS_CACHE[key]


def _sapi_tts(text: str, voice: Optional[str]) -> Optional[np.ndarray]:
    if platform.system() != "Windows":
        return None
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "tts.wav"
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
            "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
            "[System.Speech.AudioFormat.AudioChannel]::Mono); "
            + (f"$s.SelectVoice('{voice}'); " if voice else "")
            + f"$s.SetOutputToWaveFile('{wav}', $f); $s.Rate = -1; "
            f"$s.Speak('{text}'); $s.Dispose()"
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           check=True, capture_output=True, timeout=60)
            with wave.open(str(wav), "rb") as w:
                return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()
        except Exception:
            return None


def pad(audio: np.ndarray, seconds: float = 4.0) -> np.ndarray:
    """Lead-in and tail of low noise, so the response spans a few windows."""
    rng = np.random.default_rng(len(audio))
    n = int(seconds * SR)
    lead = (rng.normal(0, 30, int(0.6 * SR))).astype(np.int16)
    out = np.concatenate([lead, audio.astype(np.int16)])
    if out.size < n:
        out = np.concatenate([out, rng.normal(0, 30, n - out.size).astype(np.int16)])
    return out


def run_trial(transcriber, audio_for: Callable[[List[str]], Optional[np.ndarray]], seed: int) -> Optional[dict]:
    cfg = dataclasses.replace(
        settings.challenge, cm_llr_max=None, asv_min=None, require_audio=False,
        client_text_drill=False, max_per_session=1, trigger_consecutive=1,
    )
    sess = CallSession(
        f"p1c-{seed}",
        SessionContext(claimed_identity="user_1"),
        pipeline=_ScriptedPipeline([BORDERLINE] + [CLEAN] * 20),
        fusion_engine=RiskFusionEngine(RiskConfig(), settings.fusion),
        challenge_cfg=cfg,
        challenge_transcriber=transcriber,
    )
    sess.challenge._seed = seed  # vary the digits per trial, reproducibly
    cr = None
    while cr is None or not cr.trigger_challenge:
        cr = sess.push_chunk(np.zeros(SR, dtype=np.int16))
    digits = list(sess._active_challenge.expected_digits)
    audio = audio_for(digits)
    if audio is None:
        return None
    for i in range(0, audio.size, SR):
        sess.push_chunk(audio[i:i + SR])
    # The scripted client always sends the right digits with a human latency.
    grade = sess.grade_challenge(" ".join(digits), 1000)
    return {"digits": "".join(digits), "ok": grade["ok"], "reason": grade["reason"],
            "transcript": grade["transcript"], "asr_error": grade["asr_error"],
            "latency_ms": grade.get("latency_ms")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--provider", default=None, help="default: VR_CHALLENGE_ASR_PROVIDER")
    ap.add_argument("--tts", action="store_true", help="add the SAPI TTS attack and control")
    ap.add_argument("--genuine-proxy", action="store_true",
                    help="measure correct TTS answers by voice x language x channel instead")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    transcriber = ChallengeTranscriber(provider=args.provider)
    if not transcriber.available:
        print(f"no local ASR engine: {transcriber.load_error}", file=sys.stderr)
        return 2

    def silence(d):
        return np.random.default_rng(1).normal(0, 20, 4 * SR).astype(np.int16)

    def noise(d):
        return np.random.default_rng(2).normal(0, 3000, 4 * SR).clip(-32767, 32767).astype(np.int16)

    def vocoder(d):
        x = synthetic_spoofed(duration_sec=4.0, sr=SR, rng=np.random.default_rng(3))
        return (np.clip(x, -1, 1) * 32767).astype(np.int16)

    def tts_wrong(d):
        wrong = [WORDS[(int(x) + 1) % 10] for x in d]
        a = sapi_tts(", ".join(wrong))
        return None if a is None else pad(a)

    def tts_correct(d):
        a = sapi_tts(", ".join(WORDS[int(x)] for x in d))
        return None if a is None else pad(a)

    if args.genuine_proxy:
        return genuine_proxy(transcriber, args)

    attacks: Dict[str, Callable] = {"silence": silence, "noise": noise, "vocoder": vocoder}
    controls: Dict[str, Callable] = {}
    if args.tts:
        attacks["tts_wrong"] = tts_wrong
        controls["tts_correct"] = tts_correct

    report: dict = {
        "criterion": "P1-C: each attack fails in >= 90% of trials (docs/13 §1.2, docs/15 N2)",
        "provider": transcriber.provider,
        "trials": args.trials,
        "cm_check": "off (transcript gate measured alone)",
        "date": time.strftime("%Y-%m-%d"),
        "attacks": {},
        "controls": {},
    }
    for group, cases in (("attacks", attacks), ("controls", controls)):
        for name, fn in cases.items():
            rows = [r for r in (run_trial(transcriber, fn, 1000 + i) for i in range(args.trials)) if r]
            if not rows:
                report[group][name] = {"skipped": "audio source unavailable"}
                continue
            fail = sum(not r["ok"] for r in rows) / len(rows)
            reasons: Dict[str, int] = {}
            for r in rows:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            entry = {"n": len(rows), "fail_rate": round(fail, 4), "reasons": reasons,
                     "examples": rows[:3]}
            if group == "attacks":
                entry["p1c_pass"] = fail >= 0.90
            else:
                entry["pass_rate"] = round(1 - fail, 4)
            report[group][name] = entry
            print(f"{group[:-1]:7s} {name:12s} n={len(rows):3d} fail={fail:6.1%}  {reasons}")

    graded = [v for v in report["attacks"].values() if "p1c_pass" in v]
    report["p1c_pass"] = bool(graded) and all(v["p1c_pass"] for v in graded)
    print("P1-C (transcript gate):", "PASS" if report["p1c_pass"] else "FAIL")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if report["p1c_pass"] else 1


def genuine_proxy(transcriber, args) -> int:
    voices = sapi_voices()
    if not voices:
        print("no SAPI voices (Windows only)", file=sys.stderr)
        return 2
    langs = {"en": WORDS, "hi_roman": HINDI_WORDS}
    channels = {
        "clean": lambda a, i: a,
        "g711": lambda a, i: g711(a),
        "g711_snr15": lambda a, i: g711(a, 15.0, seed=i),
    }
    report: dict = {
        "what": "genuine-answer PROXY: SAPI voices saying the right digits (docs/14 W2.c)",
        "not_measured": "real people, real Hindi speech, AMR codecs",
        "provider": transcriber.provider,
        "trials": args.trials,
        "latency_band_ms": [settings.challenge.latency_min_ms, settings.challenge.latency_max_ms],
        "cm_check": "off (transcript and latency gates only)",
        "date": time.strftime("%Y-%m-%d"),
        "cells": {},
    }
    for voice in voices:
        for lang, words in langs.items():
            for ch_name, ch in channels.items():
                counter = [0]

                def audio_for(d, words=words, voice=voice, ch=ch, counter=counter):
                    a = sapi_tts(", ".join(words[int(x)] for x in d), voice)
                    counter[0] += 1
                    return None if a is None else ch(pad(a), counter[0])

                rows = [r for r in (run_trial(transcriber, audio_for, 1000 + i)
                                    for i in range(args.trials)) if r]
                reasons: Dict[str, int] = {}
                for r in rows:
                    reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
                ok = sum(r["ok"] for r in rows) / max(1, len(rows))
                cell = f"{voice} / {lang} / {ch_name}"
                report["cells"][cell] = {
                    "n": len(rows),
                    "pass_rate": round(ok, 4),
                    "reasons": reasons,
                    "failed_examples": [{k: r[k] for k in ("digits", "transcript", "reason", "latency_ms")}
                                        for r in rows if not r["ok"]][:3],
                }
                print(f"{cell:45s} n={len(rows):3d} pass={ok:6.1%}  {reasons}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
