"""Centralised configuration.

All environment-driven knobs live here. ``run.py`` and the FastAPI app
both import from this module so a single env var moves every layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = _env_int("VR_SAMPLE_RATE", 16000)
    window_seconds: float = _env_float("VR_WINDOW_SECONDS", 4.04)
    hop_seconds: float = _env_float("VR_HOP_SECONDS", 1.0)


@dataclass(frozen=True)
class RiskConfig:
    allow_threshold: float = _env_float("VR_ALLOW_THRESHOLD", 35.0)
    challenge_threshold: float = _env_float("VR_CHALLENGE_THRESHOLD", 70.0)
    ema_alpha: float = _env_float("VR_EMA_ALPHA", 0.65)
    fast_rise_threshold: float = _env_float("VR_FAST_RISE_THRESHOLD", 0.85)

    # Deterministic impostor override.
    #
    # The fused score is a weighted sum, so one piece of overwhelming evidence
    # gets diluted by the others: a caller whose voice matched the anchor at
    # 0.09 landed at risk 67 — CHALLENGE, just under the 70 BLOCK line — and the
    # transaction interlock never fired on a confirmed identity mismatch.
    #
    # Fraud policy is not purely probabilistic. If the speaker consistency is
    # decisively below this value, the tier is forced to BLOCK regardless of the
    # fused score. Set to <= 0 to disable.
    impostor_block_below: float = _env_float("VR_IMPOSTOR_BLOCK_BELOW", 0.10)


@dataclass(frozen=True)
class FusionConfig:
    """Logistic-regression weights for the calibrated risk fusion.

    Defaults are derived from a synthetic ground-truth prior (see
    ``vanirakshak/risk/fusion.py``). Real deployment re-fits these on
    the ASVspoof 5 dev partition.
    """

    intercept: float = _env_float("VR_FUSION_BIAS", -1.6)
    w_cm: float = _env_float("VR_FUSION_W_CM", 2.4)
    w_asv: float = _env_float("VR_FUSION_W_ASV", 1.6)
    w_prosody: float = _env_float("VR_FUSION_W_PROSODY", 0.7)
    w_channel: float = _env_float("VR_FUSION_W_CHANNEL", 0.9)
    w_context: float = _env_float("VR_FUSION_W_CONTEXT", 0.5)

    # Weight for the ASR/keyword semantic signal (vanirakshak/semantic/).
    # Deliberately smaller than the acoustic weights: this is a parallel,
    # advisory context term (scam-phrase keyword hits in the live transcript),
    # not a primary detector — the acoustic evidence stays authoritative.
    # 0.0 (and semantic_risk absent) reproduces the pre-ASR fusion exactly.
    w_semantic: float = _env_float("VR_FUSION_W_SEMANTIC", 0.4)


# A more aggressive preset for demos where the synthetic detector can't
# always tell genuine from spoofed on the first window. Override by
# setting VR_FUSION_PRESET=aggressive.
_DEMO_OVERRIDES = {
    "demo": {
        "w_cm": 3.5,
        "w_asv": 2.5,
        "w_prosody": 1.2,
        "w_channel": 1.5,
        "w_context": 1.0,
        "w_semantic": 0.6,
        "intercept": -2.4,
    },
}

# Real-detector-backend preset (VR_ASV_BACKEND=ecapa / VR_CM_BACKEND=wavlm).
#
# Swapping in real ECAPA-TDNN and WavLM features changes each detector's
# score distribution completely — the stub-era weights fail
# scripts/integration_check.py's tier-match gate outright when real backends
# are active. w_prosody/w_channel are left at the stub-era defaults — that
# script's fitting exercise found the real anchor (formal speeches) vs.
# other-speaker (casual audio) recordings differ systematically in prosody
# for reasons unrelated to genuine-vs-spoofed, so refitting those two from
# that dataset would encode the confound rather than a real signal.
#
# w_asv=2.0 came from scripts/fit_fusion_weights.py's isolated ASV-axis fit
# (real anchor vs. real other-speaker recordings), capped well below the raw
# MLE — see that script's printed caveat for why the raw number isn't
# trustworthy at this sample size.
#
# w_cm did NOT end up near that script's fitted value. The CM axis was fit
# on synthetic_speech()/synthetic_spoofed() clips, which separate very
# cleanly under WavLM's anomaly heuristic — but the actual demo clips (real
# recordings) showed the heuristic reads a higher baseline "anomaly" on real
# genuine speech than the synthetic generator does, so the synthetic fit's
# weight pushed genuine real clips into CHALLENGE. w_cm=1.5 (below even the
# stub's 2.4) was reached by checking against scripts/integration_check.py's
# tier gate directly, not by the isolated synthetic fit — an honest sign
# that a synthetic-only calibration set doesn't fully transfer to real audio
# for this detector. See docs/archive/AUDIT_REPORT.md for the residual gap this leaves:
# demo_02 (degraded copy) and demo_03 (splice) still land one tier hotter
# than their hand-labeled expectation, because a real anomaly-sensitive
# detector plausibly does react somewhat to real distortion/splice
# artifacts — a legitimate difference from the stub, not necessarily a
# miscalibration, but not validated against real labeled data either.
_REAL_BACKEND_OVERRIDES = {
    "w_cm": 1.5,
    "w_asv": 2.0,
}


@dataclass(frozen=True)
class AuditConfig:
    # Durable JSONL path for the hash-chained audit log. Empty disables
    # persistence (in-memory ring only). The default lives under the repo's
    # data/ dir and is gitignored — it is a runtime artifact, not source.
    path: str = _env(
        "VR_AUDIT_PATH",
        str(Path(__file__).resolve().parent.parent.parent / "data" / "audit_log.jsonl"),
    )
    max_events: int = _env_int("VR_AUDIT_MAX_EVENTS", 10_000)


@dataclass(frozen=True)
class ServerConfig:
    host: str = _env("VR_HOST", "127.0.0.1")
    port: int = _env_int("VR_PORT", 8000)
    # Comma-separated allowlist for browser dashboard origins.
    cors_origins: str = _env(
        "VR_CORS_ORIGINS",
        "http://localhost:3001,http://127.0.0.1:3001",
    )


@dataclass(frozen=True)
class Settings:
    audio: AudioConfig = field(default_factory=AudioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    def __post_init__(self):
        preset = os.environ.get("VR_FUSION_PRESET")
        if preset in _DEMO_OVERRIDES:
            object.__setattr__(self, "fusion", FusionConfig(**_DEMO_OVERRIDES[preset]))
            return

        real_backend_active = (
            os.environ.get("VR_ASV_BACKEND", "stub").lower() == "ecapa"
            or os.environ.get("VR_CM_BACKEND", "stub").lower() == "wavlm"
        )
        if real_backend_active:
            # An explicit VR_FUSION_W_CM / VR_FUSION_W_ASV env var always wins —
            # this preset only fills in the ones the user didn't set themselves.
            overrides = {
                k: v
                for k, v in _REAL_BACKEND_OVERRIDES.items()
                if os.environ.get(f"VR_FUSION_W_{k[2:].upper()}") is None
            }
            if overrides:
                object.__setattr__(
                    self, "fusion", FusionConfig(**{**self.fusion.__dict__, **overrides})
                )


settings = Settings()
