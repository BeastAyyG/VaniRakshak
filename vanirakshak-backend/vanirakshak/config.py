"""Centralised configuration.

All environment-driven knobs live here. ``run.py`` and the FastAPI app
both import from this module so a single env var moves every layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
        "intercept": -2.4,
    },
}


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

    def __post_init__(self):
        preset = os.environ.get("VR_FUSION_PRESET")
        if preset in _DEMO_OVERRIDES:
            object.__setattr__(self, "fusion", FusionConfig(**_DEMO_OVERRIDES[preset]))


settings = Settings()
