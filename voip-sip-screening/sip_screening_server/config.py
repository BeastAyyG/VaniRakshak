"""Configuration via environment variables. No secrets are hardcoded anywhere.

Copy .env.example to .env and fill in values for the provider you use.
Everything runs without provider credentials (health endpoint, state
machine, mock bridge), so this module never fails on missing keys.
"""

from __future__ import annotations

import os


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class Settings:
    # Scorer backend on lab-beast (existing FastAPI service).
    SCORER_HOST: str = _get("SCORER_HOST", "100.89.248.108")
    SCORER_PORT: int = int(_get("SCORER_PORT", "8000"))
    SCORER_WS_PATH: str = _get("SCORER_WS_PATH", "/ws/stream")

    # Verdict policy. OPERATING_THRESHOLD is the scorer's verified operating
    # point (LLR 21.9308). CHALLENGE_MARGIN and the timing values are
    # proposals to tune with live test calls (see design doc section 5).
    OPERATING_THRESHOLD: float = _get_float("OPERATING_THRESHOLD", 21.9308)
    CHALLENGE_MARGIN: float = _get_float("CHALLENGE_MARGIN", 10.0)
    MIN_SPEECH_SECONDS: float = _get_float("MIN_SPEECH_SECONDS", 4.0)
    MAX_SCREEN_SECONDS: float = _get_float("MAX_SCREEN_SECONDS", 20.0)
    VERDICT_TIMEOUT_SECONDS: float = _get_float("VERDICT_TIMEOUT_SECONDS", 10.0)
    CHALLENGE_AGGREGATION_FRAMES: int = int(_get("CHALLENGE_AGGREGATION_FRAMES", "5"))

    # This server's public base URL, used to build WSS media URLs returned
    # to providers. Must be https/wss in production.
    PUBLIC_BASE_URL: str = _get("PUBLIC_BASE_URL", "https://screen.example.com")

    # Active provider: "exotel" | "plivo" | "twilio".
    PROVIDER: str = _get("PROVIDER", "exotel").lower()

    # Provider credentials (all optional at import time; required only when
    # the matching provider is active and webhooks arrive).
    EXOTEL_ACCOUNT_SID: str = _get("EXOTEL_ACCOUNT_SID")
    EXOTEL_API_KEY: str = _get("EXOTEL_API_KEY")
    EXOTEL_API_TOKEN: str = _get("EXOTEL_API_TOKEN")
    PLIVO_AUTH_ID: str = _get("PLIVO_AUTH_ID")
    PLIVO_AUTH_TOKEN: str = _get("PLIVO_AUTH_TOKEN")
    TWILIO_ACCOUNT_SID: str = _get("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: str = _get("TWILIO_AUTH_TOKEN")

    # Number to bridge allowed calls to (user's phone), E.164.
    USER_BRIDGE_NUMBER: str = _get("USER_BRIDGE_NUMBER")

    # Screening notice text version, logged per call as the consent record.
    NOTICE_VERSION: str = _get("NOTICE_VERSION", "v1-en")

    @property
    def scorer_ws_url(self) -> str:
        return f"ws://{self.SCORER_HOST}:{self.SCORER_PORT}{self.SCORER_WS_PATH}"

    @property
    def media_ws_base(self) -> str:
        base = self.PUBLIC_BASE_URL
        if base.startswith("https://"):
            return "wss://" + base[len("https://"):]
        if base.startswith("http://"):
            return "ws://" + base[len("http://"):]
        return base


settings = Settings()
