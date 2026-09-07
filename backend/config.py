"""
VaniRakshak — Backend Configuration

Loads settings from environment variables / .env file.
Never hardcodes secrets.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


# Locate project root .env
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # --- Tavily ---
    tavily_api_key: str = Field(
        default="",
        description="Tavily API key for web evidence retrieval.",
    )

    # --- Verification tunables ---
    max_tavily_results: int = Field(
        default=5,
        description="Maximum search results to request from Tavily per query.",
    )
    tavily_timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for each Tavily search request.",
    )
    tavily_search_depth: str = Field(
        default="basic",
        description="Tavily search depth: basic | advanced.",
    )

    # --- Risk engine ---
    risk_threshold_low: float = 0.1
    risk_threshold_medium: float = 0.3
    risk_threshold_high: float = 0.6

    model_config = {
        "env_file": str(_ENV_PATH),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
