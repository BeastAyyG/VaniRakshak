"""Pytest configuration and shared fixtures for VaniRakshak E2E test suite."""

from __future__ import annotations

import io
import json
import math
import os
import struct
import sys
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Tuple

import numpy as np
import pytest

# Ensure repository root and backend package are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = REPO_ROOT / "vanirakshak-backend"

for p in [str(REPO_ROOT), str(BACKEND_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

SAMPLE_RATE = 16000


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def backend_root() -> Path:
    return BACKEND_ROOT


@pytest.fixture
def pcm_generator() -> Callable[[float, float, float], bytes]:
    """Generate mono 16-bit PCM audio bytes at 16 kHz."""
    def _gen(seconds: float = 1.0, freq: float = 440.0, amp: float = 8000.0) -> bytes:
        frames = int(seconds * SAMPLE_RATE)
        return b"".join(
            struct.pack(
                "<h",
                int(amp * math.sin(2 * math.pi * freq * (i / SAMPLE_RATE))),
            )
            for i in range(frames)
        )
    return _gen


@pytest.fixture
def wav_generator(pcm_generator) -> Callable[[float, float, float], bytes]:
    """Generate in-memory WAV byte stream."""
    def _gen(seconds: float = 1.0, freq: float = 440.0, amp: float = 8000.0) -> bytes:
        pcm = pcm_generator(seconds=seconds, freq=freq, amp=amp)
        bio = io.BytesIO()
        with wave.open(bio, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        return bio.getvalue()
    return _gen


@pytest.fixture
def make_valid_checkpoint() -> Callable[..., Dict[str, Any]]:
    """Produce a dictionary matching the AASIST-L checkpoint contract."""
    def _make(
        architecture: str = "aasist-l",
        frontend: str = "microsoft/wavlm-base-plus",
        eer_threshold: float = 0.25,
        dev_eer: float = 0.031,
        dev_min_dcf: float = 0.087,
        state_dict: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if state_dict is None:
            # Minimal state dict placeholder or real weights
            state_dict = {"proj.weight": np.zeros((64, 768), dtype=np.float32)}
        return {
            "architecture": architecture,
            "frontend": frontend,
            "sample_rate": 16000,
            "window_s": 4.0,
            "state_dict": state_dict,
            "eer_threshold": eer_threshold,
            "dev_eer": dev_eer,
            "dev_min_dcf": dev_min_dcf,
        }
    return _make


@pytest.fixture
def test_client() -> Generator:
    """FastAPI TestClient with isolated settings."""
    from fastapi.testclient import TestClient
    from vanirakshak.server.app import app
    client = TestClient(app)
    yield client
