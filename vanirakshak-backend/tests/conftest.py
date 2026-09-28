"""Shared fixtures for the backend tests."""

from __future__ import annotations

import pytest


class NoEngineTranscriber:
    """Stands in for the challenge ASR engine: never loads a model."""

    def transcribe(self, pcm_int16, sample_rate):  # noqa: ARG002
        return None, "no ASR engine in unit tests"


@pytest.fixture(autouse=True)
def _no_real_challenge_asr(monkeypatch):
    """Challenge grading transcribes response audio (docs/15 N2). Unit tests
    must not download or run a real model, so the session default is replaced;
    a test that needs a transcript injects its own ``challenge_transcriber``."""
    import vanirakshak.session as session_module

    monkeypatch.setattr(session_module, "CHALLENGE_TRANSCRIBER", NoEngineTranscriber())
