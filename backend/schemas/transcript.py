"""
Transcript schema — represents a timestamped segment of speech.

Every segment preserves start/end times so the frontend can locate
claims on the audio waveform/timeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A single timestamped segment from a speech-to-text transcription."""

    start: float = Field(
        ...,
        ge=0.0,
        description="Start time in seconds within the audio.",
    )
    end: float = Field(
        ...,
        ge=0.0,
        description="End time in seconds within the audio.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Transcribed text for this segment.",
    )
    speaker: str | None = Field(
        default=None,
        description="Optional speaker label (e.g. 'caller', 'agent').",
    )
