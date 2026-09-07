"""
Evidence schema — normalized representation of web evidence.

Tavily (or any future provider) results are converted into this
project-level representation immediately after retrieval.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Classification of the evidence source."""

    WEB = "WEB"
    NEWS = "NEWS"
    OFFICIAL = "OFFICIAL"
    UNKNOWN = "UNKNOWN"


class Evidence(BaseModel):
    """A single piece of web evidence used for claim verification."""

    title: str = Field(
        ...,
        description="Title of the source page / article.",
    )
    url: str = Field(
        ...,
        description="URL of the source.",
    )
    content: str = Field(
        ...,
        description="Relevant text excerpt from the source.",
    )
    source_type: SourceType = Field(
        default=SourceType.WEB,
        description="Classification of the source.",
    )
    retrieved_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the evidence was retrieved.",
    )
    relevance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Relevance score from the search provider, if available.",
    )
