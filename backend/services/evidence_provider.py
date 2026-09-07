"""
Evidence Provider — abstract interface + Tavily implementation.

The claim verification engine depends on WebEvidenceProvider (Protocol),
NOT directly on Tavily.  This makes it trivial to swap providers later.

TavilyEvidenceProvider wraps the already-installed tavily-python SDK.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from backend.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider-agnostic result
# ---------------------------------------------------------------------------

@dataclass
class WebResult:
    """Raw result from a web evidence provider (before normalization)."""

    title: str
    url: str
    content: str
    score: float | None = None  # relevance score if provided
    raw: dict | None = None     # original provider response for debugging


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

@runtime_checkable
class WebEvidenceProvider(Protocol):
    """
    Abstract web evidence provider.

    Any implementation must expose a synchronous `search` method.
    (Async wrapper can be built on top if needed.)
    """

    def search(
        self,
        query: str,
        topic: str | None = None,
        time_range: str | None = None,
        max_results: int = 5,
    ) -> list[WebResult]:
        ...


# ---------------------------------------------------------------------------
# Tavily implementation
# ---------------------------------------------------------------------------

class TavilyEvidenceProvider:
    """
    Concrete WebEvidenceProvider using the Tavily Python SDK.

    Loads API key from environment via Settings.
    Never hardcodes or logs the key.
    """

    def __init__(self, api_key: str | None = None):
        from tavily import TavilyClient

        key = api_key or get_settings().tavily_api_key
        if not key:
            raise ValueError(
                "TAVILY_API_KEY is not set. "
                "Set it in .env or as an environment variable."
            )
        self._client = TavilyClient(api_key=key)
        self._settings = get_settings()

    def search(
        self,
        query: str,
        topic: str | None = None,
        time_range: str | None = None,
        max_results: int | None = None,
    ) -> list[WebResult]:
        """
        Search using Tavily and return normalized WebResult list.

        Handles: timeout, API errors, rate limits, empty results.
        NEVER converts failures into CONTRADICTED.
        """
        max_results = max_results or self._settings.max_tavily_results

        try:
            response = self._client.search(
                query=query,
                search_depth=self._settings.tavily_search_depth,
                topic=topic or "general",
                time_range=time_range,
                max_results=max_results,
                include_answer=True,
                timeout=self._settings.tavily_timeout,
            )
        except Exception as exc:
            logger.error("Tavily search failed for query '%s': %s", query, exc)
            raise EvidenceProviderError(
                f"Web search failed: {type(exc).__name__}: {exc}"
            ) from exc

        # Parse results
        results: list[WebResult] = []
        raw_results = response.get("results", [])

        for item in raw_results:
            results.append(
                WebResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score"),
                    raw=item,
                )
            )

        # Also capture the synthesized answer if available
        answer = response.get("answer")
        if answer and not results:
            results.append(
                WebResult(
                    title="Tavily Summary",
                    url="",
                    content=answer,
                    score=None,
                    raw={"answer": answer},
                )
            )

        return results


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class EvidenceProviderError(Exception):
    """Raised when the evidence provider fails."""
    pass
