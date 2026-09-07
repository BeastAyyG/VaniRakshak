"""
VaniRakshak — FastAPI Application Entry Point

Recorded-audio forensic prototype.
This module provides the factual claim verification API.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.claims import router as claims_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VaniRakshak — Factual Claim Verification",
    description=(
        "Detect factual claims in recorded conversation transcripts, "
        "verify them against current web evidence, and return "
        "explainable verification results with risk contributions."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permissive for development, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(claims_router)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/health",
    tags=["health"],
    summary="Health check",
)
async def health() -> dict:
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "service": "vanirakshak-claim-verification",
        "version": "0.1.0",
    }
