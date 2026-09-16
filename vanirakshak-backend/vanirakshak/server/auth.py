"""Authentication and rate-limiting for VaniRakshak API endpoints.

Implements G5 specifications (Product P0):
  - Bearer / X-API-Key credentials on REST and WS
  - Privilege separation: admin key required for POST /v1/enroll and GET /v1/audit
  - In-memory sliding-window rate limiting for /analyze and /enroll
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import HTTPException, Request

from ..config import settings


class SlidingWindowRateLimiter:
    """Thread-safe in-memory rate limiter using a sliding 60-second window."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._records: Dict[str, Deque[float]] = defaultdict(deque)

    def is_allowed(self, key: str, max_requests: int) -> bool:
        if max_requests <= 0:
            return True
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            timestamps = self._records[key]
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            if len(timestamps) >= max_requests:
                return False
            timestamps.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


RATE_LIMITER = SlidingWindowRateLimiter(window_seconds=60.0)


def extract_token(request: Request) -> Optional[str]:
    """Extract auth credential from Authorization header, X-API-Key, or query params."""
    auth_hdr = request.headers.get("Authorization")
    if auth_hdr:
        parts = auth_hdr.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
        elif len(parts) == 1:
            return parts[0]

    api_key_hdr = request.headers.get("X-API-Key")
    if api_key_hdr:
        return api_key_hdr.strip()

    token_param = request.query_params.get("token") or request.query_params.get("api_key")
    if token_param:
        return token_param.strip()

    return None


def verify_client_auth(request: Request) -> str:
    """Verify that the caller has a valid client or admin credential.

    Raises:
        HTTPException(401): If no token provided or token does not match known keys.
    """
    if not settings.auth.enabled:
        return "anonymous"

    token = extract_token(request)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    valid_keys = {settings.auth.api_key, settings.auth.admin_key}
    if token not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return "admin" if token == settings.auth.admin_key else "client"


def verify_admin_auth(request: Request) -> str:
    """Verify administrative privileges (required for POST /v1/enroll and GET /v1/audit).

    Raises:
        HTTPException(401): If no token provided or token unrecognized.
        HTTPException(403): If standard client key is provided instead of admin key.
    """
    if not settings.auth.enabled:
        return "admin"

    token = extract_token(request)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token == settings.auth.api_key:
        raise HTTPException(
            status_code=403,
            detail="Administrative privileges required for this endpoint.",
        )

    if token != settings.auth.admin_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return "admin"


def check_rate_limit(request: Request, route_name: str, max_requests: int) -> None:
    """Enforce per-client sliding window rate limits."""
    if not settings.auth.enabled or max_requests <= 0:
        return

    client_ip = request.client.host if request.client else "127.0.0.1"
    token = extract_token(request) or "anonymous"
    key = f"{route_name}:{token}:{client_ip}"

    if not RATE_LIMITER.is_allowed(key, max_requests):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {route_name} ({max_requests} requests/min limit). Retry after 60 seconds.",
            headers={"Retry-After": "60"},
        )


def verify_ws_token(token: Optional[str]) -> bool:
    """Validate token for WebSocket stream."""
    if not settings.auth.enabled:
        return True
    if not token:
        return False
    return token in (settings.auth.api_key, settings.auth.admin_key)
