"""API-key authentication, scopes, and rate limiting.

Every REST route and the WebSocket accepted unauthenticated connections until
this module existed: anyone who could reach the port could enroll a voiceprint
under someone else's ``user_id``, stream audio, read the audit chain, or
exhaust the box (docs/09-post-cm-roadmap.md P0 #1).

**Posture.** Disabled when ``VR_API_KEYS`` is unset -- the same honest-by-
default stance the detector backends take -- so the demo, `npm run dev:all` and
`scripts/integration_check.py` keep working untouched. The server logs a loud
warning at startup in that state. Set ``VR_API_KEYS`` and every protected route
enforces; set ``VR_REQUIRE_AUTH=1`` and the server refuses to start without
keys, which is what a deployment should do.

**Key format** (``VR_API_KEYS``, comma-separated):

    key_id:secret[:scopes[:users]]

``scopes`` is ``|``-separated from {stream, analyze, enroll, audit}, or ``*``
for all; it defaults to ``stream|analyze|enroll`` -- note ``audit`` is NOT in
the default, because the audit chain is a log of who was scored and when.
``users`` is a ``|``-separated allowlist of ``user_id`` values the key may
enroll or claim, or ``*`` (the default). Restricting it is what stops one
client enrolling a voiceprint under another client's identity, which is an
identity attack rather than a misconfiguration.

Secrets are compared with :func:`hmac.compare_digest` and never logged.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ALL_SCOPES = ("stream", "analyze", "enroll", "audit")
_DEFAULT_SCOPES = ("stream", "analyze", "enroll")


@dataclass(frozen=True)
class ApiKey:
    key_id: str
    secret: str
    scopes: Tuple[str, ...]
    users: Tuple[str, ...]  # ("*",) means any

    def may(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes

    def may_act_as(self, user_id: str) -> bool:
        if "*" in self.users:
            return True
        return bool(user_id) and user_id in self.users


class AuthError(Exception):
    """Raised with the HTTP status the caller should receive."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _parse_keys(raw: str) -> Dict[str, ApiKey]:
    """Parse VR_API_KEYS. A malformed entry is skipped with a warning rather
    than silently widening access."""
    keys: Dict[str, ApiKey] = {}
    for entry in (e.strip() for e in raw.split(",")):
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
            logger.warning("VR_API_KEYS: skipping malformed entry (need key_id:secret)")
            continue
        key_id, secret = parts[0].strip(), parts[1].strip()
        raw_scopes = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        raw_users = parts[3].strip() if len(parts) > 3 and parts[3].strip() else "*"

        if raw_scopes == "*":
            scopes: Tuple[str, ...] = ("*",)
        elif raw_scopes:
            scopes = tuple(s for s in (x.strip() for x in raw_scopes.split("|")) if s)
            unknown = [s for s in scopes if s not in ALL_SCOPES]
            if unknown:
                logger.warning("VR_API_KEYS: key %s lists unknown scope(s) %s", key_id, unknown)
        else:
            scopes = _DEFAULT_SCOPES

        users = ("*",) if raw_users == "*" else tuple(
            u for u in (x.strip() for x in raw_users.split("|")) if u
        )
        if key_id in keys:
            logger.warning("VR_API_KEYS: duplicate key_id %s; last one wins", key_id)
        keys[key_id] = ApiKey(key_id=key_id, secret=secret, scopes=scopes, users=users)
    return keys


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Token bucket, per (key_id, route), in-process.

    In-process is the honest scope: it protects a single box from one noisy
    client, and it does not survive a restart or coordinate across replicas.
    A deployment behind more than one worker needs a shared store; that is a
    deployment decision, not something this file can fake.
    """

    def __init__(self) -> None:
        self._buckets: Dict[Tuple[str, str], _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key_id: str, route: str, per_minute: int) -> None:
        if per_minute <= 0:
            return
        now = time.monotonic()
        ident = (key_id, route)
        with self._lock:
            b = self._buckets.get(ident)
            if b is None:
                self._buckets[ident] = _Bucket(tokens=per_minute - 1.0, updated=now)
                return
            # Refill continuously rather than in fixed windows, so a client
            # cannot burst 2x the limit across a window boundary.
            b.tokens = min(float(per_minute), b.tokens + (now - b.updated) * per_minute / 60.0)
            b.updated = now
            if b.tokens < 1.0:
                raise AuthError(429, f"rate limit exceeded for {route} ({per_minute}/min)")
            b.tokens -= 1.0

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


class Authenticator:
    """Holds the configured keys and answers "may this caller do this?"."""

    def __init__(
        self,
        raw_keys: Optional[str] = None,
        require_auth: Optional[bool] = None,
        rate_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        raw = os.environ.get("VR_API_KEYS", "") if raw_keys is None else raw_keys
        self.keys = _parse_keys(raw)
        if require_auth is None:
            require_auth = os.environ.get("VR_REQUIRE_AUTH", "").strip().lower() in (
                "1", "true", "yes", "on",
            )
        self.require_auth = bool(require_auth)
        self.rate_limits = rate_limits or {}
        self.limiter = RateLimiter()

        if not self.keys and self.require_auth:
            raise RuntimeError(
                "VR_REQUIRE_AUTH is set but VR_API_KEYS is empty: refusing to start "
                "an open server. Set VR_API_KEYS=key_id:secret[:scopes[:users]]."
            )

    @property
    def enabled(self) -> bool:
        return bool(self.keys)

    def warn_if_open(self) -> None:
        if not self.enabled:
            logger.warning(
                "AUTH DISABLED - every route is open, including POST /v1/enroll "
                "and GET /v1/audit. Fine for the demo, never for a deployment. "
                "Set VR_API_KEYS to enable, VR_REQUIRE_AUTH=1 to make it mandatory."
            )

    def authenticate(self, presented: Optional[str]) -> Optional[ApiKey]:
        """Resolve a presented secret to a key. ``None`` when auth is disabled.

        Raises AuthError(401) when auth is on and the secret is absent or wrong.
        """
        if not self.enabled:
            return None
        if not presented:
            raise AuthError(401, "missing API key (Authorization: Bearer <key>, "
                                 "X-API-Key header, or 'api_key' in the WS handshake)")
        # Constant-time against every configured secret: comparing only against
        # a key the caller names would leak which key ids exist.
        matched: Optional[ApiKey] = None
        for key in self.keys.values():
            if hmac.compare_digest(key.secret, presented):
                matched = key
        if matched is None:
            raise AuthError(401, "invalid API key")
        return matched

    def authorize(
        self,
        key: Optional[ApiKey],
        scope: str,
        route: str,
        acting_as: Optional[str] = None,
    ) -> None:
        """Check scope, per-user allowlist and rate limit. No-op when disabled."""
        if not self.enabled:
            return
        assert key is not None  # authenticate() raises before this when enabled
        if not key.may(scope):
            raise AuthError(403, f"key '{key.key_id}' lacks the '{scope}' scope")
        if acting_as is not None and not key.may_act_as(acting_as):
            raise AuthError(
                403,
                f"key '{key.key_id}' is not permitted to act as user '{acting_as}'",
            )
        limit = self.rate_limits.get(route, 0)
        if limit:
            self.limiter.check(key.key_id, route, limit)


def extract_secret(headers) -> Optional[str]:
    """Pull the presented secret from ``Authorization: Bearer`` or ``X-API-Key``.

    Accepts anything with a case-insensitive ``.get`` (Starlette Headers, dict).
    """
    try:
        auth = headers.get("authorization") or headers.get("Authorization")
    except Exception:
        auth = None
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return auth.strip()
    try:
        return headers.get("x-api-key") or headers.get("X-API-Key")
    except Exception:
        return None
