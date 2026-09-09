"""DPDP Act 2023 compliant feature-only audit log.

We log:
  * session_id, caller_id, claimed_identity
  * timestamp (UTC ISO 8601)
  * per-window SCALAR features (no raw audio, no invertible embeddings)
  * risk score, tier, and a SHA-256 "receipt hash" that chains each
    event to the previous one for tamper evidence.

The actual log lives in an in-memory ring by default. The 73-hour
hackathon prototype keeps the last ``max_events`` events and exposes
``GET /v1/audit`` to the dashboard.

Pass ``path`` (a JSONL file) to durably persist every appended event: each
event is appended as one canonical JSON line, and an ``AuditLog`` opened on
an existing file reloads it and re-verifies the chain. The in-memory ring
stays a capped hot window; the file is the durable record (rotate it with
normal log tooling — the process never truncates it).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Union


@dataclass
class AuditEvent:
    session_id: str
    caller_id: str
    claimed_identity: str
    timestamp: float
    risk_score: float
    tier: str
    features: Dict[str, float]
    receipt_hash: str = ""
    prev_hash: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))
        return d


class AuditLog:
    """Append-only, hash-chained audit log.

    Each event's ``receipt_hash`` is ``SHA-256(prev_hash || canonical_json(event_without_hashes))``.
    The chain is verifiable offline: any tampering breaks the chain.
    """

    def __init__(self, max_events: int = 10_000, path: Optional[Union[str, Path]] = None) -> None:
        self._events: Deque[AuditEvent] = deque(maxlen=max_events)
        self._path: Optional[Path] = Path(path) if path else None
        if self._path is not None:
            self._load()

    # -- persistence ----------------------------------------------------
    def _event_to_line(self, ev: AuditEvent) -> str:
        return json.dumps(asdict(ev), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _event_from_dict(d: Dict[str, Any]) -> AuditEvent:
        return AuditEvent(
            session_id=str(d.get("session_id", "")),
            caller_id=str(d.get("caller_id", "")),
            claimed_identity=str(d.get("claimed_identity", "")),
            timestamp=float(d.get("timestamp", 0.0)),
            risk_score=float(d.get("risk_score", 0.0)),
            tier=str(d.get("tier", "")),
            features={k: float(v) for k, v in dict(d.get("features", {})).items()},
            receipt_hash=str(d.get("receipt_hash", "")),
            prev_hash=str(d.get("prev_hash", "")),
        )

    def _save_event(self, ev: AuditEvent) -> None:
        """Best-effort durable append. A logging failure must never break
        the verdict path, so errors are reported to stderr and swallowed."""
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(self._event_to_line(ev) + "\n")
        except OSError as e:
            print(f"audit persistence write failed ({self._path}): {e}", file=sys.stderr)

    def _load(self) -> None:
        """Reload a previously persisted chain. Corrupt lines are skipped
        (counted on stderr); a broken hash link after load surfaces through
        the existing ``verify_chain()`` / ``GET /v1/audit`` reporting."""
        assert self._path is not None
        if not self._path.is_file():
            return
        skipped = 0
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._events.append(self._event_from_dict(json.loads(line)))
                    except (ValueError, TypeError, AttributeError):
                        skipped += 1
        except OSError as e:
            print(f"audit persistence read failed ({self._path}): {e}", file=sys.stderr)
            return
        if skipped:
            print(f"audit persistence: skipped {skipped} corrupt line(s) in {self._path}", file=sys.stderr)
        if self._events and not self.verify_chain():
            print(
                f"audit persistence: WARNING chain in {self._path} does not "
                f"verify — possible tampering or partial write",
                file=sys.stderr,
            )

    @property
    def _last_hash(self) -> str:
        return self._events[-1].receipt_hash if self._events else ""

    def append(
        self,
        *,
        session_id: str,
        caller_id: str,
        claimed_identity: str,
        risk_score: float,
        tier: str,
        features: Dict[str, float],
    ) -> AuditEvent:
        prev = self._last_hash
        payload = {
            "session_id": session_id,
            "caller_id": caller_id,
            "claimed_identity": claimed_identity,
            "ts": time.time(),
            "risk": round(float(risk_score), 4),
            "tier": tier,
            "features": {k: round(float(v), 6) for k, v in features.items()},
            "prev": prev,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        receipt = hashlib.sha256(canonical).hexdigest()
        ev = AuditEvent(
            session_id=session_id,
            caller_id=caller_id,
            claimed_identity=claimed_identity,
            timestamp=payload["ts"],
            risk_score=round(float(risk_score), 4),
            tier=tier,
            features=payload["features"],
            receipt_hash=receipt,
            prev_hash=prev,
        )
        self._events.append(ev)
        self._save_event(ev)
        return ev

    def tail(self, n: int = 50) -> List[AuditEvent]:
        return list(self._events)[-n:]

    def verify_chain(self) -> bool:
        """Re-compute the hash chain and return True iff no tampering occurred."""
        prev = ""
        for ev in self._events:
            payload = {
                "session_id": ev.session_id,
                "caller_id": ev.caller_id,
                "claimed_identity": ev.claimed_identity,
                "ts": ev.timestamp,
                "risk": ev.risk_score,
                "tier": ev.tier,
                "features": {k: round(float(v), 6) for k, v in ev.features.items()},
                "prev": prev,
            }
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            expected = hashlib.sha256(canonical).hexdigest()
            if expected != ev.receipt_hash or ev.prev_hash != prev:
                return False
            prev = ev.receipt_hash
        return True
