"""Persistent, encrypted enrolment store (docs/15 N4, closes G-D).

Enrolments used to live only in the ASV model's in-process dict, so a restart
erased every voiceprint and per-user calibration. This keeps one record per
user on disk:

* ``embedding`` -- the voiceprint vector (never audio),
* ``asv_backend`` -- which model produced it; a record from another backend is
  not loaded, since a stub projection is meaningless to ECAPA and vice versa,
* ``calibration`` -- the per-user ASV score calibration, if any,
* ``qa`` -- the enrolment QA summary (voiced seconds, SNR, CM LLR),
* ``consent_ref`` -- the caller's reference to the consent record,
* ``created_at``.

Each record is sealed with AES-256-GCM under ``VR_ENROL_STORE_KEY`` and written
atomically. File names are a keyed hash of the user id, so the directory
listing does not reveal who is enrolled. Erasure unlinks the file. Only
ciphertext was ever written, so any blocks left behind on disk are unreadable
without the key.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_SUFFIX = ".vre"
_AAD = b"vanirakshak-enrolment-v1"


class EnrolmentStoreError(RuntimeError):
    pass


@dataclass
class EnrolmentRecord:
    user_id: str
    embedding: np.ndarray
    asv_backend: str
    calibration: Optional[Dict[str, float]] = None
    qa: Dict[str, object] = field(default_factory=dict)
    consent_ref: Optional[str] = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_json(self) -> bytes:
        return json.dumps({
            "user_id": self.user_id,
            "embedding": base64.b64encode(np.asarray(self.embedding, dtype=np.float32).tobytes()).decode(),
            "embedding_dim": int(np.asarray(self.embedding).size),
            "asv_backend": self.asv_backend,
            "calibration": self.calibration,
            "qa": self.qa,
            "consent_ref": self.consent_ref,
            "created_at": self.created_at,
        }).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "EnrolmentRecord":
        d = json.loads(raw.decode("utf-8"))
        emb = np.frombuffer(base64.b64decode(d["embedding"]), dtype=np.float32).copy()
        return cls(
            user_id=d["user_id"],
            embedding=emb,
            asv_backend=d["asv_backend"],
            calibration=d.get("calibration"),
            qa=d.get("qa") or {},
            consent_ref=d.get("consent_ref"),
            created_at=d.get("created_at", ""),
        )


def parse_key(raw: str) -> bytes:
    """``VR_ENROL_STORE_KEY``: 32 bytes as base64 (urlsafe or standard) or hex."""
    raw = raw.strip()
    for decode in (lambda s: bytes.fromhex(s), base64.urlsafe_b64decode, base64.b64decode):
        try:
            key = decode(raw)
        except Exception:
            continue
        if len(key) == 32:
            return key
    raise EnrolmentStoreError("VR_ENROL_STORE_KEY must be 32 bytes, base64- or hex-encoded")


class EnrolmentStore:
    """One AES-GCM-sealed file per enrolled user under ``root``."""

    def __init__(self, root: Path, key: bytes) -> None:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as e:  # pragma: no cover - dependency is in requirements.txt
            raise EnrolmentStoreError(
                "the enrolment store needs the 'cryptography' package"
            ) from e
        if len(key) != 32:
            raise EnrolmentStoreError("enrolment store key must be 32 bytes")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._aead = AESGCM(key)
        # Separate subkey for file names, so the name never reuses the AEAD key.
        self._name_key = hmac.new(key, b"file-names", hashlib.sha256).digest()

    def _path(self, user_id: str) -> Path:
        name = hmac.new(self._name_key, user_id.encode("utf-8"), hashlib.sha256).hexdigest()
        return self.root / f"{name}{_SUFFIX}"

    def save(self, record: EnrolmentRecord) -> None:
        nonce = os.urandom(12)
        blob = nonce + self._aead.encrypt(nonce, record.to_json(), _AAD)
        path = self._path(record.user_id)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read(self, path: Path) -> EnrolmentRecord:
        blob = path.read_bytes()
        return EnrolmentRecord.from_json(self._aead.decrypt(blob[:12], blob[12:], _AAD))

    def load(self, user_id: str) -> Optional[EnrolmentRecord]:
        path = self._path(user_id)
        return self._read(path) if path.exists() else None

    def contains(self, user_id: str) -> bool:
        return self._path(user_id).exists()

    def erase(self, user_id: str) -> bool:
        """Delete the user's record. True if a record existed and is now gone."""
        path = self._path(user_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return not path.exists()

    def load_all(self) -> "LoadReport":
        report = LoadReport()
        for path in sorted(self.root.glob(f"*{_SUFFIX}")):
            try:
                report.records.append(self._read(path))
            except Exception as e:  # wrong key, truncated file, bad JSON
                report.unreadable.append(f"{path.name}: {type(e).__name__}")
        return report


@dataclass
class LoadReport:
    records: List[EnrolmentRecord] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)


def store_from_env(path: str, key: str) -> Optional[EnrolmentStore]:
    """``None`` when no path is set (in-memory only, the previous behaviour).
    A path without a key refuses to start: plaintext voiceprints are not an
    option."""
    if not path:
        return None
    if not key:
        raise EnrolmentStoreError("VR_ENROL_STORE_PATH is set but VR_ENROL_STORE_KEY is not")
    return EnrolmentStore(Path(path), parse_key(key))
