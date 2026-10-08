"""AudioBridge: pipe provider audio to the scorer and yield verdict frames.

Real parts:
  - chunking 16 kHz PCM into 1-second frames,
  - a simple energy-based voice gate (placeholder for a real VAD),
  - the async plumbing: connect to the scorer WebSocket, send chunks,
    receive frames.

Stubbed parts (clearly marked TBD):
  - the exact request/response frame schema of the scorer's /ws/stream
    endpoint on lab-beast. send_chunk() and _parse_verdict() carry the
    best-guess schema (raw PCM bytes out, JSON with an "llr" field back)
    and must be aligned with the real implementation before live use.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class VerdictFrame:
    llr: float
    scored_speech_seconds: float
    received_at: float


class ScorerUnreachable(Exception):
    """Raised when the scorer WebSocket cannot be opened or dies."""


class AudioBridge:
    """One bridge per call. Owns the scorer WebSocket for the call's life.

    Usage:
        bridge = AudioBridge(scorer_ws_url)
        async for frame in bridge.pipe_to_scorer(pcm_iter):
            session.record_llr(frame.llr, frame.scored_speech_seconds)
    """

    CHUNK_SECONDS = 1.0
    SAMPLE_RATE = 16000
    BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16-bit mono

    # Naive energy gate: chunks quieter than this are treated as silence
    # and are neither sent nor counted as speech. Replace with a real VAD.
    SILENCE_RMS_THRESHOLD = 150.0

    def __init__(self, scorer_ws_url: str):
        self.scorer_ws_url = scorer_ws_url
        self._ws = None
        self._chunks_sent = 0

    async def pipe_to_scorer(
        self, pcm_iter: AsyncIterator[bytes]
    ) -> AsyncIterator[VerdictFrame]:
        """Connect to ws://SCORER_HOST:8000/ws/stream and yield verdicts.

        pcm_iter yields 16 kHz 16-bit mono PCM chunks of any size; this
        method re-chunks to 1-second frames, gates silence, sends speech
        frames, and yields one VerdictFrame per scorer response.
        """
        try:
            import websockets  # local import: module stays importable without it
        except ImportError as exc:
            raise ScorerUnreachable(
                "the 'websockets' package is required for live scoring; "
                "install requirements.txt"
            ) from exc

        try:
            self._ws = await websockets.connect(self.scorer_ws_url)
        except Exception as exc:
            raise ScorerUnreachable(
                f"could not open scorer WebSocket {self.scorer_ws_url}: {exc}"
            ) from exc

        buffer = bytearray()
        try:
            async for pcm in pcm_iter:
                buffer += pcm
                while len(buffer) >= self.BYTES_PER_SECOND:
                    chunk = bytes(buffer[:self.BYTES_PER_SECOND])
                    del buffer[:self.BYTES_PER_SECOND]
                    if self._is_silence(chunk):
                        continue
                    frame = await self._send_chunk(chunk)
                    if frame is not None:
                        yield frame
        finally:
            await self._close()

    def _is_silence(self, chunk: bytes) -> bool:
        n = len(chunk) // 2
        if n == 0:
            return True
        total = 0
        for i in range(n):
            s = int.from_bytes(chunk[i * 2:(i + 1) * 2], "little", signed=True)
            total += s * s
        rms = (total / n) ** 0.5
        return rms < self.SILENCE_RMS_THRESHOLD

    async def _send_chunk(self, chunk: bytes) -> VerdictFrame | None:
        """Send one 1-second PCM chunk; return the parsed verdict frame.

        TBD: align this schema with the real /ws/stream implementation on
        lab-beast. Current assumption: raw PCM bytes are sent as a binary
        message and the server replies with a JSON text message carrying
        the LLR, e.g. {"llr": 24.7, ...}. If the real protocol differs
        (envelope JSON, base64, batched windows), change this method only.
        """
        assert self._ws is not None
        await self._ws.send(chunk)  # binary PCM frame
        try:
            reply = await asyncio.wait_for(self._ws.recv(), timeout=30)
        except asyncio.TimeoutError:
            return None
        return self._parse_verdict(reply)

    def _parse_verdict(self, reply: str | bytes) -> VerdictFrame | None:
        """TBD: confirm the real verdict-frame schema on lab-beast.

        Best guess: JSON text with an "llr" key. Returns None when the
        frame cannot be understood, so the caller can keep streaming.
        """
        try:
            text = reply.decode("utf-8") if isinstance(reply, bytes) else reply
            data = json.loads(text)
            llr = float(data["llr"])  # KeyError/ValueError -> None below
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        return VerdictFrame(llr=llr, scored_speech_seconds=self.CHUNK_SECONDS,
                            received_at=time.time())

    async def _close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
