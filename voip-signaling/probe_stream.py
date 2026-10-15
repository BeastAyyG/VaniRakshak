#!/usr/bin/env python3
"""Ground-truth probe of the scorer /ws/stream frame schema (Phase 2, Part B).

Reproduces the Android VaniStreamingClient wire format exactly:
  1. JSON text handshake first
  2. binary int16 LE mono PCM @16kHz frames
  3. records every server message verbatim

Also verifies that a JSON text frame sent AFTER the handshake is ignored
(vs. raw binary being the only accepted audio format).

Usage: python probe_stream.py [ws_url]
Default: ws://127.0.0.1:8000/ws/stream
"""

import asyncio
import json
import math
import struct
import sys
import uuid

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000/ws/stream"
SR = 16000


def sine_pcm(seconds, freq=440.0, amp=8000):
    n = int(SR * seconds)
    return struct.pack("<%dh" % n,
                       *[int(amp * math.sin(2 * math.pi * freq * i / SR))
                         for i in range(n)])


async def recv_json(ws, timeout):
    raw = await asyncio.wait_for(asyncio.ensure_future(ws.recv()), timeout)
    assert isinstance(raw, str), "expected text frame, got %r" % type(raw)
    return json.loads(raw)


async def main():
    ws = await websockets.connect(URL, max_size=10 << 20)
    print("connected to", URL, flush=True)

    # 1. handshake (same shape as VaniStreamingClient.kt; no api_key:
    #    VR_API_KEYS is unset on lab-beast so auth is disabled)
    sid = "probe-" + uuid.uuid4().hex[:8]
    handshake = {
        "session_id": sid,
        "sample_rate": SR,
        "caller_id": "probe",
        "claimed_identity": "",
        "transaction_value_inr": 0,
        "origin_country": "IN",
        "prior_fraud_score": 0,
    }
    print("C->S handshake:", json.dumps(handshake), flush=True)
    await ws.send(json.dumps(handshake))
    ack = await recv_json(ws, 10)
    print("S->C:", json.dumps(ack), flush=True)
    assert ack.get("type") == "session_ack", "expected session_ack, got %r" % ack

    # 2. binary PCM: 7.5 s of 440 Hz sine @16kHz int16 LE, sent in 100 ms chunks.
    #    Server windows are 4.04 s with 1.0 s hop, so this yields 3 verdicts.
    pcm = sine_pcm(7.5)
    chunk = SR // 10 * 2
    print("C->S: %d binary PCM frames (%d bytes each, %d bytes total)"
          % (len(range(0, len(pcm), chunk)), chunk, len(pcm)), flush=True)
    verdicts = []
    for i in range(0, len(pcm), chunk):
        await ws.send(pcm[i:i + chunk])
        while True:  # drain anything the server already emitted
            try:
                msg = await recv_json(ws, 0.2)
            except asyncio.TimeoutError:
                break
            verdicts.append(msg)
            print("S->C verdict #%d: %s" % (len(verdicts), json.dumps(msg)), flush=True)

    # 3. final drain: GPU scoring lags behind the sends
    try:
        while len(verdicts) < 3:
            msg = await recv_json(ws, 30)
            verdicts.append(msg)
            print("S->C verdict #%d: %s" % (len(verdicts), json.dumps(msg)), flush=True)
    except asyncio.TimeoutError:
        print("drain timeout; verdicts received:", len(verdicts), flush=True)
    print("TOTAL_VERDICTS:", len(verdicts), flush=True)

    # 4. variant check: JSON text audio frame after the handshake
    await ws.send(json.dumps({"type": "audio", "pcm_b64": "AAAA"}))
    try:
        msg = await recv_json(ws, 2)
        print("S->C after JSON text frame:", json.dumps(msg), flush=True)
    except asyncio.TimeoutError:
        print("S->C after JSON text frame: nothing (text frames ignored)", flush=True)

    await ws.close()
    print("PROBE_DONE verdicts=%d" % len(verdicts), flush=True)
    return 0 if verdicts else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
