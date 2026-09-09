# VaniRakshak — Third-Party Integration Guide

For mobile apps, PBX/telephony gateways, and any client that isn't the
bundled Next.js dashboard and wants to pipe live call audio (or a recorded
file) into VaniRakshak's forensic pipeline.

**This document summarizes; it does not redefine the contract.**
[`docs/PROTOCOL.md`](PROTOCOL.md) is the single source of truth for the
WebSocket wire format — read it alongside this guide, and trust it over this
one if they ever disagree (`scripts/check_docs.py` exists specifically to
catch that from happening). [`docs/API_SPEC.md`](API_SPEC.md) is the full
REST reference.

> **If you have an older copy of this guide** (or a doc titled
> `LIVE_STREAM_INTEGRATION_GUIDE.md` describing port `8084`,
> `/ws/live-call-stream`, `threat_score`, or a "Live Ground-Truth Web
> Evidence Search" stage): none of that exists in this codebase and never
> has. That guide described a different, aspirational design that was never
> implemented — the endpoints, message types, and fields below are what the
> server (`vanirakshak-backend/vanirakshak/server/app.py`) actually does.
> `docs/FLOWCHARTS_CORRECTED.md` explains why a web-search fact-checking
> stage specifically isn't part of this system's design, not just that it
> happens to be unbuilt.

## 1. What you're integrating with

A FastAPI service exposing:
- one WebSocket endpoint for live audio (`/ws/stream`, aliased as `/v1/stream`
  and `/api/v1/stream`) that streams back a risk verdict every ~1 second,
- REST endpoints for one-shot file analysis, speaker enrollment, and a
  identity-verification-only endpoint mobile clients can call without
  streaming at all.

Default port `8000`, host `127.0.0.1` (both overridable — `VR_PORT`,
`VR_HOST`). Default bind means the server only accepts local connections;
change `VR_HOST` and open a firewall rule if your PBX or mobile backend runs
elsewhere. There is no authentication on any endpoint — put this behind your
own gateway/VPN before exposing it beyond a trusted network.

## 2. The live-stream flow, end to end

1. Open a WebSocket to `ws://<host>:8000/ws/stream`.
2. **Send the JSON handshake as the very first frame — as text, not binary.**
   If the first frame is binary, the server closes the connection with code
   `1003` immediately. Minimal handshake:
   ```json
   { "sample_rate": 16000 }
   ```
   Every other field is optional; see PROTOCOL.md §1 for the full set
   (`caller_id`, `claimed_identity`, `transaction_value_inr`,
   `origin_country`, `prior_fraud_score`) and what each does to risk scoring.

    Supplying your own `session_id` is recommended, but no longer required.
    If you omit it, the server generates one internally (`ws-<8 hex chars>`)
    and returns it in the first message back — a `session_ack`
    (`{"type": "session_ack", "session_id": ..., "sample_rate": ...}`),
    which arrives before any verdict and carries no `tier`/`risk_score`, so
    verdict parsers skip it by shape. Adopt the acked id: §4 below needs a
    `session_id` you actually know to grade a challenge over REST. Full
    contract in PROTOCOL.md §1.
3. Stream raw 16-bit signed little-endian mono PCM as **binary** WebSocket
   frames, at the sample rate you declared. Any frame size works — the
   server buffers internally and only emits a verdict once it has
   accumulated a full 4.04-second window (then again every 1.0 second of new
   audio after that).
4. Read verdicts as they arrive — JSON text frames, one per completed window.
   Each carries `risk_score`, `tier` (`ALLOW`/`CHALLENGE`/`BLOCK`),
   `interlock_active`, and more — full schema in PROTOCOL.md §3. **Trust
   `tier` and `interlock_active` as authoritative** — don't re-derive your
   own tier from `risk_score` alone, because a deterministic override
   (`VR_IMPOSTOR_BLOCK_BELOW`) can force `BLOCK` independent of the numeric
   score.
5. On `tier: "CHALLENGE"`, the verdict also carries a `challenge` object
   (phrase + expected digits). Play or display `challenge.phrase` to the
   caller, capture their spoken response, and grade it — either by posting
   the transcribed text to `POST /v1/sessions/{sid}/challenge` (see §4), or
   locally if you don't have a session id handy (matching
   `challenge/engine.py`'s digit-order + 400–4000ms latency rule, PROTOCOL.md
   §3).
6. Close the socket when the call ends. There is no explicit "end of call"
   message — closing the WebSocket is the signal. Reconnecting starts a
   **new session**; the server does not resume mid-call state (PROTOCOL.md §6).

### Telephony audio realities

- If your PBX delivers G.711 (µ-law/A-law, 8 kHz) rather than 16-bit linear
  PCM at 16 kHz, **you must transcode before sending** — the server does not
  decode codec'd audio itself; it expects raw int16 samples already at the
  declared `sample_rate`. `vanirakshak/data/augment.py` has the G.711/AMR-WB
  encode/decode math this system uses on its own evaluation side, if you need
  a reference implementation.
- The server does not resample. Whatever `sample_rate` you declare in the
  handshake is trusted as the truth about the frames that follow — send the
  wrong value and every downstream metric is wrong, silently.

## 3. Worked example (Python)

A complete, synchronous SDK already ships in this repo — use it rather than
hand-rolling the WebSocket framing:

```python
from vanirakshak.sdk import VaniRakshakClient

client = VaniRakshakClient(host="http://127.0.0.1:8000")
print(client.health())

with client.stream(caller_id="+91-9xxxxxxxxx", claimed_user="narendra_modi") as s:
    for pcm_chunk in your_audio_source():  # yields raw int16 PCM bytes
        verdict = s.send_chunk(pcm_chunk)
        if verdict is None:
            continue  # not enough audio accumulated yet for a new verdict
        if verdict.is_blocked:
            print("BLOCK:", verdict.reason)
            break
        if verdict.trigger_challenge:
            print("Challenge:", verdict.challenge["phrase"])
```

Source: `vanirakshak-backend/vanirakshak/sdk/client.py` (`VaniRakshakClient`,
`StreamClient`, `Verdict`). `vanirakshak-backend/examples/sdk_demo.py` is a
runnable end-to-end script covering health/enroll/analyze against a live
server.

## 4. Worked example (raw WebSocket, any language)

For a PBX gateway or mobile client not in Python, the wire protocol is
simple enough to hand-roll. Pseudocode:

```
ws = connect("ws://host:8000/ws/stream")
ws.send_text(json.encode({"sample_rate": 16000, "claimed_identity": "narendra_modi"}))

loop:
    pcm_bytes = read_next_audio_frame()   # int16 LE mono @ 16kHz
    ws.send_binary(pcm_bytes)
    if ws.has_message_waiting():
        verdict = json.decode(ws.receive_text())
        handle_verdict(verdict)

ws.close()
```

Grading a challenge from a non-Python client, over REST rather than through
the same socket:

```
POST /api/v1/sessions/{session_id}/challenge
{ "response_text": "8 4", "response_latency_ms": 1200 }
→ { "ok": true, "reason": "ok", "confidence": 0.6, "blocked": false }
```

`session_id` must be one the server assigned — either the one you sent in
the WS handshake, or the one returned by `POST /v1/sessions` if you're
managing session/challenge state over REST separately from the audio socket.

## 5. Mobile clients that can't stream continuously

`vanirakshak-android/` is the reference for this pattern today: when the
backend is reachable it holds a live WebSocket open
(`vanirakshak-android/app/src/main/kotlin/com/vericall/audio/VaniStreamingClient.kt`
— handshake-first, 16-bit PCM @16kHz,
server-authoritative `tier`/`interlock_active`, same reconnect backoff as §6
below) and grades challenges via `POST /v1/sessions/{session_id}/challenge`
(`vanirakshak-android/app/src/main/kotlin/com/vericall/audio/VaniChallengeGrader.kt`), with `receipt_hash` shown and persisted —
see `vanirakshak-android/DESIGN.md` §4. When the backend is unreachable it
falls back to an on-device heuristic clearly labeled as an offline estimate.
Its one-shot REST call for the enrolled-voice identity check is:

```
POST /v1/verify-speaker
  multipart form field: audio (WAV)
→ { "available": true, "label": "modi_ji", "is_target": true, "llr": 4.74, "prob_target": 0.991 }
```

This scores a short recorded chunk against the trained `ml/` identity model
directly — no acoustic risk pipeline, no tier, no interlock. It answers "is
this the enrolled voice?", not "is this call safe?". If you need the full
forensic pipeline (spoofing detection, risk tiering, the transaction
interlock) from a client that can't hold a long-lived WebSocket open, use
`POST /v1/analyze` (or `/analyze/audio`) instead — it runs the same
acoustic+ASV+fusion pipeline as the live stream, one-shot, over a complete
recorded file. See API_SPEC.md for both.

## 6. Reconnect and failure handling

Match the dashboard's policy (PROTOCOL.md §6) unless you have a reason not
to: on an unexpected close, retry with exponential backoff (1s → 2s → 4s →
8s → 15s, capped, with jitter), give up after 5 attempts, and **always
re-send the handshake** on reconnect — the server has no memory of a
disconnected session. A reconnect is a new session, not a resume.

## 7. What this system explicitly does not do

Worth stating plainly, since the guide this one replaces claimed several of
these:

- No web search or live fact-checking of caller claims.
- No LLM in the decision path, anywhere. The advisory semantic layer
  (`vanirakshak/semantic/`, PROTOCOL.md §3.1) runs ASR and a rule-based
  keyword/claim scorer — never an LLM classifying the call.
- No `threat_score`/`threat_level`/`scam_type` fields, no `event`-keyed
  message envelope, no server-initiated `stream_connected` handshake message.
  The fields are exactly those in PROTOCOL.md §3, and the client speaks
  first.
- No port `8084`, no `/api/stream/call-chunk`, no query-string session
  parameters on the WebSocket URL.

If your integration plan depends on any of the above, it was written against
the guide this one replaces — re-derive the plan from PROTOCOL.md instead.
