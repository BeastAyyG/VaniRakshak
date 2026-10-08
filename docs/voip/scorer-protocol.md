# Scorer /ws/stream Frame Schema (ground truth, verified live)

Probe run 2026-10-09 against lab-beast `ws://127.0.0.1:8000/ws/stream`
(backend PID 27448, model `aasist_l_wavlm_dev_eer_0.045`). Raw websocket
client reproduced the Android `VaniStreamingClient.kt` wire format exactly:
JSON handshake, then binary int16 PCM. 75 binary frames sent, 4 verdict
frames received. Full probe output:
`hidden_files/voip-build/scorer-probe-output.txt`.

Android client source:
`C:\VaniRakshak-voip\vanirakshak-android\app\src\main\kotlin\com\vericall\audio\VaniStreamingClient.kt`
Backend endpoint source:
`C:\VaniRakshak-voip\vanirakshak-backend\vanirakshak\server\app.py` (`stream_endpoint`)

## Endpoint

| Where | URL |
|---|---|
| lab-beast localhost | `ws://127.0.0.1:8000/ws/stream` |
| lab-beast over Tailscale | `ws://100.89.248.108:8000/ws/stream` |

Aliases on the same handler: `/v1/stream`, `/api/v1/stream`.

## Client to server

### 1. Handshake (REQUIRED first frame, JSON text)

If the first frame is not JSON text the server closes with code 1003.

```json
{
  "session_id": "probe-4937794c",
  "sample_rate": 16000,
  "caller_id": "probe",
  "claimed_identity": "",
  "transaction_value_inr": 0,
  "origin_country": "IN",
  "prior_fraud_score": 0
}
```

Field notes (verified against source + live):

* `session_id`: optional. Client-generated (the Android app uses
  `"android-<uuid>"`). If omitted the server generates `ws-<8 hex>` and
  reports it in the ack. The server echoes the resolved id in every verdict
  the client can read.
* `sample_rate`: optional, default 16000. The server calls
  `set_sample_rate(int(...))`, so other rates are accepted.
* `caller_id`, `claimed_identity`, `transaction_value_inr`,
  `origin_country` (default `"IN"`), `prior_fraud_score`: context fields for
  the risk session; all optional.
* `api_key`: optional IN THE CURRENT LAB-BEAST STATE. Auth is disabled
  because `VR_API_KEYS` is unset (the server logs a loud warning at startup
  in this state). If keys are ever configured, the handshake must carry
  `api_key` in the body (or the key in the `Authorization` header, since a
  browser cannot set headers on a WebSocket); otherwise the server closes
  with code 1008.

### 2. Audio frames (binary only)

* Raw binary WebSocket frames.
* Format: mono 16-bit signed PCM, little-endian, at the handshake
  `sample_rate` (16 kHz in the app).
* Chunk size is arbitrary. The probe used 100 ms chunks (1600 samples,
  3200 bytes); the server accumulates them into scoring windows.
* A JSON text frame sent AFTER the handshake (e.g.
  `{"type":"audio","pcm_b64":"..."}`) is silently ignored: no error, no
  verdict, connection stays open. Raw binary is the only accepted audio
  format.

### 3. Windowing (server side)

* Window 4.04 s, hop 1.0 s (`VR_WINDOW_SECONDS`, `VR_HOP_SECONDS`;
  defaults in `vanirakshak/config.py`).
* First verdict arrives after 4.04 s of audio is buffered, then one verdict
  per 1.0 s hop. The probe sent 7.5 s and received 4 verdicts (windows
  ending at ~4.04 / 5.04 / 6.04 / 7.04 s of audio).

## Server to client

### session_ack (first server message, always)

```json
{"type": "session_ack", "session_id": "probe-4937794c", "sample_rate": 16000}
```

Carries no `tier`/`risk_score`; verdict parsers must skip it by shape (the
Android client does exactly this).

### verdict (one per hop window, JSON text)

Real captured frame (verdict #1 of the probe, whitespace trimmed):

```json
{
  "timestamp_ms": 1791487133103,
  "risk_score": 72.61,
  "p_spoof": 0.7261,
  "tier": "BLOCK",
  "metrics": {
    "cm_score": 20.793534716796874,
    "cm_genuine_prob": 0.9999999990678567,
    "asv_consistency": 2.0518203454966407e-08,
    "asv_evaluated": true,
    "asv_synthetic": true,
    "channel_anomaly": 0.24999999490118208,
    "prosody_unnatural": 0.5,
    "snr_db": 46.446044269676634,
    "tier_override": "impostor",
    "codec_guess": "gsm_efr_or_amr_wb"
  },
  "trigger_challenge": false,
  "interlock_active": true,
  "receipt_hash": "4c0b0151b55caec7b8c90dfb3d16ce75c7de24dba151297fbb585380ad9cbade",
  "challenge": null,
  "semantic": {
    "available": false,
    "transcript": "",
    "hinglish_text": "",
    "language": null,
    "matches": [],
    "categories_hit": [],
    "semantic_risk": 0.0,
    "keyword_risk": 0.0,
    "claim_risk": 0.0,
    "claim_findings": [],
    "provider": null,
    "reason": "not started"
  }
}
```

Key semantics (verified):

* `timestamp_ms`: wall-clock epoch milliseconds at scoring time, NOT an
  audio offset. (Probe values decoded to the probe run time.)
* `risk_score`: 0-100 float. `p_spoof`: 0-1 float (`risk_score / 100` in the
  observed frames).
* `tier`: `"ALLOW"`, `"CHALLENGE"`, or `"BLOCK"`. The Android client treats
  these authoritatively for its SAFE/WARNING/DANGER display.
* `metrics`: per-window diagnostics. Keys observed: `cm_score`,
  `cm_genuine_prob`, `asv_consistency`, `asv_evaluated`, `asv_synthetic`,
  `channel_anomaly`, `prosody_unnatural`, `snr_db`, `tier_override`,
  `codec_guess`.
* `trigger_challenge`: bool. `challenge`: null normally; populated (with
  `phrase` and `instructions`, which the Android client renders) only the
  first time a challenge fires.
* `interlock_active`: bool, drives the client's DANGER display.
* `receipt_hash`: hex string, audit receipt for the window.
* `semantic`: present but `available: false` by default. Enabling it needs
  `VR_SEMANTIC_ENABLED=1`, which uploads each audio window to Groq
  (third-party) and turns off zero retention.

### session_closed (revocation path, from source, not probed)

If the claimed identity is erased mid-call the server sends
`{"type": "session_closed", "session_id": "<sid>", "reason": "<text>"}`
then closes with code 1000.

## Behavioral note for the VoIP workstream (verified, important)

The probe streamed a plain 440 Hz sine wave (obviously not spoofed audio).
The CM leg scored it genuine (`cm_genuine_prob` ~ 1.0), yet the verdict was
`BLOCK` with `risk_score` 72.61 because `metrics.tier_override` was
`"impostor"`: `asv_consistency` was ~2e-08, i.e. the audio matched no
enrolled voiceprint, so the ASV leg overrode the tier to impostor/BLOCK.

Consequence: with the backend in its current configuration, ANY caller whose
voice does not match an enrolled voiceprint gets BLOCK, genuine CM score or
not. For the VoIP screening use case (screening unknown callers, where no
voiceprint is enrolled) this behavior needs a product decision before the
Android workstream wires verdicts to call UI: either enroll per-contact
voiceprints, relax the impostor override for the screening profile, or map
impostor-driven BLOCK to a CHALLENGE-style step-up instead of a hard block.

## TBD

* Verdict shape when a challenge fires (`challenge` non-null): not captured
  in this probe. The Android client reads `challenge.phrase` and
  `challenge.instructions`; confirm against a live CHALLENGE tier when the
  product decision above is made.
* Exact `metrics` key set can vary by backend config (ASV on/off, semantic
  on/off). The list above is what the current lab-beast config emits.
* Behavior with `VR_API_KEYS` set (1008 close path) is per source only, not
  probed; lab-beast currently runs with auth disabled.
* `timestamp_ms` is wall-clock; there is no per-verdict audio-offset field.
  If the Android workstream needs to align verdicts to call time, it should
  stamp arrival locally.
