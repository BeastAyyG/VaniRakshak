# VaniRakshak — API Specification

## WebSocket API

### `WS /ws/stream` (alias `/v1/stream`)

Real-time forensic analysis endpoint. Full contract in [PROTOCOL.md](PROTOCOL.md).

**Connection flow:**
1. Client opens WebSocket to `ws://<host>:8000/ws/stream`
2. Client sends JSON text handshake (must be first frame)
3. Client streams binary 16-bit mono PCM frames at 16 kHz
4. Server responds with JSON verdict messages per analysis window (4.04s sliding window, 1.0s hop)

### Handshake Schema

```json
{
  "session_id": "string (optional, auto-generated if omitted)",
  "sample_rate": 16000,
  "caller_id": "string (optional)",
  "claimed_identity": "string (optional)",
  "transaction_value_inr": 0,
  "origin_country": "IN",
  "prior_fraud_score": 0
}
```

### Verdict Response Schema

```json
{
  "timestamp_ms": 1699999999999,
  "risk_score": 43.2,
  "p_spoof": 0.55,
  "tier": "CHALLENGE",
  "metrics": {
    "cm_score": 0.61,
    "cm_genuine_prob": 0.39,
    "asv_consistency": 0.68,
    "channel_anomaly": 0.12,
    "prosody_unnatural": 0.3,
    "snr_db": 18.4,
    "codec_guess": "wideband"
  },
  "trigger_challenge": true,
  "interlock_active": false,
  "receipt_hash": "9f2c...",
  "challenge": {
    "phrase": "Say: Mango 8 nadi 4 blue",
    "expected_digits": "84",
    "expected_full": "Mango 8 nadi 4 blue",
    "instructions": "Repeat the phrase exactly.",
    "issued_at_ms": 1699999999999,
    "seed": 42
  }
}
```

## REST API

### `GET /api/v1/health`

Returns backend status and hardware information.

```json
{
  "ok": true,
  "status": "healthy",
  "version": "0.1.0-sprint1",
  "active_sessions": 0,
  "hardware": { "device": "cpu", "device_name": "CPU (Acoustic Forensic Core)" },
  "models": { "wavlm_model": "AASIST / ECAPA Forensic Pipeline", "vad_enabled": true, "zero_retention": true }
}
```

### `POST /v1/enroll` (and `/api/v1/enroll`)

Speaker enrollment with user_id + audio file.

### `POST /v1/analyze` (and `/api/v1/analyze`, `/v1/analyze/audio`, `/api/v1/analyze/audio`)

One-shot forensic file analysis.

### `POST /v1/sessions` (and `/api/v1/sessions`)

Initialize call session with caller metadata.

### `POST /v1/sessions/{sid}/challenge` (and `/api/v1/sessions/{sid}/challenge`)

Grade challenge response.

### `GET /v1/audit` (and `/api/v1/audit`)

Tail SHA-256 audit log.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `VR_PORT` | `8000` | Backend listen port |
| `VR_HOST` | `127.0.0.1` | Backend bind host |
| `VR_CORS_ORIGINS` | `http://localhost:3001,http://127.0.0.1:3001` | Allowed browser CORS origins |
| `VR_ALERT_WEBHOOK_URL` | *(unset)* | Alert webhook target; unset disables dispatch |
| `NEXT_PUBLIC_BACKEND_WS` | `ws://localhost:8000/ws/stream` | Dashboard WebSocket target |

## Error Codes

| WebSocket Close Code | Meaning |
|---|---|
| `1000` | Normal closure |
| `1003` | First frame was binary instead of JSON handshake |
| `4000` | Pipeline error |
