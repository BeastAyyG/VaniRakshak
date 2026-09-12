# VaniRakshak — API Specification

## WebSocket API

### `WS /ws/stream` (alias `/v1/stream`, `/api/v1/stream`)

Real-time forensic analysis endpoint. Full contract in [PROTOCOL.md](PROTOCOL.md) —
this section only summarizes; PROTOCOL.md is the source of truth for the wire
format, not this file.

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
  "claimed_identity": "string (optional — falls back to the loaded anchor's user_id if omitted)",
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
    "asv_evaluated": true,
    "asv_synthetic": false,
    "channel_anomaly": 0.12,
    "prosody_unnatural": 0.3,
    "snr_db": 18.4,
    "codec_guess": "wideband"
  },
  "trigger_challenge": true,
  "interlock_active": false,
  "receipt_hash": "9f2c...",
  "challenge": {
    "phrase": "mango, 8, nadi, 4.",
    "expected_digits": ["8", "4"],
    "expected_full": "mango 8 nadi 4",
    "instructions": "Now repeat ONLY the digits, in order, quickly.",
    "issued_at_ms": 1699999999999,
    "seed": 42
  },
  "semantic": {
    "available": true,
    "transcript": "...",
    "hinglish_text": "...",
    "language": "hi",
    "matches": [{"category": "otp_pin_exfiltration", "term": "otp"}],
    "categories_hit": ["otp_pin_exfiltration"],
    "keyword_risk": 0.3,
    "claim_risk": 0.75,
    "semantic_risk": 0.75,
    "claim_findings": [
      {"rule_id": "otp_request", "claim": "...", "finding": "...", "authority": "RBI ...", "severity": 0.95, "matched_terms": ["otp"]}
    ],
    "provider": "groq",
    "reason": null
  }
}
```

`metrics.asv_evaluated` (bool) — whether ASV ran a real comparison this
window; when `false`, `asv_consistency` is the fusion-neutral `0.5`
placeholder, not a measured similarity. `metrics.asv_synthetic` (bool, G7) —
whether the ASV backend behind `asv_consistency` is the synthetic
random-projection stub rather than real biometrics; `true` under the default
`VR_ASV_BACKEND=stub`, `false` only with the real ECAPA-TDNN backend. A client
must not render `asv_consistency` as a voiceprint when this is `true`.
`challenge.expected_digits` is a
`List[str]` of single-digit strings; `expected_full` is the full spoken
phrase (words and digits), not the digits alone.

`semantic` is the parallel ASR/keyword/claim-verification signal — advisory
only, a small additive term in the fusion, never the primary evidence, and
**always present** with the same twelve keys whether idle, degraded, or
populated. Full schema and the honest scope of `claim_findings` (rule-based
against known policy, not open-ended fact-checking) in
[PROTOCOL.md](PROTOCOL.md) §3.1. Degrades to
`{"available": false, "semantic_risk": 0.0, "reason": "..."}` when
`VR_SEMANTIC_ENABLED` is unset (the default), or `GROQ_API_KEY` or the optional ASR stack (`requirements-asr.txt`) is absent —
every other field is unaffected.

## REST API

### `GET /v1/health` (and `/api/v1/health`)

Returns backend status, hardware, active detector backends, and the loaded
speaker anchor.

```json
{
  "ok": true,
  "status": "healthy",
  "version": "0.1.0-sprint1",
  "active_sessions": 0,
  "hardware": { "device": "cpu", "device_name": "CPU (Acoustic Forensic Core)" },
  "models": {
    "wavlm_model": "synthetic_cm_v1 / synthetic_asv_v1",
    "vad_enabled": true,
    "zero_retention": true,
    "third_party_audio_processors": [],
    "cm_backend": "synthetic_cm_v1",
    "asv_backend": "synthetic_asv_v1"
  },
  "anchor": {
    "user_id": "narendra_modi",
    "enrolled": true,
    "calibrated": true,
    "source_clips": 5,
    "error": null
  }
}
```

`wavlm_model` is named for whatever is actually running (`cm_backend` /
`asv_backend` duplicated for clarity) — never asserts "AASIST" or "ECAPA"
unless the real backends (`VR_CM_BACKEND=wavlm`, `VR_ASV_BACKEND=ecapa`) are
active, in which case it reports `wavlm_heuristic_v1` / `ecapa_tdnn_voxceleb`.

`models.asv_synthetic` (bool, G7) is `true` when `asv_backend` is the synthetic
random-projection stub and `false` only for the real ECAPA-TDNN backend — read
from the running instance, so it stays truthful if `VR_ASV_BACKEND=ecapa` was
requested but degraded to the stub. The dashboard uses it to relabel the
"TARGET ANCHOR ASV" tile as synthetic rather than let a projection read as a
voiceprint. See PROTOCOL.md §3.

`zero_retention` and `third_party_audio_processors` are declared from
configuration, not inferred after the fact: with `VR_SEMANTIC_ENABLED=1`
the server reports `zero_retention: false` and `["groq"]` before any call
is made, because the semantic layer uploads caller audio to Groq. The same
pair appears in `/analyze`'s `dpdp_compliance` block as
`zero_retention_verified` and `third_party_audio_processors`. Off by default.

`anchor` describes the single enrolled target speaker loaded at server
startup (`vanirakshak/anchor.py::load_anchor`) — `user_id` is `null` and
`error` is set when the anchor file is missing, corrupt, or was embedded by a
different ASV backend than the one currently running (loading a mismatched
vector space is refused, not silently misinterpreted). This is also the
identity the WS handshake and `/analyze`'s `claimed_identity` fall back to
when the caller sends none — see PROTOCOL.md §1.

### `GET /v1/enroll` (and `/api/v1/enroll`)

Lists enrolled speakers — the multi-user/"family" surface.

```json
{ "enrolled": ["narendra_modi"], "asv_backend": "synthetic_asv_v1" }
```

Empty `enrolled` doesn't mean broken; it means nobody has been enrolled on
this running instance yet.

### `POST /v1/enroll` (and `/api/v1/enroll`)

Form fields: `user_id` (string, required), `audio` (file, required, ≥0.5s).

```json
{
  "status": "enrolled",
  "user_id": "narendra_modi",
  "embedding_dimension": 192,
  "audio_retained": false,
  "created_at": "2026-09-09T02:20:18Z"
}
```

Any number of distinct `user_id`s may be enrolled (`EnrolledSpeakersPanel.tsx`
is the dashboard UI for this). With the stub ASV backend (default) the
resulting "voiceprint" is a random projection, not real biometrics — real
multi-speaker verification requires `VR_ASV_BACKEND=ecapa`.

### `POST /v1/analyze` / `/v1/analyze/audio` (and `/api/v1/` variants)

One-shot forensic file analysis. Form fields: `file` or `audio` (either name
accepted), `caller_id` (optional), `claimed_identity` (optional, same anchor
fallback as the WS handshake).

Response (`AnalyzeResponse`) — abbreviated to the fields not already covered
by the verdict schema above:

```json
{
  "status": "success",
  "file_duration_sec": 10.0,
  "calibrated_risk": 33.9,
  "p_spoof": 0.34,
  "tier": "ALLOW",
  "receipt_hash": "...",
  "breakdown": {
    "codec_guess": "wideband",
    "cm_score": 1.2,
    "asv_consistency": 0.73,
    "asv_evaluated": true,
    "asv_synthetic": false,
    "channel_anomaly": 0.1,
    "prosody_unnatural": 0.2
  },
  "session_id": "analyze-abc123",
  "audio_sha256": "...",
  "duration_sec": 10.0,
  "device_used": "CPU (Pure NumPy Forensic Pipeline)",
  "windows_count": 6,
  "overall_risk_score": 33.9,
  "verdict": "AUTHENTIC_TARGET",
  "windows": ["... per-window breakdown, see ForensicVerdictCard.tsx ..."],
  "mean_acoustic_anomaly": 0.33,
  "max_acoustic_anomaly": 0.35,
  "asv_evaluated": true,
  "mean_speaker_sim": 0.73,
  "mean_micro_tremor": 0.8,
  "mean_spectral_flatness": 0.24,
  "mean_high_band_flatness": 0.24,
  "mean_zcr_variance": 0.05,
  "vocoder_phase_discontinuities_detected": 0,
  "detected_boundary_timestamps": [],
  "dpdp_compliance": {
    "receipt_sha256": "...",
    "statutory_act": "India Digital Personal Data Protection (DPDP) Act 2023 Section 8(7)",
    "zero_retention_verified": true,
    "third_party_audio_processors": [],
    "timestamp": "2026-09-09T02:20:18Z"
  },
  "target_speaker": {
    "available": true,
    "label": "modi_ji",
    "is_target": true,
    "prob_target": 0.991,
    "llr": 4.74,
    "reason": null
  },
  "semantic": { "...": "same shape as the WS verdict's semantic block" }
}
```

`asv_evaluated` (top-level and in `breakdown`) mirrors the WS path's
`metrics.asv_evaluated`; `asv_synthetic` (likewise top-level and in
`breakdown`) mirrors `metrics.asv_synthetic` and `health.models.asv_synthetic`
— see the note on the verdict schema above. `target_speaker` is the trained `ml/` identity model
— an **identity check** ("is this the enrolled voice?"), explicitly **not** a
clone/synthetic detector; reported alongside the acoustic `verdict`, never
folded into it, so a missing checkpoint degrades to `available: false`
without breaking the rest of the report. `semantic` is computed once for the
whole file and fed into every window's fusion pass (unlike the WS path, where
it necessarily lags one window — see PROTOCOL.md §3.1). Multi-window risk score
aggregation is configurable via `VR_ANALYZE_POOLING` (`last` by default, `max`,
or `top_k`).

### `POST /v1/verify-speaker` (and `/api/v1/verify-speaker`)

The one-shot REST call the Android client makes (`VaniSpeakerVerifier.kt`).
Form fields: `audio` or `file` (either name accepted, WAV). Scores against the
trained `ml/` identity model directly — the same model `/analyze`'s
`target_speaker` block reports, without running the acoustic pipeline.

```json
{
  "available": true,
  "label": "modi_ji",
  "is_target": true,
  "llr": 4.74,
  "prob_target": 0.991,
  "reason": null
}
```

Never returns HTTP 500 for a missing checkpoint or missing `torch`/`librosa`
— degrades to `{"available": false, "label": "unavailable", "reason": "..."}`.

### `POST /v1/sessions` (and `/api/v1/sessions`)

Initialize a call session with caller metadata (the REST equivalent of the WS
handshake, for clients that grade challenges over REST without streaming
audio through this same connection). Body matches the WS handshake schema.
Returns `{"session_id": "call-..."}`.

### `POST /v1/sessions/{sid}/challenge` (and `/api/v1/sessions/{sid}/challenge`)

Grade a challenge response. Body: `{"response_text": "...", "response_latency_ms": 1200}`.

```json
{ "ok": true, "reason": "ok", "confidence": 0.6, "blocked": false }
```

`404` if `sid` doesn't match an active session; `{"ok": false, "reason": "no_active_challenge"}`
if the session has no challenge outstanding (including after a challenge has
already been graded — each challenge is single-use). Every graded response
(pass or fail) is recorded in the audit chain. **A failed challenge always
escalates the session to `BLOCK`, permanently. A passed challenge never
un-blocks or re-arms** (see PROTOCOL.md §3 "Challenge outcomes and the
interlock"): `blocked` in the response reflects the session's one-way latch,
not a reversible state.

### `GET /v1/audit` (and `/api/v1/audit`)

Tail the SHA-256 hash-chained audit log. Query param `n` (default 50).

```json
{ "chain_valid": true, "events": [ { "session_id": "...", "receipt_hash": "...", "prev_hash": "...", "risk_score": 43.2, "tier": "CHALLENGE", "features": {}, "timestamp_iso": "..." } ] }
```

The log is a capped in-memory ring (default 10,000 events) backed by a
durable JSONL file (`VR_AUDIT_PATH`, default `data/audit_log.jsonl`) that is
reloaded and re-verified on restart. Feature-only: no raw audio or embeddings are
ever logged.

## Authentication

Disabled unless `VR_API_KEYS` is set, in which case every route below except
`GET /v1/health` requires a key and returns **401** without one, **403** for a
key lacking the needed scope or not permitted to act as the requested
`user_id`/`claimed_identity`, and **429** past the per-key rate limit. Send it
as `Authorization: Bearer <key>` or `X-API-Key: <key>`; on the WebSocket, as
`api_key` in the handshake. Full key format and scopes in
[PROTOCOL.md](PROTOCOL.md) §8.

`GET /v1/health` stays open deliberately: a load balancer cannot hold a
credential, and it exposes only backend names and whether an anchor loaded.

## Environment Variables

See [PROTOCOL.md](PROTOCOL.md) §8 for the complete, current table — this file
does not duplicate it to avoid the two drifting apart again.

## WebSocket Close Codes

| Code | Meaning |
|---|---|
| `1000` | Normal closure |
| `1003` | First frame was binary instead of a JSON handshake |
| `1008` | Authentication failed (missing/invalid `api_key`) — only when `VR_API_KEYS` is set |

`stream_endpoint`'s only `ws.close(code=...)` call in the codebase is the
`1003` above. There is no `4000`/pipeline-error close code — an exception
during streaming is caught, logged, and the connection falls through to
cleanup rather than closing with a distinct code.
