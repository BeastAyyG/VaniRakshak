# VaniRakshak — Operations Runbook

## Starting the Stack

### One command (recommended)

```bash
npm run dev:all
```

`scripts/dev.js` creates `vanirakshak-backend/.venv` on first run, boots the
backend on :8000, waits for `/api/v1/health`, then starts the frontend on
:3001 with `NEXT_PUBLIC_BACKEND_URL`/`NEXT_PUBLIC_BACKEND_WS` pointed at it.
This is what `.claude/launch.json` uses.

### Backend only (port 8000)

```bash
cd vanirakshak-backend
pip install -r requirements.txt
python run.py
```

### Frontend only (port 3001)

```bash
npm install --prefix frontend
npm run dev --prefix frontend
```

### Docker Compose (both services)

```bash
docker-compose up --build
```

## Verification

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### WebSocket Protocol Conformance

```bash
cd vanirakshak-backend
python tests/test_stream_e2e.py
```

### Backend Unit Tests

```bash
cd vanirakshak-backend
python -m pytest tests/ -v
```

### Full Integration Check (what the dashboard actually depends on)

```bash
python scripts/integration_check.py
```

Boots the app in-process (no running server needed) and exercises health,
`/analyze/audio` on all four demo clips, `/verify-speaker`, the audit chain,
and a real WebSocket session replaying the browser's exact framing —
including the challenge object shape, `metrics.asv_evaluated`, the
`semantic` block's key set, and the `VR_IMPOSTOR_BLOCK_BELOW` tier override.
Pass `--live --base-url http://127.0.0.1:8000` to run it against an
already-running server instead.

### Real detector backends (optional, verified working)

```bash
cd vanirakshak-backend
pip install -r requirements-asv.txt --extra-index-url https://download.pytorch.org/whl/cpu
set VR_ASV_BACKEND=ecapa
set VR_CM_BACKEND=wavlm
python ..\scripts\integration_check.py
```

This swaps the zero-dependency stubs for real SpeechBrain ECAPA-TDNN
embeddings and WavLM features (weights download from Hugging Face once).
Measured on the demo corpus (first 4.04 s window, anchor `narendra_modi`):
genuine `asv=0.854`, impostor `asv=0.044` — clean target/impostor
separation, versus the stub's uninformative projection. Two caveats this
surfaced, both by design: (1) the tier-match gates in
`integration_check.py` are calibrated for stub-era operating points and
currently fail under real backends (`degraded_copy` → CHALLENGE,
`partial_splice` → BLOCK — the system gets *more* sensitive, not less;
recalibrate the gates before treating them as pass/fail under `ecapa`);
(2) the WavLM CM decision on top of the real features is still a
hand-tuned heuristic (`cm` scored the impostor clip *higher* than the
genuine one), not a trained anti-spoofing classifier — do not cite
spoof-detection accuracy for either backend.

### Demo → working system (activation checklist)

```powershell
# 1. Real detectors (verified working — see above)
pip install -r vanirakshak-backend/requirements-asv.txt
$env:VR_ASV_BACKEND="ecapa"; $env:VR_CM_BACKEND="wavlm"
# 2. Trained countermeasure — NOT available: needs an ASVspoof-style
#    labeled corpus + GPU training; nothing in-repo substitutes for it.
# 3. Enroll real speakers (10–20 s genuine 16 kHz mono WAV each)
#    Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/enroll `
#      -Form @{ user_id="alice"; audio=[IO.File]::OpenRead("alice_20s.wav") }
#    then pass "claimed_identity": "alice" in the WS handshake (PROTOCOL.md §1).
# 4. Semantic layer: pip install groq; $env:GROQ_API_KEY="gsk-..."
# 5. Android: needs Android Studio (SDK + JDK 17); point
#    VaniBackendConfig at the LAN host, e.g. ws://192.168.1.10:8000/v1/stream.
# 6. Harden: behind gateway/VPN + auth (none built in); VR_HOST / VR_PORT
#    to bind, VR_AUDIT_PATH to durable storage (default data/audit_log.jsonl).
```

### Doc/Code Drift Check

```bash
python scripts/check_docs.py
```

Verifies every file path and `VR_*` environment variable a doc mentions
actually exists, and that documented WebSocket verdict keys match what
`test_stream_e2e.py` asserts. Run after touching `docs/PROTOCOL.md`,
`docs/API_SPEC.md`, or `vanirakshak/config.py`.

### Synthetic Evaluation Harness

```bash
cd vanirakshak-backend
python -m vanirakshak eval
```

### Frontend Lint + Build

```bash
cd frontend
npm run lint
npm run build
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Dashboard shows "Demo mode" | Backend unreachable at `ws://localhost:8000/ws/stream` | Start backend first; check `VR_PORT` and `VR_CORS_ORIGINS` |
| WebSocket closes with code 1003 | First frame was binary, not JSON handshake | Ensure client sends JSON text handshake before streaming PCM |
| `BLOCK` verdict but no webhook fires | `VR_ALERT_WEBHOOK_URL` not set | Export `VR_ALERT_WEBHOOK_URL` with your webhook endpoint |
| `httpx` import error in alerting | httpx not installed | `pip install httpx` (optional dependency) |
| Reconnect loop after 5 attempts | Backend crashed or network issue | Restart backend; click status badge to start fresh session |

## Monitoring

- **Audit log**: SHA-256 hash-chained events in `AuditLog` (in-memory ring + durable `VR_AUDIT_PATH` JSONL; export via `/api/v1/audit`)
- **Event log**: Frontend displays last 20 events with severity badges
- **Alert delivery**: Logged at WARNING level on failure; best-effort, never breaks the stream

## Known Limitations

Recorded here rather than silently carried, so nobody rediscovers these the
hard way:

- **Challenge is a one-shot gate, not a recovery path (decided).** A failed
  challenge always escalates to `BLOCK` permanently; a *passed* challenge
  never un-blocks a session or re-arms a future challenge — `_blocked` and
  the one-challenge-per-session gate are both one-way by design
  (`grade_challenge()`), because grading is digit-order + latency only with
  no voice biometrics. Each challenge is single-use: re-grading returns
  `no_active_challenge`.
- **WS handshake `session_id` handling.** The server always opens with a
  `session_ack` (`docs/PROTOCOL.md` §1) echoing the resolved `session_id`,
  so an omitted id is recoverable for REST challenge-grading. Supplying
  your own id still works (see `docs/INTEGRATION_GUIDE.md`).
- **The `/analyze` per-window report fields are lightweight measurements.**
  `inference_ms` is a real per-`push_chunk` wall-clock timing;
  `speech_ratio`/`is_voiced` come from a simple short-time-energy VAD and
  `zcr_variance` from per-frame zero-crossing variance
  (`vanirakshak/detectors/voice.py`) — honest signal-energy heuristics, not
  a trained VAD, so treat them as indicative, not ground truth.
- **The impostor label and the impostor interlock use different
  thresholds (decided).** `/analyze`'s `IMPOSTOR_SPEAKER` verdict label fires below
  `mean_speaker_sim < 0.40` (report-level identity: "not the enrolled voice");
  the actual `BLOCK` override (`VR_IMPOSTOR_BLOCK_BELOW`, default `0.10` in
  `vanirakshak/config.py`, enforced in `risk/fusion.py`) fires only on decisive
  mismatch. A clip in `[0.10, 0.40)` is therefore *labelled* `IMPOSTOR_SPEAKER`
  while its *tier* stays `CHALLENGE` — label states identity, tier states
  interlock action. Kept split deliberately; do not "fix" by aligning the
  numbers without a product decision.
- **The audit chain persists to JSONL.** Every event is appended to
  `VR_AUDIT_PATH` (default `data/audit_log.jsonl`, gitignored) and the chain
  is reloaded and re-verified on restart; empty `VR_AUDIT_PATH` opts back
  out to the in-memory ring only. The process never truncates the file —
  rotate it with normal log tooling.
- **Client resampling is linear-interpolation only.** A hardware rate
  mismatch is detected and corrected in-client, but the resampler is a
  simple interpolator, not a proper anti-aliased filter — fine for keeping
  detectors on-rate, not lab-grade.
- **`WaveformEvidenceTimeline` falls back to a synthetic pattern only when
  audio can't be decoded.** Bar heights come from per-bar peaks of the real
  decoded audio (`decodeAudioData`); the sine fallback shows while decoding
  or for unfetchable (no-CORS) sources. Colors and boundary markers were
  always data-driven.
- **Standalone prototypes are a different product from the acoustic
  pipeline above, and none is wired into it.** `vanirakshak_sotto/` —
  **VaniRakshak SACH** — is now verified working end-to-end (Groq Whisper +
  Gemini threat detection + Crawl4AI web grounding + Gemini fact
  verification, `GEMINI_API_KEY` set, `google-genai`/`crawl4ai` installed;
  run with `cd vanirakshak_sotto && python server.py`, port 8082).
  `vanirakshak_indic_conformer/` now boots too — verified via TestClient:
  `/api/health` returns 200 as "VaniRakshak IndicConformer Edition v2.2"
  (Groq Whisper LPU with an Indic adaptation), and its Gemini client
  constructs, once a `.env` carrying `GEMINI_API_KEY` sits beside it (its
  loader walks `.` then two parents; there is no repo-root `.env`, so the
  local one is required). An earlier revision of this entry claimed its
  `from google import genai` import was still broken — that was wrong:
  both that form and `import google.genai as genai` resolve fine under
  `google-genai` 2.x, so the only thing it was ever missing was the key.
  `ayush_agnihotri/` holds no code at all — two markdown files describing
  a different product (Tavily + Gemini claim verification), carrying a
  banner to that effect; there is nothing there to audit or run. None of
  these is wired into the live pipeline, backend, web frontend, or
  phone-app call flow.
