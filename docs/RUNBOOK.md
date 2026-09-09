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

- **Audit log**: SHA-256 hash-chained events in `AuditLog` (in-memory; export via `/api/v1/audit`)
- **Event log**: Frontend displays last 20 events with severity badges
- **Alert delivery**: Logged at WARNING level on failure; best-effort, never breaks the stream

## Known Limitations

Recorded here rather than silently carried, so nobody rediscovers these the
hard way:

- **Challenge recovery semantics are undecided.** A failed challenge always
  escalates to `BLOCK` permanently; whether a *passed* challenge un-blocks a
  session or re-arms a future challenge is an open `TODO(human)` in
  `vanirakshak/session.py::grade_challenge`. As shipped, it does neither —
  `_blocked` and the one-challenge-per-session gate are both one-way.
- **Auto-generated `session_id` is never returned to the client.** If a WS
  handshake omits `session_id`, the server assigns one internally but never
  sends it back — grading a challenge over REST afterward is impossible
  without it. Always set your own `session_id` (see `docs/INTEGRATION_GUIDE.md`).
- **The `/analyze` response has several fabricated fields**, kept exactly as
  named for frontend compatibility rather than removed outright:
  `inference_ms` is a constant `14.5`, not a measurement; `speech_ratio` is a
  constant `0.95`; `is_voiced` is always `true` (no VAD runs in the serving
  path); `zcr_variance` is a constant `0.05`. Do not treat these as real
  telemetry.
- **The impostor label and the impostor interlock use different
  thresholds.** `/analyze`'s `IMPOSTOR_SPEAKER` verdict label fires below
  `mean_speaker_sim < 0.40`; the actual `BLOCK` override
  (`VR_IMPOSTOR_BLOCK_BELOW`) fires below `0.10`. A clip can be *labelled*
  `IMPOSTOR_SPEAKER` while its *tier* is only `CHALLENGE`.
- **The audit chain is in-memory only.** A server restart loses it entirely;
  there is no persistence layer.
- **No client-side audio resampling.** The client requests 16 kHz on
  `getUserMedia` and `AudioContext`, but if a browser honors a different
  hardware rate, nothing downstream corrects for it — the handshake's
  `sample_rate` is trusted, not verified.
- **`WaveformEvidenceTimeline`'s waveform display is synthetic.** Bar
  heights come from a fixed sine pattern, not decoded audio — only the
  colors and boundary markers are data-driven.
