# VaniRakshak — Operations Runbook

## Starting the Stack

### Backend (port 8000)

```bash
cd vanirakshak-backend
pip install -r requirements.txt
python run.py
```

### Frontend Dashboard (port 3001)

```bash
npm install --prefix frontend
npm run dev
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
