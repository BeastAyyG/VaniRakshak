# VaniRakshak — Architecture Overview

## System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Client (Browser)                             │
│  ┌──────────────────┐   ┌───────────────────┐   ┌───────────────┐ │
│  │  AudioWorklet     │   │  WebSocket Client  │   │  React 19 UI  │ │
│  │  with fallback    │──▶│  Binary + JSON     │──▶│  Dashboard    │ │
│  │  16-bit @ 16kHz   │   │  Streaming         │   │  + Interlock  │ │
│  └──────────────────┘   └───────────────────┘   └───────────────┘ │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ ws://host:8000/ws/stream
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Forensic Backend                          │
│  ┌────────────┐ ┌────────────┐ ┌─────────┐ ┌──────────────────┐   │
│  │ CM (AASIST)│ │ ASV (ECAPA)│ │ Channel │ │ Prosody          │   │
│  │ Spoof Det. │ │ Speaker V. │ │ Codec   │ │ Micro-tremor     │   │
│  └─────┬──────┘ └─────┬──────┘ └────┬────┘ └────────┬─────────┘   │
│        └───────────────┴─────────────┴───────────────┘             │
│                          ▼                                          │
│              Calibrated Bayesian Fusion                             │
│              (sigmoid + EMA + fast-rise)                            │
│                          ▼                                          │
│    ALLOW (<35) │ CHALLENGE (35-69) │ BLOCK (>=70)                  │
│                          ▼                                          │
│  ┌────────────┐ ┌──────────────┐ ┌──────────────────────────────┐  │
│  │ Challenge  │ │ SHA-256      │ │ Alert Dispatcher              │  │
│  │ Engine     │ │ Audit Log    │ │ (server-side webhook)         │  │
│  └────────────┘ └──────────────┘ └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Layer Descriptions

| Layer | Module | Purpose |
|---|---|---|
| Ingestion | `server/app.py` | WebSocket endpoint, JSON handshake, binary PCM framing |
| Buffering | `privacy/buffer.py` | RAM-only sliding window, DPDP-compliant, zero disk retention |
| Detection | `detectors/` | CM, ASV, channel, prosody — parallel forensic scoring |
| Fusion | `risk/fusion.py` | Calibrated Bayesian posterior, EMA smoothing, risk 0-100 |
| Decision | `session.py` | ALLOW/CHALLENGE/BLOCK tiers, dynamic phrase engine, interlock |
| Audit | `privacy/audit.py` | SHA-256 hash-chained append-only log |
| Alerting | `alerts/dispatcher.py` | Best-effort webhook POST on CHALLENGE/BLOCK events |

## Key Design Decisions

1. **Acoustic-only detection** — no STT, no NLP, no LLM. All analysis operates on raw PCM waveforms.
2. **Server-authoritative verdicts** — the backend owns ALLOW/CHALLENGE/BLOCK; the frontend only displays.
3. **Honest synthetic stubs** — detectors ship as deterministic stubs; real weights are drop-in replacements.
4. **Privacy-first** — raw audio is never persisted; DPDP Act 2023 §8(7) compliance by design.
