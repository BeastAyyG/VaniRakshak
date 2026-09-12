# VaniRakshak — Architecture Overview

> Three clients exist: the Next.js web console (below, the fully-wired
> reference implementation), **VaniRakshak Kavach** — a native Android
> call-guard client (`vanirakshak-android/`, see its own `DESIGN.md` — real,
> working, but with its own separate on-device heuristics not shown here) —
> and a standalone fact-checking prototype, **VaniRakshak SACH**
> (`vanirakshak_sotto/`), a working Groq Whisper + Gemini + Crawl4AI voice
> transcription and claim-verification server (port 8082) once
> `GEMINI_API_KEY` is set — not wired into the acoustic pipeline above.
> `vanirakshak_indic_conformer/` is a second fact-checking prototype of the
> same shape (Groq Whisper with an Indic adaptation + Gemini + Crawl4AI);
> it also boots and answers `/api/health` once a `.env` carrying
> `GEMINI_API_KEY` exists beside it. Neither prototype is wired into the
> acoustic pipeline. This document covers the web console + backend.

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
│  │ CM stub /  │ │ ASV stub / │ │ Channel │ │ Prosody          │   │
│  │ real WavLM │ │ real ECAPA │ │ Codec   │ │ Micro-tremor     │   │
│  └─────┬──────┘ └─────┬──────┘ └────┬────┘ └────────┬─────────┘   │
│        └───────────────┴─────────────┴───────────────┘             │
│                          │         ┌─────────────────────────────┐  │
│                          │         │ Semantic layer (parallel,    │  │
│                          │◀────────│ advisory — Groq Whisper ASR  │  │
│                          │         │ → keyword score + claim      │  │
│                          │         │ verification vs. policy)     │  │
│                          ▼         └─────────────────────────────┘  │
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
| Detection | `detectors/` | CM, ASV, channel, prosody — parallel forensic scoring. Stub by default; `VR_ASV_BACKEND=ecapa` / `VR_CM_BACKEND=wavlm` swap in real speechbrain ECAPA-TDNN / WavLM features |
| Semantic (parallel, advisory) | `semantic/` | Groq Whisper transcription → `keywords.py` (scam-vocabulary scoring) + `claims.py` (verified contradictions of published policy, e.g. "no bank asks for an OTP") → small additive `semantic_risk` term in fusion, never the primary evidence |
| Fusion | `risk/fusion.py` | Calibrated Bayesian posterior, EMA smoothing, risk 0-100 |
| Decision | `session.py` | ALLOW/CHALLENGE/BLOCK tiers, dynamic phrase engine, interlock |
| Enrollment | `detectors/asv.py`, `server/app.py` | Multi-speaker: `POST/GET /v1/enroll`, any number of enrolled `user_id`s, not one hardcoded anchor |
| Audit | `privacy/audit.py` | SHA-256 hash-chained append-only log |
| Alerting | `alerts/dispatcher.py` | Best-effort webhook POST on CHALLENGE/BLOCK events |

## Key Design Decisions

1. **Acoustic-first, not acoustic-only.** The acoustic pipeline (CM/ASV/channel/prosody →
   fusion) is the primary, authoritative detector and alone decides ALLOW/CHALLENGE/BLOCK.
   A parallel ASR/keyword/claim-verification layer (`semantic/`) was added as a small,
   advisory *additional* signal — real Groq Whisper transcription feeding a
   scam-vocabulary scorer and a rule-based policy-contradiction checker. It contributes
   one weighted term to the fusion and can never dominate it. See `docs/PROTOCOL.md`
   §3.1 and `docs/FLOWCHARTS_CORRECTED.md` for why a *transcript-first* (STT →
   NLP-only) redesign was rejected — this is additive, not a replacement.
2. **Server-authoritative verdicts** — the backend owns ALLOW/CHALLENGE/BLOCK; the frontend only displays.
3. **Honest stubs, real backends optional.** Detectors default to deterministic
   synthetic stubs (zero dependencies, run anywhere). `VR_ASV_BACKEND=ecapa` and
   `VR_CM_BACKEND=wavlm` swap in real pretrained models — see
   `docs/02-technical-strategy-and-architecture.md`'s Production Upgrade Path table
   and `archive/AUDIT_REPORT.md` §12 for what's real, what's still heuristic, and the honest
   partial recalibration result.
4. **Privacy-first** — raw audio is never persisted; DPDP Act 2023 §8(7) compliance by design.
