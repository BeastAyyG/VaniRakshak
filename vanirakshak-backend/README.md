# VaniRakshak — Real-Time Voice Cloning Impersonation Detection Backend

The one live backend for [VaniRakshak](../README.md). Runs the WebSocket
streaming pipeline, REST one-shot analysis, speaker enrollment, the challenge
engine, and the DPDP-compliant audit chain.

By default this runs entirely on **deterministic, zero-dependency stub
detectors** (numpy/scipy only, no GPU or pretrained weights needed) — so the
whole pipeline, including calibrated risk fusion, the challenge engine, and
WebSocket streaming, works on any laptop out of the box. Real detector
backends (real ECAPA-TDNN speaker embeddings, real WavLM self-supervised
features) are opt-in behind environment variables, not a future TODO — see
§"Real detector backends" below. There is no trained anti-spoofing
classifier (AASIST or otherwise) anywhere in this repo, stub or real; see
`docs/02-technical-strategy-and-architecture.md`'s Production Upgrade Path
table for what training one would take.

## Quick start

```bash
# From the repo root — one command, boots both backend and frontend:
npm run dev:all
```

Or run just this backend:

```bash
# 1. Install (pure-Python + FastAPI; ~30s on a fresh venv)
pip install -r requirements.txt

# 2. Run the offline demo (no server, no GPU, no audio needed)
python run.py demo

# 3. Run the full FastAPI server (WebSocket + REST + SDK)
python run.py serve
# Then in another shell:
python examples/sdk_demo.py
```

`python run.py` with no subcommand also defaults to `serve`. The demo prints
a stream of risk verdicts, a triggered challenge, and a final minDCF number
on synthetic telephony data.

## What is here

| Component | File | Status |
|---|---|---|
| Streaming PCM buffer (RAM-only, DPDP-compliant) | `vanirakshak/privacy/buffer.py` | ✅ working |
| Sliding-window inference loop | `vanirakshak/detectors/pipeline.py` | ✅ working |
| Acoustic countermeasure (CM) | `vanirakshak/detectors/cm.py` (default heuristic stub), `cm_wavlm.py` (`VR_CM_BACKEND=wavlm`, real WavLM features, still a heuristic decision) | 🟡 stub default; real features opt-in |
| Speaker verification (ASV) | `vanirakshak/detectors/asv.py` (default numpy stub), `asv_ecapa.py` (`VR_ASV_BACKEND=ecapa`, real ECAPA-TDNN embeddings) | 🟡 stub default; real backend opt-in |
| Enrolled target-speaker anchor loading | `vanirakshak/anchor.py` | ✅ working |
| Telephony codec & prosody features | `vanirakshak/detectors/channel.py`, `prosody.py` | ✅ working |
| Calibrated Bayesian risk fusion (0–100) | `vanirakshak/risk/fusion.py` | ✅ working |
| Dynamic challenge engine + response analyzer | `vanirakshak/challenge/engine.py` | ✅ working |
| Advisory ASR + scam-keyword/claim signal | `vanirakshak/semantic/` (Groq Whisper, optional — `GROQ_API_KEY`) | ✅ working, advisory-only |
| Trained identity-check model (`ml/`) | `vanirakshak/ml_bridge.py` | ✅ working, optional (`requirements-ml.txt`) |
| DPDP Act 2023 privacy + SHA-256 audit receipts | `vanirakshak/privacy/audit.py` | ✅ working (in-memory only — a restart loses the chain) |
| Server-side alert dispatch | `vanirakshak/alerts/` | ✅ working (no-op until `VR_ALERT_WEBHOOK_URL` set) |
| Codec augmentation (G.711 a/μ-law, AMR-WB sim, noise) | `vanirakshak/data/augment.py` | ✅ working |
| ASVspoof 5 metric harness (minDCF/actDCF/CLLR/EER) | re-uses `benchmarks/compute_mindcf.py` | ✅ working |
| FastAPI server | `vanirakshak/server/app.py` — see endpoints below | ✅ working |
| 5-line Python SDK | `vanirakshak/sdk/client.py` | ✅ working |
| OpenAPI docs (auto) | `/docs` on the served URL | ✅ working |

### Endpoints

`WS /ws/stream` (aliased `/v1/stream`, `/api/v1/stream`) — live streaming,
see `../docs/PROTOCOL.md`. REST, each also aliased under `/api/v1/`:
`GET /health`, `GET`/`POST /enroll`, `POST /analyze` (+ `/analyze/audio`),
`POST /verify-speaker`, `POST /sessions`, `POST /sessions/{id}/challenge`,
`GET /audit`. Full reference: `../docs/API_SPEC.md`.

### Real detector backends (opt-in, not future work)

```bash
pip install -r requirements-asv.txt      # speechbrain ECAPA-TDNN
pip install -r requirements-asr.txt      # groq + torch, for the semantic/ layer
pip install -r requirements-ml.txt       # the ml/ identity-check model

VR_ASV_BACKEND=ecapa VR_CM_BACKEND=wavlm python run.py serve
```

Both real backends degrade to their stub on any load failure (missing
package, missing weights) with only a warning logged — the server never
refuses to start. `docs/02-technical-strategy-and-architecture.md`'s
Production Upgrade Path table has the honest calibration numbers this
produces; `docs/archive/AUDIT_REPORT.md` §12 has the full history of how
these were validated.

## Architecture

```
Incoming PCM (1.0 s chunks)
       │
       ▼
┌──────────────────────────┐
│ RAM-only sliding buffer  │   ◀── DPDP §8(7): zero retention
│ (4.04 s window, 1.0s hop)│
└──────────┬───────────────┘
           ▼
┌────────────────────────────────────────┐
│ Multi-layer scoring                    │
│  • CM (stub, or real WavLM features)   │
│  • ASV (stub, or real ECAPA-TDNN)      │
│  • Telephony codec detect → s_ch       │
│  • Prosody micro-tremor → s_pros       │
│  • Advisory: ASR + keyword/claim signal│
└──────────┬─────────────────────────────┘
           ▼
┌────────────────────────────────────────┐
│ Calibrated risk fusion                 │
│  P(spoof|x) = σ(w·x + b)               │
│  risk = round(100·P, 1)                │
│  tier = ALLOW | CHALLENGE | BLOCK      │
│  + deterministic impostor override     │
└──────────┬─────────────────────────────┘
           ▼
┌────────────────────────────────────────┐
│ 35 ≤ risk < 70  → Challenge engine     │
│ risk ≥ 70       → BLOCK + interlock    │
│ Always          → SHA-256 audit receipt│
└────────────────────────────────────────┘
```

See `../docs/ARCHITECTURE.md` for the system-wide design and
`../docs/02-technical-strategy-and-architecture.md` for the acoustic/fusion
math in detail.

## License

Apache 2.0 — see `../LICENSE`.
