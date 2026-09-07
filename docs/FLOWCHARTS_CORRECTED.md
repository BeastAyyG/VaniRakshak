# VaniRakshak - Corrected Architecture Flowcharts

> **If your Markdown viewer fails to render the Mermaid below**, open
> `docs/flowcharts.html` in a browser - it bundles Mermaid and renders both
> diagrams directly. The syntax here was also hardened (no `&` junctions,
> ASCII-only labels) so it parses on strict renderers.

**Why these were rewritten:** the two original flowcharts described a
*semantic scam-classification system* (Speech-to-Text -> claim extraction ->
web-search fact-checking -> LLM-RAG). That does **not** match the VaniRakshak
codebase. VaniRakshak (per `README.md`, `docs/PROTOCOL.md`,
`vanirakshak-backend/README.md`, `CHANGELOG.md`) is a **Target-Conditioned
Recorded-Audio Forensic System** for **real-time detection of voice-cloning /
deepfake-audio fraud on live calls**, ending in an **autonomous transaction
interlock**. There is **no STT, no claim extraction, no web search, and no
LLM-RAG anywhere in the system.** Detection is purely **acoustic + biometric +
Bayesian**.

---

## Flowchart 1 - Real-Time Inference / Detection Pipeline (corrected)

```mermaid
flowchart TD
    subgraph Client["Client - User Device"]
        A["User starts call or live mic stream"]
        B["AudioWorklet PCM capture<br/>16-bit mono PCM at 16 kHz<br/>Web Audio API"]
    end
    A --> B

    subgraph Ingest["Ingestion and Streaming"]
        C["JSON handshake<br/>session and call context"]
        D["Binary PCM frames<br/>WebSocket ws stream endpoint"]
        E["RAM-only sliding buffer<br/>raw audio is not persisted"]
    end
    B --> C
    C --> D
    D --> E

    subgraph Score["Multi-Layer Forensic Scoring"]
        F["CM anti-spoofing<br/>AASIST stub<br/>CM score and genuine probability"]
        G["ASV speaker verification<br/>ECAPA-TDNN stub<br/>similarity to claimed identity"]
        H["Channel and codec detector<br/>channel anomaly, codec and SNR"]
        I["Prosody detector<br/>pitch jitter and micro-tremor"]
    end
    E --> F
    E --> G
    E --> H
    E --> I

    subgraph Fusion["Calibrated Risk Fusion"]
        J["Evidence vector<br/>CM spoof, ASV mismatch,<br/>prosody, channel and context"]
        K["Bayesian sigmoid fusion<br/>EMA smoothing and fast-rise override"]
        L["Risk score from 0 to 100<br/>ALLOW, CHALLENGE or BLOCK"]
    end
    F --> J
    G --> J
    H --> J
    I --> J
    J --> K
    K --> L

    subgraph Decision["Decision and Autonomous Interlock"]
        M{"Risk tier"}
        N1["LOW RISK<br/>risk below 35"]
        N2["SUSPICIOUS<br/>risk from 35 to 69"]
        N3["CRITICAL<br/>risk at least 70"]
        O1["ALLOW<br/>transaction proceeds"]
        O2["CHALLENGE<br/>issue a dynamic phrase"]
        P2["Grade response<br/>digit order and latency"]
        O3["BLOCK<br/>engage transaction interlock"]
    end
    L --> M
    M --> N1
    N1 --> O1
    M --> N2
    N2 --> O2
    O2 --> P2
    P2 -->|Pass| O1
    P2 -->|Fail| O3
    M --> N3
    N3 --> O3

    subgraph Audit["Audit and Privacy"]
        Q["SHA-256 audit receipt<br/>append-only hash chain"]
        R["Webhook alerting<br/>server-side dispatch"]
    end
    O1 --> Q
    O2 --> Q
    O3 --> Q
    Q --> R
```

---

## Flowchart 2 - Data Synthesis, Training-Prep & Evaluation Pipeline (corrected)

```mermaid
flowchart TD
    subgraph Data["Honest Synthetic Data"]
        A1["Synthetic genuine speech<br/>broadband and jittered pitch"]
        A2["Synthetic spoofed speech<br/>vocoded and low-pass cues"]
        A3["Synthetic silence"]
        A4["Speaker enrolment audio<br/>reference identity embedding"]
        B["Labelled synthetic corpus<br/>genuine, spoofed and silence"]
    end
    A1 --> B
    A2 --> B
    A3 --> B

    subgraph Aug["Telephony Augmentation"]
        C1["G.711 A-law and mu-law"]
        C2["AMR-WB and GSM-EFR simulation"]
        C3["Noise and babble augmentation"]
        C4["Room impulse response"]
        C5["Random gain"]
        D["Augmented evaluation set"]
    end
    B --> C1
    B --> C2
    B --> C3
    B --> C4
    B --> C5
    C1 --> D
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D

    subgraph Models["Detector Models"]
        E1["CM anti-spoofing<br/>AASIST stub"]
        E2["ASV speaker verification<br/>ECAPA-TDNN stub"]
        E3["Channel and codec detector"]
        E4["Prosody micro-tremor detector"]
        E5["Calibrated Bayesian fusion<br/>sigmoid plus EMA"]
    end
    D --> E1
    D --> E2
    D --> E3
    D --> E4
    A4 --> E2
    E1 --> E5
    E2 --> E5
    E3 --> E5
    E4 --> E5

    subgraph Eval["Honest Evaluation"]
        F1["Detector and fused scores<br/>on synthetic augmented data"]
        F2["ASVspoof metrics<br/>minDCF, actDCF, CLLR and EER"]
        F3["Latency benchmark<br/>p50 and p95 per window"]
    end
    E5 --> F1
    F1 --> F2
    E5 --> F3

    subgraph Monitor["Reporting and Future Updates"]
        G1["Log risk, tier and features"]
        G2["Review thresholds and fusion weights"]
        G3["Report synthetic results honestly<br/>replace stubs before real claims"]
    end
    F2 --> G1
    F3 --> G1
    G1 --> G2
    G2 --> G3
    G3 -->|Future tuning| E5
```

---

## Discrepancy Report - Original Flowchart 1 vs Repository

| Original node | Problem | Correction |
|---|---|---|
| Audio Capture (Live call, Recorded file, Screen-captured) | VaniRakshak captures live mic via AudioWorklet AND accepts recorded file uploads via POST /v1/analyze; no screen-captured path. | Live mic PCM capture at 16 kHz via Web Audio API, streamed over WebSocket |
| Speech-to-Text Service (Whisper / ASR) | **No STT exists** in the system | Removed - operates on raw waveform, never transcripts |
| Transcript Preprocessing / Speaker & time segmentation | No transcript is produced | Removed |
| Claim Extraction (payments, KYC, prizes, threats, urgency) | Detection is **acoustic**, not semantic/NLP | Replaced by Multi-Layer Forensic Scoring (CM, ASV, channel, prosody) |
| Web Search / Scraping (RBI, NPCI, banks, news, fact-checkers) | **No web/LLM fact-checking** | Removed - replaced by calibrated Bayesian fusion over acoustic evidence |
| Fact Comparison Engine (Rules + LLM) | Not present | Removed |
| Risk Scoring (# contradictory urgent claims, scam patterns) | Risk is **Bayesian fusion** of cm/asv/prosody/channel + context | Calibrated Bayesian fusion (Layer 5) |
| Risk Tier -> Low / Medium / High / Critical | Actual tiers are **ALLOW / CHALLENGE / BLOCK**, bands LOW RISK (<35) / SUSPICIOUS (35-69) / CRITICAL (>=70) | Tiers and thresholds corrected |
| Overlay / Block payment apps / cybercrime report | Client shows verdict + interlock; alerts are webhook+email **stubs**; no external block or report API | ALLOW/CHALLENGE/BLOCK + interlock + SHA-256 audit; alerting stubbed |
| Logging / Analytics / Feedback -> Retrain | Audit + honest-eval + bench exist, but **no live user-feedback loop** | Audit (SHA-256) + DPDP; monitoring is the eval harness, not a live feedback loop |

## Discrepancy Report - Original Flowchart 2 vs Repository

| Original node | Problem | Correction |
|---|---|---|
| Real scam call recordings (anonymized) | Repo uses **honest synthetic audio only** | Removed; data is synthetic_speech vs synthetic_spoofed |
| Synthetic scam dialogues (GPT-2 / LLM) | No LLM text dialogues; audio is DSP-synthesized | Removed |
| Synthetic audio (xTTS / TTS models) | Synthetic audio is **numpy DSP** (vocoded-style) | Replaced with synthetic_speech / synthetic_spoofed generators |
| Public scam transcripts / forums | N/A | Removed |
| Audio preprocessing: Noise reduction, VAD | Sliding buffer + per-window analysis; no VAD/ASR stage | Replaced with sliding-window buffering + per-window detection |
| ASR/STT (WhisperX), Diarization, Text cleaning | None (single-stream, no transcript) | Removed |
| NER, Regex urgency, BERT embeddings | No text features | Removed; features are acoustic |
| Speaker embeddings (Resemblyzer / ECAPA-TDNN) | ASV uses **ECAPA-TDNN stub (192-dim)** for speaker **verification** vs claimed_identity | Kept ECAPA, reframed as enrolment + verification vs claimed identity |
| Prosodic features + Emotion cues | Prosody = **pitch jitter + micro-tremor consistency** (not emotion) | Prosody definition corrected |
| Call metadata, User history | Context = origin_country, transaction_value_inr, prior_fraud_score (from CRM) | Context inputs corrected |
| Text classifier (BERT / RoBERTa) | No text classifier; CM is AASIST, ASV is ECAPA | Replaced with CM (AASIST) + ASV (ECAPA) |
| Audio classifier (CNN / XVector) | CM is an AASIST stub | Replaced |
| Hybrid Fusion (weighted average) | Real fusion is **calibrated Bayesian** (sigmoid + EMA) | Fusion corrected |
| LLM-RAG Module | None | Removed |
| Risk bands Low / Medium / High / Critical | Bands are LOW / SUSPICIOUS / CRITICAL (<35 / 35-69 / >=70) | Corrected |
| Drift detection, model updates | Monitoring is **ASVspoof-5 eval harness** (minDCF/actDCF/CLLR/EER) + honest reporting | Reframed |

---

## Gaps Both Original Flowcharts Missed

1. **The CHALLENGE interlock** - the signature feature. A dynamic phrase issued mid-call so a replay of a genuine voice cannot answer, and a live TTS is caught by latency. Absent from both charts.
2. **SHA-256 audit + DPDP Act 2023 privacy** - append-only hash-chained audit receipts; RAM-only buffer, raw audio never persisted. Absent from both.
3. **WebSocket streaming contract & server-owned decision** - ws://host:8000/ws/stream, JSON handshake then binary PCM; the server owns ALLOW/CHALLENGE/BLOCK (client thresholds exist only for demo mode). Absent from both.
4. **Augmentation stack (R1)** and the **ASVspoof-5 evaluation harness** - central to Flowchart 2, missing there.
5. **Honest-reporting standard** - all CM/ASV scores are explicitly synthetic until real AASIST/ECAPA weights are dropped in.

## Status of Repository Gaps & Implementation State

All major previously identified gaps have been addressed:
- Architecture & technical strategy: Documented in `docs/02-technical-strategy-and-architecture.md`, `docs/ARCHITECTURE.md`, `docs/API_SPEC.md`, and `docs/RUNBOOK.md`.
- Codec augmentation: Documented in `docs/03-codec-augmentation-and-telephony.md` and implemented in `vanirakshak/data/augment.py`.
- Metrics & evaluation: Documented in `docs/04-metrics-and-honest-evaluation.md` and implemented in `benchmarks/compute_mindcf.py` (`python -m vanirakshak eval`).
- Privacy & DPDP compliance: Documented in `docs/07-privacy-compliance-and-apis.md` and implemented in `vanirakshak/privacy/audit.py` and `buffer.py`.
- `Makefile` and `docker-compose.yml`: Fully written and operational.
- WebSocket reconnect logic: Implemented with 5-attempt exponential backoff and jitter in `frontend/app/page.tsx`.
- Server-side alerting: Implemented in `vanirakshak/alerts/dispatcher.py` and invoked on `CHALLENGE`/`BLOCK`.

---

## Canonical ML-side pipeline (use THIS, not the "fixed" variant)

> **Heads-up:** a diagram titled **"VaniRakshak - ML Pipeline (Fixed Mermaid
> Syntax)"** has been circulating. It is **syntactically valid** (it renders) but
> it describes a **different product** - a *semantic scam-call classifier*
> (Whisper STT -> NER/BERT -> CNN/XVector -> LLM-RAG -> weighted scam-score 0-1).
> VaniRakshak's real ML side **never transcribes audio** and **never scores
> "scam probability."** Do **not** paste that diagram into the README as this
> system's design. The correct ML-side architecture is Flowchart 1 above; the
> condensed ML-only view is below.

```mermaid
flowchart LR
    A["Raw 16 kHz PCM<br/>sliding audio window"] --> B["CM anti-spoofing<br/>AASIST stub"]
    A --> C["ASV speaker verification<br/>ECAPA-TDNN stub"]
    A --> D["Channel and codec detector"]
    A --> E["Prosody micro-tremor detector"]
    B --> B1["CM score and genuine probability"]
    C --> C1["ASV consistency to claimed identity"]
    D --> D1["Channel anomaly, codec and SNR"]
    E --> E1["Prosody unnaturalness"]
    B1 --> F["Calibrated Bayesian fusion<br/>sigmoid plus EMA"]
    C1 --> F
    D1 --> F
    E1 --> F
    G["Context risk<br/>country, value and prior fraud"] --> F
    F --> H["Risk score from 0 to 100<br/>fast-rise override"]
    H --> I{"ALLOW, CHALLENGE or BLOCK"}
```

### Why the "fixed" ML diagram is wrong (ML side only)

| "Fixed" diagram claims | What the repo actually has |
|---|---|
| S1 Whisper/WhisperX STT -> transcript + speaker tags | No STT anywhere; raw PCM flows straight into detectors |
| S2 text NER / regex / BERT embeddings | No text processing at all |
| S2 audio: Resemblyzer/ECAPA embeddings, prosody (pitch/energy/rate), emotion | ECAPA is a **verification** stub vs `claimed_identity`; prosody = **pitch-jitter + micro-tremor** (a spoof cue), **no emotion**; missing the AASIST CM and channel/codec branches entirely |
| S3 BERT text classifier, CNN/XVector audio classifier, LLM-RAG -> P_text/P_audio/P_pattern | No classifiers, no LLM; outputs are `cm_score`, `asv_consistency`, `channel_anomaly`, `prosody_unnatural`, `snr_db` |
| S4 weighted avg of scam probs (0-1) | Calibrated **Bayesian** fusion of **spoof** evidence, EMA + fast-rise override, output **risk 0-100** |
| S5/S6 bands 0.3/0.6/0.85 + "block payment apps"; retrain from feedback | Tiers **ALLOW/CHALLENGE/BLOCK** at <35/35-69/>=70; **CHALLENGE** (dynamic phrase) is the signature; logging is SHA-256 audit + ASVspoof-5 eval, **no live feedback loop / auto-retrain** |
