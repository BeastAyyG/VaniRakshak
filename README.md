# 🛡️ VaniRakshak — Factual Claim Verification System

> **"Vani" = Voice/Speech | "Rakshak" = Guardian/Protector**
> Guard every spoken word with real-time factual verification.

---

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Gemini AI](https://img.shields.io/badge/Gemini-AI-orange?logo=google)](https://deepmind.google/technologies/gemini/)
[![Tavily](https://img.shields.io/badge/Tavily-Search-purple)](https://tavily.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 👨‍💻 Creator

| | |
|---|---|
| **Name** | Ayush Agnihotri |
| **GitHub** | [@ayushagnihotrii](https://github.com/ayushagnihotrii) |
| **Project** | VaniRakshak — AI-powered factual claim verification |

> This project was conceptualized, designed, and built by **Ayush Agnihotri**.
> VaniRakshak is Ayush's original contribution to the open-source AI verification space.

---

## 📖 What is VaniRakshak?

VaniRakshak is an **AI-powered factual claim verification system** that detects, parses, and verifies factual claims in spoken or written content — in real time — against live internet sources.

Whether it's a podcast, a customer service call, a debate, or a news transcript, VaniRakshak guards every factual statement and tells you whether it's **supported**, **contradicted**, or **unverifiable**.

---

## 🚀 How It Works

```
Call Audio (Mono, 16 kHz)
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   VaniRakshak Speech & Scam Pipeline                   │
│                                                                        │
│  1. 🎧 Audio Preprocessing — decode, stereo→mono, resample to 16 kHz   │
│  2. ⏱️ 4-Second Chunking    — windowing with temporal timestamp buffer │
│  3. 🇮🇳 IndicConformer ASR   — AI4Bharat 600M multilingual Hindi STT   │
│  4. 🔤 Hinglish Normalizer — preserve KYC, OTP, SBI, bank terminology  │
│  5. 🚨 Scam Threat Matrix  — Gemini 2.5 Flash urgency & fraud scoring  │
│  6. 🌐 Crawl4AI Grounding  — Playwright headless Chromium live search  │
└────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   SUPPORTED ✅ | CONTRADICTED ❌ | CRITICAL SCAM 🚨
```

---

## 🎙️ AI4Bharat IndicConformer Speech-to-Text (ASR) Subsystem

VaniRakshak features a dedicated **Hindi and Hinglish Speech-to-Text (ASR) subsystem** powered by **AI4Bharat IndicConformer** (`ai4bharat/indic-conformer-600m-multilingual`), built by IIT Madras for superior recognition across 22+ Indian languages.

### 📐 Architecture & Features
- **Primary Canonical Output**: Native Devanagari Hindi text (`hindi_text`).
- **Secondary Normalized Representation**: Colloquial Hinglish (`hinglish_text`) generated via a lightweight, deterministic transliteration layer.
- **Strict Fraud Terminology Preservation**: Critical terms like `KYC`, `OTP`, `UPI`, `PIN`, `SBI`, `RBI`, `bank account`, `suspend`, `expire`, `headquarters`, `customer care` are preserved in English rather than mangled into unnatural Hindi.
- **4-Second Chunk Windowing**: Real-time audio streaming buffer with precise millisecond timestamps.
- **Singleton Lifecycle**: Loaded once during application startup and kept warm in memory (never reloaded per 4-second chunk).
- **Hardware Acceleration**: Automatically detects CUDA GPUs (`Using device: cuda`) with graceful CPU fallback (`Using device: cpu`).
- **Zero-Disruption Fallback**: The existing Whisper engine is preserved behind the unified `ASRProvider` interface (`ASR_PROVIDER=indicconformer` or `whisper`).

### 📦 Installation
```bash
# Install core ASR dependencies:
pip install -r requirements.txt
```

### 🚀 Running the Standalone ASR Test
```bash
# Test Native Hindi audio:
python scripts/test_indicconformer.py test_audio/hindi_sample.wav

# Test Hinglish audio:
python scripts/test_indicconformer.py test_audio/hinglish_sample.wav

# Test real-world Scam Call phrasing (multiple 4s chunks):
python scripts/test_indicconformer.py test_audio/scam_sample.wav
```

### 🧪 Running Automated Unit Tests
```bash
pytest tests/test_asr.py -v
```

### ⚙️ Configuration & Environment Variables
Configure in your `.env` file:
```env
# ASR Provider Configuration
ASR_PROVIDER=indicconformer                      # 'indicconformer' or 'whisper'
ASR_MODEL=ai4bharat/indic-conformer-600m-multilingual
ASR_LANGUAGE=hi
ASR_SAMPLE_RATE=16000
ASR_DEVICE=auto                                  # 'auto', 'cuda', or 'cpu'
ASR_CHUNK_SECONDS=4.0
ASR_FALLBACK_TO_WHISPER=true                     # Fallback to faster-whisper if gated access not configured

# Hugging Face Access Token (for IndicConformer weights download)
HF_TOKEN=hf_your_token_here
```

### 🛠️ Troubleshooting Guide

1. **Hugging Face Gated Repo (401 Error)**:
   - `ai4bharat/indic-conformer-600m-multilingual` is a gated repository.
   - **Resolution**: Visit [Hugging Face IndicConformer](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual), accept user terms, and add `HF_TOKEN=your_token` in `.env` or run `huggingface-cli login`.
   - If not authenticated, the system automatically falls back to `ExistingWhisperASR` so transcription continues seamlessly.

2. **CUDA Out of Memory (OOM)**:
   - IndicConformer 600M requires ~1.5 GB of GPU VRAM.
   - **Resolution**: Set `ASR_DEVICE=cpu` in `.env` to run on system RAM.

3. **Audio Preprocessing Errors**:
   - Audio must be valid audio (`.wav`, `.mp3`, `.m4a`, `.flac`) and at least 0.1 seconds in duration.
   - VaniRakshak automatically downmixes stereo to mono and resamples any sample rate (8kHz, 44.1kHz, etc.) to 16,000 Hz.

---

## ✨ Features

- ✅ **Real-time web search** via Tavily API
- ✅ **Gemini AI** verification with confidence scores
- ✅ **8+ claim types** — weather, finance, government, company, person, event, location, and more
- ✅ **Private claim detection** — won't expose personal data to web search
- ✅ **Explainable results** — every verdict comes with reason + evidence sources
- ✅ **REST API** — clean FastAPI endpoints, Swagger docs included
- ✅ **Single question mode** — ask any factual question and get a verified answer

---

## 🗂️ Project Structure

```
VaniRakshak/
├── backend/
│   ├── api/
│   │   └── routes/
│   │       ├── claims.py          # Batch transcript verification
│   │       └── question.py        # Single question endpoint
│   ├── services/
│   │   ├── factual_question_handler.py   # Core pipeline
│   │   ├── claim_detector.py
│   │   ├── claim_parser.py
│   │   ├── claim_verifier.py
│   │   ├── query_builder.py
│   │   ├── evidence_provider.py
│   │   ├── evidence_normalizer.py
│   │   └── verification_pipeline.py
│   ├── schemas/
│   └── main.py
├── tests/
│   ├── test_factual_question_handler.py  # All verification cases
│   └── ...
├── ayush_agnihotri/                      # Creator's notes & showcase
│   ├── README.md
│   └── project_notes.md
├── requirements.txt
└── README.md                             # ← You are here
```

---

## 🔌 API Endpoints

### Single Factual Question
```http
POST /api/v1/question/
Content-Type: application/json

{
  "question": "What is the current RBI repo rate?"
}
```

**Response:**
```json
{
  "answer": "The RBI repo rate is currently 6.5%...",
  "verification_status": "SUPPORTED",
  "verification_reason": "Multiple sources confirm...",
  "verification_confidence": 0.92,
  "query": "RBI repo rate 2024",
  "evidence": [{"title": "...", "url": "...", "content": "..."}],
  "verification_method": "MODEL_VERIFICATION"
}
```

### Batch Transcript Verification
```http
POST /api/v1/claims/verify
POST /api/v1/claims/verify-text
```

### Health Check
```http
GET /api/v1/health
```

---

## 🧪 Verification Outcomes

| Status | Meaning |
|--------|---------|
| ✅ `SUPPORTED` | Web evidence confirms the claim |
| ❌ `CONTRADICTED` | Web evidence contradicts the claim |
| ⚠️ `UNVERIFIABLE` | Not enough evidence, private claim, or search failed |

---

## 🛠️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/ayushagnihotrii/VaniRakshak.git
cd VaniRakshak

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Add TAVILY_API_KEY and GEMINI_API_KEY in .env

# 4. Start the server
uvicorn backend.main:app --reload

# 5. Open API docs
# → http://localhost:8000/docs
```

---

## 🧪 Run Tests

```bash
python -m pytest -q
# 76 passed ✅
```

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI |
| AI Verification | Google Gemini |
| Web Search | Tavily Search API |
| NLI Scoring | HuggingFace Transformers |
| Testing | Pytest |
| Language | Python 3.11+ |

---

## 📄 License

MIT License © 2024 **Ayush Agnihotri** ([@ayushagnihotrii](https://github.com/ayushagnihotrii))

---

<div align="center">
  <strong>Built with ❤️ by <a href="https://github.com/ayushagnihotrii">Ayush Agnihotri</a></strong><br/>
  <em>VaniRakshak — Because every word deserves to be verified.</em>
</div>
