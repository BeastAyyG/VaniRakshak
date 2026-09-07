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
User question / transcript
        │
        ▼
┌────────────────────────────────────────────────┐
│               VaniRakshak Pipeline             │
│                                                │
│  1. 🔍 Claim Detection   — find factual claims │
│  2. 🧠 Claim Parsing     — classify & enrich   │
│  3. 🔎 Query Building    — smart search query  │
│  4. 🌐 Tavily Search     — live web evidence   │
│  5. 📊 Evidence Filter   — relevance gating    │
│  6. 🤖 Gemini Verify     — AI verdict          │
└────────────────────────────────────────────────┘
        │
        ▼
   SUPPORTED ✅ | CONTRADICTED ❌ | UNVERIFIABLE ⚠️
```

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
