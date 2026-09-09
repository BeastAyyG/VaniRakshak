"""
VaniRakshak Kavach Sotto Edition — Local OpenVINO Dictation & Scam Shield Server
Engine: Sotto OpenVINO Hardware Accelerated Whisper (Intel CPU/GPU/NPU)
Port: 8082 (Runs side-by-side with original Faster-Whisper on 8080)
"""

import os
import io
import sys
import re
import time
import json
import logging
import urllib.parse
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

env_paths = [
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path(__file__).resolve().parent.parent.parent / ".env"
]
for p in env_paths:
    if p.exists():
        load_dotenv(p)
        break

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from google import genai
from google.genai import types

from sotto_engine import transcribe_with_sotto, get_openvino_device, get_sotto_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vanirakshak_sotto")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up Sotto OpenVINO model on server launch
    get_sotto_model()
    yield

app = FastAPI(
    title="VaniRakshak Kavach Sotto Edition — OpenVINO Voice Scam Shield",
    version="2.1.0",
    description="Local Sotto OpenVINO Whisper Speech Recognition & Crawl4AI Fact Grounding",
    lifespan=lifespan
)

STATIC_DIR = Path(__file__).resolve().parent / "fact_app" / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing. Set it in .env")
    return genai.Client(api_key=api_key)

class FactCheckRequest(BaseModel):
    claim: str = Field(..., min_length=3, description="The claim or statement to fact check")
    urls: Optional[List[str]] = Field(default=None, description="Optional custom target URLs to crawl")

def is_error_or_404_page(text: str, status_code: int) -> bool:
    if status_code and status_code >= 400:
        return True
    
    text_sample = text[:1500].lower()
    error_indicators = [
        "404 not found",
        "page not found",
        "fuera de juego",
        "error 404",
        "página no encontrada",
        "this page could not be found",
        "404 error",
        "we can't find that page",
        "nothing found"
    ]
    for ind in error_indicators:
        if ind in text_sample:
            return True
    return False

def build_search_query(text: str, is_scam: bool = False, scam_type: str = "") -> str:
    if is_scam and scam_type:
        clean_type = re.sub(r"[^\w\s]", "", scam_type)
        return f"{clean_type} warning alert advisory"
    
    clean = re.sub(r"[^\w\s\$\€\₹\.\-]", " ", text).strip()
    words = clean.split()
    query = " ".join(words[:7])
    return query

async def discover_and_crawl_sources(query_text: str, user_urls: Optional[List[str]] = None, is_scam: bool = False, scam_type: str = "") -> List[dict]:
    scraped_results = []
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
        page_timeout=15000,
    )

    urls_to_crawl = []
    async with AsyncWebCrawler() as crawler:
        if user_urls and len(user_urls) > 0:
            urls_to_crawl = [u.strip() for u in user_urls if u.strip().startswith("http")]
        else:
            query = build_search_query(query_text, is_scam=is_scam, scam_type=scam_type)
            search_url = f"https://search.yahoo.com/search?p={urllib.parse.quote(query)}"
            logger.info(f"Crawl4AI querying search engine: {search_url}")
            
            try:
                search_res = await crawler.arun(url=search_url, config=config)
                if search_res.success and not is_error_or_404_page(search_res.markdown or "", search_res.status_code):
                    search_md = (search_res.markdown or "").strip()
                    search_snippet = search_md[:400].replace("\n", " ").strip()
                    scraped_results.append({
                        "url": search_url,
                        "title": f"Sotto Web Grounding: {query}",
                        "status": search_res.status_code or 200,
                        "scraped_chars": len(search_md),
                        "markdown": search_md[:8000],
                        "snippet": search_snippet + "..." if len(search_snippet) >= 300 else search_snippet
                    })

                    authority_domains = [
                        "rbi.org.in", "sbi.co.in", "cybercrime.gov.in", "pib.gov.in", "wikipedia.org",
                        "screener.in", "moneycontrol.com", "reuters.com", "bbc.com", "laliga.com"
                    ]
                    discovered_links = []
                    for l in search_res.links.get("external", []):
                        href = l.get("href", "")
                        if href.startswith("http") and not any(bad in href.lower() for bad in ["yahoo.com", "yimg.com", "advertising", "privacy", "help", "terms", "facebook.com", "twitter.com", "x.com"]):
                            if href not in discovered_links:
                                discovered_links.append(href)

                    sorted_links = sorted(discovered_links, key=lambda u: 0 if any(d in u.lower() for d in authority_domains) else 1)
                    urls_to_crawl = sorted_links[:1]
            except Exception as e:
                logger.warning(f"Search engine crawl timed out or failed: {e}")

        for url in urls_to_crawl:
            try:
                logger.info(f"Crawl4AI scraping destination URL: {url}")
                res = await crawler.arun(url=url, config=config)
                raw_text = (res.markdown or "").strip()
                
                if is_error_or_404_page(raw_text, res.status_code or 200):
                    logger.warning(f"URL {url} is 404/error page. Skipping.")
                    continue

                if res.success and len(raw_text) > 100:
                    title = res.metadata.get("title") or url
                    snippet = raw_text[:400].replace("\n", " ").strip()
                    scraped_results.append({
                        "url": url,
                        "title": title,
                        "status": res.status_code or 200,
                        "scraped_chars": len(raw_text),
                        "markdown": raw_text[:5000],
                        "snippet": snippet + "..." if len(snippet) >= 300 else snippet
                    })
                    logger.info(f"Successfully scraped {url} ({len(raw_text)} chars)")
            except Exception as exc:
                logger.error(f"Error scraping {url}: {exc}")

    return scraped_results

def evaluate_threat_and_scam(transcript: str, client: genai.Client) -> dict:
    prompt = f"""You are VaniRakshak Sotto's elite cyber-fraud defense engine.
Analyze the following phone audio transcript (in English, Hindi, or Hinglish) for voice scams, financial fraud, impersonation, or social engineering tactics.

TRANSCRIPT:
"{transcript}"

SCAM PATTERNS TO IDENTIFY:
1. Bank KYC / Debit Card / Account Expiration / OTP / Block threats
2. Electricity Bill / Water Bill Disconnection threats ("aaj raat 9 baje bijli cut")
3. Digital Arrest / Fake Police / CBI / FedEx customs drugs parcel scams
4. Urgent Coercion / Demand for money / Gift cards / UPI / APK app installation
5. Lottery, Telegram part-time job, investment doubling

Return ONLY valid JSON matching this schema:
{{
  "is_scam": true or false,
  "threat_level": "CRITICAL_SCAM" (if score >= 75) | "SUSPICIOUS" (if score 40-74) | "SAFE" (if score < 40),
  "risk_score": 0 to 100,
  "scam_type": "Bank KYC Phishing" | "Utility Disconnection Scam" | "Digital Arrest / Impersonation" | "Investment / Lottery Fraud" | "None (Normal Speech)",
  "urgency_detected": true or false,
  "red_flags": [
    "Specific trigger phrase or psychological pressure tactic"
  ],
  "action_advice": "Clear, urgent advice to the user (e.g., 'HANG UP IMMEDIATELY. Banks never ask for OTP or threaten cutoff.')"
}}
"""
    fallback_models = ["gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.6-flash"]
    gen_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )

    for candidate_model in fallback_models:
        try:
            res = client.models.generate_content(
                model=candidate_model,
                contents=prompt,
                config=gen_config
            )
            return json.loads(res.text)
        except Exception as exc:
            logger.warning(f"Threat analysis failed with model {candidate_model}: {exc}")
            
    lower_t = transcript.lower()
    scam_keywords = ["kyc", "expired", "blocked", "electricity", "bijli", "police", "arrest", "fedex", "cbi", "otp", "debit card", "freeze"]
    matched = [kw for kw in scam_keywords if kw in lower_t]
    if matched:
        return {
            "is_scam": True,
            "threat_level": "CRITICAL_SCAM",
            "risk_score": 90,
            "scam_type": "Suspicious Social Engineering Call",
            "urgency_detected": True,
            "red_flags": [f"Contains high-risk triggers: {', '.join(matched)}"],
            "action_advice": "HANG UP IMMEDIATELY. Do not share OTP, banking details, or transfer funds."
        }
    return {
        "is_scam": False,
        "threat_level": "SAFE",
        "risk_score": 5,
        "scam_type": "Normal Speech",
        "urgency_detected": False,
        "red_flags": [],
        "action_advice": "No threat indicators detected in this audio segment."
    }

def evaluate_fact_with_gemini(claim: str, scraped_sources: List[dict], client: genai.Client) -> dict:
    valid_sources = [s for s in scraped_sources if s.get("scraped_chars", 0) > 100]
    evidence_text = "\n\n".join([
        f"--- EVIDENCE SOURCE: {s['title']} ({s['url']}) [Status: {s['status']}] ---\n{s['markdown']}"
        for s in valid_sources
    ])

    if not evidence_text.strip():
        return {
            "verdict": "UNVERIFIABLE",
            "confidence": 0,
            "summary": "No active web pages could be retrieved by Crawl4AI.",
            "reasoning": "Crawl4AI could not find or access live pages for this claim.",
            "key_facts": []
        }

    prompt = f"""You are an elite, objective fact-checking intelligence system.
Analyze the following claim or statement against the external real-time evidence scraped directly by Crawl4AI.

CLAIM TO VERIFY:
"{claim}"

EVIDENCE SCRAPED BY CRAWL4AI:
{evidence_text}

CRITICAL GROUNDING RULES:
1. Base your verdict SOLELY on the Crawl4AI evidence provided above.
2. Verdict options:
   - "SUPPORTED": The evidence confirms the claim is true / accurate.
   - "CONTRADICTED": The evidence shows the claim is false / inaccurate / refuted.
   - "FRAUDULENT_CLAIM": The evidence confirms this is a known scam or fraudulent message.
   - "UNVERIFIABLE": The evidence does not contain sufficient information to prove or disprove the claim.

Return ONLY a valid JSON object matching this structure (no markdown, no preamble):
{{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "FRAUDULENT_CLAIM" | "UNVERIFIABLE",
  "confidence": 95,
  "summary": "Brief 1-2 sentence executive summary of the verdict",
  "reasoning": "Detailed, evidence-backed explanation citing exact quotes, numbers, or facts from the scraped pages.",
  "key_facts": [
    {{"fact": "Core extracted fact or number", "source": "Source URL or title"}}
  ]
}}
"""
    fallback_models = ["gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.6-flash"]
    gen_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )

    for candidate_model in fallback_models:
        try:
            res = client.models.generate_content(
                model=candidate_model,
                contents=prompt,
                config=gen_config
            )
            return json.loads(res.text)
        except Exception as exc:
            logger.warning(f"Fact evaluation failed with {candidate_model}: {exc}")

    return {
        "verdict": "UNVERIFIABLE",
        "confidence": 0,
        "summary": "Gemini analysis error.",
        "reasoning": "Failed to complete Gemini verification.",
        "key_facts": []
    }

# ==============================================================================
# HTTP ENDPOINTS (Port 8082)
# ==============================================================================

@app.get("/")
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(index_file)

@app.get("/api/health")
async def health_check():
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    ov_device = get_openvino_device()
    return {
        "status": "online",
        "app": "VaniRakshak Sotto Edition v2.1",
        "stt_engine": f"Sotto Whisper (Groq Cloud LPU + OpenVINO Host: {ov_device})",
        "model": "whisper-large-v3-turbo",
        "openvino_device": ov_device,
        "crawler": "Crawl4AI v0.9.3 (Playwright Headless Chromium)",
        "gemini_key_configured": bool(gemini_key and len(gemini_key) > 10)
    }


@app.get("/api/audio/samples")
async def get_audio_samples():
    samples = [
        {
            "id": "scam_60s",
            "title": "1-Min Extended Bank Scam Call",
            "lang": "English",
            "tag": "⚡ Sotto OpenVINO (60s Max)",
            "filename": "sample_60s_scam.wav",
            "url": "/static/samples/sample_60s_scam.wav",
            "description": "Full 1-minute continuous bank fraud call transcribed via Sotto OpenVINO."
        },
        {
            "id": "kyc_en",
            "title": "Bank KYC Expiry Scam",
            "lang": "English",
            "tag": "🚨 Financial Fraud",
            "filename": "sample_kyc_en.mp3",
            "url": "/static/samples/sample_kyc_en.mp3",
            "description": "Urgent fake bank alert threatening account freeze."
        },
        {
            "id": "bijli_hi",
            "title": "Bijli Bill Cut Scam",
            "lang": "Hindi (हिंदी)",
            "tag": "⚡ Utility Scam",
            "filename": "sample_bijli_hi.mp3",
            "url": "/static/samples/sample_bijli_hi.mp3",
            "description": "Electricity disconnection threat today night."
        },
        {
            "id": "fedex_hinglish",
            "title": "FedEx / Police Digital Arrest",
            "lang": "Hinglish / English",
            "tag": "👮 Digital Arrest",
            "filename": "sample_fedex_hinglish.mp3",
            "url": "/static/samples/sample_fedex_hinglish.mp3",
            "description": "Narcotics / CBI arrest coercion call."
        },
        {
            "id": "chandrayaan_en",
            "title": "Chandrayaan-3 Moon Landing",
            "lang": "English",
            "tag": "🚀 Fact / News",
            "filename": "sample_chandrayaan_en.mp3",
            "url": "/static/samples/sample_chandrayaan_en.mp3",
            "description": "Factual speech claim regarding lunar touchdown."
        }
    ]
    return {"samples": samples}

@app.post("/api/audio/verify-chunk")
async def verify_audio_chunk(
    file: UploadFile = File(...),
    verify_fact: bool = Form(default=True)
):
    """
    Sotto OpenVINO Pipeline:
    1-Minute Audio Chunk -> Sotto OpenVINO Whisper -> Hindi/English transcript -> Scam Matrix -> Crawl4AI live grounding.
    """
    t_start = time.time()
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio chunk received.")
        
        # 1. Transcribe with Sotto OpenVINO Engine
        transcription = transcribe_with_sotto(audio_bytes)
        transcript_text = transcription["transcript"]

        if not transcript_text:
            return {
                "transcription": transcription,
                "threat": {
                    "is_scam": False,
                    "threat_level": "SAFE",
                    "risk_score": 0,
                    "scam_type": "No Speech Detected",
                    "red_flags": [],
                    "action_advice": "No audible voice or intelligible speech detected in this chunk."
                },
                "fact_check": None,
                "sources": [],
                "pipeline_time_sec": round(time.time() - t_start, 2)
            }

        # 2. Threat & Scam Intelligence with Gemini
        client = get_gemini_client()
        threat_data = evaluate_threat_and_scam(transcript_text, client)

        # 3. Live Grounding with Crawl4AI
        sources = []
        fact_decision = None
        if verify_fact:
            is_scam = threat_data.get("is_scam", False)
            scam_type = threat_data.get("scam_type", "")
            
            sources = await discover_and_crawl_sources(
                query_text=transcript_text,
                user_urls=None,
                is_scam=is_scam,
                scam_type=scam_type
            )
            
            if sources:
                fact_decision = evaluate_fact_with_gemini(transcript_text, sources, client)
            else:
                fact_decision = {
                    "verdict": "FRAUDULENT_CLAIM" if is_scam else "UNVERIFIABLE",
                    "confidence": 85 if is_scam else 30,
                    "summary": threat_data.get("action_advice", "Analyzed using threat intelligence patterns."),
                    "reasoning": f"Threat detection identified pattern: {scam_type}",
                    "key_facts": []
                }

        total_time = round(time.time() - t_start, 2)
        return {
            "transcription": transcription,
            "threat": threat_data,
            "fact_check": fact_decision,
            "sources": [
                {
                    "url": s["url"],
                    "title": s["title"],
                    "status": s["status"],
                    "scraped_chars": s["scraped_chars"],
                    "snippet": s["snippet"]
                }
                for s in sources
            ],
            "pipeline_time_sec": total_time
        }
    except Exception as exc:
        logger.error(f"Error in Sotto verify_audio_chunk: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/verify")
async def verify_claim(req: FactCheckRequest):
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Claim cannot be empty")

    client = get_gemini_client()
    scraped_sources = await discover_and_crawl_sources(claim, req.urls)

    if not scraped_sources:
        return {
            "claim": claim,
            "target_urls": [],
            "verdict": "UNVERIFIABLE",
            "confidence": 0,
            "summary": "Could not retrieve live web evidence.",
            "reasoning": "Crawl4AI was unable to locate valid, non-404 web pages for this claim.",
            "key_facts": [],
            "sources": []
        }

    decision = evaluate_fact_with_gemini(claim, scraped_sources, client)

    return {
        "claim": claim,
        "target_urls": [s["url"] for s in scraped_sources],
        "verdict": decision.get("verdict", "UNVERIFIABLE"),
        "confidence": decision.get("confidence", 0),
        "summary": decision.get("summary", ""),
        "reasoning": decision.get("reasoning", ""),
        "key_facts": decision.get("key_facts", []),
        "sources": [
            {
                "url": s["url"],
                "title": s["title"],
                "status": s["status"],
                "scraped_chars": s["scraped_chars"],
                "snippet": s["snippet"]
            }
            for s in scraped_sources
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8082, reload=False)
