# VaniRakshak

Recorded-audio forensic prototype for detecting suspicious/AI-generated voice
impersonation and presenting explainable evidence.

## Factual Claim Verification Module

This module detects factual claims in recorded conversation transcripts, verifies
them against current web evidence, and returns explainable verification results.

See [docs/claim_verification.md](docs/claim_verification.md) for full documentation.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and set your TAVILY_API_KEY
```

### 3. Run the demo

```bash
# With live Tavily API
python run_demo.py

# With mock evidence (no API key needed)
python run_demo.py --mock
```

### 4. Run the API server

```bash
python -m uvicorn backend.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

### 5. Run tests

```bash
# Unit tests (no internet needed)
python -m pytest tests/ --ignore=tests/test_integration_tavily.py -v

# All tests including real Tavily integration
python -m pytest tests/ -v
```

## Project Structure

```
VaniRakshak/
  backend/
    main.py                  # FastAPI app
    config.py                # Environment settings
    schemas/                 # Pydantic data models
      transcript.py          # TranscriptSegment
      claims.py              # ClaimType, StructuredClaim, DetectedClaim
      evidence.py            # Evidence, SourceType
      verification.py        # VerificationStatus, VerificationResult
      risk.py                # RiskLevel, ClaimRiskResult
    services/                # Core business logic
      claim_detector.py      # Factual claim detection
      claim_parser.py        # Claim structuring
      query_builder.py       # Search query generation
      evidence_provider.py   # WebEvidenceProvider + TavilyProvider
      evidence_normalizer.py # Tavily -> Evidence normalization
      claim_verifier.py      # Claim vs evidence comparison
      risk_engine.py         # Factual risk scoring
      verification_pipeline.py # Full pipeline orchestration
    api/routes/
      claims.py              # POST /api/v1/claims/verify
  tests/                     # All tests (40 unit + 2 integration)
  docs/                      # Module documentation
  run_demo.py                # CLI demo script
```

## License

Internal project — VaniRakshak team.
