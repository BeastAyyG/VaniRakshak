# VaniRakshak — Factual Claim Verification Module

## Purpose

Detects factual claims in recorded conversation transcripts, verifies them against
current web evidence using Tavily, and returns explainable verification results with
risk contributions.

This module is one component of the VaniRakshak recorded-audio forensic prototype.
It produces a **factual risk signal** that the larger system can combine with
voice/deepfake analysis and speaker similarity signals.

## Architecture

```
Recorded Call Transcript
        |
        v
  Claim Detector          (rule-based: filters greetings, detects factual assertions)
        |
        v
  Claim Parser            (extracts location, time, org, condition, event)
        |
        v
  Query Builder           (claim-type-aware search query generation)
        |
        v
  Tavily Evidence Provider (web search via installed Tavily SDK)
        |
        v
  Evidence Normalizer     (Tavily format -> project Evidence schema)
        |
        v
  Claim Verifier          (multi-signal heuristic comparison)
        |
  +-----+-----+
  |     |     |
  v     v     v
 SUPPORTED  CONTRADICTED  UNVERIFIABLE
        |
        v
  Risk Engine             (factual risk contribution scoring)
        |
        v
  FastAPI JSON Response
```

## Claim Types

Extensible enum (`backend/schemas/claims.py`):

| Type | Examples |
|------|----------|
| WEATHER | "It is raining in Delhi" |
| FINANCE | "RBI increased the repo rate" |
| GOVERNMENT | "Government announced a new scheme" |
| NEWS | "Breaking news: ..." |
| TRANSPORT | "Delhi metro is delayed" |
| COMPANY | "The company launched a new product" |
| PERSON | "The minister was arrested" |
| EVENT | "The summit started today" |
| PRODUCT | "Apple released a new iPhone" |
| LAW | "The Supreme Court ruled..." |
| LOCATION | "An earthquake hit..." |
| TIME | Time-specific claims |
| GENERAL_FACT | Other factual claims |

## Input Format

POST `/api/v1/claims/verify`

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 4.0,
      "text": "Hello sir, how are you?"
    },
    {
      "start": 4.0,
      "end": 8.0,
      "text": "It is raining heavily in Delhi right now."
    }
  ]
}
```

## Output Format

```json
{
  "transcript": [...],
  "claims": [
    {
      "claim_id": "claim_001",
      "start": 4.0,
      "end": 8.0,
      "text": "It is raining heavily in Delhi right now.",
      "claim_type": "WEATHER",
      "structured_claim": {
        "location": "Delhi",
        "condition": "heavy rain",
        "time_reference": "current",
        "certainty": "ASSERTED"
      },
      "verification": {
        "status": "SUPPORTED",
        "verification_confidence": 0.72,
        "reason": "External evidence supports the claim...",
        "query": "current weather Delhi heavy rain",
        "evidence": [
          {
            "title": "...",
            "url": "https://...",
            "content": "...",
            "source_type": "WEB",
            "retrieved_at": "..."
          }
        ]
      },
      "risk_contribution": 0.0
    }
  ],
  "risk": {
    "claim_risk_score": 0.036,
    "risk_level": "NONE"
  }
}
```

## Tavily Integration

The module uses the Tavily Python SDK (already installed) for web evidence retrieval.

- **Abstraction**: `WebEvidenceProvider` protocol in `backend/services/evidence_provider.py`
- **Implementation**: `TavilyEvidenceProvider` wraps `tavily.TavilyClient`
- **Configuration**: `TAVILY_API_KEY` environment variable (loaded from `.env`)
- **Parameters used**: `search_depth`, `topic`, `time_range`, `max_results`, `include_answer`

Downstream code never imports `tavily` directly. The provider can be swapped without
changing any verification logic.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TAVILY_API_KEY` | Yes | Tavily API key for web search |

Set in `.env` file (never committed) or as system environment variable.

## Verification Logic

### SUPPORTED
Available relevant evidence agrees with the claim.

### CONTRADICTED
Available relevant evidence **conflicts** with the claim.
Requires actual contradictory evidence — not merely absence of evidence.

### UNVERIFIABLE
- No relevant evidence found
- Evidence is insufficient
- Claim is private/personal (e.g., "your bank account is frozen")
- Web search failed (timeout, auth error, etc.)
- Evidence sources conflict

**Critical**: "Not found" does NOT equal "false". Web retrieval failure
does NOT mean the claim is false.

## Risk Calculation

| Status | Risk Contribution |
|--------|-------------------|
| SUPPORTED | 0.0 |
| CONTRADICTED | 0.3 - 0.9 (scaled by confidence) |
| UNVERIFIABLE | 0.1 |

**Risk levels** (configurable thresholds):
- NONE: score < 0.1
- LOW: 0.1 - 0.3
- MEDIUM: 0.3 - 0.6
- HIGH: >= 0.6

**Important**: A factual contradiction does NOT automatically prove fraud.
This is a risk signal, not a verdict.

## Testing

Run all unit tests (no internet required):
```
python -m pytest tests/ --ignore=tests/test_integration_tavily.py -v
```

Run real Tavily integration test (requires API key):
```
python -m pytest tests/test_integration_tavily.py -v
```

Run all tests:
```
python -m pytest tests/ -v
```

## How to Run

### Demo (with mock evidence):
```
python run_demo.py --mock
```

### Demo (with live Tavily):
```
python run_demo.py
```

### FastAPI server:
```
python -m uvicorn backend.main:app --reload
```

Then visit `http://localhost:8000/docs` for the interactive API documentation.

## Known Limitations

- **Claim detection is rule-based**: Uses pattern matching, not a trained ML model.
  Can miss claims that don't match known patterns, and may occasionally flag
  non-factual statements. Can be replaced by a fine-tuned classifier.

- **Semantic comparison is heuristic**: Uses keyword overlap and negation detection,
  not NLI/entailment. May miss subtle contradictions or support signals.

- **Not every factual statement can be verified publicly**: Private/personal
  information (bank accounts, KYC status, etc.) cannot be verified through web search.

- **Search results can be incomplete**: Web search may not return the most relevant
  or most current information.

- **Different sources can disagree**: When sources conflict, the system returns
  UNVERIFIABLE rather than forcing a verdict.

- **Current information changes over time**: Weather conditions, stock prices, etc.
  change continuously. Evidence retrieved at one moment may not reflect conditions
  at another.

- **Entity and time resolution can make mistakes**: The rule-based parser may
  misidentify locations, organizations, or time references.

- **Performance depends on language/domain**: Currently optimized for English text
  with Indian context (Indian cities, RBI, etc.).

- **Web retrieval failure does not mean the claim is false**.

- **Claim detection can make mistakes**: Both false positives and false negatives
  are possible with the baseline detector.
