# VaniRakshak — Project Notes by Ayush Agnihotri

## Core Idea

**Problem:** Misinformation spreads fast. Podcasts, customer service calls,
news broadcasts — nobody fact-checks in real time.

**Solution:** VaniRakshak listens, detects claims, searches the internet,
and gives a verdict — SUPPORTED, CONTRADICTED, or UNVERIFIABLE — with evidence.

---

## Key Design Decisions

### 1. Why Tavily over Google Search API?
Tavily is purpose-built for AI agents. It returns clean, summarized content
rather than raw HTML, which is exactly what Gemini needs for evidence evaluation.
Lower cost, faster, better AI-native format.

### 2. Why separate parse → query → search → verify instead of one LLM call?
Each step is independently testable and replaceable. If Tavily goes down,
swap in a different provider. If Gemini fails, fall back to heuristics.
Modularity > monolith.

### 3. Private claim detection
If someone says "My account balance is ₹5000" — that should NEVER be
searched on the web. We detect private claim types early in the pipeline
and return UNVERIFIABLE without touching Tavily at all.

### 4. Multiple UNVERIFIABLE reasons
Not all UNVERIFIABLE results are equal:
- `PRIVATE_CLAIM_UNVERIFIABLE` — intentional, by design
- `INVALID_QUERY_UNVERIFIABLE` — query was degenerate
- `PROVIDER_FAILURE` — Tavily had an error
- `NO_EVIDENCE_UNVERIFIABLE` — search returned nothing
- `IRRELEVANT_EVIDENCE_UNVERIFIABLE` — evidence was off-topic

This granularity matters for debugging and user trust.

---

## Verification Flow

```
question / claim
     │
     ▼
parse_claim()          ← extract entity, location, org, time, type
     │
     ▼
build_query()          ← generate targeted search query
     │
     ▼
validate_query()       ← reject degenerate queries like "weather"
     │
     ▼
TavilyEvidenceProvider.search()   ← live web search
     │
     ▼
normalize_evidence()   ← clean & structure results
     │
     ▼
verify_claim()         ← Gemini reads evidence, returns verdict
     │
     ▼
{ answer, status, confidence, reason, evidence, query }
```

---

## Test Coverage Philosophy

I wrote tests for every failure mode, not just the happy path:
- What if the provider throws?
- What if the query is garbage?
- What if the claim is private?
- What if Gemini says CONTRADICTED?

76 tests. 0 failures.

---

## Future Ideas

- [ ] Voice input (Whisper transcription → VaniRakshak pipeline)
- [ ] Browser extension for real-time YouTube fact-checking
- [ ] Multi-language support (Hindi claims verified in Hindi sources)
- [ ] Live dashboard showing verification stats

---

*By Ayush Agnihotri — [@ayushagnihotrii](https://github.com/ayushagnihotrii)*
