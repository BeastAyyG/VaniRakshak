"""
VaniRakshak — CLI Demo Script

Runs the claim verification pipeline on example transcripts.
First milestone: "It's raining heavily in Delhi right now"
  → structured claim → Tavily search → actual web results → verdict.

Usage:
    python run_demo.py
    python run_demo.py --mock    (uses mock provider, no API key needed)
"""

from __future__ import annotations

import sys
import json
import argparse
import logging

# Ensure the project root is on the path
sys.path.insert(0, ".")

from backend.schemas.transcript import TranscriptSegment
from backend.services.verification_pipeline import VerificationPipeline, PipelineResult
from backend.services.evidence_provider import WebResult
from tests.conftest import MockEvidenceProvider


def get_demo_transcript() -> list[TranscriptSegment]:
    """The primary demo transcript from the master prompt."""
    return [
        TranscriptSegment(start=0.0, end=3.0, text="Hello sir, how are you?"),
        TranscriptSegment(start=3.0, end=5.0, text="I am calling from the bank."),
        TranscriptSegment(start=5.0, end=9.0, text="Your KYC has expired."),
        TranscriptSegment(start=9.0, end=13.0, text="Your account will be blocked today."),
        TranscriptSegment(start=13.0, end=18.0, text="It is raining heavily in Delhi right now."),
        TranscriptSegment(start=18.0, end=22.0, text="RBI increased the repo rate yesterday."),
        TranscriptSegment(start=22.0, end=24.0, text="Okay."),
        TranscriptSegment(start=24.0, end=26.0, text="Thank you."),
    ]


def get_mock_provider() -> MockEvidenceProvider:
    """Create a mock provider with canned responses."""
    mock = MockEvidenceProvider()

    mock.set_response("delhi", [
        WebResult(
            title="Delhi Weather Update - IMD",
            url="https://mausam.imd.gov.in/delhi",
            content="Heavy rain is currently reported in Delhi NCR. The Indian Meteorological Department has issued an orange alert for continuous rainfall.",
            score=0.95,
        ),
        WebResult(
            title="Delhi Rain Today",
            url="https://weather.example.com/delhi",
            content="Delhi is experiencing heavy rain and waterlogging in several areas. Continuous precipitation reported since morning.",
            score=0.88,
        ),
    ])

    mock.set_response("rbi", [
        WebResult(
            title="RBI Monetary Policy Decision",
            url="https://rbi.org.in/press-release",
            content="The Reserve Bank of India (RBI) has increased the repo rate by 25 basis points to 6.75% in its latest monetary policy review.",
            score=0.92,
        ),
    ])

    return mock


def print_result(result: PipelineResult):
    """Pretty-print the pipeline result (ASCII-safe for Windows console)."""
    print("\n" + "=" * 70)
    print("  VANIRAKSHAK -- FACTUAL CLAIM VERIFICATION RESULTS")
    print("=" * 70)

    if not result.verified_claims:
        print("\n  No factual claims detected in the transcript.")
        print("=" * 70)
        return

    for i, vc in enumerate(result.verified_claims, 1):
        sc = vc.claim.structured_claim
        v = vc.verification

        print(f"\n  +-- CLAIM {i}: {vc.claim.claim_id}")
        print(f"  |")
        print(f"  |  Time:     {vc.claim.start:.1f}s - {vc.claim.end:.1f}s")
        print(f"  |  Text:     \"{vc.claim.text}\"")
        print(f"  |  Type:     {vc.claim.claim_type.value}")
        print(f"  |")

        if sc.location:
            print(f"  |  Location:     {sc.location}")
        if sc.condition:
            print(f"  |  Condition:    {sc.condition}")
        if sc.organization:
            print(f"  |  Organization: {sc.organization}")
        if sc.event:
            print(f"  |  Event:        {sc.event}")
        if sc.time_reference:
            print(f"  |  Time Ref:     {sc.time_reference}")
        print(f"  |  Certainty:    {sc.certainty.value}")

        print(f"  |")
        print(f"  |  Query:     \"{vc.query}\"")
        print(f"  |")

        # Verdict
        status_icon = {
            "SUPPORTED": "[OK]",
            "CONTRADICTED": "[X]",
            "UNVERIFIABLE": "[?]",
        }.get(v.status.value, "?")

        print(f"  |  +-- VERDICT: {status_icon} {v.status.value}")
        print(f"  |  |   Confidence: {v.verification_confidence:.0%}")
        print(f"  |  |   Reason:     {v.reason}")
        print(f"  |  |   Risk:       {vc.risk_contribution:.3f}")

        # Evidence
        if v.evidence:
            print(f"  |  |")
            print(f"  |  |   Evidence ({len(v.evidence)} source(s)):")
            for j, ev in enumerate(v.evidence, 1):
                print(f"  |  |   {j}. [{ev.source_type.value}] {ev.title}")
                if ev.url:
                    print(f"  |  |      URL: {ev.url}")
                content_preview = ev.content[:120] + "..." if len(ev.content) > 120 else ev.content
                print(f"  |  |      \"{content_preview}\"")

        print(f"  |  +--")
        print(f"  +--")

    # Overall risk
    print(f"\n  +-- OVERALL FACTUAL RISK")
    print(f"  |   Score: {result.risk.claim_risk_score:.3f}")
    print(f"  |   Level: {result.risk.risk_level.value}")
    print(f"  +--")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="VaniRakshak Claim Verification Demo")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock evidence provider (no API key needed)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Build pipeline
    if args.mock:
        print("\n  [MOCK MODE] Using canned evidence responses.")
        provider = get_mock_provider()
    else:
        print("\n  [LIVE MODE] Using real Tavily API.")
        provider = None  # Will create TavilyEvidenceProvider from settings

    try:
        pipeline = VerificationPipeline(evidence_provider=provider)
    except ValueError as exc:
        print(f"\n  ERROR: {exc}")
        print("  Set TAVILY_API_KEY in .env or use --mock flag.")
        sys.exit(1)

    # Run
    transcript = get_demo_transcript()
    print(f"\n  Processing {len(transcript)} transcript segments...")

    result = pipeline.run(transcript)

    # Print
    print_result(result)

    # Also dump raw JSON
    print("\n  Raw JSON output:")
    output = {
        "claims": [
            {
                "claim_id": vc.claim.claim_id,
                "start": vc.claim.start,
                "end": vc.claim.end,
                "text": vc.claim.text,
                "claim_type": vc.claim.claim_type.value,
                "structured_claim": {
                    "location": vc.claim.structured_claim.location,
                    "condition": vc.claim.structured_claim.condition,
                    "organization": vc.claim.structured_claim.organization,
                    "event": vc.claim.structured_claim.event,
                    "time_reference": vc.claim.structured_claim.time_reference,
                    "certainty": vc.claim.structured_claim.certainty.value,
                },
                "query": vc.query,
                "verification": {
                    "status": vc.verification.status.value,
                    "confidence": vc.verification.verification_confidence,
                    "reason": vc.verification.reason,
                },
                "risk_contribution": vc.risk_contribution,
            }
            for vc in result.verified_claims
        ],
        "risk": {
            "claim_risk_score": result.risk.claim_risk_score,
            "risk_level": result.risk.risk_level.value,
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
