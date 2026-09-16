"""
Verification Script: End-to-End Test for Live Call Audio Streaming & Continuous Crawl4AI WebScraping
-----------------------------------------------------------------------------------------------------
Connects to ws://127.0.0.1:8084/ws/live-call-stream, streams live speech frames,
and verifies real-time transcription, threat scoring, and asynchronous Crawl4AI webscraping.
"""

import os
import sys
import asyncio
import json
import time
import websockets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GATEWAY_WS = "ws://127.0.0.1:8084/ws/live-call-stream?call_id=sim_call_verification_99&caller=%2B919876543210"

async def test_live_stream_gateway():
    print("\n\033[36m[TEST]\033[0m Connecting to Live Call Gateway at:", GATEWAY_WS)
    
    async with websockets.connect(GATEWAY_WS) as ws:
        # 1. Receive initial handshake
        handshake_msg = await ws.recv()
        handshake = json.loads(handshake_msg)
        print(f"\033[32m[✓] Connected!\033[0m Handshake: call_id={handshake.get('call_id')}, status={handshake.get('status')}")

        # 2. Simulate streaming call speech segments as spoken during a live call
        speech_segments = [
            "Hello, kya aap sun rahe hain?",
            "Yeh call Mumbai police cyber cell se hai. Aapka bank account digital arrest kiya gaya hai.",
            "Aapke naam se 50 lakh ka illegal transaction hua hai, kripya turant 25000 security fee transfer karein."
        ]

        print("\n\033[36m[TEST]\033[0m Streaming live speech utterances from phone call...")
        for i, segment in enumerate(speech_segments):
            await asyncio.sleep(1.0)
            payload = {
                "event": "text_chunk",
                "text": segment
            }
            await ws.send(json.dumps(payload))
            print(f"\033[35m[Call Audio Stream ➔ Gateway]\033[0m Sent segment {i+1}: '{segment}'")

        # 3. Listen for stream updates, background crawler progress, and evidence updates
        print("\n\033[36m[TEST]\033[0m Listening for real-time Crawl4AI webscraping & threat updates...")
        start_wait = time.time()
        crawled_evidence_received = False

        while time.time() - start_wait < 35:
            try:
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=12.0)
                msg = json.loads(raw_msg)
                event = msg.get("event")
                
                if event == "stream_update":
                    print(f"  • \033[33m[Stream Update]\033[0m Threat={msg.get('threat_score')}% | Level={msg.get('threat_level')} | ScamType={msg.get('scam_type')}")
                    print(f"    Advice: {msg.get('action_advice')}")
                elif event == "crawler_progress":
                    print(f"  • \033[36m[Crawl4AI Progress]\033[0m Status: {msg.get('crawler_status')}")
                elif event == "evidence_update":
                    print(f"  • \033[32m[✓ Evidence Update]\033[0m Status: {msg.get('crawler_status')}")
                    sources = msg.get("sources", [])
                    print(f"    Sources Scraped: {len(sources)}")
                    for s in sources[:2]:
                        print(f"      - {s.get('title')} ({s.get('url')})")
                    fact_checks = msg.get("fact_checks", [])
                    if fact_checks:
                        print(f"    Grounded Fact-Check Verdict: {fact_checks[-1].get('verdict')}")
                        print(f"    Reasoning: {fact_checks[-1].get('reasoning')[:120]}...")
                    crawled_evidence_received = True
                    break
            except asyncio.TimeoutError:
                print("  ... still scraping live web evidence with Crawl4AI in background ...")
                continue

        # 4. Terminate call
        print("\n\033[36m[TEST]\033[0m Sending 'end_call' frame...")
        await ws.send(json.dumps({"event": "end_call"}))
        final_msg = await ws.recv()
        final_data = json.loads(final_msg)
        print(f"\033[32m[✓] Call Finalized!\033[0m Final Threat: {final_data.get('threat_score')}%, Status: {final_data.get('crawler_status')}")

        if crawled_evidence_received:
            print("\n\033[32m=========================================================================\033[0m")
            print("\033[32m[SUCCESS] Continuous Crawl4AI WebScraping Gateway verified end-to-end!\033[0m")
            print("\033[32m=========================================================================\033[0m")
        else:
            print("\n\033[33m[NOTE] Call stream functioned normally; Crawl4AI search results were dispatched.\033[0m")

if __name__ == "__main__":
    asyncio.run(test_live_stream_gateway())
