#!/usr/bin/env python
"""
VaniRakshak — Live Call Streaming & Continuous Crawl4AI WebScraping Gateway
----------------------------------------------------------------------------
Runs the high-speed live call ingestion gateway on Port 8084.
Streams 16kHz PCM mono audio from mobile call, performs continuous ASR & scam heuristics,
and executes asynchronous continuous Crawl4AI webscraping in the background.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_ROOT = Path(__file__).resolve().parent
INDIC_DIR = WORKSPACE_ROOT / "vanirakshak_indic_conformer"
VENV_PYTHON = WORKSPACE_ROOT / "vanirakshak-backend" / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

import threading

def run_adb_watchdog():
    """Background watchdog thread that ensures ADB reverse port forwardings remain continuously active."""
    while True:
        try:
            subprocess.run(["adb", "reverse", "tcp:8084", "tcp:8084"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["adb", "reverse", "tcp:8000", "tcp:8000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        time.sleep(4)

def setup_adb_reverse():
    """Sets up ADB reverse port forwarding for Port 8084 and Port 8000."""
    print("\n\033[36m[VaniRakshak Gateway]\033[0m Checking attached Android devices...")
    try:
        devices_out = subprocess.check_output(["adb", "devices", "-l"], text=True, stderr=subprocess.DEVNULL)
        lines = [line.strip() for line in devices_out.strip().split("\n")[1:] if line.strip()]
        attached = [l for l in lines if "device" in l and not "offline" in l and not "unauthorized" in l]

        if not attached:
            print("\033[33m[!] No active Android devices found via ADB. (Connect phone via USB with USB Debugging enabled)\033[0m")
            return

        print(f"\033[32m[✓] Detected {len(attached)} active Android device(s):\033[0m")
        for dev in attached:
            print(f"    • {dev}")

        # Reverse port 8084 (Live Call Streaming + Crawl4AI Gateway)
        res_8084 = subprocess.run(["adb", "reverse", "tcp:8084", "tcp:8084"], capture_output=True, text=True)
        # Reverse port 8000 (Forensic Core)
        res_8000 = subprocess.run(["adb", "reverse", "tcp:8000", "tcp:8000"], capture_output=True, text=True)

        if res_8084.returncode == 0 and res_8000.returncode == 0:
            print("\033[32m[✓] ADB Reverse Port Forwarding Active (Continuous Watchdog Engaged):\033[0m")
            print("    • Port 8084 (Handset ➔ PC Live Call Audio Stream & Crawl4AI Gateway)")
            print("    • Port 8000 (Handset ➔ PC Acoustic Forensic Engine)")
            t = threading.Thread(target=run_adb_watchdog, daemon=True)
            t.start()
        else:
            print(f"\033[33m[!] ADB reverse warning: {res_8084.stderr.strip()} / {res_8000.stderr.strip()}\033[0m")

    except FileNotFoundError:
        print("\033[33m[!] 'adb' not found on system PATH. Ensure Android SDK platform-tools are installed.\033[0m")
    except Exception as e:
        print(f"\033[33m[!] Could not configure ADB reverse: {e}\033[0m")

def print_banner():
    banner = r"""
==============================================================================
   _   __            _ _____       _     _           _     
  | | / /           (_) ___ \     | |   | |         | |    
  | |/ /__ _ _ __  _ _| |_/ / __ _| | __| |__   __ _| | __ 
  |    // _` | '_ \| | |    // _` | |/ /| '_ \ / _` | |/ / 
  | |\ \ (_| | | | | | | |\ \ (_| |   < | | | | (_| |   <  
  \_| \_\__,_|_| |_|_|_\_| \_\__,_|_|\_\|_| |_|\__,_|_|\_\ 
                                                           
   LIVE CALL AUDIO STREAMING & CONTINUOUS CRAWL4AI GATEWAY
==============================================================================
 • Gateway Protocol : WebSocket (ws://127.0.0.1:8084/ws/live-call-stream)
 • Audio Ingestion  : 16 kHz 16-bit Mono Linear PCM (Real-Time)
 • Multi-Lingual ASR: AI4Bharat IndicConformer / Groq Whisper LPU
 • Web Evidence Engine: Crawl4AI v0.9.3 (Headless Chromium WebScraping)
 • Grounding Mode   : Asynchronous Continuous Scrape & Claim Verification
==============================================================================
"""
    print("\033[35m" + banner + "\033[0m")

def main():
    print_banner()
    setup_adb_reverse()

    if not VENV_PYTHON.exists():
        print(f"\033[31m[ERROR] Python virtualenv not found at {VENV_PYTHON}\033[0m")
        sys.exit(1)

    print(f"\n\033[36m[VaniRakshak Gateway]\033[0m Starting Gateway Server on http://0.0.0.0:8084 ...")
    print("\033[90mPress Ctrl+C to terminate gateway.\033[0m\n")

    cmd = [
        str(VENV_PYTHON),
        "-m", "uvicorn",
        "server:app",
        "--host", "0.0.0.0",
        "--port", "8084",
        "--app-dir", str(INDIC_DIR)
    ]

    try:
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        print("\n\033[36m[VaniRakshak Gateway]\033[0m Shutting down cleanly.")

if __name__ == "__main__":
    main()
