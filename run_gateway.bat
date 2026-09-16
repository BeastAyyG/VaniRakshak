@echo off
title VaniRakshak Live Call Streaming & Crawl4AI Gateway
cd /d "%~dp0"
vanirakshak-backend\.venv\Scripts\python.exe run_live_gateway.py
pause
