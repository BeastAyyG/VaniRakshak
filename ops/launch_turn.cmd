@echo off
set "PATH=C:\VaniRakshak\.venv\Scripts;%PATH%"
cd /d C:\VaniRakshak-voip
C:\VaniRakshak\.venv\Scripts\python.exe C:\vr-turn\turn_server.py --port 3478 --external-ip 10.0.2.2 >> C:\vr-out\voip-build-logs\turn.log 2>&1
