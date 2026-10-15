@echo off
rem VaniRakshak VoIP signaling server launcher (Phase 2)
rem Launched detached via WMI (Win32_Process.Create). Do NOT run with Start-Process over SSH.
setlocal
set "PATH=C:\VaniRakshak\.venv\Scripts;%PATH%"
cd /d C:\VaniRakshak-voip\voip-signaling
C:\VaniRakshak\.venv\Scripts\python.exe server.py >> C:\vr-out\voip-build-logs\signaling.log 2>&1
