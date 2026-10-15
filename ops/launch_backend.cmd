@echo off
rem VaniRakshak scoring backend launcher (voip worktree)
rem Launched detached via WMI (Win32_Process.Create). Do NOT run with Start-Process over SSH.
setlocal
set "PATH=C:\VaniRakshak\.venv\Scripts;C:\Users\singu\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin;%PATH%"
set "VR_CM_AASIST_WEIGHTS=C:\VaniRakshak\data\cm_docs17_R42_Tmc3_deploy.pt"
set "VR_CM_BACKEND=aasist"
cd /d C:\VaniRakshak-voip\vanirakshak-backend
C:\VaniRakshak\.venv\Scripts\python.exe run.py serve --host 0.0.0.0 --port 8000 >> C:\vr-out\voip-build-logs\backend_serve.log 2>&1
