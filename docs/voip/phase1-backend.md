# Phase 1: VaniRakshak scoring backend relaunched on lab-beast
Date: 2026-10-09 ~00:40 IST. Machine: lab-beast (TON-618, user singu).

## Entry point found
`C:\VaniRakshak-voip\vanirakshak-backend\run.py` -> `vanirakshak.cli:main`
Serve command: `python run.py serve --host 0.0.0.0 --port 8000`
(FastAPI app `vanirakshak.server.app:app` via uvicorn; GET /v1/health, WS /ws/stream.)

## Backend copy used
Worktree copy (C:\VaniRakshak-voip\vanirakshak-backend). Hash diff vs C:\VaniRakshak
shows they are NOT identical: the voip branch (feat/voip-calling) already has its
own changes (73 files vs 69; differs in vanirakshak/server/app.py, config.py,
session.py, challenge/*, risk/simulate.py, detectors/calibration.py, plus new
examples/decision_layer_fitted.env and extra tests). Ran from the worktree copy.

## Launcher
`C:\VaniRakshak-voip\ops\launch_backend.cmd` (untracked in our branch; nothing
written to C:\VaniRakshak):
- PATH prepends C:\VaniRakshak\.venv\Scripts and the WinGet ffmpeg bin
  (C:\Users\singu\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin)
- `set "VR_CM_BACKEND=aasist"` (REQUIRED: without it, pipeline.py defaults to the
  synthetic stub CM and health reports cm_backend=synthetic_cm_v1; found via smoke test)
- `set "VR_CM_AASIST_WEIGHTS=C:\VaniRakshak\data\cm_docs17_R42_Tmc3_deploy.pt"` (read-only use of the deploy checkpoint)
- cds to C:\VaniRakshak-voip\vanirakshak-backend and runs the venv python
- stdout+stderr appended to C:\vr-out\voip-build-logs\backend_serve.log

## WMI launch (detached, survives SSH disconnect)
PowerShell via -EncodedCommand (encode16.py helper; no $ inline):
```
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments `
  @{CommandLine='cmd /c "C:\VaniRakshak-voip\ops\launch_backend.cmd"';
    CurrentDirectory='C:\VaniRakshak-voip'}
```
Result: ReturnValue=0, ProcessId=29432 (WMI launcher shell; real server PID below).

## Verification (2026-10-09 ~00:41 IST, ~60s after launch)
- tasklist: python.exe PIDs 11044 (5,024 K) and 27448 (523,952 K)
- netstat: `TCP 0.0.0.0:8000  0.0.0.0:0  LISTENING  27448` (server PID 27448 holds the listener)
- GET http://127.0.0.1:8000/v1/health:
```json
{"ok":true,"status":"healthy","version":"0.1.0-sprint1","active_sessions":0,"hardware":{"device":"cpu","device_name":"CPU (Acoustic Forensic Core)"},"models":{"wavlm_model":"aasist_l_wavlm_dev_eer_0.045 / synthetic_asv_v1","vad_enabled":true,"zero_retention":true,"third_party_audio_processors":[],"cm_backend":"aasist_l_wavlm_dev_eer_0.045","asv_backend":"synthetic_asv_v1","asv_synthetic":true},"enrolment_store":{"persistent":false,"loaded":0,"skipped_other_backend":0,"unreadable":0},"challenge_grading":{"source":"server_asr","asr_provider":"local"},"anchor":{"user_id":"narendra_modi","enrolled":true,"calibrated":true,"source_clips":5,"error":null}}
```
- Log tail: clean uvicorn startup, no tracebacks; health request 200 OK.

## Stop / restart
- Stop: `ssh lab-beast 'taskkill /F /IM python.exe /T'` (kills the whole tree)
- Restart: clear `C:\vr-out\voip-build-logs\backend_serve.log` if desired, then repeat
  the WMI Invoke-CimMethod command above. Never use Start-Process over SSH
  (process tree dies on disconnect).

## Notes for later phases
- AUTH DISABLED warning in log (pre-existing dev posture; same as the last known-good run).
- Enrolment anchor narendra_modi loaded with 5 source clips (pre-existing state).
- Keep launcher in sync if new VR_* env vars are added by later VoIP phases.
