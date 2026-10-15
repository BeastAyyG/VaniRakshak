# Phase 4 follow-up: 1003-close fix + NEUTRAL banner verification

Date: 2026-10-09 ~03:40-03:55 IST. Machine: lab-beast (TON-618).
Worktree: `C:\VaniRakshak-voip`, branch `feat/voip-calling`. Nothing pushed.
Original `C:\VaniRakshak` checkout untouched.

## 1. Backend health on entry

All three services were up from the earlier session, no relaunch needed:
- :8000 LISTENING (PID 7128, backend)
- :8001 LISTENING (PID 9468, signaling)
- :3478 UDP LISTENING (PID 15476, TURN)

## 2. The 1003-close AttributeError: fixed

**Root cause (confirmed in installed sources).** uvicorn 0.30.6's
`websockets_impl.WebSocketProtocol` subclasses
`websockets.legacy.protocol.WebSocketServerProtocol` (websockets 12.0) and
performs the opening handshake manually; it never calls websockets'
`connection_open()`. The attribute `transfer_data_task` is only assigned in
`connection_open()` (websockets/legacy/protocol.py:359; in `__init__` it is a
bare annotation, line 306). So *any* `ws.close()` on this stack raises
`AttributeError: 'WebSocketProtocol' object has no attribute
'transfer_data_task'` at protocol.py:788 (`await self.transfer_data_task`).
Full traceback captured from `C:\vr-out\voip-build-logs\backend_serve.log`
(lines 330-362) before the fix.

**Fix** (`vanirakshak-backend/vanirakshak/server/app.py`, +17/-3 lines):
- Added `_safe_ws_close(ws, code, reason="")` just above `stream_endpoint`
  (line ~911): best-effort close that swallows the library exception. The
  connection is torn down when the handler returns regardless, so a failed
  close frame must never raise.
- Replaced all three close sites: 1003 (bad handshake, line 922), 1008
  (auth fail, line 937), 1000 (revoked session, line 969).
- `py_compile` passes. Backend restarted via WMI (old PID 7128 killed with
  `taskkill /F /PID 7128 /T`, relaunched with
  `C:\VaniRakshak-voip\ops\launch_backend.cmd`); health confirms
  `healthy` + `cm_backend=aasist_l_wavlm_dev_eer_0.045` (new PID 14504).

**Verification.**
- Regression probe: connected to `/ws/stream`, sent a binary frame as the
  first frame. Server closed with code **1003** as designed; the new process
  log shows a clean accept/close with **no traceback** (the two
  AttributeError entries still visible in the log are from the old process
  before restart).
- Signaling handshake suite rerun live against :8001: **17/17 PASS**
  ("ALL STEPS PASSED"), identical step list to the earlier run.

## 3. NEUTRAL banner path: VERIFIED with genuine speech

**Setup.** Added `voip_test_audio=file:<path>` support to
`VoipCallActivity.armTestAudio()` (refactored the TTS branch into
`armTtsTestAudio()`; the file branch loads any WAV through the existing
`VoipTestAudio.wavToPcm16kMono` + `loopingFeeder`). Rebuilt
(`assembleDebug` BUILD SUCCESSFUL, incremental 15 s), reinstalled on both
emulators. Fed bob (emulator-5556) the repo's own genuine clip
`data/demo/demo_01_genuine.wav` (ground_truth.json: label `genuine`, target
speaker `narendra_modi` held out of enrolment, 10 s @ 16 kHz, expected tier
ALLOW under the aasist backend). Pushed via
`run-as com.vericall.temp` to the app's private files dir (no new permission
needed). Alice (emulator-5554) kept TTS audio and auto-called bob.

**Result: NEUTRAL observed and proven.**
- Alice's log: `verdict p_spoof=0.3098 tier=ALLOW override=null banner=NEUTRAL`,
  then 0.3398, 0.2960, 0.2936, all NEUTRAL. Bob's log confirms
  `test_audio file_ready seconds=10` (the new file feeder ran).
- UI hierarchy dump taken mid-call on alice shows the rendered banner:
  title `text="No deepfake detected"`, detail
  `text="server tier=ALLOW, p_spoof=0.294"`.
- The genuine clip scored p_spoof ~0.29-0.34 (NEUTRAL) with some windows at
  0.58-0.68 (AMBER); the detector cleanly separates genuine human speech
  from the TTS speech that scored 0.64-0.88 in the earlier run.
- Notable: tier came back ALLOW with `override=null` rather than
  BLOCK/impostor, because demo_01 is the enrolled anchor speaker
  (narendra_modi) and the ASV matched it. The NEUTRAL branch of
  VoipVerdictPolicy (p_spoof < 0.4) is what rendered, which is exactly the
  path under test.

**Evidence files (this folder):**
- `phase4-neutral-banner.png` - in-call screenshot at the NEUTRAL moment
- `phase4-neutral-uidump.xml` - uiautomator hierarchy dump showing the
  "No deepfake detected" banner text and p_spoof=0.294 detail
- `phase4-neutral-verdict.png` - earlier trapped screenshot (banner state at
  capture time not text-verified; the uidump above is the ground truth)

## 4. What remains unverified

- Physical-device calling (still emulator-only; FCM `google-services.json`
  still the known user action for background ringing).
- TURN under load / multiple concurrent calls.
- `session_closed` and CHALLENGE-tier UI paths.
- Layer 2 CPaaS account/KYC/virtual number (unchanged).

## 5. Machine state on exit

- Backend :8000 LISTENING (PID 14504, with the 1003 fix, healthy/aasist).
- Signaling :8001 LISTENING (PID 9468). TURN :3478 UDP LISTENING (PID 15476).
- Both emulators attached (emulator-5554/5556) with the rebuilt test APK.
- Scratch scripts and temp screenshots removed from `C:\vr-out`; logs kept.
- Worktree `feat/voip-calling` has two new uncommitted changes on top of the
  phase-3 state: the `_safe_ws_close` backend fix and the
  `voip_test_audio=file:` test hook. Nothing committed, nothing pushed.
- No stray windows, no GUI activity, no reboots.
