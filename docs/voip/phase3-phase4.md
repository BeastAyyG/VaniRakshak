# Phase 3 + Phase 4 Report: Android WebRTC Calling and Emulator Verification

Date: 2026-10-09 ~03:05 IST. Machine: lab-beast (TON-618). Worktree: `C:\VaniRakshak-voip`, branch `feat/voip-calling`. Nothing pushed. Original `C:\VaniRakshak` checkout untouched.

## Outcome

**Phase 3 COMPLETE. Phase 4 COMPLETE and verified.** Two headless emulators placed a real WebRTC voice call through the signaling server, exchanged bidirectional audio, and both in-call screens rendered live deepfake verdict banners driven by the scoring backend. Screenshots: `phase4-alice-verdict.png`, `phase4-bob-verdict.png` in this directory.

## What changed (file list, all in C:\VaniRakshak-voip)

New package `vanirakshak-android/app/src/main/kotlin/com/vericall/voip/`:
- `VoipSignalingClient.kt` - OkHttp WebSocket client speaking the documented signaling protocol (register, presence, call, accept, reject, sdp, ice, hangup).
- `VoipCallManager.kt` - PeerConnection lifecycle, SDP/ICE exchange, mic tap, DataChannel PCM transport, scoring stream, verdict handling.
- `VoipVerdictPolicy.kt` - pure verdict-to-banner mapping (unit-testable).
- `VoipCallActivity.kt` - call screen: register, presence list, call, accept/reject, hangup, verdict banner, plus headless test hooks (intent extras).
- `VoipTestAudio.kt` - TTS-synthesized speech WAV as a deterministic mic stand-in for headless tests.

Modified:
- `vanirakshak-android/app/src/main/kotlin/com/vericall/audio/VaniStreamingClient.kt` - added `onVerdictJson` raw-verdict callback; added `handshakeSent` gate so a binary PCM frame can never overtake the JSON handshake (the server closes with 1003 otherwise; this race was hit live and fixed).
- `vanirakshak-android/gradle/libs.versions.toml` - added WebRTC dependency.
- `vanirakshak-android/app/build.gradle.kts` - `implementation(libs.webrtc)`; ABI filter experiment reverted.
- `vanirakshak-android/app/src/main/AndroidManifest.xml` - registered VoipCallActivity (exported for adb); added `ACCESS_NETWORK_STATE` (required: without it WebRTC's NetworkMonitor throws SecurityException in JNI and the process dies with SIGILL).

New infra (lab-beast, not in the Android module):
- `C:\vr-turn\turn_server.py` - minimal RFC 5766 TURN server (UDP, long-term credentials), written for this phase because the two emulators sit behind separate NATs and cannot reach each other directly. Message framing and HMAC verified by unit tests before deployment.
- `C:\VaniRakshak-voip\ops\launch_turn.cmd` - WMI launcher for the TURN server.

## Architecture note (deviation from the original plan, documented honestly)

The plan called for tapping the remote WebRTC AudioTrack's PCM via `AudioTrack.addSink()`. That API **no longer exists** in any WebRTC Android build published on Maven Central (verified against the SDK classes: no `AudioSink` class, no `addSink` method). A true remote-track tap would require a native (C++) AudioDeviceModule, which is out of scope for this phase.

Implemented instead, with equivalent user-visible outcome:
1. Each side taps **WebRTC's own mic capture** via `JavaAudioDeviceModule.setSamplesReadyCallback()` (no second AudioRecord, no mic contention).
2. Mic PCM is resampled to 16 kHz mono int16 and forwarded to the peer over a reliable WebRTC DataChannel (`voip-pcm`).
3. Each side feeds the **remote party's** incoming DataChannel PCM into its own scorer stream, so the deepfake verdict about the person you are talking to is computed on your device from their voice. No trust in peer-reported verdicts; no signaling-server change needed.

For deepfake detection this is arguably the stronger signal (pre-Opus audio, where vocoder artifacts are cleanest). The one trade-off vs. the original plan: it scores pre-transmission audio rather than post-transmission audio.

## Build/test results (numbers)

- `assembleDebug` **BUILD SUCCESSFUL**. Final APK 51.8 MB.
- WebRTC dependency: `com.dafruits:google-webrtc` was first tried as `com.infobip:google-webrtc:1.0.45036`, whose `PeerConnectionFactory$Builder` references a missing `org.webrtc.Environment` class (broken AAR, verified across 3 versions). Switched to `com.dafruits:webrtc:123.0.0` (M123), which is internally consistent.
- Crash fixed: missing `ACCESS_NETWORK_STATE` killed the process with SIGILL on WebRTC's network thread (`jni_zero crashing due to uncaught Java exception` right after `NetworkMonitor: Start monitoring`). Adding the permission fixed it.
- Race fixed: binary PCM overtook the JSON handshake on fast reconnects, server closed with 1003. Fixed with a `handshakeSent` gate in `VaniStreamingClient`.
- Server-side bug observed (not fixed, out of scope): the backend's 1003 close path throws `AttributeError: 'WebSocketProtocol' object has no attribute 'transfer_data_task'` (websockets library version mismatch in `vanirakshak/server/app.py` line 922). Harmless on the happy path; worth fixing separately.

## Phase 4 verification (2026-10-09 ~02:58-03:00 IST)

- Two headless emulators (emulator-5554 as `alice`, emulator-5556 as `bob`), both booted with `-no-window`.
- Signaling: register, presence, call, accept, SDP offer/answer, ICE all completed per protocol. `pc_state=CONNECTED`, `ice_state=CONNECTED`.
- Media: DataChannel `voip-pcm` opened; **75 seconds of PCM flowed each direction** (`mic sent: 75s, remote scored: 75s`, mic RMS ~3500 = real TTS speech, not silence).
- Scoring: **95 verdict frames** rendered on Alice's log; Bob similar. Example: `verdict p_spoof=0.874 tier=BLOCK override=impostor banner=RED`.
- Banners: Alice showed AMBER "Unknown caller (not enrolled). Uncertain" (p_spoof=0.647); Bob showed RED "Unknown caller (not enrolled). Possible AI-cloned voice" (p_spoof=0.874). Screenshots captured headlessly via `adb exec-out screencap`.
- The TTS test audio (synthetic speech) correctly scores high p_spoof: the detector is doing its job.
- Clean teardown: auto-hangup at 100s/120s, `hangup_sent`, `call_ended reason=local-hangup`, state returned to READY. No crashes in the final run.

## VoipVerdictPolicy (as implemented)

For the VoIP path, `tier_override == "impostor"` is treated as "unenrolled / unknown caller", not a hard block. The banner is driven by the CM deepfake score: p_spoof >= 0.7 RED "Possible AI-cloned voice", 0.4-0.7 AMBER "Uncertain", < 0.4 NEUTRAL "No deepfake detected". Backend untouched. This matches the judgment call flagged for user review.

## What remains unverified

- Real human speech (the test used TTS-synthesized speech; the mic path on a physical device is untested).
- Physical-device calling (emulator-only so far; real phones need the FCM step below for background ringing).
- The NEUTRAL banner path (p_spoof < 0.4 was not observed; TTS speech scored 0.64-0.88).
- TURN server under load / multiple concurrent calls (single call tested).
- `session_closed` and CHALLENGE-tier UI paths.
- The backend 1003-close AttributeError (server-side, noted above).

## User actions still open

1. **FCM background ringing**: add `google-services.json` to the app module and the Firebase Messaging dependency, then wire an incoming-call push that wakes `VoipCallActivity`. Without this, incoming calls only ring while the app is open. This is the one known user action for the calling feature.
2. **Layer 2 (real-number screening)**: CPaaS account + KYC, virtual screening number, per-minute budget, publicly reachable server. Checklist in `india-voip-setup-guide.md`.
3. **Review**: the VoipVerdictPolicy (impostor-as-unenrolled) call and the DataChannel-PCM architecture deviation above.

## Machine state on exit

- Backend :8000 LISTENING (PID 7128, healthy, aasist backend).
- Signaling :8001 LISTENING (PID 9468).
- TURN :3478 UDP LISTENING (PID 15476, started this phase via WMI, logs to `C:\vr-out\voip-build-logs\turn.log`; firewall rule "VaniRakshak TURN 3478" exists).
- Both emulators still running (emulator-5554 = AVD `vr-emu-a`, emulator-5556 = AVD `vr-emu-b`; API 34 x86_64 Google APIs images, cloned from the pre-existing `vr-pixel` AVD since no sdkmanager/avdmanager exists on the box) with the test APK installed. Relaunch recipe (WMI): `C:\Android\Sdk\emulator\emulator.exe -avd vr-emu-a -no-window -no-boot-anim -no-snapshot -gpu swiftshader_indirect -memory 2048` (same for vr-emu-b); cold boot ~60-90s.
- No stray windows; no GUI activity; no reboots. Original `C:\VaniRakshak` checkout untouched. Nothing pushed to any remote.
- Test scripts used: `/tmp/call_test.ps1`, `/tmp/turn_up.ps1` (ephemeral); launchers in `C:\VaniRakshak-voip\ops\`.
