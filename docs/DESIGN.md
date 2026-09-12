# VaniRakshak UI Design System

## Overview
- Dark theme
- Geist font
- Tailwind v4

## Dashboard Layout
- Scrolly video pipeline explainer
- Console with risk meter
- Forensic benchmark suite
- Forensic verdict card
- Waveform evidence timeline

## Color Palette
Risk tiers mapped to colors:
- Green = ALLOW
- Amber = CHALLENGE
- Red = BLOCK

## Component Architecture
- `ScrollyVideoCanvas`
- `DemoQuickSelector`
- `AudioUploadZone`
- `ForensicVerdictCard`
- `WaveformEvidenceTimeline`
- `EnrolledSpeakersPanel` — multi-speaker enrollment UI (`GET`/`POST /v1/enroll`);
  shows a "Real ECAPA-TDNN biometric" badge only when the backend reports
  `asv_backend: ecapa_tdnn_voxceleb`, otherwise an honest "Demo biometric
  (not real voiceprint)" badge

## Interlock UI States
- Normal
- Challenge active
- Transaction locked

## Real-time Streaming UI
- Audio level meter
- Live event log
- Risk score gauge

## VaniRakshak Kavach — Android Guard Screen (vanirakshak-android)
Two pipeline phases visible on the guard screen:

### ARMED (keyword-gated, default ON)
- Server is idle; on-device `SpeechRecognizer` listens for money-talk
  (bank, OTP, UPI, payment, rupee, पैसा, बैंक, खाता…).
- Screen shows **👂 "Listening for money-talk (bank, OTP, UPI, payment…) — server wakes on match."**
- No network traffic, mic owned by recognizer, battery-efficient.

### SCORING (server streaming)
- Triggered automatically when a keyword is heard, or immediately if keyword-gate is OFF.
- WebSocket to `ws://<host>:8000/ws/stream` — 16-bit mono PCM @ 16 kHz.
- Server verdicts arrive ~1/sec: `tier` (ALLOW/CHALLENGE/BLOCK), `risk_score`, `interlock_active`, `receipt_hash`, optional `challenge.phrase`.
- Screen shows **Server: BLOCK/ALLOW/CHALLENGE** badge, `receipt_hash` line, challenge box if issued.

### UI Controls
- **Contacts screen → Backend settings**: toggle "Wake server only on money-talk keywords" (persisted, default ON), host:port editable (LAN IP for real device, `127.0.0.1:8000` + `adb reverse` for emulator).
- **Guard screen**: server tier badge, receipt hash, challenge phrase + grade button, live amplitude pulse.

### Pastel Design System (vibrant pastel, modern, screen-adaptive)
- Primary violet `#6C4DF6`, deep indigo ink, washes: lavender/sky/mint/peach/lemon/rose
- Rounded cards (20dp), pill chips, 28dp top-radius bottom sheets, 14dp overlay corners
- 52dp CTA buttons, white-on-violet primaries, fluid `match_parent`/`weight`/`ScrollView` layouts
- All 6 layouts rewritten with IDs intact — zero Kotlin changes needed
