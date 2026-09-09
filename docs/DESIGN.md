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
