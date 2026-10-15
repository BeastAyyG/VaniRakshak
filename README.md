# VaniRakshak

**Real-time protection against AI voice-clone fraud on live calls.**

VaniRakshak listens to a call as it happens, decides whether the voice is
genuinely who it claims to be, and can hold a transaction mid-call when it isn't —
before the money moves.

## What it does

- **Detects synthetic and cloned speech** in a live audio stream.
- **Verifies the speaker** against an enrolled voice.
- **Acts, not just alerts** — each call is continuously graded
  `ALLOW` → `CHALLENGE` → `BLOCK`, and a block locks the transaction.
- **Privacy by design** — by default, audio is analysed in memory and never
  retained or sent to a third party; every decision leaves a tamper-evident
  audit record.

## Current state (Oct 2026)

- **Live deepfake verdicts on in-app internet calls** (eat/voip-calling, PR #20):
  Android WebRTC calling with a JSON-over-WebSocket signaling server. The remote
  party's voice is scored live and the in-call banner turns red (>= 0.7),
  amber (0.4-0.7) or neutral (< 0.4) by deepfake score. Verified end to end on
  two headless emulators: 75 s bidirectional call, 95 verdict frames.
- **Trained countermeasure**: WavLM-base-plus Layer-5 frontend + AASIST-L head,
  production threshold 21.9308. All 33 training checkpoints are backed up in the
  private Hugging Face repo BeastAgpandey/vanirakshak-weights.
- **Still ahead**: background incoming-call ringing (needs Firebase FCM) and
  PSTN call screening through a SIP/CPaaS layer.

## Getting started

Requires Python 3.11+ and Node.js 20+.

```bash
npm run dev:all
```

Then open **http://localhost:3001**. That boots the backend and the dashboard
together, using fast deterministic stub detectors (no GPU or model download
needed). For the real detectors:

```bash
npm run dev:all:real
```

See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for what that changes.

To confirm everything is wired up:

```bash
python scripts/integration_check.py
```

## Documentation

Detailed design, protocol, API and operations documentation lives in
[`docs/`](docs/) and is intended for the project team. New contributors should
start with [`docs/11-handoff-checklist.md`](docs/11-handoff-checklist.md).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
