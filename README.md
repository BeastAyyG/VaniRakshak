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

## Getting started

Requires Python 3.11+ and Node.js 20+.

```bash
npm run dev:all
```

Then open **http://localhost:3001**. That boots the backend and the dashboard
together.

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
