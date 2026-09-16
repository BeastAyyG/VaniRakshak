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

## Where things live

The tree holds two tracks that share one repo:

| Path | What it is |
|---|---|
| `vanirakshak-backend/` | **Serving backend.** FastAPI app, detectors, Bayesian fusion, interlock, audit log. Owns every verdict. |
| `frontend/` | **Dashboard.** Next.js 16 console; displays verdicts and locks controls. Decides nothing. |
| `vanirakshak-android/` | **Kavach** — native Android call-guard client. |
| `ml/` | **Research/ML.** Training (`ml/train_cm.py`, `ml/train.py`), AASIST-L architecture, feature extraction. |
| `benchmarks/` | Metric authority (`benchmarks/compute_mindcf.py` — EER / minDCF / actDCF / CLLR). |
| `scripts/` | Experiment, scoring and integration tooling (`scripts/integration_check.py`, `scripts/check_claim.py`). |
| `reports/` | Published results — immutable, one file per run, never overwritten. |
| `tests/`, `vanirakshak-backend/tests/` | Root suite (11 tests, run via `tests/run_all.py`) and backend suite (9 tests, `pytest`). |
| `data/` | Manifests, eval slice, feature caches, committed checkpoints. |
| `backend/` | Retired standalone server; some shared modules still imported by detectors. |
| `docs/` | All design, protocol, API and operations documentation — start at [`docs/README.md`](docs/README.md). |

Two standalone prototypes — `vanirakshak_sotto/` and
`vanirakshak_indic_conformer/` (Groq/Gemini/Crawl4AI voice fact-checking) — are
**not** wired into the acoustic pipeline.

For the full file-by-file map, including what is *not* in git, see
[`docs/PROJECT_INVENTORY.md`](docs/PROJECT_INVENTORY.md).

## Documentation

Detailed design, protocol, API and operations documentation lives in
[`docs/`](docs/) and is intended for the project team. There is an index at
[`docs/README.md`](docs/README.md).

- New contributors: [`docs/11-handoff-checklist.md`](docs/11-handoff-checklist.md).
- WebSocket/HTTP contract: [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — binding on both
  `frontend/` and `vanirakshak-backend/`.
- File/tool map: [`docs/PROJECT_INVENTORY.md`](docs/PROJECT_INVENTORY.md).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
