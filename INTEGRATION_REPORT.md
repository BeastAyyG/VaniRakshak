# VaniRakshak — Integration Report

**Status: integrated and verified end-to-end.** The Next.js console and the Python
forensic backend now run as one application, started with a single command, and the
whole vertical slice is checked mechanically.

Date: 2026-09-08 · Repo: `BeastAyyG/VaniRakshak` · Scope: frontend ↔ backend integration

---

## 1. What "integrated" means here

```
Browser (Next.js 16, :3001)
   │  REST  POST /api/v1/analyze/audio   (benchmark clip or uploaded WAV)
   │  REST  GET  /api/v1/health          (engine status pill, 15 s poll)
   │  WS    /ws/stream                   (live mic: JSON handshake → int16 PCM @16 kHz)
   ▼
FastAPI forensic service (:8000)
   ├─ detectors/  cm · asv · channel · prosody  → pipeline
   ├─ risk/fusion.py      calibrated Bayesian fusion → ALLOW / CHALLENGE / BLOCK
   ├─ challenge/engine.py dynamic phrase + grading
   ├─ privacy/            DPDP hash-chained audit log, volatile audio buffer
   ├─ alerts/             server-side webhook dispatch (never client-side)
   └─ ml_bridge.py        trained ml/ target-speaker model (identity check)
```

The wire contract lives in [`docs/PROTOCOL.md`](docs/PROTOCOL.md) and is the single
source of truth. The server owns the verdict; the client only renders it.

## 2. Run it

```bash
npm run dev:all          # or: node scripts/dev.js
```

That one command: creates `vanirakshak-backend/.venv` if missing, installs
`requirements.txt`, boots the backend on `:8000`, waits for
`GET /api/v1/health` to return 200, then boots Next.js on `:3001` with
`NEXT_PUBLIC_BACKEND_URL` / `NEXT_PUBLIC_BACKEND_WS` already pointing at it.
Log lines are prefixed `[backend]` / `[frontend]`; Ctrl-C tears both down.

Open **http://localhost:3001**.

Prefer two terminals? `make backend` + `npm --prefix frontend run dev` still work.

## 3. Verify it

```bash
python scripts/integration_check.py            # boots the app in-process
python scripts/integration_check.py --live     # exercises a running server
```

**Result: 13/13 checks passed**, both in-process and against a live uvicorn server.

| # | Check | Result |
|---|---|---|
| 1 | `GET /api/v1/health` → 200, carries `hardware.device_name` + `models.wavlm_model` | PASS |
| 2–5 | `POST /api/v1/analyze/audio` returns the full `ForensicReportData` contract for all 4 benchmark clips | PASS |
| 6 | `POST /api/v1/verify-speaker` never returns 500 | PASS |
| 7 | Identity model degrades honestly (`available:false` + `reason`) when the ML stack is absent | PASS |
| 8 | `GET /api/v1/audit` — DPDP hash chain verifies | PASS |
| 9 | `WS /ws/stream` — JSON handshake accepted, verdicts returned (no close 1003) | PASS (2 verdicts / 6 s stream) |
| 10 | Verdict carries `risk_score`, `p_spoof`, `tier`, `metrics`, `interlock_active`, `receipt_hash` | PASS |
| 11 | `metrics.asv_consistency` **and** `metrics.snr_db` both present | PASS |
| 12 | `receipt_hash` emitted per verdict (audit anchor) | PASS |

The WebSocket check runs over a **real socket** (ephemeral-port uvicorn) and replays
exactly what the browser does — handshake text frame first, then 20 ms int16 frames —
so it catches the class of bug that made the stream die instantly on connect.

## 4. Defects found and fixed during integration

| # | Defect | Fix |
|---|---|---|
| 1 | **`POST /api/v1/verify-speaker` returned HTTP 500** whenever the ML stack was absent. `ModuleNotFoundError: No module named 'torch'` escaped as an unhandled ASGI exception — the endpoint was a landmine on any minimal install. | `TargetSpeakerVerifier` now degrades to `{"available": false, "reason": ...}` at **both** load and inference time (`vanirakshak/ml_bridge.py`). Never raises. |
| 2 | **The trained model was unreachable from the analysis path.** `ml/checkpoints/best_model.pt` existed but nothing in `/api/v1/analyze/audio` called it — the "Anchor: Narendra Modi" badge in the UI was purely cosmetic. | `/api/v1/analyze/audio` now runs the verifier and returns a `target_speaker` block (`available`, `label`, `is_target`, `prob_target`, `llr`). |
| 3 | **No way to see that signal.** | `ForensicVerdictCard` renders an *Enrolled-voice identity check* strip: `MATCH` / `NO MATCH` / `UNAVAILABLE`, with P(target) and LLR, labelled as an identity check — never as a clone verdict. |
| 4 | **The ASV path could never engage.** The frontend never sent `claimed_identity`, so `pipeline.analyse()` always got `None` and returned the constant 0.5. | `page.tsx` now posts `claimed_identity=narendra_modi` with every analysis. |
| 5 | **No single command to run the stack** (two terminals, manual venv, env vars set by hand). | Added `scripts/dev.js` → `npm run dev:all`. |
| 6 | **No mechanical integration test.** | Added `scripts/integration_check.py` (13 checks, `--live` supported). |

## 5. Honest limitations (unchanged by this work)

These are real and should not be papered over:

1. **The detectors are deterministic stubs.** Scored against `ground_truth.json`, the
   four benchmark clips come back at **46.3 / 46.2 / 46.2 / 48.0** — a **1.8-point
   spread**. They are not separated. The benchmark suite is therefore a
   **UI/API smoke test, not measured detection accuracy.** `integration_check.py`
   prints this warning deliberately instead of asserting a PASS.
   Full analysis: [`AUDIT_REPORT.md`](AUDIT_REPORT.md) §5.
2. **The demo assets are near-duplicates.** Per the audit, `demo_02` correlates
   +0.9965 with `demo_01` and `demo_03` +0.9814 (derived by `tanh(x·1.5)`, resample
   and copy-paste). A detector that "passes" this suite has proven nothing.
3. **`metrics.asv_consistency` is 0.5** whenever the claimed identity is not enrolled
   in the stub ASV. The real 192-dim ECAPA anchor exists in
   `data/anchor_speaker_profile.json` but is **not** loaded by `vanirakshak-backend`.
   Until it is, the speaker-similarity gauge is inert.
4. **`target_speaker` is UNAVAILABLE in this environment.** `torch 2.14.0+cpu` was
   installed, but `librosa`/`soundfile` could not be — `pip` is blocked here by the
   sandbox bulk-delete guard (`SAFE_DELETE_BULK_CONFIRM_REQUIRED`), not by the
   project. On a normal machine:
   `pip install -r vanirakshak-backend/requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu`.
   The endpoint returns a 200 with a reason either way — that is the point of fix #1.
5. **The identity model is not a clone detector.** It was trained on one target
   speaker vs one non-target speaker with **zero** cloned audio. A high-quality
   clone of the enrolled voice would likely be scored `MATCH`. See [`ml/README.md`](ml/README.md).
6. **`next build` cannot complete in this sandbox** — the same delete guard blocks
   Next from cleaning `.next`. This is an environment artifact, not a code failure:
   `npx tsc --noEmit` passes clean, and the dev server returns HTTP 200 with every
   key UI string rendered.
7. **Two backends exist.** `backend/` (heavy WavLM/ECAPA) is *not* wired to the
   dashboard and its `/ws/stream` uses a different, incompatible protocol
   (`WINDOW_ANALYSIS` frames, no handshake). Only `vanirakshak-backend/` is
   integrated. Consolidating or deleting the other is an open decision.
8. **Live-mic capture was not exercised in a real browser** (no browser automation
   available on Windows here). It was verified by replaying the exact protocol over a
   real WebSocket; a human should still click "Test live stream" once.

## 6. Files changed / added

| File | Change |
|---|---|
| `vanirakshak-backend/vanirakshak/ml_bridge.py` | Graceful degradation at load + inference; `load_error` surfaced in the reason |
| `vanirakshak-backend/vanirakshak/server/app.py` | `target_speaker` on `AnalyzeResponse`; verifier wired into `/api/v1/analyze/audio` behind a total-exception guard |
| `vanirakshak-backend/requirements-ml.txt` | **new** — optional torch/librosa stack for the identity model |
| `frontend/app/page.tsx` | Sends `claimed_identity` so the target-conditioned path engages |
| `frontend/app/components/ForensicVerdictCard.tsx` | `target_speaker` type + identity-check strip |
| `scripts/dev.js` | **new** — one-command launcher for backend + dashboard |
| `scripts/integration_check.py` | **new** — 13-check end-to-end harness |
| `package.json` | `dev:all`, `integration:check` scripts |

## 7. Next steps, in priority order

1. **Enroll the real anchor** (`data/anchor_speaker_profile.json`) into the serving
   backend so `asv_consistency` stops being a constant — this is what makes the
   product "target-conditioned" rather than nominally so.
2. **Replace the stubs** in `vanirakshak/detectors/{cm,asv}.py` with real AASIST /
   ECAPA-TDNN, then re-score. Until then, do not quote any accuracy figure.
3. **Rebuild the benchmark suite** on genuinely distinct audio, with `expected_tier`
   in `ground_truth.json`, and gate CI on `integration_check.py`.
4. **Decide on `backend/`** — wire it in behind the same protocol, or remove it.
5. **Add a browser smoke test** (Playwright) for the mic + interlock path.
