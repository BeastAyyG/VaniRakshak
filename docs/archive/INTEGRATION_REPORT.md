> **ARCHIVED 2026-09-09.** A point-in-time integration snapshot (13/13
> checks passing as of the date below). Its still-true defect table has been
> folded into `docs/RUNBOOK.md`'s known-limitations section. For the current
> integration check, run `python scripts/integration_check.py` directly
> rather than reading this file's numbers as current — the check has since
> grown to 17 assertions and this report does not reflect that.

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

1. **The detectors are deterministic stubs by default** — this remains true; nothing
   changes unless you opt in. **Optional real backends now exist**:
   `VR_ASV_BACKEND=ecapa` (real `speechbrain/spkrec-ecapa-voxceleb` embeddings) and
   `VR_CM_BACKEND=wavlm` (real `microsoft/wavlm-base-plus` features, still a heuristic
   decision on top — not a trained classifier; see `detectors/cm_wavlm.py`'s docstring).
   With real backends + refit fusion weights (`scripts/fit_fusion_weights.py`,
   `config.py`'s `_REAL_BACKEND_OVERRIDES`), `integration_check.py`'s tier-match gate
   passes 2/4 demo clips (the unambiguous genuine and impostor cases) — the other two
   land one tier hotter than hand-labeled, an honest partial result, not a claimed fix.
   Full analysis: [`AUDIT_REPORT.md`](AUDIT_REPORT.md) §12 (supersedes the stub-only
   numbers in §5, which still describe the default zero-dependency path accurately).
2. **The demo assets are near-duplicates.** Per the audit, `demo_02` correlates
   +0.9965 with `demo_01` and `demo_03` +0.9814 (derived by `tanh(x·1.5)`, resample
   and copy-paste). A detector that "passes" this suite has proven nothing.
3. **`metrics.asv_consistency` is 0.5** whenever the claimed identity is not enrolled —
   unchanged, this is correct behavior for an unenrolled speaker. A real 192-dim ECAPA
   anchor is now loadable: `data/anchor_asv_embedding_ecapa.json` (generated by
   `scripts/enroll_anchor.py --backend ecapa`), used automatically when
   `VR_ASV_BACKEND=ecapa` is set. `data/anchor_speaker_profile.json` (the file this bullet
   used to point at) has been deleted — it claimed a real ECAPA vector but was never
   actually loaded by anything; keeping a doc-only file that looked like ground truth but
   wasn't wired in was the kind of drift this report exists to catch.
4. ~~`target_speaker` is UNAVAILABLE in this environment.~~ **Resolved.** `librosa` and
   `soundfile` are installed; the trained identity model loads and scores
   (`label=modi_ji, p_target=0.9913, llr=4.7361`). Note this pulled numpy to 2.5.3, so
   `requirements.txt` no longer hard-pins `numpy==1.26.4` — verified 13/13 on both.
5. **The identity model is not a clone detector** — and this was worse than documented:
   `countermeasure_cm.pt` was **byte-identical** to `best_model.pt`, so what looked like a
   trained countermeasure was the same one-speaker identity model under a second filename
   (`AUDIT_REPORT.md` §12.1). That file is deleted, `train.py` now writes one task-named
   checkpoint per run, and `thresholds.json` carries `task`/`task_note` so its `EER 0.00%`
   cannot be re-read as anti-spoofing performance. A high-quality clone of the enrolled
   voice would still score `IDENTITY MATCH` — the UI now says exactly that. The real fix
   is a countermeasure trained on a genuine spoof corpus: that path is now built and
   tested (`ml/dataset.py::ASVspoofDataset`, `train.py --asvspoof-protocol`), and needs
   only the ASVspoof download to run. See [`ml/README.md`](ml/README.md).
6. **`next build` cannot complete in this sandbox** — the same delete guard blocks
   Next from cleaning `.next`. This is an environment artifact, not a code failure:
   `npx tsc --noEmit` passes clean, and the dev server returns HTTP 200 with every
   key UI string rendered.
7. **Closed.** `backend/`'s standalone server (its incompatible `WINDOW_ANALYSIS`
   protocol included) has been deleted. Its two genuinely-working pieces —
   `app/models/ecapa_verifier.py` and `app/models/wavlm_detector.py` — are kept as the
   shared source for `vanirakshak-backend/vanirakshak/detectors/{asv_ecapa,cm_wavlm}.py`
   (imported via `sys.path`, the same pattern `ml_bridge.py` uses for `ml/`), alongside
   `services/asr/` which `vanirakshak/semantic/signal.py` already depended on. One
   integrated backend now; see `backend/README.md`.
8. **Live-mic capture was not exercised in a real browser** (no browser automation
   available on Windows here). It was verified by replaying the exact protocol over a
   real WebSocket; a human should still click "Test live stream" once.
9. **A third client exists and was undocumented here until now: `vanirakshak-android/`.**
   A compiled APK (`VaniRakshak_app-debug.apk`) implementing on-device keyword
   triggering, a *second, independent* clone-detection heuristic in Kotlin (unreconciled
   with the Python CM detectors), real identity verification against
   `/v1/verify-speaker`, and a genuinely strong coercion-shield (overlay warning +
   OTP/banking notification suppression when a banking app opens mid-call). Backend
   connectivity is dev-only (`adb reverse`), not a deployable path. Full breakdown:
   `AUDIT_REPORT.md` §12.3.

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

## 7a. Parallel semantic signal (ASR + scam-keyword) — added post-`ayush` merge

The `ayush` branch merge (commit `63c547d`) brought in a full ASR stack
(`backend/services/asr/`: Groq Whisper cloud + IndicConformer/faster-whisper
fallback, Hinglish transliteration) with no wiring into the serving backend.
Per `docs/FLOWCHARTS_CORRECTED.md`, VaniRakshak's acoustic pipeline must stay
the sole authority for the verdict — so this ASR stack is wired in as a
**parallel, advisory context signal**, not a replacement pipeline:

- `vanirakshak/semantic/keywords.py` — scam-phrase keyword lexicon (OTP/PIN
  exfiltration, KYC/account threats, impersonated authority, urgency,
  payment requests), scored against the transliterated transcript.
- `vanirakshak/semantic/signal.py` — bridges `backend/services/asr` with the
  exact degrade-never-raise pattern `ml_bridge.py` established for the
  torch/librosa identity model: missing `torch`/`groq`/`GROQ_API_KEY`
  produces `{"available": false, "reason": ...}`, never a 500 or a dead
  socket.
- `risk/fusion.py` gained a `semantic_risk` term (`VR_FUSION_W_SEMANTIC`,
  default `0.4` — a fifth of the CM anti-spoofing weight) so keyword hits
  nudge the fused risk score without ever dominating the acoustic evidence.
- On `WS /ws/stream`, transcription runs off the event loop
  (`asyncio.to_thread` + a fire-and-forget task) so a slow/absent Groq call
  never blocks the acoustic verdict stream; its result lands on the
  *following* window (`session.py`'s `pop_pending_window` / `update_semantic`).
- New `semantic` field on both the WS verdict and `AnalyzeResponse` — see
  `docs/PROTOCOL.md` §3.1. Frontend: `ForensicVerdictCard.tsx` renders a
  "Live transcript & keyword flags" panel; `target_speaker`'s sibling.
- Optional install: `vanirakshak-backend/requirements-asr.txt` (torch, groq,
  python-dotenv) + `GROQ_API_KEY`. Verified end-to-end in this environment
  with `groq` installed and no API key set: correctly degrades to
  `{"available": false, "reason": "ASR warmup failed (ValueError: GROQ_API_KEY
  environment variable is not set...)"}`, and `integration_check.py` still
  passes 13/13 with the new field present on every verdict.
- `backend/` (the second, unwired backend from gap #7 below) is untouched —
  only the `services/asr/` subtree is imported, via a `sys.path` insert
  mirroring `ml_bridge.py`'s `ml/` bridge.

## 7. Next steps, in priority order

1. ~~Enroll the real anchor into the serving backend.~~ **Done** —
   `data/anchor_asv_embedding_ecapa.json`, real ECAPA-TDNN, `VR_ASV_BACKEND=ecapa`.
2. **Replace the CM stub with a trained anti-spoofing classifier.** Partially done:
   `VR_CM_BACKEND=wavlm` swaps in real WavLM self-supervised features, but the
   decision on top is still a hand-tuned heuristic — no AASIST or other trained
   countermeasure weights exist anywhere fetchable without a labeled spoof corpus.
   This item stays open until one is trained or sourced. Do not quote a
   spoof-detection accuracy figure until then.
3. ~~Rebuild the benchmark suite with `expected_tier`.~~ **Done** — `expected_tier`
   is in `ground_truth.json`; `integration_check.py`'s tier-match gate asserts on it
   when real backends are active (2/4 pass — see `AUDIT_REPORT.md` §12). Gating CI on
   it is still open — it's a manual `--live` / env-var invocation today, not wired
   into a CI job.
4. ~~Decide on `backend/`.~~ **Done** — standalone server removed; model-loading
   code kept and reused (see gap #7 above).
5. **Assemble a real, labeled calibration corpus.** The actual blocker on closing
   #2 and getting `expected_tier` to pass 4/4 honestly: there is no corpus in this
   repo of real genuine-vs-cloned recordings of the same speaker, and the
   synthetic/real-recording mix used to fit `scripts/fit_fusion_weights.py` has
   its own confounds (see `AUDIT_REPORT.md` §12). This is the actual prerequisite,
   not another round of hand-tuning weights.
6. **Add a browser smoke test** (Playwright) for the mic + interlock path.
