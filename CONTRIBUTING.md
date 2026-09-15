# VaniRakshak — Collaboration Guidelines & Team Rules

Welcome to **VaniRakshak**! To ensure seamless collaboration, prevent merge conflicts, keep code quality high, and maintain acoustic data integrity, all contributors and AI agents must follow these rules.

---

## 1. Git & Branching Strategy

### Branch Naming Conventions
Never push directly to `main`. Always create a feature or fix branch from the latest `main`:
* **Features:** `feat/<short-description>` (e.g., `feat/prosody-spectral-flux`)
* **Bug Fixes:** `fix/<short-description>` (e.g., `fix/worklet-sample-rate-clamping`)
* **Docs & Specs:** `docs/<short-description>` (e.g., `docs/add-api-architecture-diagram`)
* **Refactoring:** `refactor/<short-description>` (e.g., `refactor/risk-fusion-weights`)
* **Testing:** `test/<short-description>` (e.g., `test/e2e-websocket-handshake`)

### Pull Request & Sync Protocol
1. **Pull Before You Build:**
   Always sync your branch with upstream `main` before starting new work and before creating a PR:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. **Commit Message Format (Conventional Commits):**
   Use standard prefixes: `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, `chore:`.
   * *Good:* `feat(backend): add spectral jitter detection to prosody analyzer`
   * *Bad:* `updated some files`
3. **No Large Binaries or Session Notes:**
   * Never commit audio files (`.wav`, `.mp3`, `.flac`) directly unless placed under `data/demo/` for testing.
   * Never commit `.opencode/`, `.workbuddy-ai/`, `.venv/`, or model weights (`.pt`, `.pth`, `.bin`, `.onnx`). Check `.gitignore` before staging.

---

## 2. Architecture & Separation of Concerns

VaniRakshak consists of three main layers. Keep changes strictly decoupled:

```
VaniRakshak/
├── frontend/             # Next.js 16 (App Router) + Tailwind CSS + AudioWorklet
├── vanirakshak-backend/  # FastAPI real-time forensic WebSocket/REST service
├── docs/                 # Single source of truth for contracts & architecture
├── backend/
├── ml/
├── benchmarks/
└── data/                 # Benchmark scripts and demo audio ground-truth
```

### Golden Rules Across Layers:
1. **The Backend Owns the Interlock Decision:**
   * Risk thresholds (`ALLOW` < 35, `CHALLENGE` 35–69, `BLOCK` >= 70) and transaction lock decisions originate from `vanirakshak/risk/fusion.py` and `vanirakshak/session.py`.
   * The frontend *must not* invent its own security verdicts except when explicitly running in offline synthetic demo mode.
2. **Docs Are the Single Source of Truth:**
   * Any change to WebSocket payloads or field names must be documented first in `docs/PROTOCOL.md`.
   * Any change to the forensic signal pipeline must update `docs/FLOWCHARTS_CORRECTED.md`.
3. **Server-Side Security Alerting:**
   * Alerts and webhooks (`vanirakshak/alerts/`) are dispatched **from the server**, never from the client browser. Never expose gateway API keys or webhook secrets in frontend client code (`NEXT_PUBLIC_*`).

---

## 3. Frontend Guidelines (`frontend/`)

* **Framework:** Next.js 16 (App Router) + React 19 + TypeScript.
* **Audio Capture:** Audio must be captured via the Web Audio API **AudioWorklet** (`public/worklets/pcm-capture.js`), delivering raw 16-bit mono PCM @ 16 kHz. Avoid deprecated `ScriptProcessorNode`.
* **Field Normalization:** The normalization is done inline in `ws.onmessage` in `frontend/app/page.tsx`. Do not bind unvalidated raw JSON keys directly to UI components.
* **Linting & Type Safety:**
  Before pushing frontend code, verify:
  ```bash
  cd frontend
  npm run lint
  npm run build
  ```

---

## 4. Backend & ML Guidelines (`vanirakshak-backend/`)

* **Environment Setup (do this first):**
  ```bash
  make setup      # pip install -r requirements-dev.txt (CPU torch wheels)
  make test       # pytest -v  -> expect 21 passed, 0 skipped
  ```
  Without `make setup`, 9 of the 21 tests silently skip. Those 9 are the *only*
  coverage of the heavy `backend/` tree, so a green board of "12 passed" is
  misleading. The missing modules are **torchaudio** and **librosa** — an older
  skip message blamed torch/soundfile, which was wrong.
  Note: `backend/` is the **deprecated, non-serving** tree — see the warning at
  the top of `backend/README.md`. All new backend work belongs in
  `vanirakshak-backend/`.
* **Python Standard:** Python 3.10+ with strict type hinting (`from __future__ import annotations`).
* **Deterministic Fallbacks ("Honest Stubs"):**
  * When real heavy checkpoints (WavLM, ECAPA-TDNN) are not loaded or GPU is absent, models must gracefully fall back to the deterministic synthetic stubs in `vanirakshak/detectors/`.
  * The stack must always be runnable on a standard developer laptop with zero GPU requirements.
* **Protocol Conformance:**
  Always test your changes against the end-to-end WebSocket test:
  ```bash
  python vanirakshak-backend/tests/test_stream_e2e.py
  python -m pytest vanirakshak-backend/tests/test_smoke.py
  ```

---

## 5. Audio Data & Privacy (DPDP Compliance)

* **Ephemeral Buffering:**
  * Raw caller audio must be processed in RAM sliding windows (`vanirakshak/privacy/buffer.py`) and cleared after scoring.
  * Never save unencrypted caller audio to disk or permanent storage without explicit user consent.
* **Audit Receipts:**
  * Every transaction verdict generates a cryptographically signed SHA-256 receipt (`vanirakshak/privacy/audit.py`). Do not tamper with hash chaining logic.

---

## 6. Pre-Push Checklist

Before opening a pull request or pushing your branch:
- [ ] `git status` shows no untracked binaries, weights (`.pt`), or agent session files.
- [ ] Frontend builds without TypeScript or ESLint errors (`npm run build` in `frontend/`).
- [ ] Backend tests pass. Run the **whole** suite — `pytest` (21 tests). Do not
      run only `pytest vanirakshak-backend/tests/`: that subset skips the heavy
      `backend/` tree entirely, so it reports "12 passed" while 9 tests never run.
- [ ] `docs/PROTOCOL.md` reflects any newly introduced payload fields or parameter changes.
- [ ] `python benchmarks/run_benchmarks.py` exits 0. It exits **non-zero on
      failure by design** — investigate the failure rather than silencing it.
      Note the demo ground-truth check compares the emitted tier against a
      required tier; see `EXPECTED_TIER` in `benchmarks/run_benchmarks.py`.
- [ ] Commits follow Conventional Commits formatting.
