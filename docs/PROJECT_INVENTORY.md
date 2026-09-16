# VaniRakshak — File, Tool & Dependency Inventory

> Orientation reference. Answers two questions: **which files matter and why**,
> and **what each tool/dependency contributes to the workflow**.
> Generated 2026-09-16 from a read of the working tree at `C:\vr-work\repo`.
> Architecture narrative lives in `docs/ARCHITECTURE.md`; this file is the map.

---

## 1. What the project is

VaniRakshak is a **target-conditioned recorded-audio forensic system** for
real-time AI voice-clone fraud detection on live calls. It streams 16-bit mono
PCM at 16 kHz over WebSockets to a FastAPI backend, scores risk, and enforces
`ALLOW → CHALLENGE → BLOCK` transaction interlocks.

It is really **two projects sharing one tree**:

| Track | Purpose | Where |
|---|---|---|
| **Serving / product** | Live call guard: streaming detection, fusion, interlock, dashboard | `vanirakshak-backend/`, `frontend/`, `vanirakshak-android/` |
| **Research / ML** | Train and honestly evaluate the anti-spoofing countermeasure (CM) against ASVspoof 5 | `ml/`, `benchmarks/`, `scripts/`, `reports/`, `data/` |

Plus two **standalone prototypes** not wired into the acoustic pipeline:
`vanirakshak_sotto/` and `vanirakshak_indic_conformer/` (Groq Whisper + Gemini +
Crawl4AI voice fact-checking servers).

---

## 2. Main files and their architectural role

### 2.1 Serving stack — backend (`vanirakshak-backend/vanirakshak/`)

| File | Role in architecture |
|---|---|
| `server/app.py` | **The API surface.** FastAPI HTTP + WebSocket app: `POST /v1/enroll`, `POST /v1/analyze`, `WS /v1/stream`, `POST /v1/sessions/{id}/challenge`, `GET /v1/audit`, `GET /v1/health`. Owns the JSON handshake then binary PCM framing. |
| `session.py` | **Per-call state container.** `CallSession` owns the audio buffer, detection pipeline, fusion engine, challenge engine and audit log; `push_chunk()` is the single entry point every endpoint calls. |
| `detectors/pipeline.py` | **Parallel detector orchestration.** Runs CM / ASV / channel / prosody over each window and emits a `WindowResult`; also hosts the `_default_cm` factory that falls back to stubs when no weights load. |
| `detectors/cm_aasist.py` | **The trained CM landing zone.** Loads an AASIST-L checkpoint, reads `ckpt.get("layer_index", -1)`, returns LLR (positive ⇒ genuine). **Gated**: `try_create()` returns `None` and reports `aasist_missing_weights` until weights are supplied — never claims "trained" without them. |
| `detectors/cm_wavlm.py`, `cm.py` | Real WavLM-feature heuristic and the zero-dependency DSP heuristic stub. |
| `detectors/asv_ecapa.py`, `asv.py` | Speaker verification: real speechbrain ECAPA-TDNN, and the deterministic stub fallback. |
| `detectors/channel.py`, `prosody.py`, `voice.py` | Channel/codec artifacts, prosodic micro-tremor, and voice-level evidence. |
| `risk/fusion.py` | **Calibrated Bayesian fusion.** Maps the evidence vector → `P(spoof|x)`, applies EMA smoothing + fast-rise, emits 0–100 risk and the tier. Accepts the advisory `semantic_risk` term. |
| `challenge/engine.py` | Dynamic challenge phrase engine and interlock grading. |
| `privacy/buffer.py` | RAM-only sliding window, zero disk retention (DPDP Act 2023 §8(7) by design). |
| `privacy/audit.py` | SHA-256 hash-chained append-only audit log. |
| `alerts/dispatcher.py` | Best-effort server-side webhook POST on CHALLENGE/BLOCK. Must never run client-side. |
| `semantic/keywords.py`, `claims.py`, `signal.py` | Parallel **advisory** ASR layer: Groq Whisper transcript → scam-vocabulary score + rule-based policy-contradiction check. Weighted additive term; can never dominate fusion. |
| `config.py` | Every `VR_*` environment knob (backend selection, fusion weights, thresholds, host/port/CORS). |
| `ml_bridge.py` | Bridges the trained `ml/` target-speaker model into serving. **Identity check only** — explicitly not a general deepfake detector. |
| `sdk/client.py` | Python client SDK for the API. |
| `cli/` (`bench.py`, `demo.py`, `eval_synth.py`) | `python -m vanirakshak <cmd>` subcommands behind `make eval` / `make bench`. |
| `run.py` | Single-file launcher: `python run.py demo`. |

### 2.2 Serving stack — frontend (`frontend/`)

| File | Role |
|---|---|
| `app/page.tsx` | **The console.** ~1,220-line main dashboard: WebSocket streaming client, live verdict rendering, interlock lock-out controls. Must stay in sync with `docs/PROTOCOL.md`. |
| `app/layout.tsx`, `app/globals.css` | Next.js App Router shell and global Tailwind styles. |
| `app/components/*.tsx` | `AudioUploadZone`, `DemoQuickSelector`, `EnrolledSpeakersPanel`, `ForensicVerdictCard`, `WaveformEvidenceTimeline`, `ScrollyVideoCanvas` — the presentation layer. Frontend only **displays**; the backend owns verdicts. |
| `next.config.ts`, `tsconfig.json`, `eslint.config.mjs`, `postcss.config.mjs` | Next.js / TypeScript / ESLint / Tailwind-v4 config. |

### 2.3 ML core (`ml/`)

| File | Role |
|---|---|
| `train_cm.py` | **The primary training pipeline** (1,222 lines). WavLM frozen frontend + AASIST-L head, online codec/telephony augmentation, class-weighted CE, AdamW + cosine LR, streaming dev validation, early stopping on minDCF, checkpoint export matching the detector schema. Supports `--layer-index` (5 = L5, -1 = L12). |
| `aasist_model.py` | **AASIST / AASIST-L graph-attention architecture** — `GraphAttentionLayer`, `AASISTLHead(in_dim=768)`, `AttentiveMerging`. Ported from Clova AI (Jung et al., ICASSP 2022), adapted for SSL front-ends. |
| `evaluate_cm.py` | Evaluation engine: minDCF, actDCF, CLLR, pooled EER, per-attack A17–A32 breakdown; supports pre-cached tensors and streaming raw audio. |
| `dataset.py` | `ForensicAudioDataset` (file + in-memory synthetic), collate for variable-length windows. |
| `augment.py` | Codec/telephony augmentation: G.711 A-law/µ-law, AMR-WB, GSM-EFR, noise, babble, RIR, gain jitter, bandpass. Wraps the backend DSP primitives. |
| `extract_cm_features.py`, `extract_features.py` | Offline fp16 feature caching from frozen WavLM over 4.0 s windows (64,000 samples). The former forces IPv4 DNS to dodge a Windows/CDN hang. |
| `build_manifest.py` | Parses ASVspoof protocol TSVs → canonical `path<TAB>label<TAB>attack_id` manifests. |
| `model.py` | Earlier target-conditioned forensic classifier: Bi-GRU + attentive stats pooling, sub-5 ms CPU latency per 2 s window. |
| `loss.py` | `AdditiveMarginSoftmaxLoss`, differentiable minDCF surrogate, `CalibratedCostWeightedBCELoss`. |
| `train.py`, `evaluate.py` | Training/eval for the earlier `model.py` line (the `ml/` identity model). |
| `checkpoints/` | **Only `thresholds.json` is committed.** `best_model.pt` is written here by `ml/train.py` at train time and is gitignored (`*.pt` with no un-ignore rule for `ml/`), so it does not exist on a fresh clone. Docs that cite `ml/checkpoints/best_model.pt` mean "the checkpoint this path produces", not a file in the repo. |

### 2.4 Metrics and experiment tooling

| File | Role |
|---|---|
| `benchmarks/compute_mindcf.py` | **The metric authority.** numpy-only minDCF / actDCF / CLLR / EER per ASVspoof 5 conventions (`P_tar=0.05`, `C_miss=C_fa=1.0`). Imported by both eval paths so numbers are comparable. |
| `benchmarks/run_benchmarks.py` | Benchmark driver; writes `latest_benchmark_report.json`. |
| `scripts/score_checkpoint_eval.py` | Scores N checkpoints in one streaming pass (~74 utt/s, ~2.4 h for 628 k). **Never overwrites a report**; aborts above `--max-missing-frac` instead of silently scoring zeros. |
| `scripts/check_claim.py` | **Mechanical pre-registration adjudicator** for all five criteria of the headline claim, every bar copied from `docs/10`. Fails strictly — no lenient grading. |
| `scripts/check_run1_gate.py` | Criteria 1–2 only (documented as such). |
| `scripts/run_from_scratch_experiment.py` | Pre-registered 4-run from-scratch experiment (S42/B42/S43/B43), interleaved by seed, never overwrites checkpoints, archives logs. |
| `scripts/run_warmstart_control.py` | Criterion-3 warm-start control. |
| `scripts/experiment_state.py`, `experiment_progress.py`, `experiment_control.py` | Run registry, progress reporting, orchestration. |
| `scripts/evaluate_*_benchmark.py` (3way/4way/finetune/meanpool/uniform) | The comparison ladders — each is one variable against a defined baseline. |
| `scripts/check_docs.py` | Docs-consistency gate (the 4th verification gate). |
| `scripts/integration_check.py` | End-to-end vertical-slice check: health, analyze, verify-speaker, audit chain, and a real WebSocket PCM round-trip. Boots in-process by default. |
| `scripts/dev.js` | One-command dev launcher: FastAPI :8000 + Next.js :3001, prefixed logs, `--real` to swap in real detectors. |
| `scripts/verify_gpu_env.py` | GPU/torch environment probe for the training box. |
| `scripts/wire_cm_into_backend.py` | Points backend detector config at a produced checkpoint. |
| `reports/*.json` | **Immutable published results.** New runs get new filenames. |

### 2.5 Tests and config

| File | Role |
|---|---|
| `tests/run_all.py` | **pytest-free test runner.** The project venv has no pip/pytest, so this installs a minimal `pytest` shim into `sys.modules` and resolves `tmp_path` / `monkeypatch` itself. Run the suite through this. |
| `tests/test_*.py` | 11 root-level tests: metrics, claim gates, ASR, experiment control, checkpoint naming. |
| `vanirakshak-backend/tests/` | 9 backend tests: smoke, stream e2e, auth, audit persistence, CM/ASV guards, semantic gate. `pytest.ini` points `testpaths` here. |
| `tests/e2e/` | Tier-1 feature-coverage e2e with its own `conftest.py`. |
| `pytest.ini` | `testpaths = vanirakshak-backend/tests`, `pythonpath = vanirakshak-backend .` |
| `Makefile` | The task surface: `setup dev backend lint test integration-check check-docs eval bench benchmarks clean`. |
| `docker-compose.yml` | Two-service deployment: backend :8000, frontend :3001, wired via `NEXT_PUBLIC_BACKEND_*`. |
| `AGENTS.md` | **Binding agent rules**: backend owns verdicts, protocol changes must match `docs/PROTOCOL.md`, alerts stay server-side, push target `BeastAyyG/VaniRakshak`. Also documents the `graft/` context graph. |
| `CONTRIBUTING.md`, `TEST_INFRA.md` | Branch/PR conventions and the four-gate verification contract. |
| `PROJECT.md` | The current ML milestone plan (M1–M4) and checkpoint/TSV/detector interface contracts. |
| `docs/` | Design, protocol, API, metrics, privacy, roadmap, runbook. `docs/10` holds the ground rules; `docs/11` is the handoff entry point; `docs/13` covers calibration vs discrimination; `docs/14` the integration runbook. |

### 2.6 Other subsystems

| Path | Role |
|---|---|
| `vanirakshak-android/` | **Kavach** — native Android call-guard client (Gradle/Kotlin). Real and working, with its own on-device heuristics; see its `DESIGN.md`. |
| `vanirakshak_sotto/` | Standalone fact-checking prototype, port 8082, needs `GEMINI_API_KEY`. Not in the acoustic pipeline. |
| `vanirakshak_indic_conformer/` | Same shape with an Indic ASR adaptation. Also not in the acoustic pipeline. |
| `backend/` | **Retired** standalone server; `backend/app/models/*` survive as shared modules imported by the detectors. |
| `data/` | Manifests, eval benchmark slice, feature caches, and the committed checkpoints (`cm_aasist.pt`, `cm_aasist_layer5.pt`, `cm_aasist_attm*.pt`, `cm_aasist_meanpool.pt`, `pre_aug_cm_aasist.pt`, `cm_scratch_S42.pt`,
`cm_scratch_B42.pt`, `cm_scratch_S43.pt`). See §2.7 for which checkpoints are
**not** in git. |
| `handoff/` | Syncthing share for run logs and cross-machine handoff. **Gitignored** — a local exchange folder, never pushed. |
| `graft/` | Repo context graph (see §3.7). **Gitignored** — regenerate with `graft build`. |
| `enroll_target.py` | Standalone target-speaker enrolment helper. |

---

## 2.7 On disk but **not** in git (verified 2026-09-16)

A fresh clone does not contain these. They are excluded by `.gitignore`, not lost:

| Path | Why it is not tracked | Where it really lives |
|---|---|---|
| `data/cm_aasist_ft4_s42.pt` (57 MB) | `*.pt` ignored; only `data/cm_aasist.pt` is un-ignored | local disk only |
| `data/cm_scratch_S44.pt`, `S45.pt`, `S46.pt` | same — later runs were never force-added | local disk only |
| `data/audit_log.jsonl` | runtime audit chain (`VR_AUDIT_PATH` default) | regenerated per run |
| `ml/checkpoints/best_model.pt` | `*.pt` ignored, no un-ignore for `ml/` | produced by `ml/train.py` |
| `handoff/` | Syncthing exchange folder | Syncthing peers |
| `graft/` | regenerable local cache | `graft build` |
| `A:\cm-corpus\` (**the whole corpus**) | external drive, never part of the repo | Drive A |
| `A:\cm-corpus\manifests\eval_available.tsv` | external drive | Drive A — **not** `manifests/` inside the repo |
| `.workbuddy-ai/`, `.opencode/` | agent working notes | per-developer |

**The most common location mistake:** docs writing `manifests/eval_available.tsv`
as if it lived in the repo. It does not — the full path is
`A:\cm-corpus\manifests\eval_available.tsv`, and it is only present on the
training box with Drive A mounted. `data/eval_benchmark/eval_slice_20k.tsv` is
the in-repo 20k stand-in.

---

## 3. Tools, frameworks and dependencies — what each contributes

### 3.1 ML / numerics

| Dependency | Contribution |
|---|---|
| **PyTorch** (`torch>=2.1`, box runs 2.11.0+cu128) | Everything: model definition, autograd training loop, CUDA inference. GPU is an RTX 5060 Ti. |
| **torchaudio** | Audio I/O and DSP primitives in the training path. |
| **transformers** (Hugging Face) | Supplies `WavLMModel` / `AutoFeatureExtractor` for `microsoft/wavlm-base-plus` — the frozen frontend; layer 5 of its hidden states is the CM's input. |
| **numpy** / **scipy** | All metric math. `compute_mindcf.py` is numpy-only by design so metrics run without torch. |
| **soundfile** | FLAC/WAV decoding for streaming corpus reads. |
| **tqdm** | Progress bars on long extraction and scoring passes. |
| **librosa** | MFCC features for the `ml/` identity model (via `ml_bridge.py`). |
| **speechbrain** | Real ECAPA-TDNN speaker verification (`VR_ASV_BACKEND=ecapa`). Imported lazily. |

### 3.2 Backend / serving

| Dependency | Contribution |
|---|---|
| **FastAPI** | HTTP + WebSocket API surface, Pydantic-validated schemas, OpenAPI UI. |
| **uvicorn[standard]** | ASGI server (with the websockets/httptools extras the stream needs). |
| **pydantic** | Request/response contracts — enforces the protocol so frontend and backend can't drift silently. |
| **websockets** | Binary PCM streaming under `/ws/stream`. |
| **httpx** | Outbound webhook dispatch (`alerts/dispatcher.py`) and integration-check client. |
| **python-multipart** | File upload parsing for `/v1/analyze` (WAV/FLAC). |
| **python-dotenv** | Loads `.env` (API keys) for the ASR/semantic layer. |

### 3.3 ASR / semantic (advisory layer)

| Dependency | Contribution |
|---|---|
| **groq** | Primary cloud ASR (Whisper Large v3 Turbo) — fast transcription feeding keyword scoring and claim verification. |
| **faster-whisper** | Local int8 fallback ASR when the cloud is unavailable. |
| **indic-transliteration** | Hinglish/Indic normalisation before keyword matching — matters for real Indian call audio. |

### 3.4 Frontend

| Dependency | Contribution |
|---|---|
| **Next.js 16** (App Router) | The console. Note: `AGENTS.md` warns this version has breaking changes vs. older conventions — read `node_modules/next/dist/docs/` before editing. |
| **React 19 / React DOM** | Component model; live verdict state and interlock UI. |
| **TypeScript 5** | Type safety across the frontend, including protocol payload types. |
| **Tailwind CSS 4** (`@tailwindcss/postcss`) | Styling for the dashboard. |
| **lucide-react** | Icon set. |
| **ESLint 9 + eslint-config-next** | Lint gate (`npm run lint`, `make lint`). |

### 3.5 Test, build and ops

| Tool | Contribution |
|---|---|
| **pytest** | Test framework for the backend suite; also the shim target in `tests/run_all.py`. |
| **`tests/run_all.py`** | The runner that actually works in this venv (no pip, no pytest installed there). |
| **Make** | Task surface — `make test`, `make integration-check`, `make eval`, `make bench`. |
| **npm workspaces-style scripts** | Root `package.json` proxies into `frontend/` and `scripts/dev.js`. |
| **Docker / docker-compose** | Containerised two-service deployment for the vertical slice. |
| **Gradle** (`gradlew`, `build.gradle.kts`) | Android build for Kavach. |
| **git 2.55.0.windows.3** | Version control — see the caveats in `docs/14` and project memory: flat local branch names only, no local remote-tracking refs, never `git add -A`. |
| **Syncthing** | Cross-machine handoff (`handoff/`, manifests). **Stop it before starting a training run** — it owns the manifests that feed every run. |
| **graft** | Repo context graph (`graft ask` / `skeleton` / `callers`). Documented in `AGENTS.md` as the preferred way to get context before grepping. **Built for this tree 2026-09-16** — see §3.7. Local regenerable cache, gitignored, $0 and no key to rebuild. |

### 3.6 Data and external resources

| Resource | Contribution |
|---|---|
| **ASVspoof 5 corpus** (`A:\cm-corpus`) | Training/dev/eval audio. Train 182,357 / dev 142,134 / eval 628,848 clips. Read on the fly — zero feature arrays written to Drive A (~8 GB budget). |
| **ASVspoof 5 protocols** (`A:\cm-corpus\protocols\*.track_1.tsv`) | Source of truth: space-separated, no header, 10 fields. Manifests are a lossy 3-column derivation. |
| **`data/eval_benchmark/eval_slice_20k.tsv`** | The 20,000-utterance benchmark slice for fast comparison runs. |
| **Hugging Face Hub** | One-time download of WavLM and ECAPA weights. |
| **Groq / Gemini / Crawl4AI** | Cloud ASR and the two fact-checking prototypes. |

---

## 3.7 The graft context graph

`graft build` has been run for this tree (2026-09-16), so the graph `AGENTS.md`
tells agents to consult now exists. `graft/` is a **local, regenerable cache** —
it self-registers in `.gitignore` (line 113) and is never committed. A teammate
on a fresh clone runs `graft build` to get their own.

Built in **12 s, $0, no API key**: 196 files → **1,670 nodes**, **3,941 edges**,
196 per-file cards, across python / tsx / typescript / javascript / kotlin.
Re-run after substantial code changes; `graft check` fails if it is stale (CI-able).

This is the cheapest way to measure the architecture. Two views that matter:

**Symbol hubs (most-called definitions)** — the true coupling centres:

| Symbol | Kind | Location | Callers |
|---|---|---|---|
| `AASISTLHead` | class | `ml/aasist_model.py:L220-379` | **21** |
| `compute_eer` | function | `benchmarks/compute_mindcf.py:L43-109` | **17** |
| `evaluate_system` | function | `benchmarks/compute_mindcf.py:L341-430` | **17** |
| `DetectionPipeline` | class | `detectors/pipeline.py:L87-141` | **15** |
| `synthetic_speech` | function | `data/synthetic.py:L31-84` | 15 |
| `prepare_audio_window` | function | `ml/augment.py:L78-132` | 13 |
| `AuditLog` | class | `privacy/audit.py:L51-188` | 11 |
| `_validate_checkpoint` | function | `detectors/cm_aasist.py:L288-313` | 10 |
| `CallSession` | class | `session.py:L76-273` | 10 |

The shape confirms the file inventory: `AASISTLHead` is the single most-depended-on
class (the model everything hangs off), while the `compute_mindcf.py` metric pair
is the most-depended-on *module* — exactly why it was built numpy-only.

**Directory clusters (files · symbols · hubs):**

| Cluster | Files | Symbols | Notable hubs |
|---|---|---|---|
| `vanirakshak-backend/` | 53 | 303 | `DetectionPipeline` (15←), `synthetic_speech` (15←) |
| `tests/` | 16 | 306 | `write_log` (10←), `make_valid_checkpoint` (6←) |
| `scripts/` | 31 | 210 | `assert_frozen_checkpoint` (8←), `record` (7←) |
| `ml/` | 13 | 127 | `AASISTLHead` (21←), `prepare_audio_window` (13←) |
| `backend/` *(retired)* | 16 | 82 | `ASRConfig`, `AudioPreprocessor` |
| `vanirakshak_indic_conformer/` | 4 | 46 | `get_gemini_client` |
| `vanirakshak_sotto/` | 4 | 38 | `get_openvino_device` |
| `frontend/` | 13 | 52 | `checkHealth`, `issueLocalChallenge` (page.tsx) |
| `vanirakshak-android/` | 37 | 271 | `optStr`, `str` (logging utils) |
| `benchmarks/` | 3 | **10** | `compute_eer` (17←), `evaluate_system` (17←) |
| `data/` | 3 | 15 | `session_id_for`, `_read_audio` |

Note the density contrast: `benchmarks/` holds only **10 symbols across 3 files**
yet owns two of the top-three hubs in the repo — a small, load-bearing module.
Conversely `tests/` carries 306 symbols but almost nothing outside it depends on
them, which is the healthy shape for a test tree.

**Querying it** (prefer this over grepping, per `AGENTS.md`):
`graft map` orient · `graft ask "<question>" --source` ranked nodes with inlined
code spans · `graft skeleton <file>` API surface only · `graft callers <symbol>`
exact call edges (add `--direction out`, `--depth N`) · `graft grep "<literal>"`
exhaustive · `graft viz` interactive browser.

**One caveat observed:** `graft` prints a token-savings line on every call and
`graft stats` reports *"no session recorded yet"* when invoked standalone — the
counter only accumulates inside an agent session, so don't treat the absence of
session stats as a build failure.

---

## 4. How a run actually flows

```
Training:  A:\cm-corpus protocols
             └─ ml/build_manifest.py ──▶ manifest TSV
             └─ ml/extract_cm_features.py ──▶ fp16 WavLM feature cache
             └─ ml/train_cm.py  (WavLM L5 frozen + AASIST-L head, augmented)
                  └─▶ data/cm_*.pt   (schema: state_dict + eer_threshold + layer_index)

Evaluation: data/cm_*.pt + eval manifest
             └─ scripts/score_checkpoint_eval.py  (streaming, all checkpoints)
             └─ benchmarks/compute_mindcf.py      (EER / minDCF / actDCF / CLLR)
                  └─▶ reports/*.json   (never overwritten)
             └─ scripts/check_claim.py            (adjudicate the pre-registered claim)

Serving:   browser ──WS int16 PCM──▶ server/app.py
             └─▶ session.CallSession
                  └─▶ detectors/pipeline.py  (CM / ASV / channel / prosody, + semantic)
                  └─▶ risk/fusion.py         → 0-100 risk → ALLOW / CHALLENGE / BLOCK
                       ├─▶ privacy/audit.py        (hash-chained log)
                       ├─▶ challenge/engine.py     (interlock)
                       └─▶ alerts/dispatcher.py    (server-side webhook)
```

---

## 5. Entry points worth knowing

| Command | What it does |
|---|---|
| `npm run dev:all` | Boot backend + frontend with stub detectors (no GPU, no downloads). |
| `npm run dev:all:real` | Same, with real ECAPA-TDNN + WavLM AASIST detectors. |
| `python scripts/integration_check.py` | Verify the whole vertical slice is wired. |
| `python tests/run_all.py` | Run the test suite (pytest-free runner). |
| `make test` / `make check-docs` | pytest suite / docs-consistency gate. |
| `python scripts/check_claim.py` | Adjudicate the headline claim against `docs/10`. |
| `python ml/train_cm.py --layer-index 5` | Train the L5 countermeasure. |
| `python scripts/score_checkpoint_eval.py` | Score checkpoints on an eval manifest. |
| `graft build` (done) then `graft ask "..."` | Build/query the repo context graph — see §3.7. |
