# 15 — Next steps after Phase 2 and Path A

Written 2026-09-27. This picks up where `docs/14-master-plan.md` §6 left off. It
records where each step stands after commits `56caf12` (steps 3–5) and `31ffa63`
(Path A), names the gaps those commits left, and sets the order of the next work.
Every pre-registered criterion in `docs/10` and `docs/13` stays binding and
unchanged.

---

## 1. Where the docs/14 §6 sequence stands

| Step | What | Status | Evidence |
|---|---|---|---|
| 1 | Dev score dumps + `calibrate_decision_layer.py` | **Not run.** No calibration report in `reports/`. | — |
| 2 | Set the fitted thresholds; `integration_check.py` on real backends | **Blocked on step 1.** The clone rule and the challenge CM gate are still off by default, so T3/T4 protection is off in practice. | — |
| 3 | W2 server-side challenge | **Partial.** Acoustic grading of the response windows is in, and the challenge payload no longer carries the expected digits. The digit check still reads `response_text` from the client (`vanirakshak-backend/vanirakshak/challenge/engine.py`, `grade`). P1-C has not been measured. | `vanirakshak-backend/tests/test_challenge.py` |
| 4 | W5 enrolment QA, per-user calibration, ECAPA required under auth | **Done in code.** Not verified under `VR_ASV_BACKEND=ecapa` (the cloud session cannot download the weights). | `vanirakshak-backend/vanirakshak/server/app.py` start-up guard |
| 5 | docs/13 Phase 2 | **Done. C-2a failed, so STOP-B applies.** | `reports/cm_phase2_criteria.json` |
| 6 | W4 Indic and telephony corpus | Not started. | — |
| 7 | W3 encoder fine-tune, larger SSL | **Not authorised yet** (see §3). | — |
| 8 | W7 and W8 for a pilot | Partly started: erasure endpoint and `tier_override` on the dashboard. | `vanirakshak-backend/tests/test_biometric_erasure.py` |

Verified in this session: `pytest vanirakshak-backend/tests` gives 107 passed
(stub backends, no torch weights).

## 2. Gaps found while reviewing the last two commits

These are defects or overstatements in work already merged. Each is small and
fixable without a GPU.

- **G-A: "local zero-retention ASR" is not wired.** `VR_SEMANTIC_PROVIDER` is
  read into `SemanticConfig.provider` (`vanirakshak-backend/vanirakshak/config.py`),
  but `vanirakshak-backend/vanirakshak/semantic/signal.py` still builds
  `ASRConfig(provider="groq")` and reports `provider: "groq"` as its fallback.
  Setting `VR_SEMANTIC_PROVIDER=local` changes nothing. The disabled-layer
  message also always says "third-party, US-hosted".
- **G-B: `zero_retention` ignores the provider.** `/v1/health` and the
  compliance surface report `zero_retention = not settings.semantic.enabled`.
  Once G-A is fixed, a local provider should keep `zero_retention: true`, and
  only a cloud provider should turn it off.
- **G-C: the challenge is still forgeable on text.** A scripted client that
  sends the right digits passes the text check; only the acoustic check can
  stop it. The phrase reveals the digits, so P1-C needs the server to transcribe
  the answer itself. The local ASR from G-A is the engine for this.
- **G-D: enrolments live only in memory.** A restart erases every voiceprint and
  per-user calibration. A pilot needs a persistent store. `DELETE /v1/enroll/{id}`
  must then erase from that store too, not just the in-process dict.
- **G-E: erasure does not reach live sessions.** A session opened for a user
  before erasure keeps scoring against the claim. Erasure should end or
  downgrade those sessions and say so in the receipt.
- **G-F: docs/14 is stale.** Its §4 scorecard and §5 W2/W5 text still describe
  the pre-`56caf12` state. docs/13 has no dated record of the STOP-B outcome.

## 3. What Phase 2 means for CM work

C-2a failed on both screens. S42+layer5 opened a new hole on A19 (+2.8 pp over
the incumbent). S43+layer5 opened a larger one on A19 (+20.2 pp) and lost to
the control. Under docs/13 §7 STOP-B, **the incumbent stays and the layer-5
Phase 3 arm does not run.**

Two findings from `reports/cm_phase2_criteria.json` shape anything after that:

1. **Seed noise is huge.** From-scratch seed SDs reach 16.6 pp (A19), 19.9 pp
   (A27) and 24.8 pp (A20). The mechanically filled Phase 3 margins are 65–97 pp
   on those attacks, so they cannot be met. A single-seed CM comparison in this
   repo says nothing. Any future CM experiment needs **three or more seeds per
   arm**, and its margins must come from those seeds.
2. **The incumbent's holes are wide.** At the Phase 2 operating threshold, its
   miss rate is 59.3 % on A28, 47.9 % on A20, 39.8 % on A19, and 33.2 % on the
   sealed A24. The CM is not a standalone detector. That confirms docs/14 §2:
   build for L1, where ASV carries most of the weight.

docs/14 W3 steps 2–4 (encoder fine-tune, larger SSL, augmentation redesign) are
a **new** hypothesis, not a continuation of docs/13. They need their own
pre-registration (a `docs/16`) before any GPU hour is spent. Write it after W4
has a first held-out Indic and telephony set, so the new CM is judged on
deployment-like audio and not only on ASVspoof 5.

## 4. The next steps, in order

### Now: no GPU, can run in a cloud session

| # | Work | Closes | Exit criterion |
|---|---|---|---|
| N1 | Pass `settings.semantic.provider` through `semantic/signal.py`. Make the disabled message and the `provider` field name the configured provider. Make `zero_retention` false only for a cloud provider. Add tests for `groq`, `local` and an unknown provider. | G-A, G-B | With `VR_SEMANTIC_PROVIDER=local`, `/v1/health` reports the local provider and `zero_retention: true`; tests pinned. |
| N2 | Server-side challenge transcription. Buffer the response windows while a challenge is outstanding, transcribe them with the N1 local engine, and grade digits from that transcript. Keep `response_text` for drills only, behind an explicit flag. Amend `docs/PROTOCOL.md` first. | G-C, W2 | **P1-C:** a scripted client sending the correct digits over silent or TTS response audio fails in ≥ 90 % of trials. The harness is committed as a test or a script. |
| N3 | Update the docs/14 §4 scorecard and §5 W2/W5 text. Add a dated STOP-B result section to docs/13. | G-F | `python scripts/check_docs.py` passes. |
| N4 | Persistent enrolment store (encrypted at rest, one record per user: embedding, calibration, QA summary, consent reference). Erasure deletes the record, ends that user's live sessions, and says both in the receipt. | G-D, G-E, part of W8 | The restart test keeps enrolments; the erasure test proves the record is gone from disk and the live session is closed. |

N1 comes before N2 because N2 reuses its engine. N3 and N4 are independent and
can run in parallel.

### Next: training box (CPU or GPU scoring only, no training)

| # | Work | Exit criterion |
|---|---|---|
| T1 | docs/14 §6 step 1, exactly as written there. Commit the calibration JSON it writes to `reports/`. | P1-S ≥ +30 pp at the worst cell and P1-F ≤ 50 %, graded as docs/13 wrote them. If they are not met in 14 days, STOP-A. |
| T2 | docs/14 §6 step 2: set the fitted `VR_CLONE_CM_LLR_MAX`, `VR_CHALLENGE_CM_LLR_MAX`, `VR_CHALLENGE_TRIGGER_CONSECUTIVE` and `operating_threshold`. Then run `VR_ASV_BACKEND=ecapa python scripts/integration_check.py`. | P1-T. demo_01 ALLOW, demo_02 not BLOCK, demo_04 BLOCK on every window under ECAPA. |

**T1 is the single most important item in this plan.** Until it is graded, L1
cannot be claimed, and the clone rule cannot be switched on.

### Then: data before more models

| # | Work | Exit criterion |
|---|---|---|
| D1 | W4 Indic spoof corpus: consenting speakers, recorded under `data/consent/`, with clones from several open TTS and VC families. Whole generator families are held out for evaluation. | Consent on file for every speaker; the family holdout is fixed in a dated commit before any scoring. |
| D2 | Real codecs: ffmpeg transcode to AMR-NB, AMR-WB and G.711. Score the incumbent on D1 × codecs. | A per-attack, per-codec report in `reports/`. This measures STOP-C for the incumbent. |
| D3 | Write `docs/16`: the pre-registration for W3 steps 2–4, with ≥ 3 seeds per arm, D1/D2 as the primary eval, and ASVspoof 5 as secondary. | Committed before any training run. |

### Pilot readiness (after T1 and T2 pass)

- **W8 consent and notice** for both parties to the call. This is the largest
  compliance gap left (docs/09 P0-3).
- **W7 latency:** p50 and p99 per concurrent call on the target CPU, with and
  without the N1 local ASR running.
- **W7 scale:** move session state and rate limits out of process before
  running more than one worker.
- **W7 one detector:** retire or relabel the Android on-device heuristic.
- **Audit chain operations:** rotation and external anchoring of the head hash.

## 5. What not to claim yet

- Not "local, zero-retention transcription" until N1 lands.
- Not "unforgeable challenge" until N2 passes P1-C.
- Not "blocks cloned voices" until T1 is graded and T2 switches the clone rule on.
- Not any CM improvement from a single seed.

---

## 6. Progress (2026-09-28)

Branch `feat/next-steps-n1-n4`, on top of this plan's commit `18bd253`.

| # | Status | Evidence |
|---|---|---|
| N1 | **Done.** The provider reaches `ASRConfig`. A local provider keeps `zero_retention: true`; Groq or an unknown name reports a transfer. An unknown name also degrades the semantic block. | `tests/test_semantic_gate.py` (9 new) |
| N2 | **Done. P1-C met for the transcript gate.** Digits are graded from a local server-side transcript; `response_text` is drill-only; no transcript fails closed. | `reports/challenge_p1c_harness.json`: silence, noise, vocoder-like, TTS-wrong-digits each fail 30/30 with correct `response_text`; TTS-right-digits control passes 30/30. `tests/test_challenge.py` |
| N3 | **Done.** docs/14 §4 scorecard and W2, W5, W6, W8 updated; docs/13 §10 records STOP-B. | `scripts/check_docs.py`: no hard failures |
| N4 | **Done.** AES-GCM enrolment store, reloaded at start-up; erasure removes the record and revokes live sessions (WS `session_closed`). | `tests/test_enrolment_store.py`: two-process restart test, live-session erasure test |

Backend suite: `pytest vanirakshak-backend/tests` gives 145 passed (107 before this work).

**Step 4 of docs/14 §6, now checked under ECAPA** (dev box, cached weights):
`VR_ASV_BACKEND=ecapa python scripts/integration_check.py` gives 29/30. The
P1-T conditions hold (demo_01 ALLOW, demo_02 ALLOW, demo_04 BLOCK on all 6
windows at the anchor threshold 0.208). The one miss is demo_03 (splice):
BLOCK at 73.7 against a hand-labelled CHALLENGE. That is the known "one tier
hotter under real backends" gap noted in `config.py`. It is not a P1-T
criterion. This is not T2: the fitted thresholds from T1 are still unset.

**What this changes in §5:** "local, zero-retention transcription" can now be
said of the semantic layer with `VR_SEMANTIC_PROVIDER=local`. Do not say
"unforgeable challenge" yet. A TTS that says the right digits passes the
transcript, and only the response CM gate stops it, which waits on T1 and T2.

**Next, in order:** T1 (unchanged, still the most important item), then T2.
Found while doing N2 and left open: challenge latency is still reported by the
client (W2.b); measure genuine pass rate on real Hindi and Hinglish answers
over codecs (W2.c).

### Follow-up, same day: the no-GPU leftovers

| Item | Status | Evidence |
|---|---|---|
| W2.b server-side latency | **Done.** Issue-to-voice-onset in audio samples; the client's value is only recorded. | `tests/test_challenge.py`; P1-C re-run: 4 attacks × 30 fail 100 %, control passes 100 % |
| W2.c genuine pass rate | **Proxy only.** English digits 80–100 % across clean, G.711 and G.711 at 15 dB SNR; romanised-Hindi digits **0 %** with whisper-tiny. | `reports/challenge_genuine_proxy.json` |
| W5 per-user calibration | **Done in code**; falls back to the anchor seed without a cohort (none on this dev box). | `tests/test_user_calibration.py` |
| W8 store key rotation | **Done.** | `tests/test_enrolment_store.py` |

**New blocker for Hindi deployments (from W2.c):** the local engine does not
hear Hindi digit words. Fix the engine (language hint, larger model or
IndicConformer), then measure on real speech, before relying on the
server-graded challenge with Hindi-speaking callers.
