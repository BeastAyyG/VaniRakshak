# 14 — Master plan: what VaniRakshak is, what it can become, every open gap

Written 2026-09-25. This is the **index** over `docs/08`, `09`, `10` and `13`.
It does not replace them. Every pre-registered criterion in `docs/10` and
`docs/13` stays binding and unchanged. This file places those criteria in one
threat model and one sequence, records what this change closed, and names the
gaps no earlier document owned.

> **Status update, 2026-09-27:** see `docs/15-next-steps.md` for where each §6
> step stands after Phase 2 (STOP-B) and Path A, and for the next steps in order.

---

## 1. What it is today (one paragraph, no spin)

VaniRakshak is a **target-conditioned voice interlock**. It streams a call's audio
(16 kHz mono PCM over WebSocket) to a backend that scores each 4.04 s window with a
countermeasure (CM: "is this audio machine-made?") and speaker verification (ASV:
"is this the enrolled person?"). It fuses those scores with channel, prosody,
transaction context and an optional scam-phrase signal into a 0–100 risk score, and
returns `ALLOW` / `CHALLENGE` / `BLOCK`. A BLOCK latches a transaction interlock.
The engineering around the detectors is strong: auth, a hash-chained audit log,
protocol conformance tests, honesty labels and pre-registered evaluation. The
detectors themselves are the weak part:

- **CM** is a trained AASIST-L head on frozen WavLM-base-plus. On 20k held-out
  ASVspoof 5 utterances it scores 12.7–14.1 % EER and 0.80–0.92 minDCF.
  Worst attacks are at 20–25 % EER (A19, A20, A27, A32).
- **ASV** is a random-projection stub by default. The real ECAPA-TDNN is
  pretrained and genuinely discriminative, but has to be switched on.
- The **"identity model"** (`ml/checkpoints/best_model.pt`) was trained on one
  speaker against one other speaker. It is not evidence of anything, and its
  weights are not in the repo.

## 2. What it can credibly become — three levels of claim

| Level | Claim | Needs | Honest today? |
|---|---|---|---|
| **L1** | "When someone claims to be an **enrolled** person, we BLOCK a voice that doesn't match, and a voice that matches but sounds synthetic." A step-up interlock for high-value confirmations (bank call-back, UPI or loan approval, a family member asking for money). | W1 calibrated on dev, W2, W5 ECAPA default, P1-S/P1-F met | **Mechanisms exist after this change; numbers pending (§6 step 1).** |
| **L2** | Carrier **family-SIM** feature: each member enrolled, every inbound "it's Dad" call verified. | L1 + W3 (a CM that survives real telephony and Indic speech) + W4 data + W7 ingestion + W8 consent | No. The CM has never seen Indic or telephone speech with clones. |
| **L3** | "Detects any AI voice, no enrollment needed." | A CM with low single-digit EER on in-the-wild audio | **No, and do not market it.** A 13 % EER CM alone is not a product. |

**Decision: position and build for L1, then L2.** Target conditioning is the
moat. The system is only strong when it knows whose voice to expect, because
ASV then carries most of the weight and the CM only has to tell "this person"
apart from "a synthetic copy of this person". L3 competes on the CM alone,
where this project is weakest.

## 3. Threat model: which layer stops which attack

| # | Attack | Primary defence | Status after this change |
|---|---|---|---|
| T1 | Different human impersonating the enrolled person | ASV → impostor rule | **Fixed (G8):** threshold derived from the anchor, so it fires on both backends. ECAPA verification pending (§7). |
| T2 | Replay of a genuine recording | Fresh challenge phrase | Works only if the answer is graded server-side (**W2 gap**) |
| T3 | Offline TTS clone of the enrolled voice | CM, **clone rule**, voice-scored challenge | **Mechanism added.** Off until thresholds are fitted on dev (§6). |
| T4 | Real-time voice conversion (live attacker, cloned voice) | Clone rule plus voice-scored challenge; ASV is fooled by design | Same as T3. This is the hardest case and the reason for W3. |
| T5 | Spliced genuine and synthetic audio | Per-window CM, EMA, BLOCK latch | demo_03 blocks (documented deviation: expected CHALLENGE) |
| T6 | Genuine voice, coerced, or a real person running a scam script | Semantic layer (scam phrases) | Off by default. The cloud path conflicts with zero retention (**W6**). |
| T7 | **Attacks on the system itself:** client-supplied challenge text, enrolment hijack, anchor poisoning, adversarial audio | Auth, scoped keys, server-side grading | Auth done (G5). **Challenge text is trusted from the client (W2).** Enrolment QA missing (W5). |

The pivotal finding (docs/13 §0) was T3/T4: with additive fusion, a *better*
clone scored *lower* risk, and no CM could BLOCK it. That finding drove this change.

## 4. Scorecard

Updated 2026-09-28 (docs/15 N3), after `56caf12`, `31ffa63` and docs/15 N1, N2, N4.

| Component | State | Measured | Gap |
|---|---|---|---|
| Decision layer | Clone rule, voice-scored challenge, re-arm, trigger persistence, G8 threshold; **thresholds fitted on dev (2026-09-28)** | **P1-S pass** (+99.75 pp worst cell), **P1-F pass**, **P1-T pass** on CUDA and CPU (docs/13 §11) | Genuine false-BLOCK 7.5–12 % per call on gate (dev budget 1 %); ASV 0.80 narrowband still challenges every genuine call |
| CM | AASIST-L on frozen WavLM-base+; incumbent kept (docs/13 §10, STOP-B) | Eval EER 12.7–14.1 %, minDCF 0.80–0.92; Phase 2 worst misses A28 59.3 %, A20 47.9 %, A19 39.8 %; **telephony: worst narrowband miss 67.4 % (G.711, A28) → STOP-C triggered** (docs/13 §12) | No telephony claim for this family. W4 data first, then a new pre-registration (docs/15 D3) judged on real codecs |
| ASV | ECAPA required under `VR_REQUIRE_AUTH=1`; stub default for the demo; enrolment QA; per-user calibration seeded from the anchor | ECAPA anchor: target LOO min 0.713 vs non-target max 0.328 (raw cosine) | Per-user calibration from a non-target cohort; multi-session enrolment; SASV evaluation |
| Enrolment store | Encrypted on disk (`VR_ENROL_STORE_PATH`), survives restart; erasure reaches disk and live sessions | Restart and erasure tests | Key management and rotation |
| Challenge | Digits graded from a **server-side local transcript**; `expected_*` off the wire; response CM/ASV scored | P1-C transcript gate: 4 attacks × 30 trials fail 100 %, TTS-correct control passes 100 % (`reports/challenge_p1c_harness.json`) | TTS of the right digits is stopped only by the response CM gate (T1/T2); latency is still client-reported |
| Semantic | Opt-in; local ASR (`VR_SEMANTIC_PROVIDER=local`) keeps zero retention, Groq discloses the transfer | Provider tests | Latency of local ASR on the target CPU (W7) |
| Fusion | Hand-set logistic weights (w_cm 1.5, w_asv 2.0) | Pinned by test | Refit on dev scores with a session-level objective (W1.3) |
| Android | Own untrained Kotlin heuristic plus backend verify-speaker | — | W7: one detector, not two |
| Compliance | Auth, audit chain, DPDP labels, erasure with receipt | — | W8: consent flow unimplemented (docs/09 P0-3) |

## 5. Workstreams — gap, fix, exit criterion, status

### W1 — Decision layer (docs/13 Phase 1). *Largest risk reduction; no GPU.*

**Done in this change** (all off by default until fitted on dev; see `docs/PROTOCOL.md` §3, §8):

- **1.1 Clone rule** (`risk/fusion.py`). BLOCK when the claimed identity is enrolled,
  ASV ≥ `VR_CLONE_ASV_MIN`, and the mean CM LLR over `VR_CLONE_MIN_WINDOWS`
  consecutive windows is ≤ `VR_CLONE_CM_LLR_MAX`. Reported as
  `metrics.tier_override = "clone"`.
- **1.2 Voice-scored challenge** (`session.py`). The windows the stream scores while
  a challenge is outstanding are the caller's answer. Their CM and ASV can now
  fail a text-correct answer. Optional re-arm after a pass
  (`VR_CHALLENGE_MAX_PER_SESSION`). No new message type was needed.
- **Trigger persistence** (`VR_CHALLENGE_TRIGGER_CONSECUTIVE`). A synthetic-score
  smoke run shows why this exists: with one-window triggering, a genuine enrolled
  caller is challenged in **55–100 %** of 60 s calls across the ASV × channel grid,
  far above P1-F's 50 % cap. One borderline window challenges the whole call.
- **1.3 `operating_threshold`** checkpoint field (`cm_aasist.py`), preferred over the EER point.
- **1.4 Session simulator** (`risk/simulate.py`). It replays per-window scores
  through the real `CallSession`, not a re-implementation, at about 40 µs per window.
- **1.5 Fusion pin** (`tests/test_decision_layer.py`). The docs/13 Amendment-1 table is
  a test; a fusion refit now fails CI instead of silently moving every gate.
- **G8** anchor-derived impostor threshold (`anchor.py`, `config.resolve_risk_config`).
  ECAPA gives about 0.209 in calibrated space, between demo_04's max of 0.135 and
  demo_02's min of 0.610. The stub gives 0.394.
- `scripts/calibrate_decision_layer.py` chooses every threshold on **dev only**
  (bisection against a per-call false-BLOCK budget) and reports P1-S and P1-F
  before and after, per attack, over the §1.5 grid.
  `scripts/score_checkpoint_eval.py --dump-scores` produces its input.

**Still open in W1:**
- W1.a Run the calibration on real dev and eval score dumps (§6 step 1). Grade P1-S,
  P1-F and P1-T exactly as docs/13 wrote them.
- W1.b Build speaker-grouped sessions once manifests carry a speaker column
  (docs/13 Amendment 1, Fatal 1). The i.i.d. bootstrap overstates the benefit of
  multi-window averaging.
- W1.c Refit fusion weights on dev with a **session-level** objective. The intercept
  and channel terms alone push narrowband genuine calls into CHALLENGE
  (ASV 0.8, channel 0.25 puts the base risk at the CHALLENGE line).

### W2 — Make the challenge unforgeable. *No GPU.*

**Done (2026-09-28, docs/15 N2).** The phrase reveals the digits, so the
protection comes entirely from the answer being spoken live.

- The session keeps the raw audio received after a challenge is issued and
  transcribes it with a **local** engine (`challenge/transcribe.py`;
  `VR_CHALLENGE_ASR_PROVIDER`, default faster-whisper; cloud providers refused).
- Digits are graded from that transcript, in English, romanised Hindi or
  Devanagari. Client `response_text` is ignored except under
  `VR_CHALLENGE_CLIENT_TEXT_DRILL=1`. No transcript fails closed.
- `expected_digits` / `expected_full` are off the wire (W2, `56caf12`).
  PROTOCOL.md §3 Amendment 2 records the change.
- **P1-C, transcript gate:** `scripts/challenge_forgery_harness.py` against real
  faster-whisper. Correct `response_text` over silence, noise, a vocoder-like
  signal, or TTS of the wrong digits failed 30/30 each. The positive control,
  TTS of the right digits, passed 30/30, so the engine does hear digits.

**Still open in W2:**
- W2.a TTS that speaks the right digits passes the transcript, as it should.
  Only the response-audio CM check (`VR_CHALLENGE_CM_LLR_MAX`) stops it, and
  that is off until T1 fits it on dev.
- ~~W2.b~~ **Done (2026-09-28).** Latency is measured by the server: audio time
  from issue to voice onset in the response (PROTOCOL.md §3 Amendment 3). The
  client's number is only recorded. P1-C still holds with it
  (`reports/challenge_p1c_harness.json`; silence now fails as `no_voice_onset`).
- W2.c **Engine fixed for Hindi (2026-09-28); real-caller measurement still open.**
  The first proxy (SAPI English voices reading romanised Hindi) passed 0 %.
  Measured again on native voices (edge-tts hi-IN Swara/Madhur, en-IN
  Neerja/Prabhat; clean, G.711, AMR-NB; `reports/w2c_hindi_asr_engines.json`):

  | Engine | Hindi | English | ms/answer | Attacks accepted |
  |---|---|---|---|---|
  | tiny, free detection (before) | 25.0 % | 96.7 % | 128 | 0 % |
  | tiny, hi+en forced | 36.2 % | 97.9 % | 126 | 0 % |
  | **small, hi+en forced (new default)** | **80.4 %** | **99.6 %** | 422 | 0 % |
  | medium, hi+en forced | 77.1 % | 96.2 % | 845 | 0 % |
  | large-v3-turbo, hi+en forced | 78.3 % | 95.0 % | 418 | 0 % |

  What changed: the answer is decoded once per expected language with that
  language forced (detection picked English on two-word Hindi answers), any
  reading with the digits passes, and Devanagari digit words are matched by
  a consonant skeleton instead of exact spelling. That normaliser was built
  from one voice's errors; the held-out voice scores 76.7 % with small. P1-C
  still holds (`reports/challenge_p1c_harness_small_hien.json`). Still open:
  about 1 in 5 Hindi answers fails, worst through AMR-NB (72.5 %);
  IndicConformer (step 3) needs gated Hugging Face access; and the number
  that matters, real people answering in Hindi over a phone line, needs D1
  recordings.

### W3 — A CM that generalises (docs/13 Phases 2–4, then docs/10 G3). *GPU.*

In this order. Each step is gated by the one before, and docs/13 §6 margins and
STOP rules apply.

1. **Phase 2** (measure what exists): scorer v2, gate split, sealed attacks A24, A26, A27, A31.
2. **Encoder fine-tuning** (`--unfreeze-top-n`, already in `ml/train_cm.py`). Every
   frozen-encoder ablation in `reports/` sits within noise of the others, so the
   frozen encoder is the likely ceiling.
3. **Larger SSL frontend** (WavLM-Large or XLS-R-300M plus the AASIST head). This
   is the configuration behind most strong published ASVspoof systems. Budget the
   CPU latency separately (W7).
4. **Augmentation redesign:** tune the `p_codec` / `p_noise` probabilities on dev,
   and add RawBoost-style signal-level augmentation. Augmentation cost the
   incumbent A19 (9.0 → 21.3 %) and A20 (6.4 → 23.2 %) while helping A24 and A28.
   The mix needs tuning, not removal.
5. **Score fusion** of complementary heads, fitted on dev. Only claim layer
   diversity if it beats the same-layer, two-seed control (docs/13 C-2a).

### W4 — Data that looks like the deployment. *Mostly not GPU.*

ASVspoof 5 is English audiobook speech. L1 and L2 are Hindi, Hinglish and regional
languages over AMR and G.711 phone lines. No amount of W3 closes that domain gap.

- **Indic spoof corpus.** Consenting speakers (consent recorded like
  `data/consent/`), their genuine speech, plus clones from several open TTS and VC
  families. **Hold out whole generator families** for evaluation.
- **Real codecs.** Transcode with ffmpeg to AMR-NB, AMR-WB and G.711 (docs/13 §2.3),
  not only the simulated band-limits in `vanirakshak/data/augment.py`.
- **In-the-wild evaluation.** Real phone recordings (with consent) and public
  in-the-wild deepfake sets, used for evaluation only.
- **Red-team set, refreshed quarterly.** Clones from current commercial
  voice-cloning services of consenting staff voices. docs/09 P1-6: nothing
  re-evaluates the CM as attacks evolve.

### W5 — ASV that is real by default

**Done in code (`56caf12`), and persisted (docs/15 N4):**
- `VR_REQUIRE_AUTH=1` refuses to start on the stub ASV (`server/app.py` start-up guard).
- Enrolment QA rejects under 3 s voiced, SNR under 10 dB, or a CM LLR under 0
  (anchor poisoning, T7). The QA summary is stored with the enrolment.
- Each enrolment gets its own calibration entry, stored in the encrypted
  enrolment store with the embedding, so both survive a restart.

- **Per-user calibration is fitted (2026-09-28)** when possible
  (`detectors/calibration.py`): leave-one-out target scores from 3 s segments
  of the enrolment audio against the non-target cohort in `VR_ASV_COHORT_DIR`,
  with the same logistic `scripts/enroll_anchor.py` fits for the anchor. With
  under 6 s of audio or no usable cohort it falls back to the anchor's
  logistic. The QA summary and `calibration_method` record which one was used.

**Still open in W5:**
- Ship a cohort. `data/other_speakers/` is gitignored and absent on this dev
  box, so every enrolment here is `seeded_from_anchor`. A consented or public
  non-target set belongs with D1.
- Enrolment from two or more sessions or devices.
- **Spoof-aware ASV (SASV):** evaluate the ASV × CM combination on ASVspoof 5's
  SASV protocol, the metric that matches L1.

### W6 — Semantic layer without breaking zero retention

**Done (docs/15 N1).** `VR_SEMANTIC_PROVIDER=local` runs the same on-box engine
as W2 and keeps `zero_retention: true`. Groq stays an explicit opt-in that turns
it off and names the processor. An unknown provider is reported as a transfer.
Open: latency of local ASR per concurrent call (W7).

### W7 — Productisation

- **Ingestion:** a SIP/RTP media fork for carriers and contact centres. For Android,
  the backend stream, not the on-device heuristic.
- **One detector:** retire `VaniLiveVoiceAnalyzer`'s untrained "is this fake" score,
  or label it as such. Two unreconciled fake detectors is a liability.
- **Latency and cost:** WavLM per 1 s hop on CPU. Measure p50 and p99 per concurrent
  call. Batch windows across sessions; consider hop 2 s with the same 4 s window.
- **Scale:** sessions live in an in-process dict, so move session state out of
  process before running more than one worker. Rate limits are per-process too
  (PROTOCOL §8).
- **Dashboard:** show `metrics.tier_override` ("BLOCK — voice matches but sounds
  synthetic" vs "BLOCK — not the enrolled voice") and the challenge grade's
  `acoustic.verdict`. The dashboard already accepts a second server challenge
  (`page.tsx` releases its pending gate after grading), so re-arm needs no UI
  change.
- **Observability:** per-tier rates, override counts (`tier_override`), challenge
  pass and fail rates, and CM score drift, all per deployment.

### W8 — Compliance (DPDP Act 2023)

- **Consent and notice** for both parties to the call (docs/09 P0-3): not
  implemented in the backend.
- **Biometric data handling:** voiceprints are sensitive personal data.
  Deletion is done (docs/15 N4): `DELETE /v1/enroll/{user}` erases memory and
  the encrypted on-disk record, ends live sessions for that identity, and
  returns an audited receipt. Enrolments carry a `consent_ref`. Store key
  rotation is done (`scripts/rotate_enrolment_key.py`, crash-safe, keys from
  the environment only). Still to define: retention period and export.
- **Audit chain operations:** rotation, external anchoring of the head hash, and
  access logging.

### W9 — Evaluation governance

- **Session-level metrics are primary:** P(BLOCK | clone call) and P(challenge |
  genuine call). Per-window EER is diagnostic only.
- Pre-register before every run (docs/10 ground rules, docs/13 §6). Never choose
  a threshold on eval data.
- Report every result against the **worst attack and the worst channel cell**,
  not pooled.

## 6. Sequence — what runs where

| Step | What | Where | Gate |
|---|---|---|---|
| 1 | Score dumps: `score_checkpoint_eval.py --dump-scores` on dev + gate split, then `calibrate_decision_layer.py` | Training box (GPU only for scoring) | P1-S ≥ +30 pp at the worst cell; P1-F ≤ 50 % — **done 2026-09-28, both pass** (docs/13 §11) |
| 2 | Set the fitted `VR_CLONE_CM_LLR_MAX`, `VR_CHALLENGE_CM_LLR_MAX`, `VR_CHALLENGE_TRIGGER_CONSECUTIVE`, `operating_threshold`; run `integration_check.py` on real backends | Dev box | P1-T — **done 2026-09-28, pass on CUDA fp16 and CPU fp32** |
| 3 | W2 server-side ASR challenge plus protocol amendment | Dev box | P1-C |
| 4 | W5 per-user calibration, enrolment QA, ECAPA required under auth | Dev box | Integration check under `VR_ASV_BACKEND=ecapa` |
| 5 | docs/13 Phase 2 (measure existing heads) | Training box | C-2a, C-2b |
| 6 | W4 Indic and telephony corpus collection (parallel with 5) | Data work | Consent recorded; generator-family holdout fixed |
| 7 | W3 encoder fine-tune, then larger SSL | Training box | docs/10 G3 bars; docs/13 no-new-hole rule |
| 8 | W7 and W8 for a first pilot (one bank flow or one carrier cohort) | Both | Consent, latency p99, on-call runbook |

Steps 1–4 need no training and deliver L1. Steps 5–7 are the CM work, and docs/13
already gates them on step 1.

**Step 1 commands** (training box; paths as in docs/13 §2):

```powershell
$py = 'C:\VaniRakshak\.venv\Scripts\python.exe'
& $py scripts/score_checkpoint_eval.py --checkpoint incumbent=data/cm_aasist.pt `
    --manifest <dev manifest> --out C:\vr-out\p1\dev_scores.json --dump-scores C:\vr-out\p1\dev
& $py scripts/score_checkpoint_eval.py --checkpoint incumbent=data/cm_aasist.pt `
    --manifest <gate manifest> --out C:\vr-out\p1\gate_scores.json --dump-scores C:\vr-out\p1\gate
& $py scripts/calibrate_decision_layer.py --dev-scores C:\vr-out\p1\dev\incumbent.npz `
    --eval-scores C:\vr-out\p1\gate\incumbent.npz --challenge-consecutive 3 `
    --out C:\vr-out\p1\decision_layer_calibration.json `
    --checkpoint data/cm_aasist.pt --write-checkpoint C:\vr-out\p1\cm_aasist_op.pt
```

## 7. Verification status of this change

- `pytest vanirakshak-backend/tests`: 98 passed with torch installed (21 new in
  `test_decision_layer.py`, plus the `operating_threshold` load test).
  `pytest tests/e2e`: 95 passed.
- `scripts/integration_check.py` (stub backends): 25/25. This includes the new
  "demo_04 BLOCKs on every window" check (G8) and a challenge-shape check that is
  no longer silently skippable.
- **Not verified here:** the G8 criteria under `VR_ASV_BACKEND=ecapa`, because
  torch and the ECAPA weights cannot be downloaded in the cloud session's network
  policy. Run `VR_ASV_BACKEND=ecapa python scripts/integration_check.py` on the dev
  box. Expected: demo_01 ALLOW, demo_02 not BLOCK, demo_04 BLOCK on every window.
- `calibrate_decision_layer.py --synthetic` ran end to end. Synthetic numbers are a
  smoke test, **not a result**, and are not recorded in `reports/`.

## 8. What not to claim

- Not "detects deepfakes" in general (L3). Say "verifies an enrolled voice and
  blocks synthetic copies of it", and only once step 1 is graded.
- Not any per-window EER as product accuracy. Quote session-level P(BLOCK | clone)
  and P(challenge | genuine) at the worst cell.
- Not the identity model's 0 % EER, anywhere.
