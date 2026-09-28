# 17 — Goals: everything still open, by outcome

Written 2026-09-28, after PRs #12, #15 and #14. Goal-wise, not time-wise:
each goal says what "done" means, who can do it, and what it waits on.
`docs/15` §6 and `docs/13` §10–12 hold the measured state this starts from.

**Owner key.** **You**: needs the owner's accounts, people, money or a
legal/product decision. **Core**: design-sensitive engineering; do it with
full context. **Agent**: small and self-contained; the task card in
`docs/18-agent-tasks.md` is enough.

## Where things stand

| Works | Does not work yet |
|---|---|
| Voice identity (ECAPA): enrolled voice vs others separates well | Spoof detector (CM): misses 58 % of A28 clean, 67 % over G.711 → **STOP-C** |
| Server-graded challenge: scripts cannot fake it; Hindi 80 % (TTS proxy) | Clone rule would BLOCK **7.5–12 % of genuine callers** |
| Decision layer: P1-S, P1-F, P1-T pass | No real Hindi caller or real phone call has ever been measured |
| Privacy: encrypted store, full erasure, local ASR | Consent and notice for both parties (DPDP) not built |

**What "workable" means here.** No detector catches every deepfake. The
product that can work is: voice is one signal; a failed check leads to a
safe step-up, never a silent BLOCK; claims match what was measured on real
Indian phone audio.

---

## G1 — A failed check leads to a safe step-up, not a wrong BLOCK
*Why:* the only way to ship before the CM improves. Turns "1 in 10 genuine
callers blocked" into "1 in 10 take one extra step". **Owner:** Core, and
**You** for the policy.
- [ ] **You:** choose the step-up for each tier and amount: in-app
      confirmation, callback to the registered number, or branch visit.
- [ ] Clone and impostor overrides can route to a new STEP_UP tier instead
      of BLOCK, one setting per override, documented in PROTOCOL.md; the
      dashboard and webhooks carry it.
- [ ] **Shadow mode** (one setting): verdicts are computed and audited, and
      nothing is enforced. Each audit event records whether it was enforced.
- [ ] Fitted thresholds (`examples/decision_layer_fitted.env`) ship with
      step-up on by default and BLOCK only for a confirmed impostor.

**Done when** a genuine caller the clone rule flags can finish the flow, and
shadow mode can run on live traffic without changing any outcome.

## G2 — Real data that looks like the deployment (docs/15 D1, D2)
*Why:* every accuracy number today comes from English audiobooks or TTS. This
blocks G3, G4 and G5. **Owner: You** (people, consent, budget), with Core
tooling.
- [ ] **You:** 50–100 consenting Hindi/Hinglish speakers (age, gender,
      region spread); consent recorded under `data/consent/`.
- [ ] Their genuine speech: read digits, free speech, phone-style answers.
- [ ] Clones of each speaker from **several** current TTS/VC families; hold
      out whole families for evaluation, fixed in a dated commit before any
      scoring.
- [ ] The same audio over **real phone lines** (G.711, AMR-NB, AMR-WB), not
      only transcoded.
- [ ] Split by speaker (train/dev/eval never share a speaker); manifests
      carry a speaker column (unblocks G5 speaker-grouped evaluation).
- [ ] Core: recording + consent capture tool; clone-generation scripts; codec
      pipeline (reuse `scripts/gpu_pipeline.py` `_ffmpeg_roundtrip`).

**Done when** a sha-pinned Indic + telephony eval set exists with consent on
file for every speaker.

## G3 — A spoof detector that works on phone lines (W3, docs/15 D3)
*Why:* STOP-C. Nothing may claim telephony coverage until this passes.
**Owner:** Core. Waits on G2.
- [ ] Write `docs/16`, the pre-registration, **before** any training: arms,
      ≥ 3 seeds per arm, margins from those seeds, G2 eval as primary and
      ASVspoof 5 as secondary, STOP rules.
- [ ] Arms to pre-register: encoder fine-tune (`--unfreeze-top-n`); larger
      SSL frontend (WavLM-Large / XLS-R-300M); codec + RawBoost augmentation
      tuned on dev; score fusion with the S42+S43 control.
- [ ] Train on this GPU box, checkpointed and on the dashboard (add stages to
      `scripts/gpu_pipeline.py`).
- [ ] Re-run T1 (thresholds) and STOP-C with the winner.

**Done when** the worst narrowband miss is < 50 % on the G2 eval set, graded
as `docs/16` wrote it, and the false-BLOCK rate at the new thresholds is
within the dev budget.

## G4 — The challenge works for real Hindi callers (W2)
**Owner:** Core, with **You** for access.
- [ ] **You:** request access to `ai4bharat/indic-conformer-600m-multilingual`
      and set `HF_TOKEN` on the GPU box.
- [ ] Add IndicConformer to the `w2c_hindi_asr` comparison; keep small+hi,en
      unless it wins on the held-out voice **and** on AMR-NB (72.5 % today).
- [ ] Measure pass rate and false-accept on **real** Hindi/Hinglish answers
      from G2 speakers, clean and over phone lines.
- [ ] Response-audio CM gate (`VR_CHALLENGE_CM_LLR_MAX`) re-fitted with the
      G3 detector: the only thing that stops TTS saying the right digits.

**Done when** genuine pass ≥ 95 % and attack false-accept ≈ 0 on real
speech over AMR-NB and G.711.

## G5 — Evaluation you can trust
**Owner:** Core.
- [ ] Speaker-grouped session simulation instead of i.i.d. windows (W1.b);
      needs the G2 speaker column.
- [ ] Refit fusion weights with a **session-level** objective (W1.c); fixes
      "ASV 0.80 narrowband challenges every genuine call".
- [ ] Ship a consented non-target cohort so per-user ASV calibration is
      `fitted_loo_vs_cohort`, not seeded (W5).
- [ ] SASV (ASV × CM) evaluation on ASVspoof 5's protocol.
- [ ] Red-team set refreshed each quarter with current commercial cloning
      services on consenting staff voices.

**Done when** every headline number is session-level, speaker-grouped and
from real-condition audio.

## G6 — Legal and compliance (W8, DPDP Act 2023)
**Owner: You** (legal review), Core for the build.
- [ ] Consent and notice for **both** parties to the call (docs/09 P0-3):
      the largest gap left.
- [ ] Retention period for voiceprints; automatic purge.
- [ ] Data-principal access/export of their record.
- [ ] Store key management: where the key lives, rotation schedule
      (`scripts/rotate_enrolment_key.py` exists).
- [ ] Audit chain: rotation and external anchoring of the head hash.
- [ ] Written DPIA / legal sign-off on claims (`docs/13` Phase 5 claim gate).

**Done when** a lawyer has signed off the consent flow and every claim.

## G7 — Runs as a product (W7)
**Owner:** Core; Agent for the pieces in `docs/18`.
- [ ] Ingestion: SIP/RTP media fork for carriers/contact centres; Android
      uses the backend stream.
- [ ] One detector: retire or clearly label the Android on-device heuristic.
- [ ] Latency: p50/p99 per concurrent call on the target CPU, with local ASR
      on; GPU sizing if needed.
- [ ] Scale: session state and rate limits out of process (e.g. Redis) before
      more than one worker.
- [ ] Observability: per-tier rates, override counts, challenge pass/fail,
      CM score drift, per deployment.
- [ ] Dashboard shows `tier_override`, challenge `grade_source`, `latency_ms`,
      `acoustic.verdict`.

**Done when** one worker pool serves a pilot's peak load with the measured
p99 and alerts on drift.

## G8 — Security hardening
**Owner:** Core.
- [ ] Production requires `VR_REQUIRE_AUTH=1`, ECAPA, the enrolment store and
      a key (fail to start otherwise).
- [ ] Threat-model review of the challenge and step-up flows (replay, relay,
      social engineering around step-up).
- [ ] External penetration test before go-live.

**Done when** the pen-test findings are closed.

## G9 — Pilot, then go-live
**Owner: You** (partner), Core.
- [ ] One bank flow or one carrier cohort in **shadow mode** (G1).
- [ ] Measure real false-BLOCK, step-up rate, challenge pass rate and fraud
      caught; re-fit thresholds on pilot data.
- [ ] Runbook and on-call; rollback trigger defined in advance.
- [ ] Switch on enforcement only when the re-fitted numbers meet the budget.

**Done when** enforcement runs on a real flow with measured outcomes inside
budget.

---

### Dependency map

```
G1 step-up ──────────────────────────────┐
G2 data ──► G3 detector ──► G4 CM gate    ├──► G9 pilot ──► go-live
      └───► G4 real speech, G5 eval       │
G6 legal ─────────────────────────────────┤
G7 product, G8 security ──────────────────┘
```

G1, G6, G7 and G8 can start now. G3 and most of G4/G5 wait on G2.
