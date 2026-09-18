# 17 — What we can build next, and what only you can decide

Compiled 2026-09-19. Grounded in three external sources plus the measured state
of this repo:

- **Speech DF Arena** (arXiv 2509.02859) — 15 systems × 14 datasets, the only
  broad out-of-domain benchmark that exists.
- **RawBoost** (arXiv 2111.04433) — the augmentation G4 is built on.
- **Commercial/vendor + India regulatory landscape** (Sep 2026).

---

## 0. The finding that should re-order your priorities

Speech DF Arena ranks detection families by average EER across 14 datasets.
This repo's stack is **WavLM frontend + AASIST-L head**. Both halves sit in the
weak tier:

| System | Params (M) | Avg EER % | Pooled EER % |
|---|---|---|---|
| Whispeak *(proprietary)* | 98.9 | **3.05** | **3.00** |
| Syntra.io *(proprietary)* | 584 | 5.76 | 11.29 |
| Resemble AI *(proprietary)* | 2112 | 10.69 | 12.37 |
| **XLSR+SLS** | 340 | **13.84** | **15.68** |
| XLSR-Mamba | 319 | 14.21 | 20.12 |
| TCM | 319 | 15.77 | 16.35 |
| Wav2Vec2-AASIST | 317.8 | 18.02 | 19.47 |
| Whisper-MesoNet | 7.6 | 28.69 | 23.76 |
| **WavLM-ECAPA** ← *our frontend* | 102 | **28.74** | **33.48** |
| HuBERT-ECAPA | 102 | 33.19 | 43.03 |
| **AASIST** ← *our head family* | 0.3 | **34.49** | **33.16** |
| RawGAT-ST | 0.44 | 34.92 | 33.93 |
| RawNet-2 | 17.6 | 35.75 | 35.66 |

Three things fall out of this:

1. **The XLSR family roughly halves the error rate of the WavLM family**
   (13.84 % vs 28.74 % average). That is a far larger gap than any
   augmentation lever in docs/10 has produced. G4 is chasing ~2 pp; a frontend
   swap is worth ~15 pp.
2. **WavLM is specifically fragile under augmentation** — the paper records its
   EER spiking to 40.67 % with music and 52.29 % on LibriSeVoc + reverb. That
   is a bad property for a system whose whole defence is codec/telephony +
   RawBoost augmentation. We may be augmenting the one frontend that tolerates
   augmentation worst.
3. **Size does not buy robustness.** Whispeak (98.9 M) beats models 20× larger.
   We do not need a bigger model; we need a better-matched one.

**Implication:** G4 should finish (it is running, and it is a real measured
result), but the highest-expected-value *next* experiment is a **frontend
ablation: XLSR-300M + the same AASIST-L head, same data, same seeds.** That is
a like-for-like swap — `ml/train_cm.py` already isolates the frontend behind
`select_wavlm_features()`, so this is a contained change, not a rewrite.

Caveat worth stating plainly: the Arena numbers are for *those* implementations
and *those* training recipes. Our exact stack has never been measured on the
Arena. The honest next step is to measure it before acting on the ranking —
see item C1.

---

## A. Detection quality — what actually moves EER

**A1. Finish and score G4 properly.** *(running now)*
Seed 42 → 43, ~35 min/epoch, ETA 07:30 / 13:30. Score it on **per-attack A19
and A20 EER**, which is the pre-registered criterion — not the pooled dev EER
the training log prints. Pooled dev EER is not the criterion.

**A2. Frontend ablation: WavLM → XLSR-300M.** *(highest expected value)*
Same head, same data, same seeds, one variable. Two seeds. ~12 h GPU. If it
reproduces even half the Arena gap, it dwarfs everything in docs/10.

**A3. Fix calibration — cheapest real win, and it's already diagnosed.**
`ml/train_cm.py` fits no Platt scaling, so shipped checkpoints are
**miscalibrated**: actDCF 2.04–2.77, CLLR 1.05–1.47. Measured minCLLR ≈ 0.42
means ~85 % of that CLLR is *calibration* loss, not discrimination — the
ranking is fine, the threshold is wrong. Calibration cannot move EER or
minDCF, so it will not make a benchmark table look better, but it is the
difference between "scores" and "deployable verdict". This is hours of work,
not days.

**A4. Attack-diversity follow-ups, informed by RawBoost's own ablation.**
The paper's best combination was **convolutive + impulsive in series** (EER
9.50 % → 5.31 %, −44 % relative) — which is exactly what
`RawBoostAugmenter` implements. It also found that adding **stationary
signal-independent noise does not help** once the other two are present. So:
do not add a third noise branch; tune the two we have instead.

**A5. Self-supervised pretraining on in-domain telephony audio.**
The best proprietary system (Whispeak) wins on *cross-domain training
strategy*, not architecture. Your A: drive corpus is Indian telephony — a
domain no public model is trained on. That is a genuine moat, and the one
thing no vendor can copy.

---

## B. Make it deployable — this is where the repo is actually weakest

docs/09 already lists these; the research independently confirms they are the
ones that matter commercially.

**B1. Authentication (G5).** Every REST route and the WS accept unauthenticated
connections. docs/10 calls this *"worth more than any countermeasure
improvement in G2–G4"*. Still true. ~1 day.

**B2. On-prem / self-hosted deployment — an actual market gap.**
Per the vendor survey, **no commercial vendor offers on-premise deployment**;
all are cloud APIs or platform integrations. For Indian banks under RBI data
localisation and DPDP Act 2023, "audio never leaves our network" is a
differentiator you can sell, not just a feature. This is the strongest
commercial wedge in the list.

**B3. Keep the interlock shape — it matches what the research recommends.**
The vendor literature is explicit: *"detection is one signal feeding a risk
score, not a gate — treating it as a gate is the most common design
mistake. The correct output of a synthetic-voice signal is friction, not
refusal."* VaniRakshak's `ALLOW / CHALLENGE / BLOCK` is already the right
shape. Do not let anyone "simplify" it into a binary.

**B4. Measure and publish the false-positive rate on ordinary customer calls.**
The survey's single most actionable line: *"the number to negotiate over is
the false-positive rate on ordinary customer calls — vendors lead with the
flattering catch rate."* A missed fake is a measured fraud loss; a wrongly
flagged customer is silent churn. Build this metric before any sales
conversation.

**B5. Latency budget.** Real-time tools operate in a few hundred ms and must
decide from partial audio; Pindrop flags within ~2 s of speech. Document
VaniRakshak's measured time-to-decision — it is a procurement question you
will be asked.

**B6. Consent, retention, and the audit chain.** docs/09 §2 records that the
semantic layer contradicts the zero-retention claim. That is a legal exposure,
not a bug.

---

## C. Credibility — turn "my numbers" into "publicly comparable numbers"

**C1. Run the Speech DF Arena evaluation on your own checkpoints.**
The tooling is open source (`github.com/Speech-Arena/speech_df_arena`) and
computes EER/accuracy/F1 on any dataset. This converts every internal number
into one the field recognises, and it will tell you whether the WavLM→XLSR
gap in §0 is real for *your* stack rather than someone else's. Highest
credibility-per-hour item in this document.

**C2. Per-attack eval as a first-class, always-on harness.**
docs/09 §6: *"nothing re-evaluates the CM as attacks evolve."* The A19/A20
blindness was found by accident. Make per-attack EER a CI job, not a
one-off analysis.

**C3. Report the honest number.** The G3 five-seed result failed criterion 2
and was published as measured, which is correct. Keep doing that. A
negative result that survives scrutiny is worth more commercially than a
positive one that doesn't — buyers in this space run their own red-team
audio.

**C4. CI.** docs/09 §10. There is none today.

---

## D. Product surface — what a buyer actually touches

- **D1.** Analyst console: per-call risk breakdown, score provenance, replay of
  the flagged window. The vendor survey notes buyers want *why*, and
  Reality Defender ships per-model explanations for exactly this reason.
- **D2.** Indian-language coverage. Hindi sample exists; the corpus is the
  asset. Every vendor in the survey is English-first — this is the second
  differentiator.
- **D3.** Integration adapters (Genesys / Five9 / NiCE / Amazon Connect, or
  SIP-level). This is how Pindrop actually gets deployed.
- **D4.** Post-hoc forensic mode. Note the survey: most vendors do real-time
  *or* batch well, rarely both. Doing both is a positioning advantage.
- **D5.** Red-team pack: generate clones from your own enrolled voices and
  publish the detection rate. Nobody else publishes this.

---

## What you can do, by time budget

**Today (while G4 runs):**
1. Decide whether A2 (XLSR swap) is the next experiment after G4 — it is the
   only item here with a double-digit pp upside.
2. Read `docs/09-post-cm-roadmap.md` §P0 and confirm B1/B6 are real for your
   target deployment. If you intend to sell to a bank, these are blocking,
   not nice-to-have.

**This week:**
3. Run C1 (Arena eval) on the existing S42 checkpoint. It is cheap and it
   decides A2.
4. Do A3 (calibration). Hours of work, removes the single clearest
   deployability blocker.

**This month:**
5. B2 (on-prem story) + B4 (FP-rate metric). Together these are the pitch.
6. A5 (in-domain pretraining) — the long-term moat.

---

## What AI can and cannot do here

**Can do, end to end:** implement and test A2–A5; build C1–C4 (eval harness,
CI, Arena integration); write B1 (auth) and B6; generate D1–D4 scaffolding;
produce the per-attack reports and keep docs in sync; monitor the G4 run and
adjudicate it against the pre-registered criterion without moving the
goalposts.

**Cannot do, and you should not delegate:** choose the deployment model
(cloud vs on-prem) — it determines the entire commercial path; decide the
acceptable false-positive rate, which is a business risk appetite, not a
technical parameter; sign off on the legal posture for consent and retention;
and decide what claim you are willing to make publicly. The last one matters
most — in this market, an overstated claim that fails a buyer's own red-team
test is worse than a modest one that holds.
