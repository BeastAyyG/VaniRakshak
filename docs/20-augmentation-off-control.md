# 20 — Pre-registration: augmentation-off matched control

**Status: pre-registered 2026-09-19; RUN and ADJUDICATED 2026-09-20 00:16 —
OVERALL PASS (all three criteria, both seeds).** Criteria were fixed before any
result existed, and were not adjusted afterwards. Full verdict and the
shipping recommendation are at the end of this document.

**Headline: the codec/telephony augmenter is not trading in-domain for
out-of-domain. It is a net loss on both.** Turning it off improved every
single metric measured, in-domain and out-of-domain, on both seeds. The
pre-registered expectation that in-domain would get worse was wrong.

## The question

Does the codec/telephony augmenter cause the out-of-domain regression measured
in docs/18?

## Why this is worth a run — the evidence as it stands

All four models scored on all 31,779 In-the-Wild rows, same harness:

| Model | augmentation | ASVspoof 5 pooled | ITW `clean` (95 % CI) | ITW `full_call` (95 % CI) |
|---|---|---|---|---|
| Pre-Aug L12 | **none** | 12.870 % | **22.173 %** [21.71, 22.65] | **33.908 %** [33.42, 34.44] |
| Post-Aug L12 | codec/telephony | 13.598 % | 33.422 % [32.92, 33.96] | 40.590 % [40.01, 41.16] |
| AttM | codec/telephony | 12.682 % | 37.379 % [36.81, 37.97] | 44.758 % [44.23, 45.30] |
| G4 s42 | codec + RawBoost | **11.212 %** | 35.308 % [34.78, 35.83] | 44.990 % [44.39, 45.56] |

Three independently trained augmented models cluster at 33–37 % (`clean`);
the one non-augmented model sits at 22 %. The ordering between in-domain and
out-of-domain **inverts**: the best in-domain model is the worst
out-of-domain, by 11–15 pp against intervals of ±0.6 pp.

**Why that is not yet proof.** "Pre-Aug" and "Post-Aug" are docs/08's labels
for two separate training runs; seed, epoch count and the exact training recipe
are not controlled between them, and this repo has no record of what precisely
differs. The clustering across four models is strong evidence for the
*direction*, but a paired, flag-flipped experiment is what actually isolates
the cause.

## Protocol — one variable only (ground rule 3)

| | arm A | arm B |
|---|---|---|
| Codec/telephony augmenter | **ON** (today's behaviour) | **OFF** |
| `--g4-rawboost` | off | off |
| Architecture | `AASISTLHead(in_dim=768, merge_layers=12)` | identical |
| Seeds | 42, 43 | 42, 43 |
| Data / epochs / batch / lr / patience | as G4 | identical |
| Encoder | frozen (`--unfreeze-top-n 0`) | frozen |

Augmentation is the single variable. RawBoost stays off in both arms so this
measures the codec/telephony augmenter alone — RawBoost is already covered by
the G4/control pair.

**Implementation required:** there is currently **no flag** to disable the
codec augmenter — `CodecTelephonyAugmenter` is constructed unconditionally at
`ml/train_cm.py:639`. An additive `--no-codec-augment` flag is needed.

**Do not add it until the G4/control sweep finishes (~19:00 on 2026-09-19).**
Queued runs launch new processes that would pick up the edit and silently
change the control arm, breaking its one-variable rule.

## Criteria (fixed in advance)

Scored on the full 20,000-row ASVspoof 5 slice **and** all 31,779 ITW rows,
with 2,000-rep bootstrap intervals (`scripts/eer_bootstrap_ci.py`):

- [x] **Primary:** ITW `full_call` EER is better with augmentation OFF by a
      **non-overlapping 95 % interval**, on **both** seeds. -> **PASS**
      (+10.562 pp seed 42, +2.383 pp seed 43, both disjoint).
- [x] **Secondary:** A19 and A20 both improve with non-overlapping intervals
      (>1.36 pp separation, per the docs/19 amendment). -> **PASS**
      (A19 +14.759 / +5.293 pp; A20 +20.330 / +11.168 pp; all disjoint).
- [x] **Recorded, not pass/fail:** in-domain pooled EER and minDCF. These were
      **expected to get worse** with augmentation off (Pre-Aug is 12.870 %
      vs G4's 11.212 %). **They got better** — see the verdict. Reported, not
      hidden, and the failed prediction is stated rather than quietly dropped.

**Falsification:** if augmentation-off does not improve ITW `full_call` beyond
the intervals on both seeds, then the observed 11 pp difference is a property
of those particular training runs rather than of augmentation, and the result
is reported as a negative. Published as measured, not re-run (ground rule 5).

## Measured — arm A (In-the-Wild), 2026-09-19 17:15

Arm A needed no new training. The docs/10 G4 **control** (`cm_g4_control_s42/s43.pt`)
is exactly "codec ON, `--g4-rawboost` omitted" with identical architecture,
seeds, data and recipe — so it *is* arm A and was reused rather than retrained.
Scored on all 31,779 ITW rows, 2,000-rep bootstrap, same harness.

| | ITW `clean` (95 % CI) | ITW `full_call` (95 % CI) |
|---|---|---|
| **Arm A seed 42** (codec only) | 40.555 % [39.99, 41.11] | 44.541 % [43.97, 45.09] |
| **Arm A seed 43** (codec only) | 32.761 % [32.26, 33.34] | 38.304 % [37.79, 38.87] |
| G4 seed 42 (codec + RawBoost) | 35.308 % [34.78, 35.83] | 44.990 % [44.39, 45.56] |
| Pre-Aug L12 (**no** augmentation) | **22.173 %** [21.71, 22.65] | **33.908 %** [33.42, 34.44] |

Two findings, and the first is a caveat that has to travel with the result:

**1. Out-of-domain seed spread is an order of magnitude worse than in-domain.**
The two arm-A seeds differ by **7.79 pp** on `clean` and **6.24 pp** on
`full_call`, with *non-overlapping* intervals — against an in-domain pooled
spread of 0.714 pp (docs/10). Any single-seed OOD number in this repo is close
to uninterpretable. The experiment is still valid because arms are compared
**within seed**, but the effect will have to be large to be unambiguous, and
this is recorded as an open item against Ground rule 1, which sets one noise
floor for all metrics. Not patched unilaterally.

**2. RawBoost helps OOD on `clean` and not on `full_call`.** Against arm A seed
42, G4 seed 42 is 5.25 pp better on `clean` (intervals disjoint) and 0.45 pp
*worse* on `full_call` (intervals overlap). This is why the primary criterion
below is written on `full_call`: `clean` is the condition that has been
flattering the augmented models all along.

## Cost — actual

**~8.5 h estimated, ~6.1 h actual.** Arm A already existed as the G4 control
and was reused, so only arm B (2 seeds) needed training. Arm B launched
2026-09-19 17:15 via `scripts/docs20_armb_runner.py`, finished seed 42 at
18:01 and seed 43 at 23:19 (an interruption and resume account for the gap).
Adjudication ran 23:52 -> 00:16 on 2026-09-20.

## What this will not answer

- Whether *RawBoost specifically* helps. That is the queued G4/control pair —
  note that both of its arms keep the codec augmenter, so it cannot isolate
  augmentation as a whole.
- Whether some *intermediate* amount of augmentation is optimal. This is a
  binary on/off test; if it confirms the effect, a dose-response sweep is the
  natural follow-up and would need its own pre-registration.

## Measured — arm B and the verdict, 2026-09-20 00:16

Arm B trained to completion on both seeds (`--no-codec-augment`). Report:
`reports/docs20_adjudication_20260920_001613.json`.

### In-domain (ASVspoof 5)

| | dev minDCF | dev EER | eval pooled EER (95 % CI) | eval minDCF |
|---|---|---|---|---|
| arm A s42 (codec ON) | 0.9965 | 15.856 % | 15.437 % [14.93, 15.99] | 0.9686 |
| **arm B s42 (OFF)** | **0.7105** | **6.728 %** | **9.241 %** [8.71, 9.76] | **0.6991** |
| arm A s43 (codec ON) | 0.9920 | 13.983 % | 12.400 % [11.89, 12.98] | 0.8759 |
| **arm B s43 (OFF)** | **0.7843** | **7.297 %** | **9.635 %** [9.14, 10.10] | **0.7003** |

### Out-of-domain (In-the-Wild, 31,779 rows)

| | ITW `clean` (95 % CI) | ITW `full_call` (95 % CI) |
|---|---|---|
| arm A s42 (codec ON) | 40.555 % [39.99, 41.11] | 44.541 % [43.97, 45.09] |
| **arm B s42 (OFF)** | **17.277 %** [16.87, 17.69] | **33.979 %** [33.45, 34.56] |
| arm A s43 (codec ON) | 32.761 % [32.26, 33.34] | 38.304 % [37.79, 38.87] |
| **arm B s43 (OFF)** | **21.107 %** [20.66, 21.57] | **35.921 %** [35.39, 36.47] |

### All nine checkpoints, out-of-domain — perfect separation

`reports/ood_comparison.json`, generated by `scripts/ood_comparison.py`.
Sorted on the primary condition:

| Checkpoint | codec | ITW `clean` | ITW `full_call` |
|---|---|---|---|
| pre-aug baseline | **off** | 22.173 % | **33.908 %** [33.42, 34.44] |
| arm B s42 | **off** | 17.277 % | **33.979 %** [33.45, 34.56] |
| arm B s43 | **off** | 21.107 % | **35.921 %** [35.39, 36.47] |
| arm A s43 | on | 32.761 % | 38.304 % [37.79, 38.87] |
| post-aug L12 | on | 33.422 % | 40.590 % [40.01, 41.16] |
| G4 s43 (codec + RawBoost) | on | 38.611 % | 41.181 % [40.60, 41.80] |
| arm A s42 | on | 40.555 % | 44.541 % [43.97, 45.09] |
| attentive merge | on | 37.379 % | 44.758 % [44.23, 45.30] |
| G4 s42 (codec + RawBoost) | on | 35.308 % | 44.990 % [44.39, 45.56] |

**The two groups do not interleave.** All three non-augmented checkpoints take
the top three ranks (worst 35.921 %); all six augmented ones sit below
(best 38.304 %). A 2.38 pp gap with no exceptions, across nine independently
trained models. That is what upgrades this from a two-seed result to a pattern.

G4 s43's In-the-Wild figures were measured to fill the gap docs/19 flagged
(`reports/eer_ci_g4_s43_ITW{,_full_call}.json`).

### Criteria

| Criterion | Seed 42 | Seed 43 | Verdict |
|---|---|---|---|
| Primary: ITW `full_call`, disjoint | +10.562 pp | +2.383 pp | **PASS** |
| Secondary: A19, disjoint & >1.36 pp | +14.759 pp | +5.293 pp | **PASS** |
| Secondary: A20, disjoint & >1.36 pp | +20.330 pp | +11.168 pp | **PASS** |

**OVERALL: PASS.** All three criteria resolve, on both seeds, with
non-overlapping intervals throughout.

**Caveat added 2026-09-21 — the seed-43 margin is thin against seed noise.**
These criteria test *bootstrap sampling error* (n = 31,779), and on that basis
every margin above is solid. They do not test *seed-to-seed* variance. Measured
afterwards (`scripts/measure_seed_spread.py`), out-of-domain seed spread is
**1.9–6.2 pp** — so docs/20's seed-43 margin of **+2.383 pp** sits inside it,
while seed 42's +10.562 pp does not.

This does **not** overturn the verdict: it passed on its own pre-registered
terms, both seeds move in the same direction, and the nine-checkpoint perfect
separation corroborates it independently. But the seed-43 arm is the weakest
link, and it is why a third arm-B seed is being trained. See docs/10
§"Proposed: a split noise floor" — the amendment is deliberately **not**
applied retroactively.

### The prediction that was wrong, stated plainly

docs/20 pre-registered that in-domain performance was "expected to get worse"
with augmentation off, on the strength of Pre-Aug 12.870 % vs G4 11.212 %.
That did not happen. Arm B is better in-domain on **every** axis: pooled EER
(−6.196 pp s42, −2.765 pp s43), minDCF (0.9686 -> 0.6991), A19, A20 and dev
EER (−9.128 pp s42, −6.686 pp s43).

So the result is not the trade-off that was hypothesised. It is not
"augmentation buys benchmark numbers at the cost of the real world". It is
**augmentation costs on both, and buys nothing.** The pre-doc/08 comparison
was confounded — Pre-Aug and Post-Aug were separate runs whose seed, epoch
count and recipe were never controlled, and docs/20 said so at the time.
The paired, flag-flipped run resolves it the other way.

### Why this is plausible rather than a fluke

ASVspoof 5 was already captured over real telephone networks, so `clean` in
this corpus **is** the call condition. Applying a synthetic codec/telephony
chain on top double-transcodes the input and pushes it off-distribution
relative to both eval and ITW. The augmenter is teaching the model invariances
to distortions that the deployment signal does not contain.

A second, consistent signal: **out-of-domain seed spread collapses when
augmentation is off** — 6.237 pp (arm A) -> 1.942 pp (arm B) on `full_call`.
Augmentation was not only shifting the mean, it was destabilising training.

### Open items (recorded, not patched)

- Ground rule 1 sets one 0.770 pp noise floor for all metrics, measured
  in-domain. Out-of-domain seed spread is 1.9–6.2 pp. The rule needs a
  separate OOD floor. **Not adjusted unilaterally here.**
- This is a binary on/off test. Whether an *intermediate* augmentation
  strength is optimal is unanswered and needs its own pre-registration.
- ~~Arm B checkpoints carry no Platt calibration block.~~ **Closed** — both
  are now calibrated and pass the G2 criterion. See the calibration section
  below. The remaining work is re-fitting the fusion weights.

## Calibration and deployment readiness — done, 2026-09-20 00:34

Both arm B checkpoints Platt-calibrated (fitted on dev only) to separate
output files, leaving the originals untouched:
`data/cm_docs20_armb_{s42,s43}_cal.pt`.

| | eval CLLR raw → cal | eval actDCF raw → cal | EER | minDCF | G2 (CLLR < 1.0) |
|---|---|---|---|---|---|
| arm B s42 | 3.2040 → **0.4271** | 8.6532 → **1.4493** | 9.249 % | 0.6990 | **PASS** |
| arm B s43 | 3.8917 → **0.4112** | 9.4448 → **1.7473** | 9.632 % | 0.7001 | **PASS** |

EER and minDCF are unchanged by design — Platt is monotone in the LLR, so it
fixes the threshold and the calibration, never the ranking. These are the
best calibrated CLLRs this repo has produced (the G4/control checkpoints came
in at 0.5097–0.6518).

**Live load proof: PASS.** `scripts/wire_cm_into_backend.py --checkpoint
data/cm_docs20_armb_s42_cal.pt --live` constructs the real backend detector
and scores a real utterance:

- detector `aasist_l_wavlm_dev_eer_0.067` — **not** `aasist_missing_weights`,
  so the trained head is in the request path and the stub is not deciding
- live LLR **+7.036** on `test_audio/hindi_sample.wav` → decides genuine-ward
- `calibration: present`

### Fusion weights — checked, no change warranted

The one open item was that `w_cm` / `w_asv` are fitted to the old head's LLR
distribution, and that calibration rescales the LLR. `fit_fusion_weights.py`
was therefore re-run with arm B s42 wired in (real dev corpus for the CM
axis, anchor enrolled). It fits only `(intercept, w_cm, w_asv)`;

| L2 | `w_cm` | `w_asv` |
|---|---|---|
| 0.10 | 2.4469 | 6.3502 |
| 0.25 | 1.5869 | 3.6372 |
| 0.50 | 1.0249 | 2.1602 |
| 1.00 | 0.6054 | 1.1902 |
| 2.00 | 0.3339 | 0.6247 |
| 5.00 | 0.1425 | 0.2568 |
| 10.00 | 0.0729 | 0.1295 |

**Read the spread, not any single row.** `w_cm` moves 33× and `w_asv` 49×
across a plausible L2 range, so there is no identifiable "best" value — only
a band. A single point estimate at the default `l2=1.0` (0.6054 / 1.1902)
looks like it indicts the deployed preset (1.5 / 2.0) as over-weighting by
~2.5×, but that is an artefact of an arbitrary default, not evidence.

The deployed preset **sits inside the band**, at roughly L2 ≈ 0.28–0.55 —
toward the conservative end, well below the aggressive l2=0.1 fit
(2.4469 / 6.3502). Training accuracy at l2=1.0 is 90.0 %.

**Conclusion: leave `w_cm=1.5` / `w_asv=2.0` alone.** There is no evidence
they need changing for arm B, and changing them on a point estimate from 170
samples would be fitting to noise. Recorded so this is not re-litigated.

Two traps found while doing this, both now guarded in the script:

- `load_anchor()` is not optional. Skip it and `asv_mismatch` is constant, so
  `w_asv` silently fits to **0.0000** — which reads as "ASV doesn't matter"
  when it is really a missing-data artefact. `fit_fusion_weights.py` now
  calls it and warns.
- Do not export `VR_FUSION_W_CM` / `VR_FUSION_W_ASV` to "fix" the fusion. An
  explicit env var overrides the backend's real-backend preset and
  re-introduces the false-alarm behaviour that preset exists to remove.

`--sweep` was added to `fit_fusion_weights.py` so the band is one command
away instead of an inline script.

## Shipping recommendation

**Ship arm B (`--no-codec-augment`), not the G4 / augmented line.**

The evidence is that arm B wins on the deployment distribution (ITW
`full_call`: 33.979 / 35.921 vs 44.541 / 38.304) *and* on the in-domain
benchmark, *and* has 3x lower seed variance. There is no axis on which the
augmented models win, so there is no trade-off to adjudicate — the decision is
unambiguous.

Two consequences that must be carried forward:

1. **The docs/10 headline metric needs revisiting.** It is written on
   ASVspoof 5 pooled EER, which in this experiment ranks models correctly
   *by luck* — arm B wins both. But the converse does not hold: a model that
   wins on pooled EER is not thereby good OOD (docs/18 measured an inversion
   across four models). Select on the deployment distribution.
2. **Before shipping, calibrate.** Run `scripts/fit_cm_calibration.py` on the
   chosen arm B checkpoint and re-fit the fusion weights `w_cm` / `w_asv`.
   EER and minDCF will not move; the decision threshold is what is currently
   wrong.

**Not recommended:** retraining anything to chase a better number. Ground
rule 5 applies — publish as measured.
