# 21 — Pre-registration: augmentation dose-response

**Status: pre-registered 2026-09-20; RUN and ADJUDICATED 2026-09-21 00:26 —
VERDICT `AT_0`.** Criteria were fixed before any result existed and were not
adjusted afterwards. See "Measured" below.

## The question

docs/20 showed the codec/telephony augmenter is a net loss: turning it off
improved out-of-domain EER by 10.56 / 2.38 pp **and** in-domain pooled EER by
6.20 / 2.77 pp. But docs/20 is a binary on/off test, so it cannot distinguish
two very different worlds:

- **augmentation is simply harmful** — the optimum is zero, and we should
  delete the augmenter;
- **some augmentation is useful, we used far too much** — the optimum is
  interior, and we should tune it down.

These have opposite engineering consequences, and docs/20 cannot tell them
apart. This experiment can.

## Dose knob

`ml/augment.py` gains `CodecTelephonyAugmenter(strength=s)`, which scales all
five probabilities (`p_codec`, `p_noise`, `p_bandpass`, `p_rir`, `p_gain`) by
`s`. Exposed as `--codec-augment-strength S`.

| `S` | `p_codec` | `p_noise` | `p_bandpass` | `p_rir` | `p_gain` |
|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.25 | 0.125 | 0.075 | 0.075 | 0.050 | 0.075 |
| 0.50 | 0.250 | 0.150 | 0.150 | 0.100 | 0.150 |
| 1.00 | 0.500 | 0.300 | 0.300 | 0.200 | 0.300 |

Scaling probabilities rather than distortion depth keeps the comparison
interpretable: at dose `s`, each effect is applied to a fraction `s` of its
usual share of utterances. Ratios between effects are held constant, so this
is one variable, not five (ground rule 3).

`S=0.0` is *equivalent to* but not *implemented as* `--no-codec-augment`;
the flag path is what docs/20 used and is reused for that endpoint.

## The endpoints already exist — do not retrain them

| Dose | Source | Status |
|---|---|---|
| **0.0** | docs/20 arm B — `cm_docs20_armb_s{42,43}.pt` | trained, scored |
| **1.0** | docs/10 G4 control — `cm_g4_control_s{42,43}.pt` | trained, scored |

Only **intermediate doses** cost GPU time. That is the entire point of
sequencing this after docs/20.

## Protocol — one variable only (ground rule 3)

| | setting |
|---|---|
| Doses | 0.0 (reused), 0.25, 0.50, 1.0 (reused) |
| Seeds | 42, 43 |
| Architecture | `AASISTLHead(in_dim=768, merge_layers=12)` — mandatory |
| Encoder | frozen (`--unfreeze-top-n 0`) |
| `--g4-rawboost` | **off** in every arm — it is a separate lever |
| Data / epochs / batch / lr / patience | as G4 |

New training: **4 runs** (2 doses × 2 seeds). At the 2h39m–3h24m measured for
docs/20's early-stopped runs, that is **~10–14 h of GPU** — roughly two
nights. Not started tonight; this repo does not leave half-finished sweeps
running into a deadline.

## Criteria (fixed in advance)

Scored on all 31,779 In-the-Wild rows and the 20k ASVspoof 5 eval slice,
2,000-rep bootstrap (`scripts/eer_bootstrap_ci.py`), primary condition is ITW
`full_call` for the same reason as docs/20: `clean` has been the flattering
one.

- [x] **Primary — is the optimum interior?** -> **AT_0** (measured 2026-09-21).
      Verdict is one of:
  - `INTERIOR_25` / `INTERIOR_50` — that dose beats **both** endpoints
    (0.0 and 1.0) with **non-overlapping 95 % intervals, on both seeds**;
  - `AT_0` — no dose beats 0.0 on both seeds (0.0 remains the choice);
  - `AT_1` — no dose beats 1.0 on both seeds (1.0 remains the choice);
  - `UNRESOLVED` — intervals overlap; reported as such, not re-run.

  This is deliberately phrased so that "augmentation is simply harmful" is a
  **valid, publishable outcome** rather than a failed experiment. docs/20
  already points that way; this must be able to confirm it rather than
  being constructed to rescue augmentation.

- [x] **Secondary — monotonicity.** -> **Not monotone** (measured 2026-09-21);
      the curve peaks at dose 0.50 and improves at 1.0. Did ITW `full_call` EER rise
  monotonically with dose? Judged on the per-seed ordering across
  0.0 → 0.25 → 0.50 → 1.0, requiring the trend to hold on **both** seeds.
  Recorded either way; a non-monotone result is informative, not a failure.

- [ ] **Recorded, not pass/fail:** in-domain pooled EER and minDCF at each
  dose, so the in-domain cost/benefit is visible alongside the OOD benefit.

**Falsification:** if no intermediate dose beats both endpoints beyond the
intervals, the conclusion is "no interior optimum" and the dose knob is not
tuned. Published as measured, not re-run (ground rule 5).

## Open items this inherits

- Ground rule 1's single 0.770 pp noise floor is measured in-domain; the
  out-of-domain seed spread is 1.9–6.2 pp. A separate OOD floor is needed.
  **Not adjusted here.**
- Two seeds is thin for a four-point curve. If the curve is flat within
  intervals, the honest conclusion is "underpowered", and a third seed is
  the fix — not a reinterpretation of the two.

## What this will not answer

- Whether the *ratios* between the five effects are right. `strength` moves
  them together by design; separating them is a different experiment.
- Whether RawBoost has a useful dose. Out of scope — it is a separate lever
  and is held off in every arm here.

## Usage

```bash
# print the plan without training anything
python scripts/docs21_runner.py --dry-run

# train the intermediate doses (seeds 42 then 43)
python scripts/docs21_runner.py --strengths 0.25 0.5 --seeds 42 43

# adjudicate against the criteria above (reuses existing reports)
python scripts/adjudicate_docs21.py
```

## Measured — adjudicated 2026-09-21 00:26, VERDICT **AT_0**

Report: `reports/docs21_adjudication_20260921_002607.json`. All four
intermediate runs completed (`all_ok=True`); both endpoints reused.

### Primary: ITW `full_call` EER (95 % CI)

| Dose | seed 42 | seed 43 |
|---|---|---|
| **0.0** (arm B) | **33.979** [33.45, 34.56] | **35.921** [35.39, 36.47] |
| 0.25 | 37.796 [37.27, 38.39] | 35.774 [35.27, 36.33] |
| 0.50 | 45.828 [45.30, 46.37] | 39.861 [39.30, 40.37] |
| **1.0** (G4 control) | 44.541 [43.97, 45.09] | 38.304 [37.79, 38.87] |

**No dose beats 0.0 on both seeds with non-overlapping intervals**, so the
optimum is at the endpoint and the verdict is `AT_0`. Dose 0.0 beats dose 1.0
on both seeds (33.979 vs 44.541; 35.921 vs 38.304), disjoint in both cases.

### Secondary: monotonicity — **FAILED, and informatively so**

The curve is **not monotone**. It rises to a peak at dose 0.50 and then
*improves* at 1.0:

- seed 42: 33.979 → 37.796 → **45.828** → 44.541
- seed 43: 35.921 → 35.774 → **39.861** → 38.304

Dose 1.0 beats dose 0.50 with **non-overlapping intervals on both seeds**
(45.09 < 45.30 and 38.87 < 39.30). That is a real effect, not noise.

Reading: full augmentation is *better than half* augmentation. A plausible
mechanism is that at full strength the model can still learn an invariance to
the distortions, whereas at half strength it sees a noisier, less consistent
input distribution and gets the cost without the invariance. Either commit to
the augmentation or don't do it — the middle is the worst place to be.

This is recorded as measured. It does not change the verdict: the best point
on the curve is still 0.0.

### Recorded (not pass/fail): in-domain

| Dose | eval pooled s42 / s43 | A19 s42 / s43 | A20 s42 / s43 |
|---|---|---|---|
| 0.0 | 9.241 / 9.635 | 7.594 / 12.568 | 8.970 / 10.000 |
| 0.25 | 10.301 / 8.822 | 13.690 / 13.356 | 16.957 / 15.435 |
| 0.50 | 11.226 / 11.927 | 17.647 / 15.615 | 18.162 / 21.119 |
| 1.0 | 15.437 / 12.400 | 22.353 / 17.861 | 29.300 / 21.168 |

In-domain is *not* monotone either and is much flatter than out-of-domain —
dose 0.25 seed 43 (8.822) is nominally the best in-domain number in the whole
table, better than dose 0.0 seed 42 (9.241). This is the same lesson as
docs/18 and docs/20: **in-domain pooled EER does not rank these models**,
which is why the criterion was written on ITW `full_call`.

### Conclusion

**Augmentation is simply harmful. There is no interior optimum and no useful
partial dose.** The dose knob exists now and the answer it gives is "off".

Consequence for the codebase: the codec/telephony augmenter should be
*removed or permanently disabled*, not tuned. That is a different action from
docs/20's finding — docs/20 could only say "this on/off setting is worse on",
docs/21 says the entire dose axis is worthless, including the points nobody
had tried.

Ground rule 5 applies: published as measured, not re-run.

## Run plan — executed 2026-09-20

Launched 10:47 IST as one chained command:

```bash
python scripts/docs21_runner.py --strengths 0.25 --seeds 42 43 \
  && python scripts/docs21_runner.py --strengths 0.5 --seeds 42 43 \
  && python scripts/adjudicate_docs21.py
```

Doses are chained (not one invocation with `--strengths 0.25 0.5`) so each
dose is observable and stoppable on its own boundary, and so a failure at
dose 0.25 does not consume 6 more hours of GPU before anyone sees it.
`docs21_runner.py` merges into an existing status file rather than resetting
it, precisely because of this chaining.

### Pre-flight (all verified before launch)

| Check | Result |
|---|---|
| GPU | RTX 5060 Ti, 17.1 GB VRAM free |
| Disk | 39.8 GB (C:), 252.3 GB (A:) |
| Endpoints 0.0 / 1.0 | all four present |
| Manifests | ITW + eval slice present |
| Backup dir `C:\vr-save\ckpt` | exists |

### Phases and gates

| Phase | Work | Gate |
|---|---|---|
| 1 | dose 0.25, seeds 42→43 | 2 checkpoints + backups |
| 2 | dose 0.50, seeds 42→43 | 2 checkpoints + backups |
| 3 | score ITW `full_call` + eval, 2,000-rep bootstrap | 8 reports |
| 4 | adjudicate | verdict JSON |
| 5 | write up, regenerate site, commit, push | suite green |

### Stopping early

Each run writes its checkpoint at every improving epoch and is copied to
`C:\vr-save\ckpt` on success, so a stop costs at most the progress since the
last improvement. If a dose is missing, the verdict is `UNRESOLVED` with the
missing doses named in the report. **Not re-run to chase a result**
(ground rule 5) — a partial curve is reported as a partial curve.

### Concurrency

Sequential, not parallel. Measured earlier: concurrent eval jobs during
training added ~2 min/epoch. Two trainings would contend for the same GPU
and both get slower, and the runs must stay comparable to the endpoints,
which were trained alone.
