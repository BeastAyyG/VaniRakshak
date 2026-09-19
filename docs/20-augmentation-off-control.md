# 20 — Pre-registration: augmentation-off matched control

**Status: pre-registered 2026-09-19, not run.** Criteria are fixed before any
result exists. Follows this repo's convention of fixing the bar in advance so
it cannot be fitted to the outcome afterwards.

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

- [ ] **Primary:** ITW `full_call` EER is better with augmentation OFF by a
      **non-overlapping 95 % interval**, on **both** seeds. The hard
      condition is the one that matters; `clean` has been the flattering one.
- [ ] **Secondary:** A19 and A20 both improve with non-overlapping intervals
      (>1.36 pp separation, per the docs/19 amendment).
- [ ] **Recorded, not pass/fail:** in-domain pooled EER and minDCF. These are
      **expected to get worse** with augmentation off (Pre-Aug is 12.870 %
      vs G4's 11.212 %). That is the cost side of the trade-off and is
      reported, not hidden.

**Falsification:** if augmentation-off does not improve ITW `full_call` beyond
the intervals on both seeds, then the observed 11 pp difference is a property
of those particular training runs rather than of augmentation, and the result
is reported as a negative. Published as measured, not re-run (ground rule 5).

## Cost

~4 h 15 m per run × 4 runs (2 arms × 2 seeds) ≈ **17 h of GPU**, matching the
G4/control sweep. Early stopping may cut it.

## What this will not answer

- Whether *RawBoost specifically* helps. That is the queued G4/control pair —
  note that both of its arms keep the codec augmenter, so it cannot isolate
  augmentation as a whole.
- Whether some *intermediate* amount of augmentation is optimal. This is a
  binary on/off test; if it confirms the effect, a dose-response sweep is the
  natural follow-up and would need its own pre-registration.

## If it confirms

The augmentation programme is buying in-domain benchmark numbers at the cost
of real-world generalisation, and the model selected by in-domain pooled EER is
the wrong one to ship. The decision to make then is explicit: optimise for the
deployment distribution (ITW / telephony) rather than the ASVspoof 5 pooled
number — which also means the headline metric in docs/10 needs revisiting.
