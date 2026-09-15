# 13 — Calibration decomposition, and the dev→eval transfer gap

**Dated finding, 2026-09-15.** Written while the full-partition re-score
(docs/10 step 5) was running. Nothing here changes a pre-registered criterion;
it measures things the criteria refer to and reports what the numbers say.

## Provenance

Everything below is measured, not estimated. Sources:

| what | file |
|---|---|
| 20 k-slice metrics, post-refactor | `handoff/smoketest/verify_20k_refactor_2026-09-15.json` |
| the committed record it reproduces | `reports/cm_from_scratch_2026-09-14.json` |
| dev-derived thresholds | `data/cm_scratch_{S42,B42,S43,B43}.pt` → `eer_threshold` |
| protocol metadata | `A:/cm-corpus/protocols/ASVspoof5.{eval,dev}.track_1.tsv` |
| eval manifest | `A:/cm-corpus/manifests/eval_available.tsv` |

The 20 k re-score was verified **bit-identical** to the committed record — all 16
headline metrics and all 64 per-attack EERs, `max |Δ| = 0.000e+00` — so these
numbers are the recorded ones, not a new measurement of a different thing.

## 1. The headline: ~70 % of the CLLR is calibration, not discrimination

| model | EER % | minDCF | actDCF | actDCF@τ_Bayes | CLLR | minCLLR | cal. loss | cal. share |
|---|---|---|---|---|---|---|---|---|
| S42 | 11.336 | 0.6515 | 2.2670 | 7.9191 | 1.4038 | **0.4031** | 1.0007 | 71.3 % |
| B42 | 13.775 | 0.8688 | 2.7550 | 8.2892 | 1.6905 | **0.4671** | 1.2234 | 72.4 % |
| S43 | 11.855 | 0.7261 | 2.3713 | 8.2517 | 1.4815 | **0.4138** | 1.0677 | 72.1 % |
| B43 | 12.050 | 0.8196 | 2.4100 | 5.9417 | 1.2718 | **0.4154** | 0.8564 | 67.3 % |

Mean **minCLLR 0.4248**, mean **calibration loss 1.0371**.

So the from-scratch CM *discriminates* at a minCLLR of ~0.40–0.47 — a respectable
figure — while its raw-LLR CLLR sits at ~1.27–1.69. Roughly seven tenths of the
CLLR is the cost of **not calibrating**, and `ml/train_cm.py` fits no Platt
scaler by design (`"calibration": "none (raw LLR)"` in every report).

This is the first measurement of the criterion docs/10 pre-registers as the
replacement yardstick for calibration work (`CLLR − minCLLR`, the `minCLLR`
checklist item). It now has a value: **≈ 1.04**.

## 2. Why this is *not* a path to G3 — and must not be sold as one

EER and minDCF are both defined at a *threshold chosen by an oracle sweep over
the same scores*. Both are therefore invariant under any monotone transform of
the scores. Calibration is exactly such a transform (the BOSARIS/FoCal **affine**
family, `s' = a·s + b`, `a ≥ 0`).

Consequences, stated plainly:

- A perfect affine calibration drives `CLLR → minCLLR` and `actDCF@τ_Bayes →
  minDCF`. It moves **no** EER and **no** minDCF. Not by one basis point.
- The G3 bar pre-registered in docs/10 (pooled eval **EER < 11.0 %**, **minDCF <
  0.75**) is a **discrimination** bar. Calibration cannot advance it.
- So calibration is a **reporting and product** win (honest CLLR, a defensible
  actDCF, and a usable fused-tier threshold), **not** a route to G3.

The practical value is real but different: `actDCF` falling from 2.27 toward
0.65 and `actDCF@τ_Bayes` from 7.92 toward 0.65 is the difference between a
system whose operating point has to be re-tuned per deployment and one that can
ship a fixed threshold. That matters for the fused tier, where the tier gates
were tuned against the *current* score distribution, so swapping the CM
checkpoint is a re-baselining exercise rather than a code change.

## 3. Where the from-scratch system actually stands against G3

On the 20 k slice, S42:

| metric | value | G3 bar | verdict |
|---|---|---|---|
| pooled EER | 11.336 % | < 11.0 % | **misses by 0.336 pp** |
| minDCF | 0.6515 | < 0.75 | **meets** |

Both S42 and S43 clear the minDCF bar; both miss the EER bar by a third of a
percentage point. The full-partition re-score now running is what decides whether
that survives at 628 k. **The gap to G3 is 0.336 pp of EER, and it is a
discrimination gap** — which is precisely what the planned layer-5 run targets.

## 4. The dev→eval threshold does not transfer

The checkpoints store `eer_threshold`, fitted on dev, and `score()` centres on
it. The eval-optimal EER threshold is materially lower in every case:

| model | dev threshold (in ckpt) | eval-optimal EER threshold | gap |
|---|---|---|---|
| S42 | 8.3789 | 6.4008 | **+1.978** |
| B42 | 7.6606 | 6.8099 | **+0.851** |
| S43 | 8.2368 | 6.4357 | **+1.801** |
| B43 | 7.7192 | 6.1440 | **+1.575** |

The sign is the same in 4/4 runs: the dev-derived threshold is **0.85–1.98 nats
too high** for eval. The dev score distribution sits above the eval one, so a
threshold carried over from dev systematically over-predicts bonafide.

**Caveat on the reported actDCF.** `evaluate_system(threshold=None)` sets
`threshold_source = "eer_operating_point"` — the EER threshold computed *on the
partition being evaluated*, i.e. an **oracle** threshold, not the checkpoint's
dev threshold. The reported actDCF (2.2670 for S42) is therefore already the
best case within the EER family. The genuinely deployed figure — actDCF at the
checkpoint's stored dev threshold — is **not currently reported by any
harness**. That is a real gap in the measurement, and a small one to close: the
scorer should report actDCF at both the stored threshold and the oracle one.

## 5. Why the transfer fails: dev and eval are different distributions

Not a hypothesis — the protocols say so:

| | dev (`dev.track_1.tsv`) | eval (`eval.track_1.tsv`) |
|---|---|---|
| rows | 140,950 | 680,773 |
| attacks | **A09–A16** (8) | **A17–A32** (16) |
| codec column | `-` for **100 %** of rows | C01–C11 for 74.7 %, `-` for 25.3 % |
| codec-quality column | `-` for all rows | 0–8 |
| source-utterance column | `-` for all rows | populated for 74.7 % |

**Dev is entirely uncoded; eval is three-quarters coded. The attack sets are
disjoint.** Dev is not a held-out sample of eval — it is a different condition
set. That is the mechanism behind the threshold gap in §4 and behind the dev/eval
metric disagreement in §5.1.

### 5.1 Refinement of the "dev is systematically harder" claim

The run notes from the from-scratch experiment observed that dev looks harder
than eval. That is **true for minDCF and only half-true for EER**:

| model | dev EER | eval EER | dev worse? | dev minDCF | eval minDCF | dev worse? |
|---|---|---|---|---|---|---|
| S42 | 12.749 % | 11.336 % | yes | 0.9425 | 0.6515 | yes |
| B42 | 12.769 % | 13.775 % | no | 0.8866 | 0.8688 | yes |
| S43 | 12.683 % | 11.855 % | yes | 0.9759 | 0.7261 | yes |
| B43 | 11.772 % | 12.050 % | no | 0.8984 | 0.8196 | yes |

minDCF is worse on dev in **4/4**; EER is worse on dev in only **2/4** (the two S
runs). The robust statement is about minDCF, not EER.

## 6. A28 is not metadata-distinctive — its difficulty is attack-intrinsic

A28 is the dominant error (27.8–31.9 % EER across all four checkpoints, the worst
of the sixteen in every one). The obvious hypothesis is that A28 sits in a
harder data condition. It does not. Per-attack metadata, over the full eval
protocol:

| attack | n | % uncoded | % has source | % AC1 | % AC2 | % AC3 |
|---|---|---|---|---|---|---|
| A17 | 34,414 | 25.2 % | 74.8 % | 42.8 % | 36.8 % | 20.4 % |
| … | … | 25.1–25.2 % | 74.8–74.9 % | 41.2–43.0 % | 35.8–38.1 % | 19.5–22.4 % |
| **A28** | **34,414** | **25.2 %** | **74.8 %** | **43.3 %** | **37.2 %** | **19.5 %** |
| A32 | 33,690 | 25.2 % | 74.8 % | 42.3 % | 36.8 % | 20.8 % |
| bonafide | 138,688 | 25.3 % | 74.7 % | — | — | — |

A28 matches every other attack to within a percentage point on every field
available. **No protocol-level condition explains A28.** The difficulty lives in
the synthesis method, so the fix must be model-side — which is what the planned
levers (layer 5, encoder unfreezing) are.

### 6.1 And not a speaker effect either

The one field the table above cannot show is speaker, so it was tested directly.
All sixteen eval attacks draw from **the same 367 speakers** — the Jaccard
overlap between A28's speaker set and every other attack's is **exactly 1.000**,
with 0 speakers unique to A28 and 0 unique to any other attack:

| pair | shared | A28-only | other-only | Jaccard |
|---|---|---|---|---|
| A28 vs A17…A32 (all 15) | 367 | 0 | 0 | **1.000** |

So every attack is evaluated on an identical speaker cohort. A28's error cannot
be attributed to A28 being tested on harder speakers — there is no such subset.

**Negative result, recorded as such.** Between §6 and §6.1, A28 is now ruled out
as an artifact of attack set, speaker set, codec, codec quality, source
presence, source multiplicity, gender, or the AC axis. It closes off data-slicing
as an explanation for the A28 gap and stops it being re-tested.

## 7. Open question: what is `AC1` / `AC2` / `AC3`?

Column 7 of both protocols, present only on spoof rows (bonafide carries `-`). It
partitions the spoofs ~42.5 % / 36.5 % / 21.0 % on eval and 43.1 % / 33.1 % /
23.8 % on dev. Measured properties:

- **Orthogonal to attack** — every one of A17–A32 splits across all three in the
  same proportions.
- **Orthogonal to codec and codec quality** — all 11 codecs appear in all three;
  the uncoded fraction is 25.0–25.3 % within each.
- **Orthogonal to gender** and to whether a source utterance exists (~75 % in
  each).
- **Orthogonal to source-utterance multiplicity** (~20 % multi-source in each).
- **Constant within a source-utterance group** — all 136,453 distinct source
  utterances map to exactly one AC value.
- **Not a speaker partition** — of 367 speakers in spoof rows, 210 appear in all
  three ACs, 110 in two, 47 in one.

So it is a real, source-utterance-level axis that the manifests discard, and its
semantics are **not derivable from the protocol alone**. Recorded as an open
question rather than guessed at. It is not blocking anything.

## 8. The eval manifest is a uniform 92.4 % subsample

`manifests/eval_available.tsv` has 628,848 rows against the protocol's 680,773 —
51,926 clips (7.6 %) are absent. The absence is **uniform**, so per-attack
comparisons are unbiased:

| | protocol | manifest | missing |
|---|---|---|---|
| each of A17–A32 | 31.3 k–34.4 k | 28.5 k–31.8 k | 7.49 %–7.90 % |
| bonafide | 138,688 | 128,292 | 7.49 % |

All 628,848 manifest IDs resolve to exactly one protocol row, no duplicates. The
scorer's own startup line (`bonafide=128292 spoof=500556`) agrees with this
count independently. Any claim resting on the full partition should say
"downloaded partition (92.4 % of eval)" rather than "eval".

## 9. What this changes

**Changes:**

1. G3 is a **discrimination** bar and calibration cannot touch it. Calibration
   work should be justified by actDCF/CLLR and the fused-tier threshold, never
   as a route to the EER bar.
2. The from-scratch system is **0.336 pp of EER** from the G3 EER bar on the 20 k
   slice, with minDCF already clear. The layer-5 run is the right next lever.
3. Dev is a **different condition set**, not a held-out sample. Model selection
   and early stopping on dev are selecting for uncoded, A09–A16 data. This is
   worth revisiting before the next training run — but it is a pre-existing
   choice, so it is flagged, not changed.
4. The dev-derived threshold does not transfer (+0.85 to +1.98 nats). Any
   deployment that uses the stored `eer_threshold` will under-detect spoof.
5. A28 is attack-intrinsic; stop looking for a data-condition explanation.

**Does not change:** the G3 thresholds, the 0.8 pp noise floor, the two-seed
minimum, or any pre-registered criterion in docs/10. Nothing here is an argument
to relax a bar.

## 10. One measurement gap, stated for the record

`actDCF` at the checkpoint's **stored** dev threshold is not produced by any
harness, yet it is the only actDCF figure that corresponds to what the system
would actually do when deployed. Every actDCF in this repo is oracle-thresholded.
Closing this is a small change to `benchmarks/compute_mindcf.py` /
`scripts/score_checkpoint_eval.py` and should be done before any actDCF number is
quoted outside the project.
