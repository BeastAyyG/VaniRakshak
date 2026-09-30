# docs/17 confirmatory codec-retrain: results

Date: 2026-09-30. Plan: `docs/17-confirmatory-codec-retrain-preregistration.md`
(committed 09:44:37 IST, before training started 09:53:48 IST).

## Protocol (as preregistered)

- Symmetric on-the-fly real-codec augmentation at p=0.2, 80% clean exposure,
  uniform over G.711 u-law, G.711 A-law, AMR-NB 12.2k, AMR-WB 12.65k.
- Seeds 42, 43; 10 epochs max; patience 5; batch 64; lr 5e-4; WavLM layer 5.
- Threshold T_mc3 fixed in advance from dev bonafide scores only
  (5th percentile of pooled codec-condition dev bonafide LLRs).
- G.711 A-law added to the telephony eval conditions.

## Training outcome

- Seed 42: 10 epochs, best epoch 7 (dev minDCF 0.4486, EER 4.474%).
  Checkpoint: `data/cm_docs17_R_s42.pt`. Wall time 2.62 h.
- Seed 43: 10 epochs, best epoch 6 (dev minDCF 0.2399, EER 2.397%).
  Checkpoint: `data/cm_docs17_R_s43.pt`. Wall time 2.53 h.
- No divergence, no crash, no early stop. GPU healthy throughout.

Note: the training log prints `augmentation={'p_codec': 0.0, ...}` in the
checkpoint provenance line. This is the legacy augmenter config dict
(`--p-codec 0` etc., intentionally zero). The docs/17 augmenter ran at
`--codec-aug-p 0.2` (startup log confirms p=0.2 with all four codecs; code
wires it per-sample in `__getitem__`; checkpoint also stores
`real_codec.describe()`). Verified 2026-09-30; experiment is valid.

## Futility check

Rule: stop if worst-narrowband miss R42 > C-mean - 1.0 pp.
R42 scored 21.62% vs control mean 58.92%. Verdict: CONTINUE.

## Telephony evaluation (T_mc3 thresholds, per head)

n_gate = 20000, n_dev_bonafide = 4000. Conditions: clean, g711u, g711a,
amrnb_12k2, amrwb_12k65. Worst narrowband miss (over g711u/g711a/amrnb_12k2):

| head | clean FRR | worst NB miss | NB mean miss | passes STOP-C bar |
|------|-----------|---------------|--------------|-------------------|
| R_s43 | 12.13% | 18.20% | 3.38% | yes |
| R_s42 | 9.45% | 21.62% | 5.70% | yes |
| C_s44 | 10.03% | 49.60% | 13.87% | yes |
| C_s42 | 8.63% | 58.96% | 22.75% | no |
| C_s43 | 9.15% | 68.20% | 26.80% | no |

Control mean worst-NB miss: 58.92%. Augmented mean: 19.91%.

## Adjudication

- G17-1 (effect): 19.91 <= 58.92 - 5.0. PASS.
- G17-2 (STOP-C bar): 19.91 < 50.0. PASS.
- G17-3 (no clean regression): mean FRR(R) = 10.79 vs mean(C) + 2.0 = 11.27,
  passes on the mean; but the per-seed guard fails: R_s43 clean FRR 12.13%
  exceeds 11.27%. EER guard passes (R 8.03/7.0 vs C max 12.8 + 1.0).
  FAIL.
- Verdict: **HARMS** (G17-3 fails).

## Interpretation

Same pattern as docs/16: symmetric on-the-fly codec augmentation at p=0.2
strongly improves telephony robustness (worst narrowband miss drops from
~59% to ~20%) but regresses clean bonafide FRR beyond the 2 pp guard on one
of two seeds (R_s43 at +2.86 pp over the control mean). The preregistered
verdict stands. As with docs/16, the honest recalibration result
(docs/19, T_mc3 rule) remains the deployment recommendation; no retrain
is indicated by this protocol.
