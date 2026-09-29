# 16 — Pre-registration: real-codec augmentation for telephony robustness

**Status: pre-registered 2026-09-29, before any run in this plan started.**
Criteria below are fixed now and will not change after results exist. Any
deviation is written up as a dated amendment, never as an edit.

Owner decision recorded: the owner asked for group-2 decisions to be taken
without waiting for the Indic/telephony corpus (docs/15 D1). This plan is
therefore judged on ASVspoof 5 through real codecs; §7 says what that cannot
show.

## 0. Why this experiment

1. **STOP-C is triggered** (docs/13 §12). The incumbent misses 67.4 % of A28
   through G.711 and 63.6 % through AMR-NB at its dev operating point.
2. **The existing codec augmentation was broken.** `ml/train_cm.py` applies
   `ml/augment.py`'s augmenter *after* `prepare_audio_window`, which
   normalises to zero mean and unit variance (values reach about ±6). Measured
   on a gate-split utterance on 2026-09-29:

   | augment.py function | output max | samples clipped | relative error |
   |---|---|---|---|
   | `ulaw_encode_decode` | 1.00 (input 6.13) | 20.4 % | 0.64 |
   | `alaw_encode_decode` | 1.00 | 20.4 % | 0.63 |
   | `amr_wb_simulate` | 6.38 | 0 % | 0.75 |
   | `gsm_efr_simulate` | 6.25 | 0 % | 1.17 |

   So every "codec-augmented" model trained with that path saw clipped or
   filtered audio, not telephony. docs/20 (augmentation off beats on) and
   docs/21 (verdict `AT_0`) measured *that* augmenter. Their conclusion
   "augmentation hurts" does not extend to real codecs, which were never
   tried.

## 1. Hypothesis (H16)

Training on real codec round trips of the raw audio (G.711 µ-law, G.711
A-law, AMR-NB 12.2k, AMR-WB 12.65k), applied **before** windowing and
normalisation, lowers the worst narrowband miss on the STOP-C telephony
protocol, without hurting clean-channel detection.

## 2. Arms (fixed)

Both arms use `ml/train_cm.py` at the commit that adds this file and
`ml/real_codecs.py`. They differ **only** in `--real-codec-p`.

Common flags: `--merge-layers 12 --layer-index 5 --epochs 10 --batch-size 64
--lr 5e-4 --weight-decay 1e-4 --patience 5 --max-dev-utts 30000
--streaming-dev --workers 24 --p-codec 0 --p-noise 0 --p-bandpass 0 --p-rir 0
--p-gain 0`, no `--init-weights` (from scratch), default train and dev
manifests (`A:\cm-corpus\manifests\train.tsv`, `dev.tsv`). Frozen WavLM
base-plus (arm B architecture, docs/20).

| Arm | Extra flags | Checkpoints |
|---|---|---|
| **C** control | `--real-codec-p 0` | `data/cm_docs16_C_s{42,43,44}.pt` |
| **R** real codec | `--real-codec-p 0.5` (uniform over the four codecs) | `data/cm_docs16_R_s{42,43,44}.pt` |

Seeds 42, 43, 44. Run order: C42, R42, futility check (§5), C43, R43, C44,
R44. Driven by `scripts/gpu_pipeline.py` stages `g2_*`.

The docs/20 arm B checkpoints are **not** the control. They came from a
different trainer revision, which warm-started from `data/cm_aasist.pt` by
default. They are reported for context only.

## 3. Evaluation protocol (fixed)

The STOP-C telephony protocol of docs/13 §12, as implemented in
`scripts/gpu_pipeline.py` (`g1_telephony_heads`):

- **Gate rows:** the seeded 20,000-row gate-split subset (seed 20260928;
  4,000 bonafide, up to 1,000 per attack); unreadable files excluded.
- **Channels:** clean; G.711µ (bit-exact); AMR-NB 12.2k and AMR-WB 12.65k
  (ffmpeg). The codec is applied to the raw utterance, then
  `prepare_audio_window`.
- **Operating point per model:** the 5th percentile of that model's own raw
  LLR on a seeded 4,000-row dev-bonafide set (the T1 rule).
- **Narrowband** = G.711µ and AMR-NB. **Worst narrowband miss** = the
  highest per-attack miss over both.

## 4. Criteria (fixed)

Means are over the three seeds of an arm.

- **P16-1 (primary, effect).** mean worst-narrowband miss(R) ≤ mean(C) −
  5.0 pp, **and** every R seed is lower than every C seed.
- **P16-2 (STOP-C bar).** mean worst-narrowband miss(R) < 50.0 %.
- **P16-3 (no clean regression).** mean clean worst-attack miss(R) ≤
  mean(C) + 3.0 pp, **and** mean clean bonafide FRR(R) ≤ mean(C) + 2.0 pp.

**Verdicts.**
- `ADOPT`: P16-1, P16-2 and P16-3 pass. The best R seed becomes a
  **candidate** only. Before any runtime use it must pass, in order: T1
  re-calibration (P1-S, P1-F), P1-T on CUDA and CPU, and STOP-C on the full
  608k gate split.
- `IMPROVES_NOT_ENOUGH`: P16-1 and P16-3 pass, P16-2 fails. The effect is
  real, and STOP-C still stands. Next step: D1 data.
- `NO_EFFECT`: P16-1 fails and P16-3 passes.
- `HARMS`: P16-3 fails.

**Recorded, not pass/fail:** dev EER and minDCF, per-attack miss on every
channel, AMR-WB, the mean narrowband miss, and the same numbers for the
existing checkpoints (the incumbent, docs/20 arm B, docs/21, G4).

## 5. Futility rule

After C42 and R42 are both trained and scored: if worst-narrowband miss(R42)
> worst-narrowband miss(C42) − 1.0 pp, the remaining runs are cancelled and
the verdict is `NO_EFFECT` (early). Recorded with both numbers.

## 6. Budget and integrity

- At most **40 GPU-hours** for training and scoring. If exceeded, stop and
  record the state.
- No change to `ml/train_cm.py`, `ml/real_codecs.py`, `ml/augment.py` or the
  evaluation code while the runs are in flight. Each run log records the git
  HEAD it ran at.
- A run that crashes is re-run from scratch with the same seed. It is not
  resumed, and no checkpoint from a partial run is scored.

## 7. What this cannot show

ASVspoof 5 is English audiobook speech, and the codecs here are applied
offline, not over real networks. Passing P16-2 would mean "the CM survives
real codec round trips on this corpus". It does not mean "works for Hindi
callers on Indian phone lines": that needs docs/15 D1 and D2. No telephony
coverage claim may be made from this plan alone.

## 8. Not changed here

`ml/augment.py`'s broken codec branch is left as it is, so docs/20 and 21
stay reproducible, and both arms here run with it off. Fixing its order is a
separate change after this plan is adjudicated.

## 9. Results and verdict (recorded 2026-09-29)

All six arm-seed runs completed per protocol. Training order: C42, R42,
futility check, C43, R43, C44, R44. Checkpoints:
`data/cm_docs16_C_s4{2,3,4}.pt`, `data/cm_docs16_R_s4{2,3,4}.pt`.

Selected checkpoint per run (best dev minDCF, patience 5):

| Run | Selected epoch | Dev minDCF | Dev EER |
|-----|---------------|-----------:|--------:|
| C42 | 1 | 0.5011 | 5.18% |
| R42 | 8 | 0.4372 | 3.98% |
| C43 | 4 | 0.7037 | 7.35% |
| R43 | 4 | 0.3782 | 3.40% |
| C44 | 2 | 0.3011 | 3.06% |
| R44 | 1 | 0.2245 | 2.54% |

Futility check (after C42/R42): worst narrowband miss R42 9.8% vs C42
50.35%. stop=false, remaining seeds continued.

Registered telephony scoring (all 84 shards, 2026-09-29 22:54 IST):

- Worst narrowband miss (%): C [50.35, 75.8, 54.3], mean 60.15;
  R [9.8, 14.7, 12.1], mean 12.2.
- Clean worst-attack miss (%): C [32.7, 44.84, 25.63];
  R [5.4, 8.1, 9.6].
- Clean bonafide FRR (%): C [11.63, 5.95, 8.75];
  R [18.05, 23.63, 20.88].

P16-1 (effect): PASS. R mean worst-NB miss improves by 47.95 pp, and every
R seed beats every C seed.
P16-2 (STOP-C bar): PASS. R mean worst-NB miss 12.2% < 50%.
P16-3 (no clean regression): FAIL. Clean bonafide FRR regresses by
+6.42, +17.68, +12.13 pp per seed, against a 2 pp limit.

**Verdict: HARMS.** Real-codec augmentation substantially improves
telephony robustness but degrades clean-audio false rejection beyond the
preregistered limit. The augmentation path is rejected; no T1
recalibration, P1-T, or STOP-C follow. Full adjudication record:
`C:\vr-out\pipeline\state_g2.json` (g2_adjudicate, 2026-09-29 22:54:48).

Operational notes: lab-beast slept twice during the campaign
(~11:42-15:17 and ~19:49-20:26 IST on 2026-09-29); training processes
survived both gaps and resumed with no reboot. ~3.5 GPU-hours lost to the
first gap. No cloud credits used.
