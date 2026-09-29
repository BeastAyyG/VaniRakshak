# Post-docs/16 follow-up: honest recalibration rescues the R checkpoints; SE artifact amplification rejected

Date: 2026-09-30. Machine: lab-beast (TON-618). Branch: `feat/gpu-group1-group2`.

## 1. Context

docs/16 preregistered a real-codec augmentation experiment (control arm C vs codec-augmented arm R, seeds 42/43/44,
10-epoch / patience-5 WavLM-layer-5 protocol, fixed threshold rule = 5th percentile of clean dev bonafide LLRs,
2pp clean-FRR guard). Adjudicated verdict: **HARMS**. The R arm cut worst narrowband attack miss from ~60% to ~12%
but clean bonafide FRR regressed 6 to 18pp against the guard, so P16-3 failed.

Follow-up question: was that failure intrinsic to augmentation, or an artifact of the threshold rule? Two read-only
analyses ran 2026-09-30 against the docs/16 artifacts. Phase 0 (Pareto + recalibration). Phase 2 (SE artifact
amplification prototype, Trachu et al. 2025). No training runs; nothing under `C:\VaniRakshak` was modified.

## 2. Phase 0: Pareto tradeoff curves and honest multi-condition recalibration

Method. (a) Full FRR-vs-attack-miss Pareto sweep for all six heads over all 84 gate shards. The sweep reproduced the
registered thresholds to 4 decimals before exploring (e.g. R42 25.19665 vs 25.1967), so the machinery is trustworthy.
(b) Honest recalibration: the seeded 4000-utterance dev bonafide set was rescored under all 4 conditions (clean,
G.711u, AMR-NB 12.2k, AMR-WB 12.65k) with all 6 heads on the lab-beast GPU. Candidate rule **T_mc3** = 5th percentile
of pooled dev bonafide LLRs over the 3 codec conditions, applied identically to both arms. Alternative **T_p1** =
1st percentile of clean dev bonafide. No gate labels were used anywhere.

### 2a. Pareto (gate-label oracle; establishes what is possible)

Dominance = R operating point with clean FRR <= C's registered FRR AND worst-attack telephony miss <= C's
registered miss.

| seed | C registered (FRR / worst NB miss) | dominating R point (thr / FRR / miss) | # dominating R points |
|------|------------------------------------|---------------------------------------|-----------------------|
| s42  | 11.62 / 50.40                      | thr 21.63: 11.10 / 18.10              | 32                    |
| s43  | 5.97 / 75.80                       | thr 6.79: 5.35 / 46.40                | 18                    |
| s44  | 8.77 / 54.30                       | thr 12.44: 8.10 / 33.00               | 18                    |

On all three seeds, R operating points exist that beat C on both metrics. The registered R thresholds were simply
miscalibrated. The mechanism, confirmed by diagnosis: augmentation thickens the left tail of the bonafide score
distribution (clean bonafide FRR moved, no clean attack regressed on any seed, and gate-clean EER actually improved
for every R seed: R 6.53/8.39/7.00% vs C 12.06/12.80/8.39%), so a rule calibrated on clean bonafide tails does not
transfer to augmented models.

### 2b. Honest recalibration: both arms under T_mc3

| seed | C thr | C clean FRR | R thr | R clean FRR | FRR delta (R-C) | C worst NB miss | R worst NB miss | miss delta |
|------|-------|-------------|-------|-------------|-----------------|-----------------|-----------------|------------|
| s42  | 13.69 | 8.48        | 21.59 | 11.02       | +2.54           | 59.30           | 18.20           | -41.1      |
| s43  | 23.94 | 9.10        | 13.36 | 10.80       | +1.70           | 68.80           | 29.60           | -39.2      |
| s44  | 24.31 | 9.93        | 13.19 | 9.45        | -0.48           | 50.10           | 28.10           | -22.0      |

Mean clean-FRR delta: **+1.25pp** (inside the 2pp guard on average; s43 passes per-seed, s44 dominates outright,
s42 exceeds the per-seed guard by only 0.54pp). Mean worst-attack miss delta: **-34.1pp**. Under the registered rule,
the same FRR deltas were +6.45 / +17.68 / +12.13 (all fail). T_mc3 keeps the telephony gain and shrinks the FRR cost
to guard level.

T_p1 as a simpler alternative: s42 dominates (FRR 7.40 vs C-reg 11.62, miss 29.60 vs 50.40), s44 dominates
(8.65 vs 8.77, miss 31.00 vs 54.30), s43 is a tradeoff (+4.75pp FRR, miss 29.70 vs 75.80). T_mc3 is the principled
choice because it directly models the mechanism (codec-induced tail thickening).

### 2c. Recommendation

Adopt **T_mc3 as the deployment threshold rule for the R checkpoints** (thresholds in the table above); **no
retraining is needed**. The registered HARMS verdict stands for the preregistered protocol: history is not rewritten,
the rule was the registered rule. Caveats stated plainly: (1) four candidate rules were compared post-hoc, so for
the scientific record a preregistered confirmatory run is needed; for a deployment decision this analysis is
sufficient. (2) s42 exceeds the per-seed 2pp guard by 0.54pp: borderline, not clean. (3) Worst attacks at
recalibrated thresholds remain A28/A30/A31/A32, consistent with the registered results.

Result data committed alongside this note: `data/followup16_recal_results.json` (per-candidate per-condition
numbers), `data/followup16_pareto.json` (Pareto summaries). Raw dev LLRs and scripts: `C:\vr-out\phase0\`.

## 3. Phase 2: SE artifact amplification prototype. Verdict: NO-GO

Method. Trachu et al., "Amplifying Artifacts with Speech Enhancement in Voice Anti-spoofing" (arXiv:2506.11542,
Interspeech 2025), as an eval-only preprocessor: add noise (white, 0 dB SNR), enhance with a speech-enhancement
model, residual = noisy minus enhanced, output = enhanced + alpha * residual, alpha in {0.6, 1, 2, 3}. Scored through
the unchanged WavLM-layer-5 to AASIST-L path on a 2000-utterance stratified pilot for heads C42 and R42 (full 6-head
eval stopped per the preregistered decision gate). SpeechBrain MetricGAN+ (VoiceBank-DEMAND) stood in for the
paper's Thunder/NCSNPP/GaGNet models. Exact parameters were extracted from the paper PDF first.

Results (registered thresholds):

| head | alpha | clean FRR | worst NB miss | clean EER |
|------|-------|-----------|---------------|-----------|
| C42 (base 11.63 / ~50 / ~12) | 0.6 | 9.75 | 94-100 | 34.8 |
| C42 | 1 | 8.00 | 94-100 | 38.6 |
| C42 | 2 | 4.50 | 94-100 | 41.0 |
| C42 | 3 | 2.75 | 94-100 | 43.4 |
| R42 (base 18.05 / 9.8 / ~6.5) | 0.6 | 60.50 | 45 | 31.6 |
| R42 | 1 | 50.75 | 54 | 35.0 |
| R42 | 2 | 38.50 | 65 | 37.7 |
| R42 | 3 | 34.00 | 69 | 38.3 |

C42 clean FRR stays in the guard, but worst narrowband miss collapses to 94-100% and EER to 34.8-43.4%. R42 violates
the guard by 16 to 42pp. SE latency is 2.6 ms/utterance on the 5060 Ti: fast but moot.

Root causes, two independent problems: (1) Train/eval mismatch. The paper's gains come from *training* the
countermeasure on amplified data so it learns amplified artifact features. Applied eval-only, the 0 dB injected
noise is out-of-distribution for clean/codec-trained classifiers; threshold-independent EER of 31-46% proves no
operating point can recover separability. (2) The substitute SE model collapses at 0 dB: it outputs near-silence
(mask mean 0.006, ||xhat||=0.004 vs ||x||=0.128), and at 5-15 dB SNR it attenuates speech and increases MSE versus
no enhancement. The "extracted artifact" was therefore approximately the noisy signal itself, not a meaningful
artifact estimate.

Recommendation: do not pursue eval-only SE artifact amplification. If the paper's direction is ever revisited, it
requires (a) a working SE model at 0 dB (Thunder/NCSNPP/GaGNet or equivalent, not MetricGAN+) and (b) retraining
the countermeasure on amplified data, which is a training-time change outside eval-only scope.

Result data committed alongside this note: `data/followup16_se_amp_pilot.json`. Scripts: `C:\vr-out\phase2\`.

## 4. Next steps

1. Deployment: ship the R checkpoints with T_mc3 thresholds (section 2b) instead of the registered clean-dev rule.
2. Optional confirmatory retrain (preregistered, if the team wants the scientific record closed): on-the-fly codec
   augmentation at p=0.2 (not offline p=0.5 full replacement), uniform over G.711u/G.711a/AMR-NB/AMR-WB, symmetric
   across both classes, explicit clean exposure, 2 seeds (42, 43), same 10-epoch/patience-5 protocol; preregister
   T_mc3 as the threshold rule, keep the 2pp clean-FRR guard, and add an EER/discrimination guard so miscalibration
   can never again be mistaken for real harm. Also fix the found gap: G.711 A-law was in training but never
   evaluated.
3. Remaining ranked alternatives if more is needed: MaskedSpec spectral masking (small implementation, needs
   retraining, still an augmentation risk); paired clean/distorted consistency learning (best conceptual fit for
   anchoring clean behavior, medium effort, ~2 WavLM forwards per sample). Not recommended: compression-aware SSL
   swap (weights unavailable, reproduction too expensive), log-spectrogram/CQT frontend swap (discards the strong
   WavLM frontend, major rework).

No cloud credits were used. No training or evaluation code was modified during either phase.
