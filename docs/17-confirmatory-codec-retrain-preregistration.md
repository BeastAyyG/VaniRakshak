# 17 — Pre-registration: confirmatory codec retrain with preregistered T_mc3

**Status: pre-registered 2026-09-30, before any run in this plan started.**
Criteria below are fixed now and will not change after results exist. Any
deviation is written up as a dated amendment, never as an edit.

## 0. Why this experiment

docs/16 adjudicated **HARMS** for real-codec augmentation at p=0.5 under the
registered threshold rule (5th percentile of *clean* dev bonafide LLRs): the
telephony effect was large (worst narrowband miss 60.15% to 12.2%) but clean
bonafide FRR regressed 6 to 18 pp against the 2 pp guard.

The docs/19 follow-up showed the failure was **threshold miscalibration, not
lost discrimination**: augmentation thickens the left tail of the bonafide
score distribution, no clean attack regressed on any seed, and gate-clean EER
*improved* for every R seed. Under the honest, dev-only, multi-condition rule
**T_mc3** (5th percentile of pooled dev bonafide LLRs over the codec
conditions), the existing R checkpoints deliver ~34 pp mean worst-attack miss
reduction at ~+1.25 pp mean clean-FRR cost, inside the guard on average, and
Pareto analysis proved R operating points dominate C's registered point on
all three seeds.

That rescue was post-hoc (four candidate rules were compared), so for the
scientific record a **preregistered confirmatory run** is needed: lighter
augmentation, T_mc3 fixed in advance as the threshold rule, and an
EER/discrimination guard so miscalibration can never again be mistaken for
real harm. This plan is that run.

A correction to docs/19 §4 wording: the docs/16 mechanism was already
per-sample on-the-fly augmentation (seeded per sample and epoch in the
training loader, `ml/train_cm.py` `_load_and_augment`), not an offline corpus
transform. The material changes here are the lighter probability (p=0.2, so
80% of samples per epoch stay clean), the dedicated `--codec-aug-p` flags,
and the preregistered T_mc3 rule.

## 1. Hypothesis (H17)

Lighter on-the-fly real-codec augmentation (p=0.2, uniform over G.711u,
G.711a, AMR-NB 12.2k, AMR-WB 12.65k, symmetric across both classes, 80% clean
exposure), evaluated under the preregistered T_mc3 threshold rule, retains a
telephony effect of at least 5 pp mean worst-narrowband miss reduction while
passing the 2 pp clean-FRR guard and a 1 pp EER discrimination guard.

## 2. Arms (fixed)

Only the R arm is retrained. The control is the three existing docs/16 C
checkpoints (`data/cm_docs16_C_s4{2,3,4}.pt`), rescored under the new
protocol below. Retraining C would burn GPU-hours and add variance for no
scientific gain; the honest baseline is the already-trained control.

New runs: seeds 42, 43, `data/cm_docs17_R_s{42,43}.pt`.

Common flags (identical to docs/16): `--merge-layers 12 --layer-index 5
--epochs 10 --batch-size 64 --lr 5e-4 --weight-decay 1e-4 --patience 5
--max-dev-utts 30000 --streaming-dev --workers 24 --p-codec 0 --p-noise 0
--p-bandpass 0 --p-rir 0 --p-gain 0`, no `--init-weights` (from scratch),
default train and dev manifests (`A:\cm-corpus\manifests\train.tsv`,
`dev.tsv`). Frozen WavLM base-plus. New flags: `--codec-aug-p 0.2`
(`--codec-aug-codecs` left at the default: uniform over all four codecs).
Dev validation stays clean (no augmentation on the dev loader), so model
selection is on clean dev minDCF exactly as in docs/16.

Run order: R42, futility check (§5), R43. Driven by
`scripts/gpu_pipeline.py` stages `g3_*`. Code changes for this plan
(`--codec-aug-p` flags in `ml/train_cm.py`; G.711 A-law eval and the G3
stages in `scripts/gpu_pipeline.py`) are committed **before** any training,
per docs/16 §6.

## 3. Evaluation protocol (fixed)

The STOP-C telephony protocol of docs/13 §12, as implemented in
`scripts/gpu_pipeline.py` (`_score_heads`), with two preregistered changes:

- **Channels:** clean; G.711u (bit-exact); **G.711 A-law (bit-exact, new —
  fixes the docs/16 gap where A-law was trained but never evaluated)**;
  AMR-NB 12.2k and AMR-WB 12.65k (ffmpeg). Codec applied to the raw
  utterance, then `prepare_audio_window`.
- **Gate rows:** the same seeded 20,000-row gate-split subset (seed
  20260928; 4,000 bonafide, up to 1,000 per attack) as docs/16.
- **Operating point per model (T_mc3, fixed in advance):** the 5th percentile
  of that model's own pooled raw LLRs on the seeded 4,000-row dev-bonafide
  set scored under the four codec conditions (g711u, g711a, amrnb_12k2,
  amrwb_12k65). Dev labels only; no gate labels touch the threshold.
- **Narrowband** = g711u, g711a, amrnb_12k2. **Worst narrowband miss** = the
  highest per-attack miss over the three.

## 4. Criteria (fixed)

Means are over the seeds of an arm (R: 2 seeds; C: 3 existing seeds).

- **G17-1 (effect).** mean worst-narrowband miss(R) ≤ mean(C) − 5.0 pp,
  **and** every R seed is lower than every C seed.
- **G17-2 (STOP-C bar).** mean worst-narrowband miss(R) < 50.0 %.
- **G17-3 (no clean regression).** mean clean bonafide FRR(R) ≤ mean(C) +
  2.0 pp, **and** each R seed's clean FRR ≤ mean(C) + 2.0 pp (per-seed guard,
  compensating for n=2), **and** each R seed's gate-clean EER ≤ mean C
  gate-clean EER + 1.0 pp (threshold-independent discrimination guard).

**Verdicts** (same mapping as docs/16).
- `ADOPT`: G17-1, G17-2 and G17-3 pass.
- `IMPROVES_NOT_ENOUGH`: G17-1 and G17-3 pass, G17-2 fails.
- `NO_EFFECT`: G17-1 fails and G17-3 passes.
- `HARMS`: G17-3 fails.

**Recorded, not pass/fail:** dev EER and minDCF per run, per-attack miss on
every channel (including per-condition G.711 A-law numbers), AMR-WB, the mean
narrowband miss, per-head T_mc3 thresholds, and the EER values behind G17-3.

## 5. Futility rule

After R42 is trained and scored: if worst-narrowband miss(R42) >
mean-C worst-narrowband miss − 1.0 pp, R43 is cancelled and the verdict is
`NO_EFFECT` (early). Recorded with both numbers.

## 6. Budget and integrity

- At most **20 GPU-hours** for training and scoring. If exceeded, stop and
  record the state.
- No change to `ml/train_cm.py`, `ml/real_codecs.py`, `ml/augment.py` or the
  evaluation code while the runs are in flight. Each run log records the git
  HEAD it ran at.
- A run that crashes is re-run from scratch with the same seed. It is not
  resumed, and no checkpoint from a partial run is scored.

## 7. What this cannot show

Same corpus limits as docs/16 §7: ASVspoof 5 English audiobook speech,
codecs applied offline, not over real networks. Additionally, n=2 for the new
arm is thinner than docs/16's n=3; the per-seed FRR and EER guards in G17-3
are the compensation, and any ADOPT verdict should still be confirmed on a
third seed before runtime use.

## 8. Results and verdict

To be recorded after adjudication, in this section, following the docs/16
convention. The verdict stands as written; history is not rewritten.
