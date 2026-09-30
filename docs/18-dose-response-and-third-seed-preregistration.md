# 18 — Pre-registration: codec-augmentation dose-response (p=0.1) plus third-seed confirmation (p=0.2)

**Status: pre-registered 2026-09-30, before any run in this plan started.**
Criteria below are fixed now and will not change after results exist. Any
deviation is written up as a dated amendment, never as an edit.

## 0. Why this experiment

docs/17 (p=0.2, seeds 42/43) adjudicated **HARMS** again, but on a much
thinner margin than docs/16: telephony effect 39.0 pp mean worst-narrowband
miss reduction (58.92% to 19.91%), clean-FRR cost +1.52 pp on the arm mean
(inside the 2 pp guard), failing only the per-seed guard on R_s43 (12.13%
vs the 11.27% bar, +0.86 pp over). The failure mode is now **seed variance
in clean FRR**, not a systematic regression: R_s42 at 9.45% passed
comfortably while R_s43 at 12.13% did not (2.68 pp seed spread).

Two questions decide the next step, and both are answerable with one
registered run:

1. **Dose-response (H18a).** If the clean-FRR cost comes from codec-induced
   tail thickening (docs/19 §2 mechanism), a lighter dose (p=0.1, 90% clean
   exposure) should shrink the FRR cost while keeping the telephony effect:
   docs/17's 39 pp reduction leaves large headroom over the 5 pp effect bar
   and the 50% STOP-C bar.
2. **Seed noise vs systematic harm (H18b).** docs/17 §7 requires a third
   seed before any ADOPT verdict is trusted at runtime. A third p=0.2 seed
   (R44) re-adjudicates the docs/17 protocol on n=3: if R44's clean FRR is
   near R_s42's, the docs/17 failure was seed noise; if R44 also breaches
   the per-seed guard, the variance is systematic at p=0.2.

Deployment is unaffected by this plan either way: R42 + T_mc3 (21.9308)
remains the deployed model. docs/18 is science; it may promote a better
checkpoint only under an ADOPT verdict.

## 1. Hypotheses

- **H18a (dose-response).** Symmetric on-the-fly real-codec augmentation at
  p=0.1 (uniform over G.711u, G.711a, AMR-NB 12.2k, AMR-WB 12.65k, 90% clean
  exposure), under the preregistered T_mc3 rule, retains a telephony effect
  of at least 5 pp mean worst-narrowband miss reduction and the STOP-C bar,
  while passing the 2 pp clean-FRR guard on every seed and the 1 pp EER
  discrimination guard.
- **H18b (third seed).** The p=0.2 protocol re-adjudicated on three seeds
  (R42, R43 from docs/17, new R44) resolves whether the docs/17 per-seed
  FRR failure was seed noise.

## 2. Arms (fixed)

- **C (control):** the three existing docs/16 checkpoints
  (`data/cm_docs16_C_s4{2,3,4}.pt`), rescored fresh under this protocol.
  No retraining.
- **R (p=0.2):** docs/17 checkpoints `data/cm_docs17_R_s42.pt`,
  `data/cm_docs17_R_s43.pt` (reused, not retrained) plus one new run:
  seed 44, `data/cm_docs18_R_s44.pt`.
- **L (light, p=0.1):** two new runs with seeds **paired** to the docs/17 R
  seeds (42, 43): `data/cm_docs18_L_s42.pt`, `data/cm_docs18_L_s43.pt`.
  Pairing the seeds isolates the dose effect from initialization noise.

New runs use the docs/17 flags exactly, except the dose:
`--merge-layers 12 --layer-index 5 --epochs 10 --batch-size 64 --lr 5e-4
--weight-decay 1e-4 --patience 5 --max-dev-utts 30000 --streaming-dev
--workers 24 --p-codec 0 --p-noise 0 --p-bandpass 0 --p-rir 0 --p-gain 0`,
no `--init-weights` (from scratch), default train and dev manifests,
frozen WavLM base-plus, `--codec-aug-codecs` at the default (uniform over
the four codecs), dev validation clean. New flags: `--codec-aug-p 0.1`
for L, `--codec-aug-p 0.2` for R44.

Run order: R44, L42, futility check (§5), L43 (if not stopped), telephony
scoring, adjudication. Driven by `scripts/gpu_pipeline.py` stages `g4_*`.
The g4 stages are committed **before** any training, per docs/16 §6.

## 3. Evaluation protocol (fixed)

Identical to docs/17 §3: the STOP-C telephony protocol, conditions clean /
g711u / g711a / amrnb_12k2 / amrwb_12k65, the same seeded 20,000-row
gate-split subset (seed 20260928), T_mc3 per head (5th percentile of that
head's own pooled raw LLRs on the 4,000-row dev-bonafide set scored under
the four codec conditions; dev labels only), narrowband = g711u, g711a,
amrnb_12k2, worst narrowband miss = highest per-attack miss over the three.
All eight heads (3 C + 3 R + 2 L) are scored fresh under `C:\vr-out\g4\all`
so every comparison in this plan uses one scoring run.

## 4. Criteria (fixed)

Computed separately per arm (R on n=3, L on n=2) against the same C (n=3).

- **G18-1 (effect).** mean worst-narrowband miss(arm) ≤ mean(C) − 5.0 pp,
  **and** every arm seed is lower than every C seed.
- **G18-2 (STOP-C bar).** mean worst-narrowband miss(arm) < 50.0 %.
- **G18-3 (no clean regression).** mean clean bonafide FRR(arm) ≤ mean(C) +
  2.0 pp, **and** each arm seed's clean FRR ≤ mean(C) + 2.0 pp (per-seed
  guard), **and** each arm seed's gate-clean EER ≤ mean C gate-clean EER +
  1.0 pp (threshold-independent discrimination guard).

**Verdicts per arm** (same mapping as docs/16/17).
- `ADOPT`: G18-1, G18-2 and G18-3 pass.
- `IMPROVES_NOT_ENOUGH`: G18-1 and G18-3 pass, G18-2 fails.
- `NO_EFFECT`: G18-1 fails and G18-3 passes.
- `HARMS`: G18-3 fails.

For the R arm this is a re-adjudication of the docs/17 protocol on three
seeds; an ADOPT here redeems docs/17's HARMS on the preregistered terms,
a HARMS confirms it. The docs/17 registered verdict itself is not rewritten.

**Recorded, not pass/fail:** dev EER and minDCF per new run, per-attack
miss on every channel, AMR-WB numbers, mean narrowband miss, per-head
T_mc3 thresholds, EER values behind G18-3, and the paired seed-42/43
dose comparison (R vs L at matched initialization).

## 5. Futility rule (L arm only)

After L42 is trained and scored (with the C heads): if worst-narrowband
miss(L42) > mean-C worst-narrowband miss − 1.0 pp, L43 is cancelled and the
L verdict is `NO_EFFECT` (early). Recorded with both numbers. R44 is a
confirmation seed and carries no futility gate.

## 6. Budget and integrity

- At most **20 GPU-hours** for training and scoring. If exceeded, stop and
  record the state. Expected: ~8 h training (3 runs) + ~3 h scoring.
- No change to `ml/train_cm.py`, `ml/real_codecs.py`, `ml/augment.py` or
  the evaluation code while the runs are in flight. Each run log records
  the git HEAD it ran at.
- A run that crashes is re-run from scratch with the same seed. It is not
  resumed, and no checkpoint from a partial run is scored.
- Stopping criteria: the futility rule (§5), the 20 h cap, or a machine
  failure. Any stop is recorded with the state at stop time.

## 7. What this cannot show

Same corpus limits as docs/16 §7 and docs/17 §7 (ASVspoof 5 English
audiobook speech, offline codecs). n=2 for the L arm is the same thinness
as docs/17; the per-seed FRR and EER guards are the compensation. If the L
arm reaches ADOPT on n=2, a third L seed is required before runtime use
(same rule as docs/17 §7).

## 8. Results and verdict

To be recorded after adjudication, in this section, following the docs/16
convention. The verdict stands as written; history is not rewritten.
