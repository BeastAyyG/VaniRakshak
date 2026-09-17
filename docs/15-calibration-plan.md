# 15 — Score calibration (pre-registered before any measurement)

Written 2026-09-17, before a single score has been fitted, because docs/10
line 263 requires it: *"Replacement criterion for future calibration work:
CLLR vs minCLLR … Do this before any future calibration claim."*

---

## Why this exists

`ml/train_cm.py` fits no calibration of any kind. Every checkpoint this repo
has produced emits raw, uncalibrated LLRs, and every `actDCF` measured against
them is **2.04–2.77** while the corresponding `minDCF` sits at ~0.72. The
scores rank acceptably and are scaled badly: the decision threshold is in the
wrong place.

The decomposition already exists (`compute_mincllr`,
`compute_calibration_loss`, 2026-09-15) and has already been measured on the
20 k slice: mean **minCLLR 0.4248** against mean **calibration loss 1.0371**.
Roughly 70 % of the CLLR is a calibration defect, not a discrimination one.

---

## What calibration can and cannot do — stated now so it cannot be claimed later

- **EER and minDCF are threshold-optimised**, hence invariant under any
  monotone transform, and BOSARIS/FoCal affine calibration *is* a monotone
  transform. **A perfect calibration moves neither by one basis point.**
- So this is a **reporting and product** win: a fixed usable threshold for the
  fused tier, and an `actDCF` that means something. It is **not** an accuracy
  win.
- It therefore **cannot rescue criterion 2** (EER 4/5, §G3) and must never be
  presented as though it does.
- A surviving `actDCF`/`minDCF` gap after optimal calibration is a statement
  about discrimination — G3's problem, not this document's.

---

## Design decisions, fixed before measurement

1. **Transform family: BOSARIS/FoCal affine**, `s' = a·s + b` with `a ≥ 0`.
   The same family `compute_mincllr` already uses. Deliberately **not**
   pool-adjacent-violators: an unconstrained monotone transform would score
   better and would not be comparable to anything the ASVspoof/BOSARIS
   toolkits report.

2. **Calibration data: a held-out split of the eval partition. Never dev.**

   docs/13 measured that dev is not a held-out sample of eval — it is 100 %
   uncoded, carries no source-utterance column, and uses a disjoint attack set
   (A09–A16 vs A17–A32). The consequence was measured too: the dev-derived
   `eer_threshold` stored in every checkpoint is **0.85–1.98 nats too high**
   for eval, same sign in 4 of 4 runs. Fitting calibration on dev would import
   exactly that bias, in a known direction, and produce a model that
   under-detects spoof.

   Fitting on all of eval and reporting on all of eval is circular, so the
   eval data is split.

3. **Split rule — deterministic, no RNG:** an utterance belongs to the
   **calibration** half when `zlib.crc32(path) % 2 == 0`, else the **report**
   half. Deterministic so it is stable across runs, machines and checkpoint
   versions; no seed to lose.

4. **Fit on the calibration half, report on the report half.** The threshold
   is chosen once, on the calibration half, and is **not** re-optimised on the
   partition it is evaluated on.

---

## Criteria (fixed now, before any number exists)

- [ ] **Invariance guard — hard.** Pooled EER and minDCF of the report half are
      **unchanged by calibration** (`|Δ| ≤ 1e-9`). Both are rank-based, so an
      increasing affine map must leave them fixed in exact arithmetic; the
      tolerance exists only for floating point. If they move, the transform was
      not monotone-affine and the entire result is void, not merely weaker.
      This is the cheap check that catches the dangerous failure mode.
- [ ] **Primary.** `actDCF` on the report half, at the **fitted** threshold,
      improves substantially. Target: **actDCF within 1.5 × minDCF**. The 10 %
      target docs/10 originally set was retired there as unachievable at this
      discrimination level; 1.5 × is the replacement bar and is written down
      here rather than chosen afterwards.
- [ ] **Secondary.** `CLLR` on the report half falls toward `minCLLR`.
      Target: **CLLR − minCLLR < 0.15**.
- [ ] **`a ≥ 0`.** A negative slope means the ranking is inverted, which is a
      bug, not a result.
- [ ] Evaluate on the **20 k slice first** because it is cheap. Only if that
      passes does the full partition get a run.

---

## Method

- `scripts/score_checkpoint_eval.py --dump-scores` writes
  `path / label / attack / <model score>` per utterance. One GPU pass; every
  later question is a join.
- `benchmarks/calibrate.py` fits `(a, b)` using the Newton/IRLS objective
  already in `compute_mincllr`, returns them **in original score space**, and
  writes a JSON carrying the split rule and the fit metadata.
- Nothing is ever fitted on the report half.

---

## Explicitly not claimed by this document

- No improvement to EER, minDCF, or A28.
- No deployment change. Shipping requires a calibration block in the
  checkpoint schema and the backend detector to apply it; neither exists yet
  and both are out of scope here.
- No claim about dev.

---

## Status

- [x] Pre-registered.
- [ ] `benchmarks/calibrate.py` implemented and tested.
- [ ] Measured on the 20 k eval slice.
- [ ] Measured on the full partition (only if the slice passes).
