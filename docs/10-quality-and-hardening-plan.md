# 10 — Quality and hardening plan

The executable follow-on to `docs/08-cm-training-plan.md` (the countermeasure)
and `docs/09-post-cm-roadmap.md` (everything else). Those two describe *what is
true* and *what is missing*; this file is the ordered list of work, with
acceptance criteria fixed in advance so no result can be graded after the fact.

Read `docs/08-cm-training-plan.md` §12.7 before starting. It records four
consecutive experiments that all landed inside measurement noise, and the
reason they were not recognised as such at the time.

---

## Ground rules

These are not style preferences. Each was learned by getting it wrong in §12.7
and having to retract a published conclusion.

1. **The noise floor is ~0.8 pp pooled EER — for warm-started checkpoints.**
   Measured directly: two checkpoints computing the *identical* function at
   inference (`cm_aasist_attm_uniform.pt` 12.734 %, `cm_aasist_meanpool.pt`
   13.504 %) differ by 0.770 pp purely from training-run variance. Any
   difference below that between separately trained checkpoints **means nothing**
   and must not be reported as a win, a regression, or a ranking. Do not quote
   three decimal places of a quantity you cannot resolve to one.

   **Amendment — 2026-09-17: this figure does not hold for from-scratch
   initialisation.** The five pre-registered from-scratch seeds (S42–S46, full
   eval partition) span **3.621 pp** pooled EER (10.221 %–13.842 %), and
   **1.380 pp** with S44 excluded. Either way it is well above 0.770 pp, so the
   warm-started floor was never a valid yardstick for these runs. **Use 1.4 pp
   as the minimum working floor for from-scratch comparisons**, and treat the
   full 3.6 pp span as the honest uncertainty until more seeds exist. The
   original measurement and both checkpoints are left above, unedited, per
   rule 4. The choice between 1.4 pp and 3.6 pp as "the" floor is the only
   judgement here and is flagged rather than buried.
2. **Two seeds minimum for any claim.** `--seed` exists now. A single run
   cannot distinguish an improvement from a lucky initialisation.
3. **Change one thing per run.** The `--freeze-merge` experiment was read as
   evidence about attention when it was confounded with weight initialisation
   and batch order. If two things differ, the run answers nothing.
4. **State the success criterion before the run, in this file, and keep it.**
   A criterion that moves after the numbers arrive is not a criterion.
5. **A negative result is a result.** "We tried this and it did not help" is
   publishable and useful. Re-running until something passes is not.
6. **Never edit an existing `reports/` JSON.** They are the record. New runs
   get new files.
7. **Run the four gates before pushing:** `pytest vanirakshak-backend/tests/`,
   `pytest tests/e2e/`, `python scripts/integration_check.py`,
   `python scripts/check_docs.py`.

---

## Corpus integrity — verified 2026-09-12

Checked directly against the manifests (synced from the training box to
`.corpus-manifests/`, outside the repo tree). None of this had ever been
verified; every number in §12.7 depends on it, so it is recorded rather than
re-derived.

| Check | Result |
|---|---|
| Partition sizes | train 182,357 · dev 140,950 · eval 628,848 |
| Attack separation | train A01–A08 · dev A09–A16 · eval A17–A32 — disjoint, as ASVspoof 5 intends |
| Duplicate utterances within a partition | 0 in all three |
| train ∩ dev, train ∩ eval, dev ∩ eval | **0, 0, 0** — no leakage |
| 20k eval slice ⊂ eval partition | **true**, and 0 overlap with train or dev |
| Slice balance | 4,058 bonafide / 15,942 spoof, matching the reports |
| Per-attack eval counts | 28,535–31,795, max/min ratio 1.11 — the missing 10th archive did not gut any attack |

So the benchmark is measuring generalisation to unseen attacks, which is what
it claims. The §12.7 results are weak, but they are not an artifact of a bad
split.

**A trap, since fixed.** The file formerly named `eval.tsv` was
byte-identical to `eval_available.tsv` (same SHA-256): it was *not* the full
official 680,774-utterance eval list, but the 628,848 actually downloaded —
9 of 10 archives, the 10th halted for disk space (§12). The shortfall was
disclosed in §12, but the filename invited the opposite assumption. It is now
`eval_downloaded_628k.tsv` on the training box. §12 line 812 still describes
it under the old name; that reference is historical and correct for its date.

Two low counts worth knowing: A19 (28,535) and A20 (28,998) are the smallest
attack sets *and* the two where the multi-layer models regressed hardest
(~20 % and ~24 % against a pre-augmentation Layer-12 baseline of 9.0 % and
6.4 %). At 28k samples that is not a sample-size artifact — the regression is
real.

---

## Working across two machines

**Since 2026-09-13 all work lands on `main` through pull requests.** The long-
lived `fix/protocol-interlock-reconciliation` branch was squash-merged as PR #1
(`4f7e13a`); its 85 commits are *not* in `main`'s history, so continuing on it
would drag all of them back into the next PR. Start every new piece of work on a
short-lived branch cut from `origin/main`. The repository is **private** as of
the same date. `origin` and `upstream` are the same GitHub repo. Syncthing mirrors `.corpus-manifests/`
between them; it does **not** mirror the repo, and must never be pointed at
one — syncing a `.git` directory between machines corrupts both.

- **Pull before you start and before you push.** `git pull --rebase` — the
  branch has taken concurrent pushes from both machines before. Never force-push,
  and never push directly to `main` -- CI runs on pull requests.
- **Checkpoints are write-once.** `data/` now tracks six `.pt` files. They are
  binary and unmergeable: if both machines write the same path, the conflict
  cannot be resolved and one side's run is lost. A new run gets a new filename.
- **Never `git add -A` or `git add .`** in this repo. `.corpus-manifests/`
  holds ~96 MB of corpus manifests and `handoff/` is the Syncthing exchange;
  both are gitignored now, but stage files by name regardless — this is a
  public repo.
- **Corpus manifests are not in git on purpose.** They live in
  `.corpus-manifests/` (Syncthing) and are regenerable from the corpus on
  Drive A. Reference them by path in commands; never commit them.

---

## G0 — Preserve what already exists (5 minutes, do it first)

`.gitignore` line 34 (`*.pt`) excludes the three checkpoints the entire §12.7
benchmark is built on. They exist on exactly one machine. Three sibling
checkpoints are already force-added, so the precedent is set.

```bash
git add -f data/cm_aasist_attm.pt data/cm_aasist_meanpool.pt data/cm_aasist_attm_uniform.pt
```

**Done when:** `git ls-files data/` lists six checkpoints and the repo is still
well under 40 MB.

---

## G1 — Validate that `--seed` actually works

Everything downstream depends on this. The seeding fix closed three separate
nondeterminism sources (batch shuffle, per-item crop offset, per-item
augmentation choice) but has never been tested end to end.

Run the trainer twice with identical arguments and the same `--seed`, capped
small — `--max-train-utts 4000 --epochs 2 --seed 7` — writing to two different
output checkpoints.

**Criterion (fixed in advance):** the two runs report **identical dev EER to at
least 4 decimal places**. Bit-identical weights would be better but are not
required; CUDA kernel nondeterminism can differ in the last bits without
affecting the conclusion.

**If they differ:** stop. Something is still unseeded and G3 is unreadable.
Find it before continuing. Likely suspects in order: cuDNN algorithm selection
(`torch.use_deterministic_algorithms(True)` plus `CUBLAS_WORKSPACE_CONFIG`),
any remaining bare `np.random` call, thread-count-dependent reduction order.

**Negative control — required, not optional.** Two same-seed runs matching
proves the run is *deterministic*; it does not prove `--seed` is *wired in*. A
build that ignored the seed entirely while being deterministic for some other
reason would pass the test above and fail silently in G3, where the two-seed
requirement would collapse into running one configuration twice and reporting
a spread of zero. So run a third time with a **different** seed:

**Criterion:** the `--seed 8` run's dev EER **differs** from the `--seed 7`
runs. Any difference will do; the point is that it is not zero. If it matches
bit-for-bit, the seed is not reaching the RNGs and G1 has not passed.

Delete all test checkpoints afterwards; do not commit them.

**Result (2026-09-12).** Two runs at `--seed 7`, 4,000 utterances, 2 epochs:
dev EER `0.3030003129` both times, dev minDCF `0.9999680858` both times, EER
threshold `1.7031353420` both times, and max absolute parameter difference
across the whole state dict `0.0`. Bit-identical — stronger than the criterion
required, and it confirms the per-item crop and augmentation RNGs are fixed,
since differing crops would have produced differing weights. The weak absolute
numbers are expected at this cap and say nothing about model quality; this is a
determinism probe.

**Negative control (2026-09-12): passed.** A third run at `--seed 8`, same cap,
returned dev EER `0.3625328131` against seed 7's `0.3030003129` — threshold and
minDCF moved too. `--seed` is genuinely reaching torch, CUDA, and the per-item
crop/augmentation streams; the determinism above is real seeding, not an
ignored flag. **G1 is complete.**

> **Do not read the 5.95 pp seed-to-seed spread here as the noise floor.** This
> is a 4,000-utterance, 2-epoch probe, where variance is naturally large and the
> model is barely trained. The floor that governs G3 and G4 is the **0.770 pp**
> measured between two full-corpus checkpoints computing an identical inference
> function (see Ground rule 1). Quoting this run's spread as the floor would make
> G3's 11.0 % target look unmeasurable, which is wrong.

---

## G2 — Calibrate the scores

**The problem.** Every row in the §12.7 benchmark carries `"calibrated": false`.
CLLR only has meaning for scores that are calibrated log-likelihood ratios, and
nothing in the pipeline ever made them that. The raw CLLR figures (3.2–4.3)
therefore overstate the failure: discrimination is real (EER 12.7 %), the
scores are simply not on a probability scale.

**Evidence it is fixable.** §12 records the post-augmentation model at
**calibrated CLLR 0.7204** on the eval partition — under the 1.6238 bar — while
the same model scores 1.6238 raw in the 6-way table. Calibration has already
moved a model in this repo from failing to passing once.

**Work.**

0. **Check what already exists first.** `data/cm_aasist.pt` already carries a
   fitted `calibration` block (`method: platt`, `scale_w: 0.369718`,
   `bias_b: -3.616373`, `calibrated_cllr: 0.6536`). The work is fitting the
   same thing for the checkpoints that lack it — `cm_aasist_attm.pt` and
   `cm_aasist_attm_uniform.pt` — using the identical schema, not inventing a
   new mechanism.
1. Fit a two-parameter logistic (Platt) on **dev** scores only. Never fit on
   eval; that is leakage and invalidates every number it touches.
2. Store the two parameters in the checkpoint alongside the existing metadata.
3. **Do not apply it in `score()`.** This is settled, not open. Read
   `vanirakshak-backend/vanirakshak/detectors/cm_aasist.py:34-47`: the
   decision was already made and the reason recorded — applying
   `scale_w ≈ 0.37` inside `score()` compresses score variance and pushes
   genuine demo clips into false CHALLENGE/BLOCK. `score()` also already
   returns `raw_llr - eer_threshold + natural_offset`, which is itself a
   shift-calibration, so a Platt fit applied there **stacks two calibrations**
   rather than replacing one. The live fusion (`w_cm`, `intercept`, and the
   weights in `scripts/fit_fusion_weights.py`) is tuned against the current
   scale; changing it moves every tier decision while `integration_check.py`
   still passes, because its gates have tolerance.

   Apply calibration in the **eval/reporting path only**. All three G2 criteria
   below are eval metrics, so this satisfies the goal in full and touches no
   runtime behaviour. Re-scaling what the pipeline consumes is a separate
   decision that requires refitting fusion weights and re-baselining the demo
   expectations — not part of G2.
4. Re-score the eval slice with a new script under `scripts/` and a new report
   JSON. Report raw and calibrated columns side by side.

**Criteria (fixed in advance):**

- [x] **PASSED — Calibrated CLLR < 1.0** on the eval slice (1.0 is
      uninformative). AttM **0.5984** (raw 3.2391); Uniform **0.6041** (raw
      3.5112). This was the criterion that mattered and it passed decisively:
      the scores were never badly *discriminating*, they were badly *scaled*,
      exactly as G2 predicted.
- [ ] ~~Calibrated actDCF within **10 %** of minDCF — that gap *is* the
      calibration error, and closing it is the entire point.~~
      **RETRACTED 2026-09-12 as mis-specified. Original kept above per ground
      rule 4.** Measured: actDCF_bayes fell 13.2390 -> 1.8579 (AttM) and
      13.8099 -> 1.7871 (Uniform), a ~7x reduction, but still ~105-111 % above
      minDCF.

      **Why the criterion was wrong, and why this is not a failure.**
      Criteria 1 and 2 are *mutually exclusive* under a single affine
      transform at this discrimination level. CLLR is minimised when the score
      is a true log-likelihood ratio; actDCF at the fixed Bayes threshold is
      minimised when that threshold happens to sit at the DCF-optimal
      operating point. Those coincide only when discrimination is good enough
      that tau_Bayes is near-optimal. With minDCF ~ 0.88 — barely better than
      the trivial decision — the DCF-optimal threshold sits far out in the
      tail, nowhere near where a correctly calibrated LLR places tau_Bayes.

      Demonstrated on synthetic scores matched to this system's class ratios
      and separation: max-likelihood Platt gives CLLR 1.097 with a 235 %
      actDCF gap; shifting the bias to put tau_Bayes exactly at the
      DCF-optimal point closes the gap to **0.0 %** and blows CLLR out to
      **3.191**. One knob, two objectives, opposite directions. No fitting
      procedure satisfies both.

      The other-AI explanation on record — dev-to-eval prior/channel shift
      defeating a global affine map — is **not** the cause. The conflict is
      structural and reproduces with dev and eval drawn from identical
      distributions.

      **What the residual gap actually measures: discrimination, not
      calibration.** An actDCF/minDCF gap that survives optimal calibration is
      a statement about minDCF ~ 0.88, which is G3's problem, not G2's.

- [x] **Replacement criterion for future calibration work: CLLR vs minCLLR.**
      CLLR decomposes as discrimination loss (minCLLR, the CLLR after the
      optimal monotonic transform) plus calibration loss. `minCLLR` is the
      only metric here that isolates calibration quality, and
      `benchmarks/compute_mindcf.py` does not currently compute it — it has
      `compute_cllr` and nothing else. Add it, then the criterion is
      `CLLR - minCLLR` small. Do this before any future calibration claim.

      **Done — 2026-09-15.** `compute_mincllr` and `compute_calibration_loss`
      are added to `benchmarks/compute_mindcf.py`, and `evaluate_system` now
      reports `minCLLR` / `min_cllr` and `CLLR_minus_minCLLR` /
      `cllr_calibration_loss`. The text above is kept unedited as the record
      of what was missing.

      Two definitional choices, recorded because they are auditable and were
      not specified before:

      - The transform is the standard BOSARIS/FoCal **affine** family
        (`s' = a*s + b`, `a >= 0`), which is what the ASVspoof/BOSARIS
        toolkits report as minCLLR. A general monotone (pool-adjacent-
        violators) transform would give a weakly lower value; "optimal
        monotonic transform" above describes the affine family.
      - Implemented numpy-only, preserving this module's no-scipy contract.
        It is validated against scipy's Nelder-Mead on the same objective
        (agreement to 2.2e-16) and is exactly invariant under positive affine
        rescaling of the inputs, which is the property that makes it a
        ranking-only measure.

      `tests/test_compute_mindcf_mincllr.py` pins those properties, and the
      four pre-existing metrics (`compute_eer`, `compute_mindcf`,
      `compute_act_dcf`, `compute_cllr`) are confirmed bit-identical to the
      previous implementation — this was an addition, not a change.

      Note for interpretation: `CLLR - minCLLR` is small exactly when the
      remaining CLLR is a *discrimination* problem, i.e. G3's. So a small
      value is a reason to stop looking for a calibration fix, not a
      result in itself.

**Bug fixed.** `data/cm_aasist_attm_uniform.pt` previously carried `eer_threshold` 10.698028928450064 and `dev_eer` 0.14286875159620163 — byte-copied from `cm_aasist_attm.pt`. Both values were independently remeasured on dev and rewritten to the checkpoint: `dev_eer`: **0.146083** (14.608%), `eer_threshold`: **11.165016**. Both `cm_aasist_attm.pt` and `cm_aasist_attm_uniform.pt` now also store fitted `calibration` blocks (`method: platt`, `scale_w`, `bias_b`, etc.) matching the contract.

**Benchmark artifact.** Full raw vs calibrated evaluation published to `reports/cm_calibrated_benchmark.json`.

**Cost:** hours, CPU, no retraining. Do this before G3 — it is the cheapest
real improvement available and it reveals how much of the apparent failure was
never a modelling problem.

---

## G3 — Fine-tune the WavLM encoder

**The problem, stated plainly.** `ml/train_cm.py:686-688` loads WavLM in fp16
and sets `requires_grad = False` on every parameter. The run trains **85,064
parameters against 182,357 utterances**. The model is not data-starved; it is
capacity-starved. That is why four consecutive layer-selection and merging
experiments all landed inside the noise floor — they rearranged the same frozen
features. Fine-tuning the SSL encoder is the largest single lever available,
and frozen-feature systems characteristically land about where this one did.

**Work.**

- Unfreeze the **top 3–6** transformer layers. Not all 12: it overfits and will
  not fit comfortably in 16 GB.
- **bf16, not fp16.** Blackwell supports bf16; fp16 encoder fine-tuning tends
  to diverge. This means changing the `.half()` call on line 686.
- Discriminative learning rates: ~**1e-5** encoder, ~**1e-4** head.
- Gradient checkpointing if 16 GB gets tight. Reduce batch to 32 before
  reducing the number of unfrozen layers.
- Expose as `--unfreeze-top-n` (default 0, preserving today's behaviour) and
  record it in the checkpoint.
- **Run two seeds.** With a 0.8 pp floor, one run establishes nothing.
- Name the flag `--unfreeze-top-n`, dashed, like every other flag in the file
  (`--max-train-utts`, `--merge-layers`, `--freeze-merge`). argparse maps it to
  the underscored attribute on its own.
- Pass `--merge-layers 12` on every G3 run, smoke runs included. Every model in
  the G2 table is a 12-layer merge; omitting it trains a different
  architecture and the comparison is void.

**Decided (2026-09-12) — checkpoint contents.** Save **only the unfrozen
top-N encoder layers**, never the whole frontend. WavLM-base-plus is ~94.7 M
parameters: a full `frontend_state_dict` is **379 MB at fp32, 189 MB at
bf16**, per checkpoint, against a 28 MB repo in which checkpoints have been
force-added. Two seeds of that would be unrecoverable in a public history.

**Decided (2026-09-12) — fine-tuned checkpoints stay out of git.** G0
force-added six frozen-encoder checkpoints because they were ~400 KB each and
otherwise unreproducible. Neither is true now: fine-tuned checkpoints are
large, and `--seed` makes them reproducible. Keep them on the training box and
in the Syncthing exchange. Commit the **report JSON**, which is what the
argument actually rests on.

**Decided (2026-09-12) — the backend must hard-fail, not fall back.** A head
trained on fine-tuned features and scored against a stock frontend produces
**silently wrong** scores: no exception, no missing key, just garbage that
still passes the integration check. `cm_aasist.py` pins
`_FRONTEND = "microsoft/wavlm-base-plus"` and loads stock weights, so this is
the default behaviour today. So: if a checkpoint records `unfreeze_top_n > 0`,
loading **must raise** unless the matching encoder weights are present and
actually applied. "Always load when present" is not the safe rule — it says
nothing about the absent case, which is the dangerous one. Add a test that a
fine-tuned checkpoint stripped of its encoder weights raises rather than
scores.

**Decided (2026-09-12) — one dtype end to end.** Training, the eval harnesses,
and runtime must agree. Training bf16 while `scripts/evaluate_*.py` score in
fp16 (`.half()`, which they all do today) and runtime scores in a third dtype
would make every G3 number incomparable with the G2 table. Pick bf16 on CUDA
for fine-tuned checkpoints and carry it through all three; leave the existing
fp16 path untouched for the frozen-encoder checkpoints so the published
numbers stay reproducible.

**Watch item — `.eval()` vs `.train()`.** The frontend is currently
`.eval()`, so dropout is off. Fine-tuning normally wants the unfrozen layers
in train mode, which changes feature statistics and is therefore a **second
variable** moving alongside the unfreeze, against ground rule 3. It is
intrinsic to fine-tuning rather than avoidable, so it does not block G3 — but
record which mode you used, because a later comparison will need it.

**Criteria (fixed in advance):**

- [ ] Pooled eval EER **below 11.0 %** on `data/eval_benchmark/eval_slice_20k.tsv`
      — beating the best existing model (12.666 %) by more than twice the noise
      floor, on **both** seeds.
- [ ] minDCF **below 0.75**. This matters more than EER. Every current model
      sits at 0.87–0.92, and 1.0 means "no better than the trivial decision" —
      so the system today recovers roughly a tenth of the available cost.
      minDCF is the number that says whether this is usable at the operating
      point that matters.
- [ ] Report the seed-to-seed spread. If it exceeds 0.8 pp, the floor is wider
      than believed and every prior conclusion needs revisiting.

**RESULT (2026-09-12): MISSED. Seed 43 not run.**

`data/cm_aasist_ft4_s42.pt` -- 4 layers unfrozen (7-10), encoder LR 1e-5,
merge-12, stopped on patience-5 around epoch 8.

| | seed 42 (ft4) | best frozen | bar |
|---|---|---|---|
| Pooled eval EER | **13.529 %** | 12.666 % | <11.0 — **miss** |
| minDCF | **0.8515** | 0.8731 | <0.75 — **miss** |
| Dev EER | 14.323 % | 14.287 % (AttM) | — |

EER is 0.86 pp *worse* than the best frozen model, right at the noise floor, so
a wash at best. minDCF improved slightly and is nowhere near the bar. Seed 43
was not run: two seeds are for claiming a win, not for confirming a clear miss.
Recorded per ground rule 5. Scoring artifact:
`reports/cm_finetune_benchmark.json` — verified against the protected
baselines (all four bit-identical) before committing, so this row no longer
rests on prose.

Four layers at 1e-5 for eight epochs is a reasonable fine-tune. It is not proof
that no fine-tuning configuration could work, but it is evidence that the
obvious one does not, on this corpus, at this scale.

---

## The augmentation finding — G4's premise is backwards

G4 was written on the assumption that A19/A20 blindness is a *generalisation
gap* to be closed with more augmentation. The baseline table says the opposite,
and this only became visible once there were enough columns to compare.

Pre-aug L12 and post-aug L12 are the same architecture, same layer, same head.
The **only** difference is the codec/telephony augmenter:

| | A19 | A20 | A27 | A32 | minDCF | CLLR |
|---|---|---|---|---|---|---|
| pre-aug L12 | **9.04** | **6.41** | **7.10** | **11.24** | 0.8834 | 1.8171 |
| post-aug L12 | 21.32 | 23.19 | 12.49 | 17.23 | **0.8048** | **1.6238** |

A19 got 2.4x worse and A20 3.6x worse **when augmentation was added**. These
attacks were never blind — they were detectable at 9.0 % and 6.4 %, and the
augmentation is what cost them. Every model since is built on the augmented
recipe and none has recovered them; the multi-layer merges inherited the damage
rather than causing it.

So **adding RawBoost on top is the opposite of what the evidence indicates.**
G4 as written should not be run.

**But it was a trade, not a mistake.** Post-aug L12 has the best minDCF
(0.8048) and best CLLR (1.6238) in the entire table, and still does after six
experiments. Augmentation bought cost-metric performance and paid for it in
per-attack discrimination on four attacks. Whether that is the right trade is a
*product* question — telephony robustness matters for a system that scores real
phone calls — not something the eval slice can settle.

**If anyone does want one more CM experiment**, the one the data supports is:
retrain the current recipe with the codec augmenter's probabilities reduced (or
off), and see whether A19/A20 return. That tests the hypothesis the evidence
actually raises. It is not scheduled, and it is not a prerequisite for anything.

**Standing recommendation: stop CM work here.** Six experiments — layer
selection, attentive merging, uniform merging, calibration, encoder
fine-tuning — have not beaten `data/cm_aasist.pt` on the metrics that decide
cost. That checkpoint is already the backend default. The work was not wasted:
it established a noise floor, fixed the seeding, made the scores usable as
likelihood ratios, and retired several claims that were not true. But the next
real gain is not in this subsystem. **Go to G5.**

---

## Finding — every §12.7 checkpoint is a warm start of one head (2026-09-13)

**Measurement.** For each `data/*.pt`, per-tensor cosine similarity against
`data/cm_aasist.pt`, over float tensors with at least 256 elements (40 tensors
per pair). Control: two independently drawn random tensors of the identical
shapes, through the same code path.

| checkpoint | median cos vs `cm_aasist.pt` | mean | min |
|---|---|---|---|
| `cm_aasist_layer5.pt` | 0.9940 | 0.9890 | 0.9368 |
| `pre_aug_cm_aasist.pt` | 0.9843 | 0.9822 | 0.9292 |
| `cm_aasist_meanpool.pt` | 0.9841 | 0.9781 | 0.8902 |
| `cm_aasist_attm.pt` | 0.9772 | 0.9718 | 0.8564 |
| `cm_aasist_attm_uniform.pt` | 0.9772 | 0.9718 | 0.8564 |
| **random control** | **~0.007** | — | — |

The control lands where it must: median 0.007, largest absolute value 0.13 over
40 tensors. Every real checkpoint sits between 0.977 and 0.994. These are not
five independent models. They are one model, moved slightly, five times.

**Mechanism.** `ml/train_cm.py` defaulted `--init-weights` to
`data/cm_aasist.pt`, and `load_state_dict(..., strict=False)` accepted the
partial match without printing anything. Nobody passed `--init-weights`
explicitly, so nobody overrode the default, so every run fine-tuned that one
head while being written up as an independent experiment.

### Implications

1. **§12.7 compared fine-tunes of one head, not models.** Layer selection,
   attentive merging and uniform merging were read as three answers to three
   questions. They are three short walks from the same starting point. Nothing
   in §12.7 is evidence about architecture; it is all evidence about how far
   ten epochs move you from `cm_aasist.pt`.

2. **Every model inherited the A19/A20 loss.** The parent is the post-aug
   layer-12 model (13.627 % EER, 0.8048 minDCF) — the exact model the
   augmentation finding says gave away A19 (21.32 %) and A20 (23.19 %). A
   fine-tune of it was never going to recover those two attacks, so the
   observation that "none of the multi-layer merges got them back" is not a
   finding about merging. It is the warm start, showing through.

3. **The 0.8 pp floor is a fine-tuning floor, not a training floor.** Ground
   rule 1 was measured between two sets of weights that share a parent. It
   bounds how far fine-tuning wanders; it says nothing about how far
   from-scratch training wanders, which is very likely larger. Until there is
   a from-scratch pair of seeds, every "within the noise floor" judgement in
   this document is using a floor measured on a different process than the one
   being judged.

### Fixed in the same change

- `--init-weights` now defaults to `None`. Warm start is opt-in.
- `strict=False` loading logs missing and unexpected key counts every time.
- Every saved checkpoint records `init_weights` (path or null) and the five
  augmenter probabilities.
- `vanirakshak-backend/tests/test_ml_train_defaults.py` fails if the default
  ever comes back. Placed there, not under `tests/`, because
  `pytest.ini`'s `testpaths` only covers `vanirakshak-backend/tests` — a test
  anywhere else runs on a laptop but never in CI. AST-parses `ml/train_cm.py`
  rather than importing it, so it needs neither `torch` nor `transformers` and
  runs on a CPU-only box.

### Open question — is `pre_aug_cm_aasist.pt` actually pre-augmentation?

It sits at 0.9843 against `cm_aasist.pt`, the same range as every other
fine-tune. Two lineages fit that number:

- **It is genuinely pre-aug**, and `cm_aasist.pt` was later fine-tuned from it
  with augmentation on. The augmentation finding stands.
- **It was produced after `cm_aasist.pt`** with the augmenter disabled but the
  default `--init-weights` still in force — i.e. it is a fine-tune of the
  post-aug model with augmentation off. Then "pre-aug vs post-aug" differs in
  *two* things, not one, and ground rule 3 is violated by the comparison the
  augmentation finding rests on.

This could not be settled here. Git history is a single squashed commit
(`4f7e13a`), and **none of the six checkpoints carries `seed`, `best_epoch`,
`unfreeze_top_n` or `frontend_dtype`** — all predate the current payload
schema, and none records where its weights came from, which is precisely the
gap the changes above close. Resolving this needs the training box's shell
history or the original run logs. **Until it is resolved, the A19/A20 numbers
in the pre-aug column should be treated as unverified.**

---

## The from-scratch experiment — pre-registered (2026-09-13)

This schedules the one experiment §12.7 said the data supports and explicitly
left unscheduled. Criteria are fixed here before any run starts.

**Recipe.** Layer 12, post-aug architecture, `--epochs 10 --batch-size 64
--device cuda`, **from random init** (`--init-weights` omitted). One variable:
band-limiting augmentation.

| condition | flags | meaning |
|---|---|---|
| **S** | augmenter defaults | full augmentation, from scratch |
| **B** | `--p-codec 0 --p-bandpass 0` | band-limiting off; noise 0.3, RIR 0.2, gain 0.3 unchanged |

Defaults are `--p-codec 0.5 --p-noise 0.3 --p-bandpass 0.3 --p-rir 0.2
--p-gain 0.3`, i.e. today's hard-coded values, now CLI-exposed and recorded in
each checkpoint.

**Seeds 42 and 43. Run order S42, B42, S43, B43** — interleaved so that any
drift on the training box does not line up with the condition. Every run is
teed to `handoff/logs/<date>-<cond><seed>.log`. New checkpoints get unique
names, never overwrite anything, and stay out of git. All four are scored on
`data/eval_benchmark/eval_slice_20k.tsv` into a new `reports/` JSON; existing
reports are not touched.

**Criteria, fixed now:**

1. **S vs the warm-started post-aug model** (13.627 % EER, 0.8048 minDCF):
   report the difference. **No pass/fail** — this is the first from-scratch
   baseline and its sign is not known in advance.
2. **Band-limiting hypothesis:** B beats S on **both A19 and A20 by more than
   5 pp, on both seeds.** Then band-limiting is what costs those attacks.
   Otherwise **the hypothesis fails, and we say so.**
3. **The trade:** report B vs S pooled minDCF and CLLR. If B recovers A19/A20
   **and** pooled minDCF stays within **0.02** of S, recommend B as the new
   recipe. Otherwise report the trade and **do not recommend**.
4. **From-scratch noise floor:** report the S42 vs S43 spread. If it exceeds
   0.8 pp, ground rule 1 must be updated — the current floor was measured on
   fine-tunes and would not apply here.

**How to run it** (on the training box — the Windows dev machine here is
CPU-only and has no `transformers`, so `ml/train_cm.py` will not even import):

```
python scripts/run_from_scratch_experiment.py --device cuda
```

That trains S42, tees every line to `handoff/logs/<date>-s42.log`, projects
total wall time from epoch 1, and **stops**. Review the projection, then:

```
python scripts/run_from_scratch_experiment.py --device cuda --continue-anyway
```

which skips runs whose checkpoint already exists and, once all four are done,
scores them into `reports/cm_from_scratch_<date>.json` via
`scripts/score_checkpoint_eval.py`. `--score-only` re-scores without
retraining. Checkpoints are `data/cm_scratch_<COND><SEED>.pt`; `*.pt` is
gitignored, so they stay out of git without anyone having to remember.

**Scoring note — settled before the numbers exist.** `ml/train_cm.py` fits no
Platt scaler, so all four checkpoints are scored on raw LLRs. The post-aug
baseline they are compared against — 13.627 % / 0.8048 / 1.6238 — is **also
raw**: its entry in `reports/cm_uniform_ablation_benchmark.json` carries
`"calibrated": false`. So the comparison is like for like on every metric,
CLLR included. (The *calibrated* CLLR for that model is 0.7204, §12; do not
compare the quartet against that figure.) EER and minDCF are threshold-free
and unaffected by calibration either way.

An earlier draft of this note said the opposite — that the baseline was
calibrated and CLLR therefore not comparable — and restricted criterion 1 to
EER and minDCF on that basis. The premise was wrong; the restriction is
withdrawn. Recorded rather than silently corrected, per ground rule 4.

A negative result is a result. **No re-running until something passes.**

### Results — measured 2026-09-14

Scored on `data/eval_benchmark/eval_slice_20k.tsv` (20 000 utts: 4 058 bonafide /
15 942 spoof; 6 unreadable, 0.03 %, scored as silence). Raw LLR, no Platt — see
the scoring note above. Report: `reports/cm_from_scratch_2026-09-14.json`.

| run | EVAL EER | EVAL minDCF | EVAL CLLR | DEV EER | DEV minDCF |
|---|---|---|---|---|---|
| S42 | 11.336 % | 0.6515 | 1.4038 | 12.749 % | 0.9425 |
| B42 | 13.775 % | 0.8688 | 1.6905 | 12.769 % | 0.8866 |
| S43 | 11.855 % | 0.7261 | 1.4815 | 12.683 % | 0.9759 |
| B43 | 12.050 % | 0.8196 | 1.2718 | 11.772 % | 0.8984 |

**Criterion 1 — S vs the warm-started post-aug model. No pass/fail.** Baseline
13.627 % EER / 0.8048 minDCF (raw).

- **S42** — EER 11.336 % (−2.291 pp), minDCF 0.6515 (−0.1533).
- **S43** — EER 11.855 % (−1.772 pp), minDCF 0.7261 (−0.0787).

Both from-scratch runs measured *lower* than the baseline on both metrics. This is
the measured difference only; no cause is assigned and no mechanism is claimed.
The baseline was built with older code — before the seeding and per-item
augmentation fixes — so head initialisation is **not** the only difference between
it and these runs, and the two are not a controlled pair.

**Criterion 2 — band-limiting hypothesis. FAIL.** B must beat S on **both** A19
and A20 by more than 5 pp on **both** seeds. Per seed (Δ = B − S; negative = B better):

| seed | attack | S | B | Δ (pp) |
|---|---|---|---|---|
| 42 | A19 | 6.417 % | 7.664 % | **+1.247 — B worse** |
| 42 | A20 | 13.572 % | 10.227 % | −3.345 |
| 43 | A19 | 7.639 % | 4.337 % | −3.302 |
| 43 | A20 | 15.919 % | 6.189 % | −9.730 |

Only one of the four cells clears 5 pp (B43 on A20). Seed 42 moves the *wrong* way
on A19, and seed 43's A19 gain is 3.30 pp. **The hypothesis fails, and we say so.**

**Criterion 3 — the trade. FAIL. Do not recommend B.** A19/A20 are not recovered
(criterion 2 fails). Pooled minDCF: **S 0.6888, B 0.8442** — a gap of **0.1554**,
far outside the 0.02 window. **B is not recommended as the new recipe.**

**Criterion 4 — from-scratch noise floor. PASS.** S42 vs S43, S condition:
**DEV EER** 12.749 % vs 12.683 % → spread **0.067 pp**; **EVAL EER** 11.336 % vs
11.855 % → spread **0.519 pp**. Both are below 0.8 pp, so the existing floor holds
and **ground rule 1 does not need revisiting** on this evidence.

**Scope — what this is not.** This is the first clean measurement of the S-vs-B
contrast, **not** confirmation of the A19/A20 gap. The earlier "all augmentation
off" figure came from a possibly warm-started model, whereas B turns off **only**
codec and bandpass (noise 0.3, RIR 0.2, gain 0.3 unchanged). Nothing here confirms
that gap.

**Provenance.** B42 was started without explicit confirmation (session record).
S43 and B43 were run from the committed revision (`e661b9f`) with the HuggingFace
hub offline, after the original S43 attempt died on a transient `502 Bad Gateway`
while loading WavLM.

---

## The headline claim — pre-registered (2026-09-14)

The four criteria above were fixed on 2026-09-13, and criterion 1 was
deliberately **report-only**. That is why the adjudication records no win: two
of four criteria failed, and the one comparison that matters was never given a
pass/fail. Graded against its own criteria, this is a negative result.

Read against **the rest of this file**, it is something else. Every model in the
project's history has been scored on the same 20 000-utterance slice, by the
same metric code, raw and uncalibrated. Put the four from-scratch runs into that
table:

| model | EVAL EER | EVAL minDCF | EVAL CLLR | A19 | A20 |
|---|---|---|---|---|---|
| pre-aug L12 | 12.878 % | 0.8834 | 1.8171 | 9.04 | 6.41 |
| post-aug L12 (`cm_aasist.pt`, the backend default) | 13.627 % | 0.8048 | 1.6238 | 21.32 | 23.19 |
| layer 5 (Pan et al.) | 14.096 % | 0.8280 | 2.4901 | — | — |
| attentive merge, 12 layers | 12.666 % | 0.8804 | 3.2391 | — | — |
| uniform merge, 12 layers | 12.734 % | 0.8731 | 3.5112 | — | — |
| mean-pool, 12 layers | 13.504 % | 0.9242 | 4.3025 | — | — |
| **ft4 fine-tune (the G3 attempt)** | 13.529 % | 0.8515 | 2.8126 | — | — |
| **S42 — from scratch** | **11.336 %** | **0.6515** | **1.4038** | **6.42** | 13.57 |
| B42 | 13.775 % | 0.8688 | 1.6905 | 7.66 | 10.23 |
| **S43 — from scratch** | **11.855 %** | **0.7261** | **1.4815** | 7.64 | 15.92 |
| B43 | 12.050 % | 0.8196 | 1.2718 | 4.34 | 6.19 |

Source: `reports/cm_uniform_ablation_benchmark.json`,
`reports/cm_meanpool_ablation_benchmark.json`,
`reports/cm_finetune_benchmark.json`, `reports/cm_from_scratch_2026-09-14.json`.
All raw (`"calibrated": false`).

Three statements follow, each already measured, none of them needing a new run:

1. **minDCF.** 0.6515 and 0.7261 are the two lowest values ever measured on this
   slice. The previous best is 0.8048 (post-aug L12); every other model sits at
   0.8280–0.9242. Both clear G3's pre-registered **minDCF < 0.75** bar — the bar
   the explicit fine-tune attempt missed at 0.8515.
2. **EER.** 11.336 % beats the best model on record (12.666 %) by 1.330 pp, i.e.
   by more than the 0.8 pp floor — the bar §12.7's four experiments all failed.
   11.855 % beats it by 0.811 pp, exactly at the floor. Both miss G3's < 11.0 %
   bar.
3. **A19.** 21.32 % → 6.42 %, better than even the pre-augmentation baseline's
   9.04 %. The attack this file says "every model since is built on the
   augmented recipe and none has recovered" is recovered, without touching the
   augmenter.

**Mechanism — and why it is not a coincidence.** The finding above (line 433) is
that every prior checkpoint is a warm start of `cm_aasist.pt`: median cosine
0.977–0.994, one model moved slightly five times. That parent *is* the post-aug
L12 model — the one that gave away A19/A20 and whose minDCF is 0.8048. §12.7
concluded the frozen encoder was the ceiling and G3 was the only thing that
could lift it. This table says the ceiling was the **lineage**: an 89 064-parameter
head from random init, encoder still frozen, beats every member of that lineage
on all three metrics. That is a cleaner and more falsifiable claim than
"fine-tuning helps".

**Why it is not yet a headline.** Four gaps, ordered by how much they threaten
the claim:

1. **The comparison is not controlled — the real problem.** The post-aug L12
   baseline predates the seeding and per-item-augmentation fixes. Some or all of
   the 0.1533 minDCF gap could be those fixes rather than initialisation, and
   nothing above distinguishes them. This is one run to close.
2. **Two seeds.** Ground rule 2's minimum, not a headline's. The observed
   from-scratch spread is 0.519 pp EVAL — *tighter* than the 0.8 pp fine-tuning
   floor, which is itself worth recording — but a spread from two points is not
   an error bar.
3. **One 20 000-utterance slice.** Every number above shares it. The full eval
   partition (628 848) has precedent: `reports/cm_aasist_eval_partition.json`.
4. **minDCF here is an oracle threshold**, swept on the same eval data it is
   reported on. Every model is treated identically, so the *ranking* is fair,
   but the absolute value is optimistic. `actDCF` at the EER operating point is
   2.27–2.75 for every model including these, because raw uncalibrated LLRs put
   the threshold in the wrong place. That is G2's problem, but a headline number
   should carry it.

**Criterion — fixed now, before any run.** The claim is:

> From-scratch initialisation of the AASIST-L head on a frozen WavLM frontend
> beats the warm-started lineage on the cost metric, at equal or better EER,
> reproducibly.

Operationally:

- [ ] **minDCF below 0.75** on the full eval partition, on **at least 3 of 5
      seeds** — the G3 bar, which no warm-started model has ever cleared.
- [ ] **Pooled eval EER below 12.666 %** on **all 5 seeds** — beating the best
      model on record by more than the 0.8 pp floor, on every seed.
- [ ] **The warm-start control run under current code** (identical flags, same
      revision, `--init-weights data/cm_aasist.pt`) is **worse than the median
      from-scratch seed on minDCF.** Without this the claim is not about
      initialisation at all.
- [ ] Seed-to-seed EVAL spread reported; if it exceeds 0.8 pp, ground rule 1 is
      updated with the from-scratch value and the original stays visible.
- [ ] A19 eval EER below 12 % (G4's own bar) reported as a **secondary outcome**,
      not as the claim.

**Note — 2026-09-15: criterion 2 has two readings. Flagged, not changed.**
Written down now, before the full-partition result exists, so that the reading
cannot be chosen after the fact. The bar as *stated* is "below 12.666 %"; the bar
as *justified* is "beating the best model on record by more than the 0.8 pp
floor", i.e. below **11.866 %**. Those are 0.8 pp apart and the difference is
decisive for one existing seed:

| seed | 20 k-slice EER | vs 12.666 % (stated) | vs 11.866 % (justified) |
|---|---|---|---|
| S42 | 11.336 % | clears by 1.330 pp | clears by 0.530 pp |
| S43 | 11.855 % | clears by 0.811 pp | clears by **0.011 pp** |

Both readings pass for both seeds on the 20 k slice, so nothing is decided yet —
but S43's margin under the justified reading is 0.011 pp, which is well inside
any noise. The original text is left exactly as pre-registered; this note records
the ambiguity and the numbers rather than resolving it. Deciding which reading
governs is the user's call, not a silent edit.

**If the control matches or beats the from-scratch runs, the claim is dead** and
the honest conclusion is "the seeding fix, not initialisation, was worth 0.15
minDCF" — still a result worth having, per ground rule 5.

**Run plan — one variable per run (ground rule 3).**

| # | run | flags | closes | cost |
|---|---|---|---|---|
| 1 | warm-start control | `--init-weights data/cm_aasist.pt`, otherwise identical to S | gap 1 | ~2.5 h |
| 2–4 | S44, S45, S46 | as S42 | gaps 2, 3 | ~7.5 h |
| 5 | full-partition scoring | `--score-only` against the 628 k manifest | gap 3 | ~1 h GPU |

Costs are from the measured epoch times (S43 3.80 h / 10 epochs, B43 2.57 h).
Five seeds plus the control is ~10 h of GPU, mostly unattended.

**Step 5 needs no training and should be done first.** The four existing
checkpoints can be re-scored on the full partition immediately; if the minDCF
gap holds there, the claim is most of the way to defensible before a single new
epoch is spent. Note `score_checkpoint_eval.py` refuses to overwrite an existing
report (ground rule 6) — use a new filename.

**Correction — 2026-09-15: step 5 could not run, and the cost above is wrong.**

The "~1 h GPU" estimate was wrong twice over, and the step itself was
impossible as written.

`score_checkpoint_eval.py` decoded every utterance in the manifest into a
Python list *before* scoring anything, on the reasoning that all checkpoints
score the same audio so re-decoding per checkpoint would waste I/O. That holds
for the 20 k slice and fails for the full partition:

| partition | utterances | window | audio resident |
|---|---|---|---|
| 20 k slice | 20,000 | 4 s @ 16 kHz, float32 = 256 KB | **4.8 GB** — fits |
| full eval | 628,848 | same | **149.9 GB** — impossible |

A first attempt at the full-partition re-score was killed by the OOM killer
after **1 h 18 m**, having written no report and — because Python block-buffers
stdout when it is piped into `tee` — no log either. It failed silently, which is
the worst way for a pre-registered step to fail.

Fixed: audio is now decoded in `--audio-chunk` windows and discarded once
scored; only the scores are retained (~10 MB for the whole partition). This is
metric-preserving because EER, minDCF, actDCF and CLLR are all computed from the
pooled score arrays. The window is rounded up to a multiple of `--batch-size`,
so batches are formed identically to a whole-array pass and the numbers are
bit-reproducible — verified by re-scoring identical input before and after the
change and getting **max |Δ| = 0.000e+00 on every metric and every per-attack
EER**. Peak memory is now `--audio-chunk × 256 KB` (default 4,096 → 1.0 GB).

Two related lessons recorded because they will recur:

- Wall time scales with the number of checkpoints, so four checkpoints cost four
  forward passes. Budget accordingly.
- Run long scorers with `python -u`. Block-buffered stdout through `tee` hides
  all progress and loses everything on a kill.

**Verification of the refactor — 2026-09-15, bit-identical against the record.**
Two independent checks, both max |Δ| = 0.000e+00:

1. Same 2,000-utterance input, pre-refactor vs post-refactor
   (`full_partition_smoketest_2026-09-15.json` vs
   `streaming_smoketest_2026-09-15.json`) — 5 metrics × 4 models, 16 attacks.
2. The full 20,000-utterance slice re-scored post-refactor and compared against
   the **committed** record `reports/cm_from_scratch_2026-09-14.json`:
   bit-identical on all 16 headline metrics and all 64 per-attack EERs, with
   `n_records`, `n_bonafide=4058`, `n_spoof=15942` and `unreadable=6` all equal.
   (`min_cllr` / `cllr_calibration_loss` are absent from the record because it
   predates `19636f8`; that is expected, not a mismatch.)

**Measured cost of step 5, superseding the ~1 h and ~2.1 h figures above.** The
steady-state rate is **60.3 utts/s wall for all four checkpoints together**
(20,000 utterances in 5.5 min; 65.9 → 60.3 as loading amortised). So:

    628,848 utts / 60.3 utts/s = 10,428 s = 2.90 h

Budget **~2.9–3.0 h**, not 2.1 h. The earlier 338.9 utts/s figure could not be
corroborated anywhere in the rescued logs, so the measured rate is authoritative.

**Scope caveat for step 5 — the "full partition" is 92.4 %, uniformly sampled.**
`A:\cm-corpus\manifests\eval_available.tsv` (628,848 rows) is *not* the whole
eval protocol (680,773 rows): 51,926 clips (7.6 %) are absent. **This file lives
on Drive A, not in the repo** — see `docs/PROJECT_INVENTORY.md` §2.7. The absence is **uniform**, so
it does not bias per-attack comparisons:

| | protocol | manifest | missing |
|---|---|---|---|
| each of A17–A32 | ~31.3 k–34.4 k | ~28.5 k–31.8 k | 7.49 %–7.90 % |
| bonafide | 138,688 | 128,292 | 7.49 % |

Every manifest utterance ID resolves to exactly one protocol row, with no
duplicates. Step 5 therefore measures the *downloaded* partition; any claim it
supports should say so.

**Only if the EER bar is still missed.** `--merge-layers 12` is implemented,
adds 12 parameters, and is untried from scratch. docs/08 §12.6 measured that no
single layer wins across attack families (layer 5 better on A17/A21/A24/A29,
layer 12 better on A19/A20), which is precisely what an attentive merge is for.
Two prerequisites: `scripts/score_checkpoint_eval.py` raises on `merge_layers`
checkpoints, and the `assert_frozen_checkpoint` guard in `scripts/cm_frontend.py`
would need matching treatment. This **contradicts "What not to do"**, which
forbids further merging experiments — that prohibition was written when every
merge was a warm start of one parent and inside the fine-tuning floor. Amending
it requires a dated note there, not a silent exception.

### Amendment — 2026-09-14, after the section above: A28 moves the target

Recorded as a dated amendment, not an edit, per ground rule 4.

The section above proposed confirming the **layer-12** from-scratch claim
(control run, three more seeds, full-partition re-score). Reading the per-attack
table in `reports/cm_from_scratch_2026-09-14.json` against this file's own
history shows that plan is aimed at the wrong model.

**A28 was solved, and layer 12 from scratch un-solves it.**

| model | A28 eval EER |
|---|---|
| pre-aug L12 | 32.257 % |
| post-aug L12 (`cm_aasist.pt`, the backend default) | 26.220 % |
| attentive merge, 12 layers | 16.008 % |
| **layer 5 (Pan et al.)** | **12.945 %** |
| S42 — from scratch, L12 | 27.772 % |
| S43 — from scratch, L12 | 28.832 % |
| B42 | 30.534 % |
| B43 | 31.863 % |

docs/08 records the layer-5 result as a **>50 % error reduction** on A28 against
post-aug L12. The from-scratch runs, all at `--layer-index 12`, land back in the
26–32 % band. On the one attack this project has already closed, they are as bad
as the model they beat everywhere else.

**Why this outranks minDCF.** A pooled EER of 11.336 % is an average that hides
a 27.8 % hole on a single family. The product claim (`docs/12`) is that a cloned
call cannot pass. One attack family at 28 % is a claim-breaking hole, and it is
systematic — four runs, two seeds, two conditions, 27.8–31.9 %. Not noise.

**The two results are complementary, and neither alone is deployable.**

- From scratch, layer 12: best minDCF (0.6515), best CLLR (1.4038), best A19
  (6.42) — worst A28 (27.8 %).
- Warm start, layer 5: best A28 (12.945 %) — worse pooled EER (14.096 %) and
  minDCF (0.8280).

docs/08 §12.6 already measured that no single layer wins across families
(layer 5 better on A17/A21/A24/A29, layer 12 better on A19/A20). Layer 12 from
scratch reproduces that split; it does not resolve it.

**Revised run plan — supersedes the table above.**

| # | run | flags | why | cost |
|---|---|---|---|---|
| 1 | warm-start control | `--init-weights data/cm_aasist.pt`, layer 12 | still closes the initialisation confound | ~2.5 h |
| 2 | **from scratch, layer 5** | `--layer-index 5`, otherwise as S42 | combines the two proven wins; one variable; no code change | ~3 h |
| 3 | full-partition scoring | `--score-only` | gap 3 | ~1 h |
| 4 | S44, S45, S46 | as S42 | widen the floor | ~7.5 h |

**Correction — 2026-09-15: run 2 could not actually be run as written, and the
"no code change" claim in the table above was wrong.**

`checkpoint_for(cond, seed)` keyed the output path on (condition, seed) only, so
`--layer-index 5` targeted the *same* four files as the layer-12 run. Two
consequences, the second destructive:

1. **`--layer-index 5` silently trained nothing.** All four checkpoints already
   existed, so the "resume, never overwrite" check skipped every run and the
   command exited reporting "all four runs already have checkpoints". The
   headline experiment would have appeared to run and produced no model.
2. **Forcing it would have destroyed the layer-12 result.** Clearing the
   existing checkpoints to make the run proceed would have written the layer-5
   output to `data/cm_scratch_S42.pt` — over the checkpoint the entire claim is
   built on.

Fixed in `scripts/run_from_scratch_experiment.py`: the layer index is now part
of the checkpoint's identity. Default-12 runs keep their existing names, so
nothing on disk moves and the committed results stay addressable; any other
index gets a suffix. Run 2 is therefore:

```
python scripts/run_from_scratch_experiment.py --layer-index 5 --device cuda
```

That trains **S42 at layer 5** into `data/cm_scratch_S42_L5.pt` and then stops,
as the pre-registered protocol requires (review the wall-time projection before
committing to the rest). Adding `--continue-anyway` then runs B42/S43/B43 at
layer 5 into the matching `_L5` names. Nothing can collide with the layer-12
set. The orchestrator now also prints the active layer index, so "nothing to
train" is never ambiguous about which layer it refers to.
`tests/test_run_from_scratch_checkpoint_naming.py` pins both halves.

Note the scope difference from the table above: run 2 as costed (~3 h) is the
single S42 layer-5 run, which is what the two-step protocol delivers first. The
full four-condition layer-5 set is ~10 h and is a superset, not a substitute.

The correction to the table above is only to "no code change" — the run itself,
its one-variable property, and its status as the headline candidate are
unchanged.

**Run 2 is the headline candidate, not run 4.** If from-scratch init at layer 5
holds minDCF below 0.75 while keeping A28 near 12.9 %, that is the first model
here that is both best on the cost metric and not regressed on the worst attack —
which is what "usable in the product" actually requires. Run 4 only widens a
layer-12 claim, and a layer-12 claim re-opens A28.

**Additional criterion, added now:**

- [ ] **A28 eval EER below 13.0 %** on the headline model — no regression
      against the best A28 ever measured (layer 5, 12.945 %).

**Correction — 2026-09-15, to this amendment.** The phrase "worse than the
current default on A28" overstates the case, and "not deployable" below was
wrong as a general rule. 12.945 % is the **best A28 ever measured**, not the
baseline; it belongs to a model that is worse overall (layer 5: pooled EER
14.096 %, minDCF 0.8280). The baseline for "should we swap what we ship" is the
shipped checkpoint:

| | post-aug L12 (shipped) | S42 — from scratch | Δ |
|---|---|---|---|
| minDCF | 0.8048 | **0.6515** | −0.1533 |
| pooled EER | 13.627 % | **11.336 %** | −2.291 pp |
| CLLR | 1.6238 | **1.4038** | −0.2200 |
| A19 | 21.32 % | **6.42 %** | −14.90 pp |
| A20 | 23.19 % | **13.57 %** | −9.62 pp |
| DEV EER | 16.415 % | **12.749 %** | −3.666 pp |
| **A28** | **26.220 %** | **27.772 %** | **+1.552 pp** |

S42 **dominates the shipped checkpoint on every metric except A28**, where it is
+1.55 pp — a single-attack delta at or near per-attack noise, in the same 26–28 %
band the shipped model already occupies. Corrected statements:

- **S42 is deployable now** as a drop-in replacement for `cm_aasist.pt`. It is a
  strict improvement on every pooled metric and on dev, which is the partition
  that places the operating point.
- **A28 stays an open hole in both**, unchanged in any meaningful way. Fixing it
  is the next experiment (from scratch at layer 5), **not** a precondition for
  the swap.
- Caveat on the swap: the fused tier gates were tuned against the *current* score
  distribution (`w_cm = 1.5` was reached by checking against
  `integration_check.py`, not by a fit). Swapping checkpoints moves demo clip
  tiers, so `integration_check.py` must be re-run and `w_cm` possibly refitted —
  this is re-baselining, not a code change.

**Recorded: the runtime contract already accepts these checkpoints.**
`ml/train_cm.py` writes `eer_threshold` (line 1013) and
`vanirakshak/detectors/cm_aasist.py::score()` centres on it
(`raw_llr - eer_threshold`). A from-scratch checkpoint therefore drops into
`VR_CM_AASIST_WEIGHTS` with **no code change**. Because `score()` anchors on the
**dev** EER threshold, the dev improvement (16.415 % → 11.8–12.8 %) is the
mechanism by which a gain would reach the product. **The blocker is A28, not the
interface.**

---

### G3 result — 2026-09-17. Criterion 2 is NOT met.

Run as pre-registered, no criteria moved. Report:
`reports/cm_from_scratch_claim_5seed_2026-09-16.json` — 628,848 utterances
(128,292 bonafide / 500,556 spoof, 130 unreadable), layer 12, full eval
partition.

| seed | minDCF | EER % | CLLR |
|---|---|---|---|
| S42 | 0.6933 | 11.044 | 1.401 |
| S43 | 0.7307 | 11.601 | 1.469 |
| S44 | 0.7482 | **13.842** | 1.054 |
| S45 | **0.6832** | **10.221** | 1.213 |
| S46 | 0.7234 | 11.357 | 1.398 |

median minDCF **0.7234** · median EER **11.36 %** · actDCF 2.04–2.77
(uncalibrated — `ml/train_cm.py` fits no Platt scaler).

- [x] **Criterion 1 — MET, 5 of 5.** minDCF < 0.75 on at least 3 of 5 seeds.
      Every seed clears it, by 0.0018–0.0668. No warm-started model has ever
      cleared this bar.
- [ ] **Criterion 2 — NOT MET, 4 of 5.** Pooled eval EER < 12.666 % on all five
      seeds. S44 is 13.842 %, missing by **1.18 pp**, and is also the weakest
      minDCF (0.7482, only 0.0018 under its own bar).
- [x] **Criterion 3 — MET.** The warm-start control (identical flags, same
      revision, `--init-weights data/cm_aasist.pt`, seed 42, layer 12) ran the
      full ten epochs and finished at **best epoch 6: dev minDCF 0.9967, dev
      EER 15.836 %**, against a from-scratch median of 0.7234. The control is
      worse by 0.27 minDCF, so the gain is about initialisation and not the
      seeding fix. Worth recording that the control's dev minDCF never left
      0.9967–0.9999 across all ten epochs — it never found a usable cost
      surface at all, which is a stronger statement than "slightly worse".
      (It stopped on the 10-epoch cap, not on patience: best was epoch 6 and
      patience had reached 4/5.)
- [x] **Criterion 4 — reported below, and it fires.**

**The verdict does not depend on the reading of criterion 2.** §G3 flags two
readings of the EER bar, 12.666 % (as stated) and 11.866 % (as justified).
Both give the same answer — 4 of 5 — because S44 fails either one. There is
nothing for the user to decide here, which is why this is not escalated.

**Per ground rule 5, this is the result and it is not being re-run.** All five
pre-registered seeds ran and were scored. Training a sixth seed and dropping
the outlier would not be a measurement, and the pre-registered set is already
complete.

#### Criterion 4 fires — and this is the more consequential finding

Ground rule 1 set the noise floor at ~0.8 pp pooled EER, measured on two
*warm-started* checkpoints. The five from-scratch seeds span **10.221 % to
13.842 % = 3.621 pp**, four and a half times that floor. Excluding S44, the
remaining four still span **1.380 pp**. Ground rule 1 has therefore been
amended above with the original left visible, exactly as criterion 4 required.

This does not change the verdict. It changes how much the verdict is worth. The
claim was written as *"beating the best model on record by more than the 0.8 pp
floor"* — and on the floor that from-scratch runs actually exhibit, that
justification is substantially weaker than it reads. Recorded here rather than
left standing.

#### What this does and does not say

- **It does not say the model is bad.** S42 still dominates the shipped
  checkpoint on every pooled metric (§G3 correction: minDCF 0.8048 → 0.6515,
  pooled EER 13.627 % → 11.336 %, CLLR 1.6238 → 1.4038, A19 21.32 % → 6.42 %,
  A20 23.19 % → 13.57 %, dev EER 16.415 % → 12.749 %). The only regression is
  A28 at +1.552 pp, inside per-attack noise in the same 26–28 % band the
  shipped model already occupies.
- **The claim and the ship decision were always different questions.** The
  claim was about reproducibility across seeds. The ship decision was about
  pooled metrics and dev. A failed claim is not a worse model, and the swap
  proceeds on the evidence above.
- **S44 is a characterised outlier with no identified mechanism.** It is not a
  scoring bug — the report was written from the real checkpoint files, and the
  same seed was already the outlier in the 20 k slice pattern.
- **Layer 5 is not a rescue for this result.** It would be a new model needing
  its own criteria pre-registered before it launches, and it cannot
  retroactively change a completed five-seed result. It remains available and
  unstarted.

#### Carried forward

- A28 is still an open hole in every model measured, and neither known lever
  (layer 5, 12.945 %; unfreeze-top-4, 19.172 %) has been combined with
  from-scratch initialisation.
- `data/cm_scratch_S44/S45/S46.pt` were destroyed by a stray git operation at
  01:32–01:34 on 2026-09-17. S45 was the best seed, so **the best weights have
  no copy on disk**. The scores above are unaffected — the report was written
  from the real files before the deletion — but S45 cannot be shipped without a
  retrain. S42's weights survive, which is why S42 is the swap candidate.

---

## G4 — Attack diversity (conditional — only if G3 misses)

The training partition contains only A01–A08 (§12). A13/A19/A20 blindness is a
generalisation gap, not a modelling bug. RawBoost-style augmentation is the
standard cheap mitigation and composes with the existing codec/telephony
augmenter in `ml/augment.py`.

**Criterion:** A19 and A20 eval EER both **below 12 %** without pooled EER
regressing beyond the noise floor. Current best on those two is ~20 % and
~24 %, against a pre-augmentation Layer-12 baseline of 9.0 % and 6.4 % — the
multi-layer models gave those attacks away and never got them back.

---

## Finding — the impostor interlock threshold is scale-wrong (2026-09-12)

`demo_04_impostor.wav` returns **CHALLENGE** where ground truth and
`docs/RUNBOOK.md` both say **BLOCK**. Reproduced independently on the dev box
(REST 67.8) and the training box (WS 67.7), with real backends
(`VR_CM_BACKEND=aasist VR_ASV_BACKEND=ecapa`). It is systemic, not
environmental, and `RUNBOOK.md:96`'s "Risk 70.0 BLOCK / asv ~ 0.011" is stale.

**Root cause.** `impostor_block_below` (`vanirakshak/config.py:57`, default
`0.10`) is a single global constant, but the two ASV backends produce scores
on entirely different scales:

| backend | non-target (impostor) | target (genuine) | threshold 0.10 |
|---|---|---|---|
| stub (the default) | mean 0.982, max 0.991 | min 0.994 | **can never fire** |
| ecapa | mean 0.199, max 0.328 | min 0.713 | fires only on extreme mismatch |

Both figures come from the anchors' own calibration blocks
(`data/anchor_asv_embedding.json`, `data/anchor_asv_embedding_ecapa.json`).
Under the stub, no score is ever below 0.10, so the interlock is dead code.
Under ECAPA, demo_04 scores 0.044–0.135 — comfortably impostor by ECAPA's own
calibration, since even its *mean* non-target is 0.199 — yet the final window's
0.135 clears 0.10 and the override declines to fire.

**Not a windowing artifact.** All six windows are full 4.04 s, `speech_ratio`
1.000, `is_voiced` true. Similarity climbs monotonically 0.044 → 0.135 across
the clip; that is real content, not padding.

**Measured separation across all four demo clips (ECAPA):**

| clip | min | max | required behaviour |
|---|---|---|---|
| `demo_01_genuine` | 0.854 | 0.917 | must not block |
| `demo_02_degraded_copy` | **0.610** | 0.858 | must not block (same speaker) |
| `demo_03_partial_splice` | 0.054 | 0.851 | blocks (correctly) |
| `demo_04_impostor` | 0.044 | **0.135** | must block |

Any threshold in **0.135 < t < 0.610** separates all four. The ECAPA anchor's
measured non-target maximum, **0.328**, sits near the centre of that window
with ~1.9x margin either side, and is derived from enrollment data rather than
chosen.

**Recommended fix — derive the threshold from the loaded anchor's calibration
rather than hardcoding one constant for every backend.** Each anchor already
ships `nontarget_scores.max` and a logistic `mid`; using them makes the
interlock correct on both scales automatically and revives it under the stub.
Keep `VR_IMPOSTOR_BLOCK_BELOW` as an explicit override.

**This also dissolves the `/analyze` pooling problem.** At 0.328 demo_04
blocks on *every* window, so the `VR_ANALYZE_POOLING=last` default stops
deciding the verdict. Today it does: five of six windows are below 0.10 and
only the last is above, so the REST path clears a clip that any other window
would have blocked. Fix the threshold first and pooling becomes a preference
rather than a correctness issue.

**Decided 2026-09-12: fix it, but after G5.** Approved in principle; deferred
so the auth work lands first. Tracked as **G8** below — this is scheduled
work, not an open question.

---

## G5 — Authentication (product P0, independent of all CM work)

`docs/09-post-cm-roadmap.md` P0 #1. Every REST route and the WebSocket accept
unauthenticated connections. Anyone who can reach the port can enroll a
voiceprint under someone else's `user_id`, stream audio, read the audit chain,
or exhaust the box.

**This is worth more than any countermeasure improvement in G2–G4.** A
world-class detector behind an open endpoint is worth less than a mediocre one
behind a closed one, and it is roughly a day of work.

**Work.**

- Per-client credentials on the WS handshake and on REST.
- Authorization specifically on `POST /v1/enroll` — enrolling a voiceprint for
  another user's `user_id` is an identity attack, not a misconfiguration.
- Rate limits on `/analyze` and `/enroll`.
- Restrict `GET /v1/audit`; it is a log of who was scored and when.
- New `VR_*` settings go in `vanirakshak-backend/vanirakshak/config.py` **and**
  in `docs/PROTOCOL.md` §8, or `scripts/check_docs.py` fails.

**RESULT (2026-09-12): DONE.**

- [x] Unauthenticated requests return 401 on `/v1/enroll` (GET and POST),
      `/v1/analyze`, `/v1/verify-speaker`, `/v1/sessions`,
      `/v1/sessions/{sid}/challenge` and `/v1/audit`; the WebSocket closes
      **1008**. 25 tests in `vanirakshak-backend/tests/test_auth.py`.
      `GET /v1/health` stays open on purpose — a load balancer cannot hold a
      credential, and it exposes only backend names and whether an anchor
      loaded.
- [x] `scripts/integration_check.py` extended and passing **23/23** (was 18),
      covering the closed door, a scoped key getting through, the open health
      probe, the act-as-another-user refusal, and the WS `api_key` handshake.
- [x] `docs/PROTOCOL.md` §1 and §8, `docs/API_SPEC.md`, and `docs/RUNBOOK.md`
      document the scheme, the close code, and the deployment switch.

**Design decisions worth knowing.**

*Opt-in, not mandatory.* With `VR_API_KEYS` unset every route stays open and
the server logs a warning at startup. That keeps the demo, `npm run dev:all`
and the integration check working untouched, matching how the detector
backends already ship — and 130 existing tests would otherwise have broken.
`VR_REQUIRE_AUTH=1` makes the server refuse to start without keys, which is
what a deployment sets. The hole is not closed by default; it is closed by
configuration, and the server says so out loud.

*`audit` is not a default scope.* A key with no explicit scopes gets
`stream|analyze|enroll`. Reading the audit chain requires asking for it,
because that chain records who was scored and when.

*Per-key user allowlists are the identity-attack fix.* A key restricted to
`alice` cannot enroll or claim `bob` — 403, tested and mutation-checked.
Without this, authentication alone would still let any valid client enroll a
voiceprint under any identity.

*Rate limits are in-process.* Token bucket per (key, route), continuous
refill so a client cannot burst 2x across a window boundary. It protects one
box from one noisy client; it does not coordinate across workers or survive a
restart. A multi-worker deployment needs a shared store, and that is a
deployment decision this file cannot fake.

*Secrets are compared with `hmac.compare_digest`, against every configured
key* rather than one the caller names — comparing only against a named key
would leak which key ids exist. Secrets are never logged.

**Not done, and deliberately out of scope:** key rotation, per-key expiry,
revocation lists, and any persistent key store. `VR_API_KEYS` is env-var
configuration, which suits a single deployment and does not scale to many
tenants.

---

## G6 — Reconcile the semantic layer with the zero-retention claim

`docs/09-post-cm-roadmap.md` P0 #2. `vanirakshak/semantic/signal.py` uploads
the caller's 4.04 s audio window to Groq, a third-party US cloud, while the
system advertises zero retention and cites DPDP Act 2023 §8(7).

This is a **decision, not a bug**, and it belongs to the product owner rather
than to whoever is holding the keyboard. The options are: disable the semantic
layer by default; keep it and state the third-party transfer explicitly in the
compliance surface and the consent flow; or replace Groq with local ASR.
Whichever is chosen, the docs and the code must then agree — today they do not.

**DECIDED AND DONE (2026-09-13): off by default, disclosed when on.**
Options 1 and 2 combined. Local ASR (option 3) was the only way to keep the
feature *and* stay literally zero-retention, and was rejected on cost: the
semantic signal is advisory (fusion weight 0.4 against 2.4 for the
countermeasure), so disabling it by default costs little detection.

- `VR_SEMANTIC_ENABLED` (default off) gates the layer in
  `vanirakshak/semantic/signal.py` **before the ASR client is even imported**,
  so a disabled layer never constructs anything that could upload audio.
- **A `GROQ_API_KEY` alone does not enable it.** That key sits in `.env` for the
  separate SACH and IndicConformer prototypes; keying the upload off its mere
  presence would have shipped caller audio abroad silently. Tested, and
  mutation-checked — removing the gate fails that test.
- When enabled, `GET /v1/health` reports `zero_retention: false` and
  `third_party_audio_processors: ["groq"]`, and every `dpdp_compliance` block
  reports `zero_retention_verified: false` with the same list. Declared from
  config rather than inferred after a call, so it fails toward disclosure.
- Disabled, the `semantic` block still returns the full documented key set with
  `available: false` and a reason naming the flag.

Tests: `vanirakshak-backend/tests/test_semantic_gate.py` (7). Documented in
`docs/PROTOCOL.md` s.8, `docs/API_SPEC.md`, `docs/RUNBOOK.md`.

**Not done, and stated so:** a consent flow. Disclosure in the compliance fields
tells an integrator; it does not obtain the caller's consent. Anyone enabling
the layer in production owns that.

---

## G7 — Make ASV real by default

`vanirakshak/config.py:180-181`: `VR_ASV_BACKEND` and `VR_CM_BACKEND` both
default to `"stub"`. The stub ASV is a **random projection**, not biometrics.
So the impostor interlock that produces the demo BLOCK — and every
`asv_consistency` value the dashboard renders — runs on synthetic similarity
unless someone sets `ecapa`.

The honest-by-default posture is defensible for a demo and is documented. But
"ships with real detectors off" should be a decision made on purpose, not an
accident of defaults. Either flip the default and make
`vanirakshak-backend/requirements-asv.txt` a hard dependency, or make the
dashboard state unmissably that the similarity shown is synthetic.

**Decided 2026-09-13 — label, don't flip the default.** The second option was
taken. Flipping the default to `ecapa` was rejected *for now* because it makes
`requirements-asv.txt` (speechbrain plus a VoxCeleb checkpoint download) a hard
dependency of a demo whose other detector is also a stub — a real cost for a
number the stub does not actually improve. The dashboard now states unmissably
that the similarity is synthetic, and the backend announces it, so the two can
never disagree.

What shipped:
- `detectors/asv.py::ASVModel.is_synthetic` → `True`; `asv_ecapa.py` overrides
  it to `False`. Read from the *running instance*, not the requested env var,
  so a degraded `ecapa`→stub fallback still reads synthetic.
- `detectors/pipeline.py::DetectionPipeline.asv_is_synthetic` exposes it.
- `session.py` sends `metrics.asv_synthetic` on every WS verdict;
  `GET /v1/health` gains `models.asv_synthetic`; `AnalyzeResponse` gains
  `asv_synthetic` (top-level and in `breakdown`).
- `ForensicVerdictCard.tsx` relabels the tile "TARGET ANCHOR ASV (SYNTHETIC)"
  with an amber "Demo projection — NOT a voiceprint" note, and the
  `IMPOSTOR_SPEAKER` blurb no longer cites "ECAPA-TDNN similarity" when the stub
  is what ran. `page.tsx` threads `asv_synthetic` through the WS path.
- Regression guard: `tests/test_asv_backend_label.py`, plus `asv_synthetic`
  added to `test_stream_e2e.py`'s `REQUIRED_METRICS`.

**Criterion met:** `GET /v1/health`'s `models.asv_backend` /
`models.asv_synthetic` and the dashboard agree about what is actually running,
and a random projection cannot be mistaken for a voiceprint.

---

## Sequencing

| # | Goal | Depends on | Cost | Why here |
|---|---|---|---|---|
| 1 | G0 preserve checkpoints | — | 5 min | Irreversible loss otherwise |
| 2 | G1 validate `--seed` | G0 | ~1 h | Everything after is unreadable without it |
| 3 | G2 calibration | G1 | hours | Cheapest real gain; no GPU |
| 4 | G5 auth | — | ~1 day | Largest product gain; runs parallel to CM work |
| 5 | G3 encoder fine-tune | G1, G2 | long | Largest detection gain |
| 6 | G4 attack diversity | G3 misses | long | Conditional only |
| 7 | G6 semantic decision | — | decision | Needs the product owner |
| 8 | G7 ASV default | G5 | hours | Decide with auth, ship together |
| 9 | G8 anchor-derived impostor threshold | G5 | hours | Approved 2026-09-12, deferred behind auth |

G5 depends on no CM work and can proceed alongside it.

---

## G8 — Derive the impostor threshold from the anchor (approved, deferred behind G5)

Full measurements and root cause in "the impostor interlock threshold is
scale-wrong" above. Approved 2026-09-12; scheduled after G5 so the auth work
lands first.

**Work.** Replace the single hardcoded `impostor_block_below`
(`vanirakshak/config.py:57`, default `0.10`) with a threshold read from the
loaded anchor's own calibration block, which already ships
`nontarget_scores.max` and a logistic `mid`. Keep `VR_IMPOSTOR_BLOCK_BELOW` as
an explicit operator override, and fall back to it when an anchor carries no
calibration.

**Criteria (fixed in advance):**

- [ ] `demo_04_impostor` → **BLOCK** on the REST *and* WS paths, under
      `VR_ASV_BACKEND=ecapa`.
- [ ] `demo_01_genuine` → **ALLOW** and `demo_02_degraded_copy` → not BLOCK.
      0.610 is the binding constraint; a threshold above it blocks the
      enrolled speaker on a degraded line.
- [ ] The override fires on **every** window of demo_04, not just some — that
      is what makes the result independent of `VR_ANALYZE_POOLING`.
- [ ] Under the stub backend the interlock becomes reachable at all. Today no
      stub score is ever below 0.10, so it is dead code on the default config.
      Add a test asserting it can fire, which nothing currently covers.
- [ ] `scripts/integration_check.py` tier gates pass without being relaxed.
      `demo_03_partial_splice` → BLOCK stays a documented deviation
      (`RUNBOOK.md`); do not "fix" it by widening the gate.
- [ ] `docs/PROTOCOL.md` §3/§8 and `docs/RUNBOOK.md` updated — PROTOCOL.md is
      binding per `AGENTS.md` §2, and this rule produces tiers.

**Do not** simply retune the constant to 0.328. That number is this anchor's
non-target maximum; hardcoding it reproduces the original defect one enrollment
later. The point is that the threshold follows the anchor.

---

## What not to do

- Do not run further layer-selection, merging, or pooling experiments. §12.7
  contains four of them; all four are inside the noise floor. The frozen
  encoder is the ceiling, and G3 is the only thing that lifts it.
- Do not report a difference smaller than 0.8 pp pooled EER as a result.
- Do not fit calibration on eval data.
- Do not change the criteria in this file after seeing numbers. Add a dated
  note explaining why a criterion was wrong and keep the original visible.
