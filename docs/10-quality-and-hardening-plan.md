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

1. **The noise floor is ~0.8 pp pooled EER.** Measured directly: two
   checkpoints computing the *identical* function at inference
   (`cm_aasist_attm_uniform.pt` 12.734 %, `cm_aasist_meanpool.pt` 13.504 %)
   differ by 0.770 pp purely from training-run variance. Any difference below
   that between separately trained checkpoints **means nothing** and must not
   be reported as a win, a regression, or a ranking. Do not quote three decimal
   places of a quantity you cannot resolve to one.
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

The dev box and the training box both push to
`fix/protocol-interlock-reconciliation` on the same remote (`origin` and
`upstream` are the same GitHub repo). Syncthing mirrors `.corpus-manifests/`
between them; it does **not** mirror the repo, and must never be pointed at
one — syncing a `.git` directory between machines corrupts both.

- **Pull before you start and before you push.** `git pull --rebase` — the
  branch has taken concurrent pushes from both machines already. Never force-push.
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

- [ ] **Replacement criterion for future calibration work: CLLR vs minCLLR.**
      CLLR decomposes as discrimination loss (minCLLR, the CLLR after the
      optimal monotonic transform) plus calibration loss. `minCLLR` is the
      only metric here that isolates calibration quality, and
      `benchmarks/compute_mindcf.py` does not currently compute it — it has
      `compute_cllr` and nothing else. Add it, then the criterion is
      `CLLR - minCLLR` small. Do this before any future calibration claim.

**Bug fixed.** `data/cm_aasist_attm_uniform.pt` previously carried `eer_threshold` 10.698028928450064 and `dev_eer` 0.14286875159620163 — byte-copied from `cm_aasist_attm.pt`. Both values were independently remeasured on dev and rewritten to the checkpoint: `dev_eer`: **0.146083** (14.608%), `eer_threshold`: **11.165016**. Both `cm_aasist_attm.pt` and `cm_aasist_attm_uniform.pt` now also store fitted `calibration` blocks (`method: platt`, `scale_w`, `bias_b`, etc.) matching the contract.

**Benchmark artifact.** Full raw vs calibrated evaluation published to `reports/cm_calibrated_benchmark.json`.

**Cost:** hours, CPU, no retraining. Do this before G3 — it is the cheapest
real improvement available and it reveals how much of the apparent failure was
never a modelling problem.

---

**Measured 2026-09-19 — G2 applied to the G4 checkpoint.** Recorded result
only; nothing above was added, removed or reinterpreted.

`scripts/fit_cm_calibration.py` generalises `run_g2_calibration.py` to any
checkpoint — it reads `merge_layers` / `layer_index` from the file rather than
hardcoding 12, which the original did, so it cannot score a non-merge
checkpoint. Platt fitted on **dev only** (12,000 rows); eval used strictly for
reporting.

| Metric | raw | calibrated |
|---|---|---|
| DEV CLLR | 2.3852 | **0.4686** |
| EVAL CLLR | 2.6703 | **0.5097** |
| EVAL actDCF (Bayes) | 11.9196 | **1.7143** |
| EVAL EER / minDCF | 11.212 % / 0.7286 | unchanged — by design |

Criterion 1 (calibrated CLLR < 1.0 on eval): **PASS** at 0.5097, in line with
the AttM 0.5984 / Uniform 0.6041 precedents. So the G4 checkpoint's problem
was the same as every previous one: the scores were badly *scaled*, not badly
*discriminating*.

The actDCF/minDCF gap reproduces the retracted criterion 2 almost exactly:
actDCF fell ~7× (11.92 → 1.71) yet still sits ~135 % above minDCF (0.7286),
against the 105–111 % recorded above. Per the analysis above that residual is
**discrimination, not calibration** — G3's problem, not G2's.

Runtime `score()` deliberately untouched, per item 3. Implemented with
`scipy` (declared in requirements.txt, installed) rather than `sklearn`
(declared nowhere, absent from this venv — `run_g2_calibration.py` cannot
run here at all).

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

**If it misses:** say so, record the numbers, and do **not** chase it with
architecture changes. Go to G4.

**Cost:** the long pole. Hours per run, two runs.

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

### G4 measured — seed 42 final (2026-09-19 06:50, `eb0f13c`)

Seed 42 finished on the epoch budget (patience 3/5); best epoch **7**, dev
minDCF 0.8590 / dev EER 11.677 %. Scored on the full 20,000-row eval slice
with `adjudicate_g4.py`. Every comparison row below was re-scored on that
same modern harness, so the table is apples-to-apples.

| Checkpoint | Pooled EER | minDCF | A19 | A20 |
|---|---|---|---|---|
| **cm_g4_rawboost_s42.pt (ep 7)** | **11.212 %** | **0.7286** | 16.257 % | 17.283 % |
| pre_aug_cm_aasist.pt (L12) | 12.845 % | — | **8.920 %** | **6.330 %** |
| cm_aasist_attm.pt | 12.682 % | — | 20.210 % | 24.020 % |
| cm_aasist_ft4_s42.pt | 13.529 % | — | 20.430 % | 22.600 % |
| cm_aasist.pt (post-aug L12) | 13.598 % | — | 21.390 % | 23.150 % |

**Verdict on the stated criterion: FAIL.** A19 16.257 % and A20 17.283 %
against a 12 % bar. Regression vs the control arm is **UNRESOLVED** — the
control began training 2026-09-19 10:18 and is not finished.

### Both seeds (seed 43 finished 10:18, early stop at epoch 9)

| Seed | best ep | dev minDCF | pooled EER (95 % CI) | A19 (95 % CI) | A20 (95 % CI) |
|---|---|---|---|---|---|
| 42 | 7 | 0.8590 | 11.212 % [10.718, 11.779] | 16.264 % [14.973, 17.792] | 17.324 % [15.943, 18.680] |
| 43 | 4 | 0.9208 | **10.522 %** [9.930, 11.090] | 15.623 % [14.220, 17.000] | 14.786 % [13.480, 16.170] |

Worst-seed is the decision rule, so the verdict is A19 **16.257 %** / A20
**17.283 %** (seed 42) → **FAIL**.

**The FAIL is robust across seeds, not seed luck.** Each seed independently
clears the bar at no point: seed 43's CI floors are 14.220 (A19) and 13.480
(A20), both above 12 %.

**The two seeds are statistically indistinguishable.** Every interval
overlaps (pooled overlap [10.718, 11.090]; A20 barely overlaps at
[15.943, 16.170]). Seed 43's apparently better pooled EER is within noise.
minDCF is nearly identical — 0.7286 vs 0.7287. Seed-to-seed pooled spread is
0.714 pp, **below** the 0.770 pp floor, so no open item is raised.

**Dev-based selection did not predict eval across seeds.** Seed 43 has the
*worse* dev minDCF (0.9208 vs 0.8590) yet the *better* eval pooled EER. Within
a run, dev minDCF is the pre-registered selector and is the right one to use;
across runs it did not rank them. Do not use dev minDCF to choose between
seeds.

Seed 43's curve also stalled: after epoch 4 its dev minDCF returned to 1.0000
(trivial reject-all) and never recovered, so early stopping fired at epoch 9
(patience 5/5). The selected checkpoint is still epoch 4.

**Two honest findings, reported whether or not they flatter the run:**

1. **Pooled 11.212 % / minDCF 0.7286 is the best pooled result this repo has
   produced.** RawBoost helped the pooled score.
2. **Dev-minDCF selection appears to have cost A19, but only A19.** Epoch 3
   measured A19 13.701 / A20 16.437; the selected epoch 7 measures 16.257 /
   17.283. Pooled improved 11.237 → 11.212. Selection was fixed in advance on
   dev minDCF, so epoch 7 is the correct checkpoint — but read the size of
   that movement against the noise before repeating it as fact.

### How much of this is noise (bootstrapped, `scripts/eer_bootstrap_ci.py`)

A19 and A20 have only **935 and 920** spoof utterances in the 20k slice, so a
per-attack EER quoted to three decimals invites over-reading. 2,000-rep
percentile bootstrap on the epoch-7 scores:

| | n_spoof | EER | 95 % CI | ± |
|---|---|---|---|---|
| pooled | 15,936 | **11.212 %** | [10.718, 11.779] | 0.53 |
| A19 | 935 | 16.264 % | [14.973, 17.792] | 1.36 |
| A20 | 920 | 17.324 % | [15.943, 18.680] | 1.36 |
| A28 | 1,012 | 16.996 % | [15.810, 18.577] | 1.36 |
| A32 | 980 | 19.936 % | [18.776, 21.341] | 1.30 |

**The FAIL is robust.** A19's CI floor is 14.97 % and A20's is 15.94 % — both
above the 12 % bar at the optimistic end of the interval. The verdict does not
depend on where in the interval the true value sits.

**The pooled win is also robust.** 11.779 % ceiling vs the best prior pooled
(AttM 12.682 %) — no overlap.

**But the epoch-3 → epoch-7 movement is mostly not.** A20 moved 0.89 pp
against a ±1.36 pp interval: that is noise, and I should not have written it
as a regression. A19 moved 2.56 pp, which exceeds the individual interval, so
it is probably real — but the epoch-3 checkpoint was overwritten, so a *paired*
test is impossible and I am not claiming significance. Treat A19 as "probably
regressed" and A20 as "unchanged within noise".

(A20 counts 920 here and 921 in the adjudication: one utterance failed to
decode. Immaterial at 0.04 pp.)

### Same intervals for the comparators — so the ranking is tested, not assumed

Bootstrapping only G4 would let me compare a *point estimate* against an
*interval*, which proves nothing. All three models, same script, same slice:

| Model | pooled EER (95 % CI) | A19 (95 % CI) | A20 (95 % CI) |
|---|---|---|---|
| **G4 s42 ep7** | **11.212 %** [10.718, 11.779] | 16.264 % [14.973, 17.792] | 17.324 % [15.943, 18.680] |
| AttM | 12.682 % [12.150, 13.270] | 20.214 % [18.600, 22.350] | 24.022 % [22.280, 25.430] |
| Pre-Aug L12 | 12.870 % [12.260, 13.380] | **8.984 %** [7.930, 10.160] | **6.382 %** [5.570, 7.390] |

Now every claim is a test with non-overlapping intervals, not an assertion:

- **G4 beats AttM on pooled** — 11.779 ceiling vs 12.150 floor. Significant.
- **G4 beats AttM on A19 *and* A20** — both intervals disjoint. So RawBoost
  did recover ground on those two attacks; it just did not recover *enough*.
- **G4 still fails the bar** — its best-case A19/A20 (14.973 / 15.943) are
  both above 12 %.
- **Pre-Aug beats G4 on A19/A20 by a wide margin** — 10.160 ceiling vs 14.973
  floor. It is also the only model whose *entire interval* sits under the
  12 % bar, so its pass is real, not a lucky point estimate.
- **…but loses on pooled** — 12.260 floor vs 11.779 ceiling. Significant.

So the trade-off is measured, not anecdotal: **augmented training buys pooled
EER and costs A19/A20, in both directions, significantly.** The open question
remains *why*, and the control arm is the clean test (see above).

In-the-Wild, bootstrapped on the same script (n = 31,779; 19,963 bonafide /
11,816 spoof): **clean 35.308 % [34.780, 35.830]**, **full_call 44.990 %
[44.390, 45.560]** — ±0.54 and ±0.59. Both intervals are far tighter than the
per-attack ones, because ITW is scored pooled over all 11,816 spoofs rather
than per ~900-utterance attack family. Precision is not the limiting problem
out of domain; the 35–45 % level is.

### Does the 0.770 pp floor survive this? For pooled yes, for per-attack no

Comparing two checkpoints means comparing two noisy estimates, so the
relevant quantity is √2 × (single-measurement half-width):

| Comparison | √2 × half-width | vs 0.770 pp floor |
|---|---|---|
| pooled (0.53) | **0.75 pp** | ≈ the floor — **corroborated** |
| A19 / A20 (1.36) | **1.92 pp** | 2.5× the floor — **floor too small** |

So Ground rule 1's 0.770 pp is independently corroborated by bootstrap for
**pooled** comparisons — it was measured as a seed-to-seed spread and lands
almost exactly on √2 × 0.53. It should be kept there. But the same number is
roughly 2.5× too small for a per-attack comparison, which is why docs/19's
per-attack bar was raised to "≥1.36 pp **and** non-overlapping intervals"
(non-overlap itself requires ≈1.92 pp of separation, matching the table).

Not patched unilaterally: Ground rule 1 still states a single number for all
metrics. Recommended change is a split floor (pooled 0.770 pp / per-attack
≈1.9 pp), recorded as an open item in docs/19.

**The gap this does not close.** The pre-augmentation model remains the only
checkpoint that clears 12 % on A19 and A20 (8.920 / 6.330), and it does so by
roughly half. Every model trained on augmented data regressed there. RawBoost
recovered part of it (~20 → 16.3 on A19) but not the missing half.

**Read the pre-aug comparison as suggestive, not causal.** `pre_aug_cm_aasist.pt`
and `cm_aasist.pt` are both layer-12 `aasist-l`, and both were scored on the
*same* 20k slice (docs/08 §"Side-by-Side Eval Comparison"), so the swing is
measured on comparable terms:

| docs/08 label | A19 | A20 | Pooled EER | minDCF |
|---|---|---|---|---|
| Pre-Augmentation L12 | **9.044 %** | **6.407 %** | 12.878 % | 0.8834 |
| Post-Augmentation L12 | 21.316 % | 23.189 % | 13.627 % | 0.8048 |

That is the same trade-off seen in G4: augmentation buys pooled separation
(minDCF 0.8834 → 0.8048) and costs A19/A20 (~12 and ~17 pp). But these are
two separate training runs, not one run with a flag flipped, so seed, epochs
and data volume are not controlled between them. **It is a correlation, not a
demonstration.** The control arm — identical run, `--g4-rawboost` omitted — is
the clean test of whether augmentation is the cause, and it is the reason that
arm was queued. Until it reports, do not state "augmentation caused this".

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

**Criteria:**

- [ ] An unauthenticated request to every route returns 401, covered by a test.
- [ ] `scripts/integration_check.py` extended to exercise the authenticated
      path, and still passing.
- [ ] `docs/PROTOCOL.md` and `docs/API_SPEC.md` document the scheme. Per
      `AGENTS.md` §2 PROTOCOL.md is binding — frontend and backend must both
      match it.

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

G5 depends on no CM work and can proceed alongside it.

---

## What not to do

- Do not run further layer-selection, merging, or pooling experiments. §12.7
  contains four of them; all four are inside the noise floor. The frozen
  encoder is the ceiling, and G3 is the only thing that lifts it.
- Do not report a difference smaller than 0.8 pp pooled EER as a result.
- Do not fit calibration on eval data.
- Do not change the criteria in this file after seeing numbers. Add a dated
  note explaining why a criterion was wrong and keep the original visible.
