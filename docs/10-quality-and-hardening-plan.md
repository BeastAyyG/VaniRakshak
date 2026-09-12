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

- [ ] Calibrated CLLR **< 1.0** on the eval slice (1.0 is uninformative).
- [ ] Calibrated actDCF within **10 %** of minDCF — that gap *is* the
      calibration error, and closing it is the entire point.
- [ ] EER and minDCF **unchanged** to 3 decimal places. A monotonic transform
      cannot change a threshold-free metric. If they move, the implementation
      is wrong, not better.

**Bug to fix while you are in here.** `data/cm_aasist_attm_uniform.pt` carries
`eer_threshold` 10.698028928450064 and `dev_eer` 0.14286875159620163 — byte
-identical to `cm_aasist_attm.pt`. The uniformizing step copied the checkpoint
dict and zeroed `merge.weights` without recomputing dev metadata, so that
checkpoint's zero-centering offset belongs to a different model. It did not
affect the 6-way benchmark (that path scores head logits directly), but Gate 3
was run against this checkpoint with the wrong threshold. Recompute both values
on dev and rewrite them.

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
