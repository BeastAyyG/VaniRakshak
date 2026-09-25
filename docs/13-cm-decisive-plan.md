# 13 — Decisive plan for the countermeasure (pre-registered 2026-09-15)

This file pre-registers every criterion below **before** any of the work runs.
It is committed to `main` first; nothing in it may be changed after the numbers
it gates are seen, except by a dated amendment that keeps the original visible
(`docs/10-quality-and-hardening-plan.md` ground rule 4).

How it was built: five agents verified the load-bearing facts against the code
and reports, three agents wrote competing plans (rigour-first,
deployability-first, cost-first), and three hostile critics (statistics,
operations, product) attacked all three. This is the synthesis. Every fatal
and major critique is addressed; the ledger is at the end.

---

## Amendment 1 (2026-09-15) — completeness critique, before anything ran

The automated synthesis and completeness-critic steps failed on a usage limit,
so the first version of this file was synthesised by hand and not
independently reviewed. A completeness critic then checked it against the code
and found 2 fatal and 15 major problems. Nothing gated by this plan has run
yet, so these corrections are legitimate under ground rule 4. **Where this
amendment conflicts with the text below, this amendment wins.** The original
text is kept unchanged below it.

**§0 was overstated.** The table assumed wideband audio with prosody, channel,
context and semantic terms all at zero. Recomputed from the fusion constants
(intercept `config.py:75`, overrides `config.py:136-139`):

| clone ASV | wideband, all other terms 0 | narrowband (channel 0.25, prosody 0.5) | every term at max |
|---|---|---|---|
| 1.00 | 47.5 CHALLENGE | 61.7 CHALLENGE | 91.7 BLOCK |
| 0.95 | 50.0 CHALLENGE | 64.0 CHALLENGE | 92.4 BLOCK |
| 0.80 | 57.4 CHALLENGE | 70.6 **BLOCK** | 94.3 BLOCK |

With a fully confident CM and nothing else, any clone with ASV ≤ 0.526 already
BLOCKs. The corrected claim: **a clone that matches the enrolled voice closely
(ASV ≳ 0.9) cannot be blocked by the CM alone on most channels**, and the
one-shot, text-graded challenge is then passable. Improving the CM helps
lower-ASV clones. The ordering stands — the decision layer first — but "close to
zero value" is withdrawn.

**Fatal 1 — Phase 1 could not be evaluated.** Every manifest has only
`path, label, attack_id` (no speaker column); there are no enrolment lists; and
the audio lives on `A:`, which the dev box does not have. **New 1.0
(prerequisite):** rebuild the eval and dev manifests with a speaker column from
the ASVspoof 5 protocol files on `A:`; build target-enrolment lists; define a
session as consecutive utterances of one speaker concatenated to ≥ 60 s. Phase 1
code is written on the dev box; **its evaluation runs on the training box**
(CPU or GPU scoring, no training). P1-S grid cells use **injected** ASV and
channel values; a real-ECAPA run is reported separately and does not gate.

**Fatal 2 — checkpoints would never leave the training box.** `*.pt` is
gitignored, so a bundle carries no weights. **New 0.1a:** copy every
`cm_scratch_*.pt` to `C:\vr-save\ckpt\` and to USB with a sha256 manifest,
before any tree is retired.

**Phase 0 safety.**
- **New 0.0:** before any git command, raw-copy both trees including `.git`
  (`robocopy <tree> C:\vr-save\raw\<name> /E /DCOPY:T /COPY:DAT`). `git status`
  fails in a tree whose worktree entry was deleted.
- 0.2 uses `git diff --binary HEAD --output=…` (plain `git diff` omits staged
  changes) and `git status --porcelain --untracked-files=all --ignored`.
- 0.3's mtime rule excludes a run only if the log also lacks
  `Training finished. Best epoch N`; a later tooling edit alone does not.
- **Fewer than four valid runs:** if either condition keeps fewer than two
  runs, C-2b cannot compute a seed SD and Phase 3 cannot run.
- The landing PR (0.5) **grades docs/10's four pre-registered from-scratch
  criteria exactly as written** before anything in this file is applied.
- The bundle leaves the box on **USB**, since sync tools are stopped in 0.1.

**Phase 1 corrections.**
- **1.1 clone rule** fires when the claimed identity is **enrolled** AND
  ASV ≥ `VR_IMPOSTOR_BLOCK_BELOW` AND CM LLR ≤ −k. That closes the gap for
  clones between ASV 0.53 and a high percentile. When the caller is not
  enrolled (ASV is the fusion-neutral 0.5), the rule does not apply; that case is
  documented, not hidden.
- **1.2 changes a recorded protocol decision.** `docs/PROTOCOL.md` §3 records
  the one-shot challenge as decided, so 1.2 needs a PROTOCOL amendment, a new
  challenge-response audio message, and matching `page.tsx` and `app.py`
  changes (`AGENTS.md` §2). It gets its own criterion **P1-C**: synthetic
  response audio fails the challenge in ≥ 90 % of trials. The critic
  also flagged that the verdict sends `expected_digits` and `expected_full` to
  the client (`challenge/engine.py:43-50`, `session.py:208`). Checked: this is
  not an independent hole, because the phrase itself already contains the
  digits the caller must say. The real weakness remains text-only grading;
  1.2's audio scoring is the fix. Stop sending the redundant fields anyway as
  part of the protocol change.
- **P1-F had a ceiling.** Genuine per-call challenge probability is already
  near 100 % for a 60 s call, so "rises by ≤ 5 pp" is empty. Replaced with an
  absolute cap: P(challenge within 60 s | genuine, measured ASV, wideband)
  ≤ 50 % after Phase 1, counting **every** challenge including re-arms. If the
  incumbent already exceeds this, reducing friction (1.3) is part of Phase 1,
  not optional.

**§6 statistics.**
- Pooling within {S42,S43} and {B42,B43} gives **2** degrees of freedom:
  margin = max(stated margin, **6.09** × SD) (= t(2, 0.975) × √2). Futility
  ceiling fixed now: if the margin for per-attack miss exceeds **5 pp**, the arm
  does not run.
- **No new hole** corrected to: miss_c ≤ miss_i + 2.0 pp **and**
  miss_c ≤ **min**(2 × miss_i, miss_i + 2.0 pp) for attacks where the incumbent
  is below 2 %, else the first clause alone. (The original max-form clause was
  implied by the first and did nothing.)
- **Sealed attacks, pinned now** (numpy 2.5.3): **A24, A26, A27, A31**. These
  per-attack numbers are already public in `docs/08`, so "sealed" means sealed
  from here on, not unseen. **Incumbent top-3 worst, pinned now:** A28, A20,
  A19.

**Phase 2 / C-2a.** `data/cm_aasist_layer5.pt` is a warm-started fine-tune, so
S42 + layer5 differs from the control in layer **and** lineage. C-2a is
relabelled a **screen**, must hold for **both** S42 + layer5 and S43 + layer5,
and uses the §6 margin. Drop `data/cm_aasist_attm.pt` from 2.4 (no criterion
used it). Scorer v2 must stream audio: the gate split is ~156 GB as float32.

**Definitions (§2.2 onward).** "Miss" = the session is not BLOCKed, evaluated
with Phase 1's final k, τ_asv and operating threshold pinned. The unsourced
friction bound (ASV 0.855, channel 0.242) is replaced by the grid cell
ASV 0.80 / channel 0.25. `VR_CM_NATURAL_OFFSET` must be 0 in the pinned
settings.

**STOP-B** no longer picks "the best Phase 2 single head" on the gate split:
select on the screen split, confirm on the gate split.

**Phase 3.** Uses the **S** augmentation. With `--patience 10` every run does 10
epochs: **~4.0 GPU-h per run, ~12 h for three seeds.** Per-epoch heads, a
`.done.json` marker, layer-aware names, and outputs to `C:\vr-out\` need trainer
and orchestrator changes, scheduled as Phase 2 dev work.

**Phase 4 needs code no phase built.** Scheduled as Phase 2 dev work: runtime
head load `strict=True`, checkpoint sha in the health label, a runtime path for
a fused head. The parity bar is set from the incumbent's own measured
CPU-fp32 vs CUDA-fp16 parity, not the guessed p99 ≤ 0.01. The telephony
35 % bar applies at the security bound.

**Budget.** 30 GPU-h and 6 weeks cap Phases 2–3 only (STOP-D); §8's table
total is corrected accordingly.

**docs/11 amendment.** `docs/11` says the training box does `git pull --rebase`;
under this plan it makes no git writes after Phase 0. `docs/11` is updated in a
follow-up PR.

---

## 0. The finding that reorders everything

**A perfect countermeasure cannot block a voice clone in the shipped product.**

Under the real-backend fusion (`vanirakshak-backend/vanirakshak/config.py:136-139`:
intercept -1.6, `w_cm` 1.5, `w_asv` 2.0) the CM term adds at most 1.5 logits
(`risk/fusion.py:40-53`). For a clone that matches the enrolled voice — which is
what a good clone *is* — risk tops out below the BLOCK line even when the CM is
certain the audio is synthetic:

| clone's ASV match | max risk with a CM that says "100 % fake" | tier |
|---|---|---|
| 1.00 | 47.5 | CHALLENGE |
| 0.95 | 50.0 | CHALLENGE |
| 0.90 | 52.5 | CHALLENGE |
| 0.80 | 57.4 | CHALLENGE |

The CHALLENGE that results is issued once per session and never re-armed
(`vanirakshak-backend/vanirakshak/session.py:167-171`), and it is graded on
client-supplied digit text and response latency only — no audio is scored
(`vanirakshak-backend/vanirakshak/challenge/engine.py:89-109`). A live
real-time voice-conversion attacker reads the digits and passes. After that,
only risk ≥ 70 or ASV < 0.10 can escalate, and neither fires for a good clone.

The fusion is also additive, so **a better clone (higher ASV) gets lower
risk** — the inverse of the actual clone signature, which is "matches the
voiceprint AND sounds synthetic".

**Consequence.** Every GPU hour spent improving the CM today buys close to zero
reduction in fraud risk against the product's real threat. The decision layer
is the bottleneck, so it goes first, and CM GPU work is gated on it.

Verified independently on the dev box on 2026-09-15 by recomputing the fusion
arithmetic from the constants above.

---

## 1. Standing amendments (dated 2026-09-15)

These rules in `docs/10` rest on the warm-start-contaminated experiments of
§12.7. They are amended here rather than silently violated. The originals stay
visible in `docs/10`.

- **A1. Ground rule 1's 0.8 pp floor** is a *fine-tuning* floor. For
  from-scratch runs the floor is whatever Phase 2 measures from the four
  existing from-scratch heads, used through the formulas in §6, never the old
  constant.
- **A2. "Do not run further layer-selection, merging or pooling experiments"**
  (`docs/10`, What not to do) is suspended for **from-scratch** runs only, and
  only through the conditional arms in Phase 3. Its premise — "the frozen
  encoder is the ceiling" — was drawn from lineage-contaminated checkpoints.
- **A3. G3's bars** (EER < 11.0 %, minDCF < 0.75) were defined for encoder
  fine-tuning. They are **not** applied to from-scratch frozen-encoder models;
  claiming a from-scratch model "clears G3" is post-hoc and is withdrawn.
- **A4. The "A28 < 13.0 %" bar** proposed on the training box after seeing the
  from-scratch numbers is withdrawn. Per-attack protection is handled by the
  no-new-hole rule in §6 instead.
- **A5. act_dcf and act_dcf_bayes** are not used as criteria: raw LLRs put
  their thresholds in the wrong place, so they track 20 × EER. Threshold-free
  metrics are reported; operating-point metrics gate.

---

## 2. Roles and how instructions travel

- **Dev box, Claude Code session (`D:/VaniRakshak`)** — the only GitHub writer.
  Opens and merges every PR, verifies every relayed number against committed
  artefacts, writes all code in Phases 1–2.
- **Training box** — GPU only. Receives code as a **git bundle of a pinned
  `main` commit**, cloned fresh and detached. After Phase 0 it makes **no git
  writes at all**. All outputs go to `C:\vr-out\<phase>\`, never into a clone.
- **WorkBuddy** — read, edit, run gates; no git writes (`docs/11`).
- **Transport.** `handoff/` is **not** synced between machines (verified
  2026-09-13). Instructions travel only as the chat prompt or a committed file.
  Artefacts travel through the Syncthing folder that *is* verified working
  (`.corpus-manifests/`) or a USB drive, and are always copied to a local,
  non-synced path and sha256-verified before use.
- **Interpreter.** Every training-box command uses the explicit venv:
  `$py = 'C:\VaniRakshak\.venv\Scripts\python.exe'`, preceded by a CUDA assert.

---

## Phase 0 — Preserve and land the from-scratch results (no GPU, ~1 day)

The results commit exists only on the training box, in a git directory that is
being damaged externally (`.git/worktrees` deleted, `refs/heads/exp/` deleted
repeatedly). An unreferenced commit there is exposed to `git gc` pruning.
**Protect first, diagnose second, transfer third.**

**0.1 Protect (training box, first commands, in this order):**

1. Stop every agent and sync tool that can reach `C:\VaniRakshak-main` or
   `C:\vr-results`. Record which ones were running.
2. In the repository that holds the objects: disable garbage collection
   (`git config gc.auto 0`, `gc.pruneExpire never`,
   `gc.worktreePruneExpire never`).
3. Confirm the object exists: `git cat-file -t f374a05`. If not,
   `git fsck --unreachable --no-reflogs` and look for it.
4. Pin it with a ref: `git update-ref refs/tags/xfer-f374a05 <full sha>`.
5. Bundle that ref immediately and copy the bundle **off the box** in the same
   block (`git bundle create ...xfer.bundle refs/tags/xfer-f374a05`).
6. Before concluding any ref is "deleted", check `.git\packed-refs` and
   `git show-ref` — `git pack-refs` moves loose refs there.

**0.2 Archive uncommitted work (training box):** in **both** trees,
`git status --porcelain --untracked-files=all`, `git diff --binary --output=<file>`
(git writes the file; PowerShell `>` re-encodes and corrupts non-ASCII), then
copy every untracked file explicitly — including the four new pause/resume
tooling files (experiment_state, experiment_progress, experiment_control and
their test), which `git diff` omits because they are untracked. sha256 everything.

**0.3 Inventory (training box):** locate every `cm_scratch_*.pt`, run log and
report JSON across both trees with size and LastWriteTime. For each run:
the log must contain `Training finished. Best epoch N` (written by
`ml/train_cm.py`), the checkpoint's `best_epoch` must equal N, and the stop
epoch must equal min(N + 5, 10). Any run whose checkpoint mtime is later than
the pause/resume tooling's mtime, or whose log lacks that line, is **partial**
and excluded.

**0.4 Environment facts (training box):** `pip freeze`, `nvidia-smi`,
transformers version, `import speechbrain`, free space on `A:` and `C:`,
`ffmpeg -encoders` (for `libopencore_amrnb`, `libvo_amrwbenc`).

**0.5 Land (dev box):** fetch the bundle, verify the sha matches the relay,
create `docs/cm-from-scratch-results` from it, merge with a **merge commit**
(not squash) so `f374a05` survives in `main`'s history as provenance. Before
merging, a verification script must reproduce every relayed number from the
committed JSON, assert the exact model set, and bind each checkpoint to its
report row by `dev_eer`, `dev_min_dcf`, `seed` and `augmentation`.

**0.6 Retire the damaged trees.** No linked worktrees on the training box ever
again. Fresh clone from bundles only.

**Stop condition P0.** If `f374a05` cannot be recovered, the from-scratch
results are declared lost. They are **not** re-run to recover them without a
new pre-registration.

---

## Phase 1 — Fix the decision layer (no GPU, dev box, ~3–5 days)

This is the phase that actually reduces fraud risk. It needs no training.

**1.1 Clone rule (deterministic override, like the existing impostor rule).**
If ASV consistency ≥ τ_asv **and** CM LLR ≤ −k, force BLOCK. This is the clone
signature. τ_asv and k are chosen **on dev only**, as percentiles of dev
bonafide scores (never on eval), stored as named settings with `VR_*` env vars
documented in `docs/PROTOCOL.md` §8.

**1.2 Challenge that checks the voice, not the typed digits.** The challenge
response audio is scored by CM and ASV; a synthetic or non-matching response
fails the challenge and BLOCKs. The challenge re-arms when risk re-enters
CHALLENGE after a pass.

**1.3 Dev-anchored operating threshold.** Add an optional checkpoint field
`operating_threshold`, set from a dev-bonafide percentile (e.g. 2nd or 5th),
which the runtime prefers over `eer_threshold`. The EER threshold puts
per-window genuine friction at or above the dev EER by construction (16.4 %
for the incumbent).

**1.4 Session simulator.** Friction is per call, not per 4 s window: one
low-CM window flips a whole session to CHALLENGE (hop 1 s, EMA α 0.65,
one-shot trigger). A torch-free simulator replays per-window scores through
`RiskFusionEngine` and the session latch and reports P(challenge within a
30 / 60 / 120 s genuine call).

**1.5 Pin the operating-point arithmetic.** A torch-free unit test in
`vanirakshak-backend/tests/` pins the fusion offsets over an ASV × channel grid
{0.80, 0.90, 0.95, 1.00} × {0 (wideband), 0.25 (narrowband)}, so a fusion refit
fails CI instead of silently invalidating every gate below.

**Pre-registered criteria for Phase 1 (security, on existing checkpoints):**

- **P1-S (security).** On ASVspoof 5 spoof-of-target trials replayed through
  the full pipeline with ECAPA and the incumbent CM: P(session ends in BLOCK |
  spoof) is reported before and after. The change passes only if it rises by
  ≥ 30 pp absolute at the **worst** cell of the ASV × channel grid.
- **P1-F (friction).** P(challenge within 60 s | genuine) rises by no more than
  5 pp absolute, measured with the session simulator on bonafide recordings of
  at least 8 s.
- **P1-T (tiers).** Real-backend `scripts/integration_check.py` passes on both
  dev-box CPU fp32 and training-box CUDA fp16, with no `ground_truth.json`
  re-baseline in the same PR.

If P1-S or P1-F fails, the decision layer is iterated (dev only) until both
pass, and **no CM GPU work starts**. If they cannot both pass within 14 calendar
days, go to STOP-A.

---

**Status 2026-09-25 (no criterion changed).** Phase 1 code is built:
1.1 clone rule, 1.2 voice-scored challenge (response windows scored by CM and
ASV, optional re-arm), 1.3 `operating_threshold`, 1.4 session simulator
(`vanirakshak/risk/simulate.py`, replays through the real `CallSession`), and 1.5
fusion pin (`tests/test_decision_layer.py`). A trigger-persistence setting
(`VR_CHALLENGE_TRIGGER_CONSECUTIVE`) was added because one-window triggering
alone cannot meet P1-F. Every threshold ships **off** and is chosen on dev by
`scripts/calibrate_decision_layer.py`, whose input
`scripts/score_checkpoint_eval.py --dump-scores` now writes. P1-S, P1-F and P1-T
are **not yet graded**; that is step 1 of `docs/14-master-plan.md` §6. The
challenge-response audio needs no new message type (the answer arrives on the
same stream), so 1.2's "new challenge-response audio message" was not needed.
The PROTOCOL §3 amendment is written. P1-C additionally needs server-side ASR
of the response (docs/14 W2): today `response_text` is client-supplied.

## Phase 2 — Measure what already exists (one GPU scoring session, ~4–8 GPU-h)

No training. Every question that can be answered from existing checkpoints is
answered here first.

**2.1 Scorer v2 (dev box code, tested, merged before use).** One WavLM forward
per batch feeds every frozen head. Writes per-utterance score dumps (row index +
manifest sha256, not path strings; `np.savez_compressed`). Supports
`merge_layers` checkpoints. Records checkpoint sha256, git commit and
effective `settings` (fusion, risk, audio) in the report, and **refuses** to
emit a gate verdict if those settings differ from the pinned values.
`--max-missing-frac 0`, `--require-cuda`. Refuses to overwrite a report.
Outputs to `C:\vr-out\p2\`.

**2.2 Held-out and sealed data, fixed now.**

- **Screen split:** the committed 20k slice (data rows 1–20,000 of
  `eval_downloaded_628k.tsv`).
- **Gate split:** data rows 20,001–628,848 (608,848 rows), never scored before.
  Asserted disjoint from the slice by path set, with its bonafide count checked.
- **Sealed attacks:** four of the 13 attacks outside {A19, A20, A28}, drawn by
  `numpy.random.default_rng(20260915).choice(sorted(others), 4, replace=False)`
  and recorded in the Phase 2 PR. No trigger, screen, ranking or fusion fit may
  use them. Worst-attack miss is reported separately for open and sealed
  attacks.

**2.3 Telephony, real codecs.** ffmpeg transcodes g711u, AMR-NB 12.2k and
AMR-WB 12.65k, deterministic and sha-pinned, built from gate-split rows,
**on `C:`** with a free-space check (abort under 50 GB free) and parallel
workers. `A:` is near full and must not receive new files.

**2.4 Heads scored.** Incumbent `data/cm_aasist.pt`; `data/cm_aasist_layer5.pt`;
`data/cm_aasist_attm.pt`; S42, S43, B42, B43 (from Phase 0).

**2.5 Score fusion, fit on dev only.** Logistic fusion of S42 (layer 12) with
layer5 (layer 5), and the control S42 + S43 (two layer-12 heads). Any
"layer diversity helps" claim must beat the S42 + S43 control.

**Metrics.** Gating: per-attack miss and bonafide FRR at the **deployed**
operating point (raw − τ_dev), at both ends of the §1.5 grid — the security
bound (ASV 1.0, channel 0) and the friction bound (ASV 0.855, channel 0.242).
A scale-invariant companion (miss at matched dev-bonafide FRR, threshold fixed
on dev) must agree in sign with any mechanism claim. Reported, not gating:
EER, minDCF, official ASVspoof 5 minDCF, CLLR, minCLLR.

**Pre-registered outcomes:**

- **C-2a (complementarity).** The fusion S42 + layer5 counts as complementary
  if, on the gate split, it improves A28 miss by ≥ 3 pp over S42 **and** keeps
  A19 and A20 within the no-new-hole rule (§6), **and** beats the S42 + S43
  control on the mean of the incumbent's three worst attacks.
- **C-2b (noise floor).** The from-scratch seed SD of each gating metric is
  computed from {S42, S43} and {B42, B43}, and the §6 margins are filled in
  mechanically from it in a dated commit before Phase 3.

---

## Phase 3 — Conditional training (only if C-2a is met)

If C-2a is **not** met, record the negative result and go to STOP-B. Training
does not run to "make the idea work" (ground rule 5).

If C-2a is met, one arm runs: **from-scratch layer 5**, seeds 42, 43, 44.

- Exactly the from-scratch recipe with `--layer-index 5`; one variable.
- `--patience 10` so every run completes all 10 epochs; each epoch's head and
  dev τ are saved. The **primary** analysis re-applies today's patience-5 rule
  after the fact; a **secondary** analysis uses epoch 10. Claimed effects must
  agree in sign across both.
- Checkpoint and log names include layer and seed; a `.done.json` marker is
  written only on completion; a checkpoint without it is quarantined, never
  skipped over.
- **Futility after seed 42 (fixed now):** if seed 42's A28 miss on the screen
  split is not at least 3 pp better than S42's, the arm stops.
- The shipped seed is chosen by a dev-only rule: median dev minDCF.
- Before the arm, a CPU determinism check (`torch.use_deterministic_algorithms(True)`)
  proves the current trainer reproduces e661b9f's tensors on a tiny manifest,
  so new-code runs are comparable with the old-code S/B runs.

**Budget.** ~3.6 GPU-h per run (8.7 expected epochs × 1,436 s + load) → ~11 h
for three seeds; with futility the floor is ~3.6 h.

---

## Phase 4 — Deployment gate and rollback

A candidate (single head or dev-fit fusion) ships only if **all** hold on the
**gate split**, with the Phase 1 decision layer in place:

1. **Security.** Worst open-attack miss at the security bound no worse than the
   incumbent, and mean of the incumbent's three worst attacks improves by at
   least the §6 margin.
2. **No new hole** on every attack, open and sealed (§6).
3. **Telephony.** On every codec, worst-attack miss ≤ 35 % and per-call
   friction within P1-F's bar.
4. **Sealed attacks.** Worst sealed-attack miss ≤ incumbent + 2 pp.
5. **Runtime.** Real-backend integration_check passes on CPU and CUDA; the
   `/v1/health` `cm_backend` label carries the checkpoint's sha256 prefix and
   matches the file; runtime-vs-scorer parity on ≥ 300 utterances
   (p99 |Δ score| ≤ 0.01).
6. **Loads strictly.** The runtime loads the head with `strict=True`.

**Deploy** is a new write-once file plus a PR that points
`VR_CM_AASIST_WEIGHTS` at it for the real-backend profile. It does **not**
edit `ground_truth.json`. Wording is scoped to what exists: the Docker image
runs stub detectors (`vanirakshak-backend/Dockerfile` installs no torch), so
this is "the real-backend demo default", not a production deployment, until a
real-backend container profile exists.

**Rollback**, written before any deploy: (1) immediate —
`VR_CM_AASIST_WEIGHTS=data/cm_aasist.pt`, confirmed by the health label;
(2) permanent — `git revert` of the deploy PR. Triggers: real-backend
integration_check failure on the merge commit, or a health label that does not
name the new sha.

---

## Phase 5 — The claim gate (human, legal, no GPU)

ASVspoof 5's 16 generators are not the product's threat. **No detection-rate
claim is made externally** until a small out-of-domain clone set is built and
scored:

- consented volunteer speakers only (written consent; this is a human task);
- at least three open-source systems, including one real-time voice conversion
  and one zero-shot TTS;
- each rendered clean and through the §2.3 codecs.

Criterion: no generator with miss > 50 % at the security bound. Until then
external wording is limited to "challenges suspected synthetic voices", with
known gaps listed.

---

## 6. Criteria formulas (fixed now; numbers filled mechanically after Phase 2)

- **Margin for any operating-point metric** =
  max(stated margin, 3.93 × SD), where SD is the pooled within-condition seed SD
  from {S42, S43} and {B42, B43} (3.93 = t(4, 0.975) × √2). If the margin is too
  large to be useful, the arm does not run — a pre-registered futility outcome.
- **No new hole (per attack):** miss_candidate ≤ miss_incumbent + 2.0 pp
  **and** miss_candidate ≤ max(2 × miss_incumbent, miss_incumbent + 2.0 pp).
- **Worst-case improvement** uses the mean of the incumbent's three worst
  attacks, not the single worst: fixing only A28 cannot move a max statistic
  more than the gap to the second-worst attack (A20, 23.19 % vs A28 26.22 %).
- **Screen vs gate.** The slice only screens (loose bar: worst + 4 pp); every
  gate row is confirmed on the gate split. One trial moves per-attack miss on
  the slice by ~0.105 pp and pooled bonafide miss by ~0.025 pp; reproduction
  tolerances are at least three trials.
- **Multiplicity.** One primary hypothesis per phase (Phase 1: P1-S; Phase 2:
  C-2a; Phase 3: the layer-5 arm's A28 effect). Everything else is secondary.

---

## 7. Stop conditions

- **STOP-A — decision layer blocked.** Phase 1 cannot meet P1-S and P1-F in 14
  days → pause all CM model work; effort goes to the challenge biometrics and a
  fusion refit on labelled data.
- **STOP-B — no complementarity.** C-2a not met → record it; the incumbent (or
  the best Phase 2 single head, if it passes Phase 4) stays; no CM training.
- **STOP-C — telephony non-viable.** Every candidate's worst narrowband miss is
  ≥ 50 % → declare the frozen WavLM + AASIST-L family non-viable for PSTN and
  forbid telephony coverage claims.
- **STOP-D — budget.** Hard cap 30 GPU-h and 6 calendar weeks for Phases 2–3.
- **STOP-E — integrity.** Any HEAD change, `git fsck` failure, or unexplained
  file mtime on the training box during a run → stop, preserve, report.

---

## 8. Budget and calendar

| phase | GPU-h | calendar |
|---|---|---|
| 0 preserve + land | 0 | 1–2 days |
| 1 decision layer | 0 | 3–5 days (+ up to 14 before STOP-A) |
| 2 score existing heads | 4–8 | 3–5 days incl. scorer v2 |
| 3 layer-5 arm (conditional) | 3.6–11 | 2–4 days |
| 4 deploy gate | ~1 | 1–2 days |
| 5 OOD clone set | 0–2 | human-paced |
| **total cap** | **30** | **6 weeks** |

Operational: before any run over two hours, the user disables sleep and
pauses Windows Update for the run window (a system setting the agent must not
change). Runs log a heartbeat and GPU temperature so a reboot is identifiable.

---

## 9. Critique ledger

| critique | severity | how this plan addresses it |
|---|---|---|
| CM capped at 1.5 logits; one-shot, text-only challenge; clone passes | fatal | §0; Phase 1 precedes all GPU work; STOP-A |
| ASV × CM × channel never measured jointly | major | P1-S full-pipeline trials; §1.5 grid |
| per-window friction ≠ per-call friction | major | 1.4 session simulator; P1-F |
| dev-EER threshold forces friction ≥ dev EER | major | 1.3 `operating_threshold` from dev bonafide |
| "deploy" only changes a demo default | major | Phase 4 wording scoped; container profile named as a gap |
| only 16 ASVspoof generators; no real clones | major | Phase 5 claim gate |
| eval attacks became design targets | major | 2.2 sealed attacks, seed fixed now |
| worst-attack max capped by A20 | major | §6 top-3 mean |
| security point was optimistic | major | gate at both grid ends, security bound ASV 1.0 / ch 0 |
| fused vs single head scale confound | major | scale-invariant companion analysis |
| early-stopping epoch confound | major | `--patience 10`, per-epoch heads, primary + secondary |
| margins vs unmeasured seed SD | major | §6 margin formula from measured SD |
| slice winner's curse | major | screen on slice, gate on 608,848 held-out rows |
| training "until it works" after a negative | major | Phase 3 runs only on positive C-2a; futility after seed 42 |
| new-code vs old-code controls | major | CPU determinism check before Phase 3 |
| demo margins set after seeing probe | major | tier sanity gate only; friction measured statistically |
| f374a05 exposed to gc; worktree damage | major | 0.1 gc off, ref, bundle off-box first |
| `git diff` misses untracked files; PS re-encoding | major | 0.2 `--output`, explicit copies, sha256 |
| completion check looked for a line not in the log | major | 0.3 uses `Training finished. Best epoch N` |
| partial/relaunched runs undetectable | major | 0.3 mtime vs tooling check; partial = excluded |
| scorer writes into a dirty tree | major | outputs to `C:\vr-out\` |
| telephony build fills `A:` | major | 2.3 builds on `C:` with space check |
| ECAPA on the CUDA venv unverified | major | 0.4 checks speechbrain; P1-T on both boxes |
| scoring straight from a live sync folder | major | copy local + sha256 before use (§2) |
| `.npz` gitignored; path-string dumps too large | minor | row index + manifest sha, `savez_compressed`, `git add -f` by name |
| TSV header off-by-one | minor | data-row indices + disjointness assertion |
| bare `python` may miss the venv | minor | explicit `$py` + CUDA assert |
| env vars leak into later gate runs | minor | real-backend checks in a child scope |
| health label can't tell near-equal heads apart | minor | sha256 prefix in the label |
| reproduction tolerance below one trial | minor | ≥ 3-trial tolerances, score-level parity |
| sleep / Windows Update reboots | minor | user disables for the run window; heartbeat |
| branch naming | minor | `docs/…`, `feat/…` prefixes only |
