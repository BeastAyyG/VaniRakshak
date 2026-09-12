# 11 — Handoff checklist

Exact locations for the work that remains. Read
`docs/10-quality-and-hardening-plan.md` first — its **Ground rules** section is
binding, and every criterion below was fixed in advance there.

Branch: `fix/protocol-interlock-reconciliation`. `origin` and `upstream` are the
same GitHub repo. `git pull --rebase` before you start and before you push;
never force-push.

---

## Status

| Goal | State | Where |
|---|---|---|
| G0 preserve checkpoints | done | `data/*.pt`, 6 tracked |
| G1 validate `--seed` | done | `docs/10` G1 |
| G2 calibration | done | `reports/cm_calibrated_benchmark.json` |
| G3 encoder fine-tune | **missed, closed** | `docs/10` G3 |
| G4 attack diversity | **do not run as written** | `docs/10`, "augmentation finding" |
| G5 authentication | done | `vanirakshak-backend/vanirakshak/server/auth.py` |
| G6 semantic vs zero-retention | **done 2026-09-13 — off by default, disclosed when on** | `docs/10` G6 |
| G7 ASV default | **done 2026-09-13 — labelled, not flipped** | `docs/10` G7 |

---

## The four gates — run all of them before every push

```bash
python -m pytest vanirakshak-backend/tests/
python -m pytest tests/e2e/
python scripts/integration_check.py
python scripts/check_docs.py
```

Current baseline: **71** backend tests, **95** e2e, **23/23** integration
checks, **0** hard doc failures. If any number drops, you broke something.

`scripts/check_docs.py` hard-fails on a doc that references a file path which
does not exist, and warns on a `VR_*` env var in `config.py` that no doc
mentions. Adding a setting means editing `docs/PROTOCOL.md` §8 in the same
commit.

---

## Loose ends — both closed 2026-09-13

1. ~~Commit `reports/cm_finetune_benchmark.json`.~~ Done in `8985855`.
   `data/cm_aasist_ft4_s42.pt` correctly stays untracked.
2. ~~Discard stray local `auth.py` / `test_auth.py`.~~ None present.

**Watch out for duplicate work across worktrees.** An earlier session left a
partial G7 (5 files, no tests or docs) *uncommitted* in the main worktree while
WorkBuddy, in its own worktree, reported that work "never persisted" and wrote
G7 from scratch. Both used the same design. The partial copy was stashed, not
discarded (`git stash list`: `partial-G7-superseded-by-894eead`), and the
complete one fast-forwarded in. Before starting a goal, run `git status` in
**every** worktree (`git worktree list`), not just your own.

---

## G6 — Semantic layer vs the zero-retention claim — DONE

Decided 2026-09-13: **off by default, disclosed when on.** `VR_SEMANTIC_ENABLED`
gates the layer in `vanirakshak-backend/vanirakshak/semantic/signal.py` before
the ASR client is imported; a `GROQ_API_KEY` alone does not enable it. Enabled,
`/v1/health` and every `dpdp_compliance` block report `zero_retention: false`
and name `groq`. Full record in `docs/10` G6.

Open and unowned: a caller **consent flow**. Disclosure informs the integrator;
it does not obtain consent.

---

## G7 — ASV ships as a random projection

`vanirakshak-backend/vanirakshak/detectors/pipeline.py:62` reads
`os.environ.get("VR_ASV_BACKEND", "stub")`. The stub
(`vanirakshak/detectors/asv.py:29`, `_stable_projection`) is a **fixed random
projection of hand-crafted spectral statistics** — not biometrics.

So by default every `asv_consistency` value the dashboard renders is synthetic,
and the impostor interlock that produces the demo BLOCK
(`VR_IMPOSTOR_BLOCK_BELOW`, `risk/fusion.py`) fires on fake similarity.

That is documented and defensible for a demo. What is not defensible is that it
happens by accident of a default rather than by decision. Pick one:

- Flip the default to `ecapa` and make `vanirakshak-backend/requirements-asv.txt`
  a hard dependency; or
- Make the dashboard state unmissably that the similarity shown is synthetic
  (`frontend/app/components/ForensicVerdictCard.tsx` already renders
  "not evaluated" when `asv_evaluated` is false — the same treatment applies).

**Criterion:** `GET /v1/health`'s `models.asv_backend` and the dashboard agree
with each other about what is actually running, and a reader cannot mistake a
random projection for a voiceprint.

**Resolved 2026-09-13 — the second option (label it).** The default was not
flipped: `requirements-asv.txt` (speechbrain + a checkpoint download) stays
optional. Instead `asv_synthetic` is now computed from the running backend and
carried on `/v1/health` (`models.asv_synthetic`), every WS verdict
(`metrics.asv_synthetic`) and `AnalyzeResponse`; the dashboard relabels the
"TARGET ANCHOR ASV" tile "(SYNTHETIC)" with a "Demo projection — NOT a
voiceprint" note. Full record and the code list in `docs/10` G7.

---

## If anyone asks for more countermeasure work

The answer is no, and the reason is recorded in `docs/10` under "The
augmentation finding". Six experiments — layer selection, attentive merging,
uniform merging, calibration, encoder fine-tuning — have not beaten
`data/cm_aasist.pt` on the metrics that decide cost, and that checkpoint is
already the backend default (`vanirakshak/detectors/cm_aasist.py:84`).

Two specific traps:

- **The noise floor is 0.8 pp pooled EER.** Measured, not guessed:
  `cm_aasist_attm_uniform.pt` (12.734 %) and `cm_aasist_meanpool.pt`
  (13.504 %) compute the *identical* function at inference and differ by
  0.770 pp purely from training-run variance. A difference under that is not a
  result.
- **G4's premise is backwards.** A19/A20 were never blind — pre-aug L12 scored
  them at 9.04 % and 6.41 %. The codec/telephony augmentation is what cost
  them (21.32 % and 23.19 % on the otherwise-identical post-aug L12). Adding
  RawBoost on top is the opposite of what the evidence indicates.

The one CM experiment the data does support, if someone insists: retrain with
the codec augmenter's probabilities reduced or off
(`ml/augment.py`, `CodecTelephonyAugmenter`) and see whether A19/A20 return.
Not scheduled, not a prerequisite for anything.

---

## Scoring harnesses — which to use

`scripts/cm_frontend.py` is the only place that knows how to build the right
WavLM frontend for a checkpoint. A head trained with `--unfreeze-top-n` scored
against a stock frontend produces **silently wrong** numbers.

- Fine-tuned checkpoints: `scripts/evaluate_finetune_benchmark.py <ckpt.pt>`
- Frozen checkpoints: the older `scripts/evaluate_*.py` harnesses, which now
  call `assert_frozen_checkpoint` and refuse a fine-tuned one rather than
  mis-scoring it.

Never edit an existing `reports/*.json`. They are the record. New runs get new
files.
