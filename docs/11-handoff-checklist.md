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
| G6 semantic vs zero-retention | **open — needs a product decision** | below |
| G7 ASV default | **open** | below |

---

## The four gates — run all of them before every push

```bash
python -m pytest vanirakshak-backend/tests/
python -m pytest tests/e2e/
python scripts/integration_check.py
python scripts/check_docs.py
```

Current baseline: **60** backend tests, **95** e2e, **23/23** integration
checks, **0** hard doc failures. If any number drops, you broke something.

`scripts/check_docs.py` hard-fails on a doc that references a file path which
does not exist, and warns on a `VR_*` env var in `config.py` that no doc
mentions. Adding a setting means editing `docs/PROTOCOL.md` §8 in the same
commit.

---

## Loose ends (small, do these first)

**1. Commit the G3 benchmark artifact.** `reports/cm_finetune_benchmark.json`
exists only on the training box. Until it is committed, the 13.529 % row in
`docs/10` G3 rests on prose — the same gap the G2 uniform row had. Commit it
next to the other reports. Do **not** commit `data/cm_aasist_ft4_s42.pt`
(54.5 MB, reproducible from `--seed 42`, and `.gitignore:34` already covers it).

**2. Discard any local `auth.py` / `test_auth.py` on the training box.** G5
landed in commit `baa390a`; a local copy from an earlier session will conflict.
`git checkout -- .` or delete them, then pull.

---

## G6 — Semantic layer vs the zero-retention claim

**This is a decision, not a bug. Do not "fix" it by picking one yourself.**

`vanirakshak-backend/vanirakshak/semantic/signal.py:74` constructs
`ASRService(ASRConfig(provider="groq"))` and uploads the caller's 4.04 s audio
window to Groq, a third-party US cloud. Meanwhile the system advertises zero
retention and cites DPDP Act 2023 §8(7) — see `dpdp_compliance` in
`docs/API_SPEC.md` and `zero_retention` in `GET /v1/health`.

The code and the compliance claim currently contradict each other. Three
options, all defensible:

1. Disable the semantic layer by default (it already degrades gracefully to
   `available: false` when `GROQ_API_KEY` is unset — see
   `vanirakshak/session.py`'s `_SEMANTIC_IDLE`).
2. Keep it and state the third-party transfer explicitly in the compliance
   surface and the consent flow.
3. Replace Groq with local ASR.

Whichever is chosen, **the docs and the code must then agree.** Surface the
choice to the product owner; do not decide it in a commit.

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
