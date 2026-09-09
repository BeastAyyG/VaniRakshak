# 08 — Trained countermeasure plan (AASIST)

**Status: plan, not implementation.** No trained anti-spoofing weights exist
anywhere in this repo. This document is the executable recipe for producing
them the moment GPU access + a labeled corpus are available. Until a
checkpoint with a *measured* dev-set EER is loaded via `VR_CM_BACKEND=aasist`,
every claim in `docs/02-technical-strategy-and-architecture.md` about the CM
being heuristic-only remains the truth — see the honesty guardrails in §10.

## 0. Goal

Replace the hand-tuned spoof/genuine decision (DSP stub in
`vanirakshak-backend/vanirakshak/detectors/cm.py`, WavLM-feature heuristic in
`vanirakshak-backend/vanirakshak/detectors/cm_wavlm.py`) with a trained
AASIST-L classifier head on frozen WavLM features, behind the existing
`score(x, sr) -> float` LLR contract so fusion (`VR_FUSION_W_CM`,
`VR_FUSION_BIAS`), the `VR_IMPOSTOR_BLOCK_BELOW` override, and the WS verdict
shape in `docs/PROTOCOL.md` all keep working unchanged.

**Measured baseline it must beat:** on the demo corpus the WavLM heuristic
scored the impostor clip *higher* (more genuine) than the genuine clip
(RUNBOOK, "Real detector backends"). Any trained head that cannot separate
bonafide from spoof on every attack family in §6 is a failed run, not a
shippable checkpoint.

## 1. Corpus

- **Primary: ASVspoof 5** (~50 GB, free for research after the license form).
  Use the shipped train / dev / eval partitions untouched — no re-splitting,
  no speaker overlap across partitions.
- **Fallback: ASVspoof 2019 LA** if ASVspoof 5 access is slow. Smaller and
  older vocoders, so expect optimistic numbers; re-run on ASVspoof 5 before
  any production claim.
- **Location: outside the repo** (e.g. `~/cm-corpus/`). `AGENTS.md` keeps
  datasets out of git; the corpus never enters `data/`. Build a manifest TSV
  (`path<TAB>label`, label in `{bonafide, spoof}` + attack id) from the
  corpus protocol files — the training script reads the manifest, never a
  hardcoded file list.

## 2. Front-end contract (must match inference exactly)

Train / inference mismatch here silently voids the whole run. These values
mirror the constants in
`vanirakshak-backend/vanirakshak/detectors/cm_aasist.py`:

- Resample everything to **16 000 Hz mono int16→float32** first (same
  `resampleToStreamRate` semantics as the frontend; training data at other
  rates is resampled, never trusted).
- Slice/pad to **4.0 s windows** (64 000 samples) — the production analysis
  window is 4.04 s; 4.0 s is the model aperture, the extra 0.04 s is pipeline
  framing, not model input.
- Features: **frozen microsoft/wavlm-base-plus** (Hugging Face model id), last hidden state
  (768-dim), no fine-tuning. Freezing is deliberate: it keeps training to a
  single-GPU job and keeps the production dependency identical to what
  `VR_CM_BACKEND=wavlm` already loads.
- Normalize per-utterance (mean/zero, unit variance) before the head.

## 3. Model + training recipe

- **Head: AASIST-L** (light variant, ~0.3 M params): spectral + temporal
  graph-attention branches over the WavLM frame sequence, graph pooling,
  single genuine logit. Standard public implementations port directly; pin
  whichever you use by commit hash in the run log.
- **Loss:** weighted cross-entropy (spoof weight = bonafide/spoof train
  ratio). OC-Softmax is an accepted substitute — record the choice, don't
  mix results across choices.
- **Optimizer:** AdamW, lr 1e-4, cosine decay, weight decay 1e-4, batch 32
  (gradient-accumulate to 32 if VRAM forces smaller micro-batches), fp16
  mixed precision, ~100 epochs, **early-stop on dev EER** (patience 10).
- **Augmentation (on the fly, probability each):** codec sims from the
  repo's own `vanirakshak-backend/vanirakshak/data/augment.py` (G.711 a/μ-law,
  AMR-WB simulation) — this is why that module exists, see
  `docs/03-codec-augmentation-and-telephony.md` — plus MUSAN-style noise at
  5–20 dB SNR and telephony bandpass (300–3400 Hz) on a fraction of batches.
  Never augment the dev/eval partitions.
- **Hardware:** 1× 24 GB card (RTX 4090) is enough with a frozen frontend;
  40 GB (A100) is comfortable. Budget **12–48 h** wall-clock incl. one
  restart for a bad seed. Log every run (git hash of this repo + training
  repo, seed, loss curves, dev EER per epoch).

## 4. Eval protocol (report all of it, not just the headline)

Conventions follow `docs/04-metrics-and-honest-evaluation.md`:

1. **EER on the eval partition** — primary metric. Also report per-attack
   EER: a head at 2% pooled EER that collapses on one attack family is not
   shippable.
2. **minDCF** (ASVspoof-5 cost parameters) — secondary; measures operating
   usefulness, not just separability.
3. **Calibration:** actDCF at the exported threshold vs minDCF. If they
   diverge, the `eer_threshold` export in §5 is miscalibrated — fix, don't
   ship.
4. **In-repo gates after integration (§6):** recalibrate with
   `scripts/fit_fusion_weights.py` under `VR_ASV_BACKEND=ecapa`
   `VR_CM_BACKEND=aasist`, then `scripts/integration_check.py` — its
   tier-match gates were calibrated for stub-era operating points and will
   need recalibration (same as happened under `ecapa`/`wavlm`); record the
   new gates, don't delete them.
5. **Demo-clip sanity:** genuine must outscore impostor on the shipped demo
   set — the minimum bar the heuristic failed.

## 5. Checkpoint export (schema is contractual)

The training run must write exactly what
`vanirakshak-backend/vanirakshak/detectors/cm_aasist.py::_validate_checkpoint`
accepts — a torch.save dict with keys `architecture` (`aasist-l`),
`frontend` (microsoft/wavlm-base-plus), `sample_rate` (16000), `window_s`
(4.0), `state_dict`, `eer_threshold` (dev-set EER operating point on the
genuine logit — `score()` centers on this, so export it from the *dev*
partition, never eval), `dev_eer`, `dev_min_dcf` (measured numbers, surfaced
in the health report so any operator can see what the weights proved).

Name the file data/cm_aasist.pt in prose only here: `*.pt` is gitignored
(`.gitignore`), so it can never be committed — transfer it to the inference
host out of band and point `VR_CM_AASIST_WEIGHTS` at it (or drop it at the
default location). Keep the run log + eval report next to it; weights
without the report are unverifiable and will not be merged into any claim.

## 6. Integration (the day the weights land)

1. Place the checkpoint, set `VR_CM_BACKEND=aasist` (and
   `VR_CM_AASIST_WEIGHTS` if off-default).
2. Health endpoint must report the `aasist` CM name with the measured
   `dev_eer` — if it reports the stub, the weights didn't load; verdicts are
   unchanged by design, so this step is silent-failure-proof only if you
   check the label.
3. Recalibrate fusion (`scripts/fit_fusion_weights.py`), update the
   integration-check tier gates, run the full battery (pytest,
   integration_check, check_docs).
4. Update docs to the measured numbers: RUNBOOK real-backend section,
   `docs/02-technical-strategy-and-architecture.md` CM row, and this file's
   status line. The "heuristic-only" statements get retired *then*, with the
   EER attached — not before.

## 7. Budget

| Item | Estimate |
|---|---|
| Corpus access + download (~50 GB) | 1–2 days wall-clock |
| GPU (1× A100 40 GB cloud) | 12–48 h compute, ~$50–150 |
| Eval + calibration + report | 1 day |
| In-repo integration + recalibration (§6) | 1–2 days |
| **Total** | **~1–2 weeks, ~$50–150** |

## 8. Acceptance criteria

- [ ] Eval EER + minDCF + per-attack breakdown published in the run report.
- [ ] actDCF ≈ minDCF (threshold calibrated).
- [ ] Beats the heuristic baseline on every attack family and on demo clips.
- [ ] Checkpoint loads via `VR_CM_BACKEND=aasist`; health shows measured EER.
- [ ] Fusion recalibrated; full test battery green; docs updated to numbers.

## 9. If training fails

A trained head that doesn't beat the heuristic is a result, not a waste:
record the EER, keep `VR_CM_BACKEND=wavlm` as the production setting, and do
not lower the bar (no shipping dev-tuned thresholds, no eval-peeking). The
next step would be unfreezing the frontend or more attack-diverse data —
a new run, not a reinterpretation of this one.

## 10. Honesty guardrails

- Never cite spoof-detection accuracy for any backend until §8 is checked
  off with a named checkpoint + report.
- `cm_aasist.py` reports `aasist_missing_weights` and the factory falls back
  to the stub while weights are absent — `VR_CM_BACKEND=aasist` without a
  checkpoint changes nothing but a log line. That is intentional.
- This file is a plan. Its existence improves nothing by itself.
