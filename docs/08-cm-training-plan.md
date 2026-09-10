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

- **Primary: ASVspoof 5** [R4][R5] — crowdsourced speech from ~2 000
  speakers, 20+ spoofing attacks, and (a first for the series) seven
  adversarial attacks. Distributed via Zenodo; **read the README and LICENSE
  files bundled with the download before downloading and before publishing
  any number from it** — the licence, not this document, governs what you
  may do with the data and how it must be cited. Mirror also on Hugging
  Face [R8].
  Use the shipped train / dev / eval partitions untouched — no re-splitting,
  no speaker overlap across partitions. Follow the Phase-2 evaluation plan
  [R4] if you want numbers comparable to the literature; deviating is
  allowed but must then be stated next to every figure.
- **Fallback: ASVspoof 2019 LA** if ASVspoof 5 access is slow. Smaller and
  older vocoders, so expect optimistic numbers; re-run on ASVspoof 5 before
  any production claim.
- **Held-out generalization set: In-the-Wild** [R6] — ~20.8 h bonafide and
  ~17.2 h spoofed audio scraped from social media and video platforms
  (English-speaking public figures, real background noise). Never train or
  tune on it. It exists solely to answer "does this checkpoint survive
  contact with audio nobody curated?" — see §4.4.
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

- **Head: AASIST-L** (light variant, **85 k params** per the original paper
  [R1]): spectral + temporal graph-attention branches over the WavLM frame
  sequence, graph pooling, single genuine logit. Port from the official
  implementation [R2] and pin it by commit hash in the run log.
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

> **Known weakness of this recipe — read before running it.** §2 freezes the
> WavLM front-end. That is a deliberate cost/complexity trade (single-GPU
> job, production dependency identical to `VR_CM_BACKEND=wavlm`), but it is
> *not* the configuration that produced the field's headline results. Tak et
> al. [R3] fine-tuned an SSL front-end (wav2vec 2.0 / XLS-R) jointly with an
> AASIST back-end and reported ~90% relative improvement over their
> baseline, and state explicitly that fine-tuning the front-end was
> necessary — the SSL model is pre-trained on bonafide speech only, so a
> frozen version has never been asked to represent spoofing artefacts.
> Expect the frozen variant here to underperform published numbers. If §8
> acceptance fails, unfreezing the front-end (§9) is the first thing to try,
> not the last; budget a larger GPU for it.

## 4. Eval protocol (report all of it, not just the headline)

Conventions follow `docs/04-metrics-and-honest-evaluation.md`:

1. **minDCF** — the **primary** metric, matching the ASVspoof 5 evaluation
   plan's Track 1 (stand-alone bonafide-vs-spoof detection) [R4][R5]. Report
   it with the challenge's own cost parameters so the number is comparable
   to published work. This repo's earlier framing made EER primary; that was
   wrong relative to the challenge protocol and is corrected here.
2. **EER, pooled and per-attack** — secondary but non-negotiable for
   shipping: a head at 2% pooled EER that collapses on one attack family is
   not shippable, and pooled EER hides exactly that.
3. **Calibration: Cllr and actDCF** at the exported threshold vs minDCF.
   ASVspoof 5 Track 1 scores Cllr and actDCF alongside minDCF precisely to
   catch systems that discriminate well but are badly calibrated [R4]. If
   actDCF and minDCF diverge, the `eer_threshold` export in §5 is
   miscalibrated — fix, don't ship.
   (Track 2's a-DCF / t-EER metrics apply to spoofing-robust ASV, i.e. a
   CM+ASV tandem. This system does fuse CM with ASV, so a-DCF is the more
   honest end-to-end number *if* you evaluate the fused pipeline rather than
   the CM alone — worth doing, out of scope for this checkpoint's gate.)
4. **Cross-dataset generalization — the gate that matters most in
   production.** Evaluate the frozen checkpoint, unchanged, on a corpus it
   was not trained on: In-the-Wild [R6] is the standard choice. Published
   surveys report models at sub-1% EER on ASVspoof degrading to **30–60%
   EER** on in-the-wild audio [R6][R7] — i.e. from usable to worse than a
   coin flip on some systems. Record this number even when it is
   embarrassing; it is the honest predictor of behaviour on a real call,
   and a checkpoint that is excellent in-domain and useless out-of-domain
   must be documented as such rather than shipped on its in-domain figure.
5. **In-repo gates after integration (§6):** recalibrate with
   `scripts/fit_fusion_weights.py` under `VR_ASV_BACKEND=ecapa`
   `VR_CM_BACKEND=aasist`, then `scripts/integration_check.py` — its
   tier-match gates were calibrated for stub-era operating points and will
   need recalibration (same as happened under `ecapa`/`wavlm`); record the
   new gates, don't delete them.
6. **Demo-clip sanity:** genuine must outscore impostor on the shipped demo
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

## 11. References

Verified by web search on 2026-09-10. Where a claim in this document rests
on one of these, it is cited inline as `[Rn]`. Anything not cited here is
this repo's own reasoning or measurement, not an external result — the
distinction matters, because the whole point of this file is that we do not
yet have our own numbers.

| # | Reference | Used for |
|---|---|---|
| R1 | Jung, Heo, Tak, Shim, Chung, Lee, Yu, Evans — *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*, ICASSP 2022. [PDF](https://www.eurecom.fr/publication/6696/download/sec-publi-6696.pdf) · [Semantic Scholar](https://www.semanticscholar.org/paper/83ed100306b892da76a7468e6eca6f99074fbd02) | The architecture. AASIST-L = **85 k params**; the paper reports ~20% relative improvement over the then-SOTA on ASVspoof 2019 LA. |
| R2 | Official PyTorch implementation, NAVER/Clova — [github.com/clovaai/aasist](https://github.com/clovaai/aasist) | The reference code to port. Pin by commit hash (§3). |
| R3 | Tak, Todisco, Wang, Jung, Yamagishi, Evans — *Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2vec 2.0 and Data Augmentation*, Odyssey 2022. [ISCA](https://www.isca-archive.org/odyssey_2022/tak22_odyssey.pdf) · [arXiv:2202.12233](https://arxiv.org/abs/2202.12233) | The SSL-front-end + AASIST combination, and the finding that **fine-tuning the front-end was necessary** — the basis for §3's warning that our frozen recipe is the weaker configuration. |
| R4 | *ASVspoof 5 Evaluation Plan (Phase 2)* — [asvspoof.org](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf) | The protocol and metrics. **Track 1 primary metric = minDCF**, with Cllr and actDCF scored alongside to catch miscalibration. Track 2 (spoofing-robust ASV) uses min a-DCF, with min t-DCF / t-EER as supplements. |
| R5 | Wang, Yamagishi, Todisco, Delgado, Nautsch, Evans et al. — *ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale*, ASVspoof Workshop 2024. [arXiv:2408.08739](https://arxiv.org/pdf/2408.08739) · [journal version](https://www.sciencedirect.com/science/article/pii/S0885230825000506) | The corpus itself: ~2 000 crowdsourced speakers, 20+ attacks, 7 adversarial attacks. |
| R6 | Müller, Czempin, Diekmann, Froghyar, Böttinger — *Does Audio Deepfake Detection Generalize?* (the **In-the-Wild** dataset), Interspeech 2022. [arXiv:2203.16263](https://arxiv.org/abs/2203.16263) | The held-out generalization set (§1) and the generalization-gap figure in §4.4. |
| R7 | *Audio Anti-Spoofing Detection: A Survey* — [arXiv:2404.13914](https://arxiv.org/html/2404.13914v1); and *Robustness in deepfake speech detection: a survey of failure mechanisms* — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0885230826001002) | Corroboration for the sub-1% → 30–60% EER in-the-wild degradation, and the taxonomy of *why* it happens. |
| R8 | ASVspoof 5 on Hugging Face — [huggingface.co/datasets/jungjee/asvspoof5](https://huggingface.co/datasets/jungjee/asvspoof5) | Alternate download mirror. Licence still governs; check it there too. |

### What these references change about this plan

Three corrections landed from reading them, recorded rather than silently
folded in:

1. **AASIST-L is 85 k parameters**, not the ~0.3 M this document previously
   claimed (R1).
2. **minDCF is the primary metric, not EER** (R4). The earlier framing here
   inverted the challenge's own protocol.
3. **Freezing the front-end is the weaker recipe** (R3). Kept anyway for
   cost reasons, but now labelled as a known handicap with an explicit
   escalation path, instead of being presented as the natural choice.

And one thing they confirm that should temper expectations for everyone
reading this repo: a countermeasure that scores beautifully on ASVspoof can
still be near-useless on real-world audio (R6, R7). Landing this checkpoint
is a real step forward, but it is **not** the point at which VaniRakshak
becomes trustworthy against arbitrary real calls. See `docs/09-post-cm-roadmap.md`
for what remains after it.
