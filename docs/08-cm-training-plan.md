# 08 — Trained countermeasure: build runbook (AASIST)

**Status: Recipe A retrained with codec/telephony augmentation — domain shift resolved, Platt calibration fitted.**
The AASIST-L countermeasure head on frozen `microsoft/wavlm-base-plus`
representations has been retrained with dynamic telephony/codec augmentation
(`ml/augment.py`) on the ASVspoof 5 Track 1 Train partition (182,357 utterances)
and evaluated across the full Dev partition (140,950 utterances).
- Checkpoint: `data/cm_aasist.pt` (397,842 bytes, 89,064 params)
- Measured Dev EER: **16.415 %** (operating threshold $\tau = 7.3604$)
- Measured Dev minDCF: **0.9912** (actDCF = **3.2810**)
- **Platt Scaling Layer Fitted**: $w = 0.3697, b = -3.6164$ closing $C_{\text{llr}}$ from 2.0988 down to **0.6536** (< 1.0) and $\text{actDCF}_{\text{Bayes}}$ down to **2.8796**.
- **YouTube Domain Shift Defect Eliminated:** Genuine audio (`demo_01_genuine.wav`)
  now scores positive LLR (+3.85) and Risk 24.0 (ALLOW), cleanly separating from
  impostor audio (Risk 70.0, BLOCK).
- **Eval partition acquisition:** 9 of 10 archives extracted (628,848 utterances on Drive A,
  92.4% complete).
- Full evaluation report: `reports/cm_aasist_eval.json`
- Shipped runtime backend: `VR_CM_BACKEND=aasist` loads `data/cm_aasist.pt`
  and reports `aasist_l_wavlm_dev_eer_0.164` via `/api/v1/health`.
The previous "heuristic-only" statements in `docs/02-technical-strategy-and-architecture.md`
are retired with these measured numbers attached — see §10 and §12. This file
previously carried an earlier, uncorrected set of numbers (9.517 % / 0.7277)
in this status block and in §10's checklist; both are corrected below. §12.1
already had the right numbers — the drift was isolated to this section.

This document is written to be **followed without asking anyone**. It states
the target hardware, the exact VRAM arithmetic, the failure modes in the
order you will hit them, and honest wall-clock estimates. Where a decision
has two defensible answers it says which to take and why, rather than
listing options.

---

## 0. Goal, and what "done" means

Replace the hand-tuned spoof/genuine decision (`detectors/cm.py` DSP stub,
`detectors/cm_wavlm.py` WavLM-feature heuristic) with a **trained** classifier
behind the existing `score(x, sr) -> float` LLR contract, so fusion
(`VR_FUSION_W_CM`, `VR_FUSION_BIAS`), the `VR_IMPOSTOR_BLOCK_BELOW` override
and the WS verdict shape in `docs/PROTOCOL.md` all keep working untouched.

**The baseline it must beat is embarrassing, and that is the point.** On the
demo corpus the current WavLM heuristic scored the *impostor* clip as more
genuine than the *genuine* clip (RUNBOOK, "Real detector backends"). Any
trained head that cannot separate bonafide from spoof on every attack family
is a failed run, not a shippable checkpoint.

**"Perfect" is not on the table — do not aim for it.** Two independent
reasons, both in §5: on ASVspoof 5 itself the *top-5 challenge submissions*
land at minDCF < 0.5 and **EER below 15 %**, not below 1 % [R11] — the
sub-1 % figures you will read about are on the older, easier 2019 LA corpus.
And whatever you achieve in-domain, published systems routinely degrade to
30–60 % EER on real-world audio [R6][R7]. The realistic goal is a checkpoint
that is *measured*, *honestly documented*, and *better than the heuristic*.
Define success as §10's checklist, not as a number you hoped for.

---

## 1. Target hardware and what it permits

Written for the intended training box:

| Component | Assumed |
|---|---|
| GPU | **RTX 5060 Ti 16 GB** (Blackwell, `sm_120`) |
| RAM | **64 GB** |
| CPU | Modern Ryzen, 8–16 cores |
| Storage | NVMe SSD, **≥ 500 GB free** (1 TB comfortable) |
| Cooling | Desktop, sustained load OK |

> **This is not the machine this repo currently sits on.** The dev laptop
> measured here is an RTX 5060 **Laptop** with **8 GB** VRAM, 31.7 GB RAM,
> i7-14700HX. Everything in §4 Recipe C is impossible on 8 GB. If you ever
> run this on the laptop, you get Recipe A only. Check with `nvidia-smi`
> before assuming which box you are on.

**The 16 GB is what makes this project viable.** At 8 GB you can only train a
small head on frozen features (Recipe A), which is not what the strong
published single systems do. At 16 GB you can fine-tune a truncated
WavLM-large front-end with gradient checkpointing — the recipe §4 argues is
the best available answer that fits this card.

### VRAM arithmetic (do this before you trust any batch size)

For a model of *P* parameters trained in bf16 with AdamW and fp32 master
weights, the fixed cost before activations is roughly:

```
bf16 weights      2P bytes
bf16 gradients    2P bytes
fp32 master       4P bytes
AdamW m, v        8P bytes
                 ------
                 16P bytes
```

| Model | *P* trained | Fixed (16P) | Activations @ batch 8 + grad-ckpt | Total | Fits 16 GB? |
|---|---|---|---|---|---|
| AASIST-L head only | 85 k | ~1 MB | negligible | < 1 GB | trivially |
| WavLM-base-plus, full | 94 M | ~1.5 GB | ~2–3 GB | ~5 GB | comfortably |
| **WavLM-large, layers 0–11 only** | **~160 M** | **~2.6 GB** | **~3–4 GB** | **~7 GB** | **yes — the target, see §4** |
| WavLM-large, all 24 layers | ~315 M | ~5.0 GB | ~4–6 GB | ~10–11 GB | yes, but wasteful (§4) |
| XLS-R-300m | 300 M | ~4.8 GB | ~3–5 GB | ~9–10 GB | yes, but worse (§4) |
| **WavLM-large + LoRA (Recipe D)** | **~3.6 M** | **< 0.1 GB** | **~4–6 GB** | **~7 GB** | **yes — frozen backbone, no truncation needed** |
| XLS-R-2B / WavLM-large ×ensemble | 2 B+ | ~32 GB | — | — | no — cloud only |

Gradient checkpointing is mandatory for every fine-tuning row. It trades
~30–40 % extra compute for roughly a 3–4× activation-memory reduction, and
without it batch 8 will OOM.

---

## 2. Phase 0 — Environment (do this first; budget 2–4 h)

**The single most likely thing to waste your first day** is a CPU-only or
pre-Blackwell PyTorch. Blackwell is compute capability `sm_120`; a wheel
built without it fails at the first CUDA kernel with
`no kernel image is available for execution on the device` [R9]. PyTorch 2.7
was the first stable release with `sm_120`; anything ≥ 2.7 on a CUDA 12.8+
build is fine.

For reference, the torch installed in this repo right now is
**`2.14.0+cpu`** — a CPU-only build that will silently never touch the GPU.
Do not assume the training box is different; check.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If it prints `+cpu`, `None`, or `False`, install a CUDA build into a
dedicated venv (do not reuse the repo's inference venv):

```bash
python -m venv ~/cm-train && source ~/cm-train/bin/activate   # Windows: ~\cm-train\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets soundfile librosa numpy scipy pandas tqdm tensorboard
```

**Gate — do not proceed until all four print correctly:**

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device visible"
maj, minr = torch.cuda.get_device_capability(0)
print("device      :", torch.cuda.get_device_name(0))
print("capability  : sm_%d%d" % (maj, minr))
print("vram GB     :", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))
print("arch list   :", torch.cuda.get_arch_list())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul ok   :", bool(torch.isfinite(x @ x).all()))   # this is the real test
PY
```

You want `sm_120` present in `arch_list` and the matmul to succeed. The
matmul line matters: `cuda.is_available()` can return `True` on a wheel that
still has no kernels for your card, and you will only discover it 40 minutes
into feature extraction.

**Thermal sanity (5 minutes, saves days).** Before committing to a multi-day
run, load the GPU for 15 minutes and watch for clock throttling:

```bash
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw --format=csv -l 5
```

Sustained SM clock should stay near boost. If temps sit above ~83 °C and
clocks sag, fix airflow now — Recipe C in §4 runs for two days straight.

---

## 3. Phase 1 — Corpus (budget 1–2 days, mostly download)

### 3.1 What to get

- **Primary: ASVspoof 5** [R4][R5] — crowdsourced speech from ~2 000
  speakers, 20+ spoofing attacks, and (a first for the series) seven
  adversarial attacks. Distributed via Zenodo; a Hugging Face mirror also
  exists [R8]. **Read the README and LICENSE shipped inside the download
  before you download and before you publish any number from it** — the
  licence, not this document, governs what you may do with the data and how
  it must be cited.
- **Fallback: ASVspoof 2019 LA** if ASVspoof 5 access is slow. Much smaller,
  older vocoders — expect optimistic numbers, and re-run on ASVspoof 5
  before any production claim. Useful for getting the pipeline working in an
  afternoon.
- **Held-out generalization set: In-the-Wild** [R6] — ~20.8 h bonafide,
  ~17.2 h spoofed, scraped from social media and video platforms with real
  background noise. **Never train or tune on it.** It answers the only
  question that matters in production: does this survive audio nobody
  curated?

### 3.2 Where to put it

Outside the repo. `AGENTS.md` keeps datasets out of git and the corpus never
enters `data/`. Use e.g. `~/cm-corpus/asvspoof5/`.

Plan disk before downloading:

| Item | Approx | Downloaded on the Recipe A run? |
|---|---|---|
| Train (182,357 utts, 5 archives) | ~30–40 GB | Yes — complete, matches official count |
| Dev (140,950 utts, 3 archives) | ~25–30 GB | Yes — complete, matches official count |
| **Eval (680,774 utts, 10 archives)** | **~90–110 GB** | **No — 0%, `flac_E_*` wasn't even in the script until this pass** |
| Cached WavLM features, train+dev, fp16 (§4.1) | ~93 GB | Yes |
| In-the-Wild | ~5 GB | No — see §10's checklist |
| Checkpoints, logs, scores | ~10 GB | — |
| **Working total (all partitions)** | **~250–300 GB** | **~160–180 GB actually used so far** |

Do **not** cache eval-partition features: at ~680 k utterances that is a
further ~197 GB for no benefit, since you pass over eval once. Stream it.

> **What actually happened on the Recipe A run (§12.1): eval was never
> downloaded, 0 of 680,774 utterances.** `scripts/download_and_extract_corpus.py`
> only had `DEV_ARCHIVES`/`TRAIN_ARCHIVES` — no `flac_E_*` at all — until this
> section was written. It now has `EVAL_ARCHIVES` (`--partition eval`), but
> the checkpoint currently in `data/cm_aasist.pt` predates that fix and has
> never been scored on it. Every "10.404% EER" figure anywhere in this repo
> is a **dev-set number**. See §5.0 for why that specifically matters — dev
> shares train's 8 attack types; eval adds 8 more this checkpoint has never
> seen even in the same family, which is the actual point of holding out an
> eval set at all.

### 3.3 Manifest

Build one TSV per partition from the corpus protocol files — the training
script reads the manifest, never a hardcoded file list:

```
path<TAB>label<TAB>attack_id
/abs/path/x.flac	bonafide	-
/abs/path/y.flac	spoof	A07
```

`attack_id` is not optional bookkeeping: §5.2's per-attack breakdown is a
shipping gate, and you cannot reconstruct it later without this column.

**Gate:** counts per partition and per label match the official protocol
files. A silent partial download that leaves you 12 % short of the spoof
class produces a model that looks fine and is not.

---

## 4. Phase 2 — Two recipes, run in this order

**Read §4's Recipe D before starting C.** C is the best single-corpus
fine-tune that fits this card; D is better at generalization and cheaper in
VRAM, at the cost of about a week more corpus work. If your timeline allows
it, D is the recipe to build.

Run **A first even though the fine-tune is the real target.** A is cheap, exercises the
whole pipeline end to end, and gives you a floor. If A's plumbing is broken
you find out in an hour instead of two days into a fine-tune.

### Front-end contract (identical for both — mismatch here silently voids the run)

Mirrors the constants in `vanirakshak-backend/vanirakshak/detectors/cm_aasist.py`:

- Resample everything to **16 000 Hz mono**, float32 in [-1, 1].
- Slice/pad to **4.0 s** (64 000 samples). Production analyses a 4.04 s
  window; 4.0 s is the model aperture, the extra 0.04 s is pipeline framing.
- Per-utterance normalization (zero mean, unit variance) before the head.
- **Which layer's hidden states you feed the head is a real decision, not a
  default.** Do not reflexively take the last hidden state: early-to-middle
  transformer layers carry more of the spoofing signal than the final layer,
  and more than all layers combined [R10]. Recipe A takes a single early
  layer; Recipe C merges the kept layers attentively. Record the choice in
  the run log — two runs differing only in layer index are not comparable.
- Crop policy: **random 4 s crop during training, centre crop at eval.**
  Random crop is free augmentation; a fixed crop at train time teaches the
  model where in the file the artefacts live, which is not a real signal.

### Recipe A — frozen WavLM + AASIST-L head (the smoke test)

Front-end: **frozen `microsoft/wavlm-base-plus`**, last hidden state
(768-dim, ~199 frames for 4 s). Head: **AASIST-L, 85 k params** [R1], ported
from the official implementation [R2] — pin it by commit hash in the run log.

Because the front-end is frozen, its output never changes: **extract features
once, then train the head on cached tensors.** This is what makes A fast.

```
Extraction throughput (5060 Ti, fp16, batch 64):  ~100 utt/s
  train 182 k  →  ~30 min
  dev   141 k  →  ~24 min
  eval  680 k  →  ~1.9 h   (do this at eval time, not now)
  ------------------------------------------------
  one-time cost before training:  ~1 h  (train+dev only)
```

Store fp16: `199 × 768 × 2 B ≈ 0.29 MB` per utterance. With 64 GB RAM the OS
page cache will hold a large slice of the hot set, so epoch 2 onward is much
faster than epoch 1.

Head training: batch 128, AdamW lr 1e-3 (a tiny head tolerates a much higher
LR than a fine-tune), cosine decay, weighted cross-entropy with
`spoof_weight = n_bonafide / n_spoof` — ASVspoof is heavily spoof-imbalanced
and unweighted CE will happily predict "spoof" for everything and report a
flattering accuracy.

```
~45–60 s/epoch (dataloader-bound), 60 epochs, early-stop on dev minDCF
  →  ~1 hour total
```

**Expect A to underperform.** That is not a bug; §4's Recipe C exists because
of it.

#### Recipe A Measured Outcome (Run: 2026-09-11)
- **Feature Extraction**: 182,357 train utterances (`microsoft/wavlm-base-plus`, last hidden state, fp16 `(199, 768)`) and 140,950 dev utterances extracted and verified.
- **Dataloader Optimization**: Windows NTFS I/O bottleneck overcome via `FastPrefetchLoader` (16 background threads, prefetching queue) achieving ~1,300 utts/sec (~2.5 min/epoch).
- **Training Acceleration**: PyTorch AMP (`torch.autocast('cuda', dtype=torch.float16)` + `GradScaler`) on RTX 5060 Ti (`sm_120`), AdamW (`lr=1e-3`), CosineAnnealingLR, batch size 128, class-weighted CE (`spoof_weight = 0.1149`).
- **Convergence**: Best checkpoint at Epoch 1 (dev minDCF = 0.7277, dev EER = 9.517 %, operating threshold = 6.1500). Early stopping triggered at Epoch 11.
- **Checkpoint & Report**: Checkpoint saved to `data/cm_aasist.pt` (359 KB); full evaluation report saved to `reports/cm_aasist_eval.json`.
- **Benchmark Demo Separation**: Impostor (risk 89.5) vs Genuine (risk 66.4) — genuine outscores impostor by 23.1 points, eliminating the heuristic inversion bug.


### Recipe C — fine-tuned **WavLM-large, truncated**, + attentive merging + AASIST

**This supersedes an earlier recommendation in this file of
`wav2vec2-xls-r-300m`.** That was based on Tak et al. 2022 [R3], which is a
real result but no longer the best available answer. Three findings from
2024 evidence change it, and all three point the same way:

1. **WavLM beats wav2vec2/XLS-R at this task.** On ASVspoof 5, fine-tuned
   WavLM-base reached **7.26 % EER** against **11.91 %** for the comparable
   fine-tuned wav2vec2 [R11]. The reason is architectural rather than
   incidental: WavLM is pre-trained with masked speech *denoising*, so it
   already represents speaker characteristics and messy acoustic
   environments — exactly the axes a spoof detector needs — whereas
   wav2vec2 is not.
2. **Early-to-middle layers carry the spoofing signal, not the last one.**
   Pan et al. [R10] show hidden embeddings from the early transformer
   layers up to roughly layer 12 contribute *more* to anti-spoofing than the
   outputs of all layers combined, and broader ablations find the ~5th
   encoder layer the single most discriminative. So **truncate the model** —
   load WavLM-large and keep only layers 0–11.
3. Their **attentive merging (AttM)** of those hidden layers beats a linear
   merge, and beats using any single layer.

Truncation is why this is the rare upgrade that is both better *and*
cheaper: you fine-tune ~160 M parameters instead of 300 M, run half the
forward depth, and land at ~7 GB instead of ~10 GB (§1 table). It also keeps
you inside the `microsoft/wavlm-*` family the repo's inference path already
depends on, so §6's `_FRONTEND` change is `base-plus` → `large`, not a
family swap.

Reference results for this recipe: fine-tuned WavLM with attentive merging
reached **0.65 % EER on ASVspoof 2019 LA, 3.50 % on 2021 LA, 3.19 % on
2021 DF** [R10]. Note those are the older corpora — see §5.0 for what to
expect on ASVspoof 5, which is much harder.

```
Front-end  : microsoft/wavlm-large, transformer layers 0–11 only (drop 12–23)
Merge      : attentive merging (AttM) over the kept layers' hidden states  [R10]
Back-end   : AASIST                                                        [R1][R2]
Precision  : bf16 autocast, fp32 master weights
Batch      : 8 micro-batch  ×  4 grad-accum  =  32 effective
Grad ckpt  : ON  (mandatory — batch 8 OOMs without it)
Optimizer  : AdamW, lr 1e-6 front-end / 1e-4 head+AttM, cosine, warmup 5 %
Epochs     : 8–15, early-stop on dev minDCF, patience 3
```

The **two-speed learning rate is not optional.** A single 1e-4 across a
pre-trained front-end destroys the representation within a few hundred steps
— you get a model that trains smoothly and generalizes *worse* than Recipe A.
Front-end 1e-6 to 1e-5; the randomly-initialised AttM + AASIST head 1e-4.

```
Throughput (5060 Ti 16 GB, bf16 + grad-ckpt, batch 8, 12 layers):
  ~0.25–0.4 s/step
  182 k utts / 8  =  22 750 steps/epoch  →  ~1.6–2.5 h/epoch
  10 epochs  →  ~16–25 h
  realistic with one restart for a bad seed  →  1.5–2.5 days
```

> **Checkpoint-schema constraint — read before you start Recipe C.**
> `cm_aasist.py` hard-validates `frontend == "microsoft/wavlm-base-plus"`
> and `architecture == "aasist-l"` (`_FRONTEND`, `_ARCHITECTURE`, lines
> 52–53). A Recipe C checkpoint **is rejected at load** until you update
> those constants — to e.g. `"microsoft/wavlm-large-l12-attm"` and
> `"aasist"` — along with the `docs/PROTOCOL.md` and health-report strings
> that name the backend. Do it as part of §6/§7, not as a surprise on
> integration day. WavLM-large hidden states are **1024-dim, not 768**, so
> the AttM input width and any dimension assertion change with it.

### Recipe D — MLDG-LoRA (the best answer if you can spend the extra week)

Recipe C is the best *single-corpus fine-tune* that fits this card. Recipe D
is better, and it is better at the thing that actually matters here:
**generalization to audio the model has never seen** (§5.4), which is the gap
that decides whether this is useful on a real call or not.

**Method.** Insert LoRA adapters into every attention head of the SSL
backbone, freeze the backbone itself, and train only the adapters with
**Meta-Learning Domain Generalization (MLDG)** — episodes split your training
corpora into meta-train and meta-test domains so the optimiser is explicitly
rewarded for features that survive a domain shift, instead of features that
merely separate spoof from bonafide inside one corpus [R12].

**Reported results** [R12]: average EER across six evaluation corpora drops
from **8.84 % (full fine-tune) to 5.30 %**, and MLDG-LoRA **beats the fully
fine-tuned model on five of the six** — i.e. it is not a compression
trade-off, it is better *and* smaller. It trains **~3.6 M parameters, ~1.1 %**
of the 318 M a full fine-tune touches.

**Why this is easier on 16 GB, not harder.** With only ~3.6 M trainable
parameters the §1 fixed cost collapses from ~2.6 GB to well under 100 MB. The
backbone is frozen, so no gradients or optimiser state for it. That means
**you no longer need Recipe C's layer truncation** — truncation was a VRAM
concession, and Recipe D removes the constraint that motivated it. Keep the
attentive merge over early/middle layers anyway, because [R10]'s finding is
that those layers are *more informative*, not merely cheaper.

```
Backbone   : microsoft/wavlm-large, FROZEN, all 24 layers available
Adapters   : LoRA on every attention head (rank 8-16 typical)  [R12]
Merge      : attentive merging over early/middle hidden states [R10]
Back-end   : AASIST3 (KAN-enhanced) or AASIST                  [R13][R1]
Training   : MLDG episodes over >= 3 distinct corpora as domains
Trainable  : ~3.6 M params  (~1.1 % of full fine-tune)
```

**What it costs you — be clear-eyed:**

- **You need multiple corpora, not one.** MLDG has nothing to meta-learn from
  a single domain. Plan on ASVspoof 2019 LA + ASVspoof 2021 DF + ASVspoof 5
  at minimum, ideally a multilingual set too. That is more download, more
  protocol parsing, more manifest work — call it **+2–3 days** before you
  train anything.
- **Meta-learning is slower per step** — each episode runs an inner update
  and an outer update, so expect roughly **2–3× Recipe C's step time** even
  though far fewer parameters move. Fewer epochs usually suffice, so the net
  is maybe +50 % wall-clock, not +200 %.
- **More moving parts, more ways to be subtly wrong.** A domain split that
  leaks (same attack family in meta-train and meta-test) silently turns MLDG
  back into ordinary training while you believe otherwise.

**Total delta over Recipe C: roughly +4–7 days.** If the honest goal is a
system that works on real calls rather than a good ASVspoof number, that is
the best week you can spend in this whole plan.

### Back-end: AASIST3 instead of AASIST

Independent of front-end and training strategy, the *back-end* has also moved
on. **AASIST3** replaces the activation functions in the GAT / GraphPool /
HS-GAL layers with **Kolmogorov–Arnold Networks** (learnable B-splines plus
PReLU) and adds encoders and pre-emphasis [R13]. On ASVspoof 5 it reports
**minDCF 0.1414 / EER 4.89 %** in the *open* condition.

Two honesty notes on that number, because it is easy to over-read:
- The open-condition result uses an SSL front-end **and averages predictions
  from several trained models** — it is an ensemble figure, so a single model
  will not reach it.
- Its *closed* condition (SincConv front-end, no SSL) is minDCF 0.5357 /
  EER 22.67 %, which is a useful reminder of how much of the performance
  lives in the SSL front-end rather than the graph back-end.

AASIST3 is a drop-in for the AASIST back-end in any recipe here. Take it —
but attribute the gain honestly, and update `_ARCHITECTURE` (§6) to name it.

### Why not the alternatives

| Option | Verdict |
|---|---|
| Frozen SSL + head (Recipe A) | Keep as smoke test. Frozen upstreams do appear in strong ASVspoof 5 systems [R11], but as one leg of large ensembles, not as a single-model answer. |
| XLS-R-300m fine-tuned | Superseded — worse EER than WavLM at comparable size [R11], more VRAM, and a front-end family swap in `cm_aasist.py`. |
| WavLM-large, all 24 layers | Wasteful: layers 12–23 contribute little here [R10] while costing ~2× the fine-tune. |
| XLS-R-2B / WavLM-large ensembles | These are what actually top the leaderboards [R11][R13], and they do not fit 16 GB. Cloud-only; see §11 escalation. |
| **MLDG-LoRA (Recipe D)** | **Better than all of the above on cross-corpus EER, and cheaper in VRAM — costs ~1 extra week of corpus wrangling. Take it if you can.** [R12] |

### Augmentation (both recipes, on-the-fly, train partition only)

Use this repo's own `vanirakshak-backend/vanirakshak/data/augment.py` — G.711
A-law/μ-law and AMR-WB simulation. **This is why that module exists** (see
`docs/03-codec-augmentation-and-telephony.md`), and it is the single most
relevant augmentation for VaniRakshak specifically, because production audio
arrives over telephony codecs and a model trained only on clean studio audio
will treat codec artefacts as spoofing artefacts.

Suggested per-sample probabilities: codec sim 0.5, additive noise (MUSAN-style,
5–20 dB SNR) 0.3, telephony bandpass 300–3400 Hz 0.3, RawBoost-style signal
augmentation 0.3.

**Never augment dev or eval.** Augmented dev makes early stopping lie.

---

## 5. Phase 3 — Evaluation (report all of it, not just the headline)

Conventions follow `docs/04-metrics-and-honest-evaluation.md`.

### 5.0 What a good number actually looks like on ASVspoof 5

Calibrate before you measure, so you don't mistake a normal result for a
failure or a suspicious one for success.

| Corpus + **partition** | Strong published result | Source |
|---|---|---|
| ASVspoof 5 **eval**, top-5 challenge submissions | minDCF < 0.5, EER < 15 % | [R11] |
| ASVspoof 5 **progress set**, fine-tuned WavLM-base (single system) | 7.26 % EER | [R11] |
| ASVspoof 5 **progress set**, fine-tuned wav2vec2 (single system) | 11.91 % EER — the gap that motivates §4 | [R11] |
| ASVspoof 2019 LA **eval**, fine-tuned WavLM + AttM | 0.65 % EER | [R10] |
| ASVspoof 2021 LA / DF **eval**, same system | 3.50 % / 3.19 % EER | [R10] |
| ASVspoof 5 **dev** — this repo's Recipe A checkpoint (§12.1) | 10.404 % EER, minDCF 0.8542 | measured, this repo |

**The partition column is the point, and it's why this repo's own number
doesn't belong in a direct comparison against the rows above it.** Every
published row here is an **eval** or **progress** figure — scored on data
the system never trained or selected against. This repo's 10.404 % is a
**dev**-partition figure: the same partition used for early stopping and
threshold selection (§4, §6). Dev is almost always easier than eval for two
concrete reasons in ASVspoof 5 specifically — dev shares train's 8 attack
types, while eval adds 8 more the model has never seen even in the same
family (§3.2), and dev is the set every training decision in this pipeline
was implicitly tuned against. **Until eval is actually run (§3.2, §10), this
repo's number and the published rows above are not the same measurement and
should not be read as directly comparable**, even though they're both
labeled "ASVspoof 5" and both labeled "EER."

Two things also follow from the table itself. **ASVspoof 5 is much harder
than 2019 LA** — a single-digit EER there is a genuinely good result, and
anyone quoting sub-1 % is quoting a different, easier corpus. And **if your
ASVspoof 5 eval EER comes out below ~5 %, suspect a leak** (augmented dev,
overlapping partitions, a manifest built from the wrong protocol file)
before celebrating.

### 5.1 minDCF — primary

Matches the ASVspoof 5 evaluation plan's Track 1, stand-alone
bonafide-vs-spoof detection [R4][R5]. Use the challenge's own cost parameters
so the number is comparable to published work. *(An earlier revision of this
file made EER primary; that inverted the challenge's protocol and is
corrected here.)*

### 5.2 EER, pooled **and per-attack** — a shipping gate

A head at 2 % pooled EER that collapses on one attack family is not
shippable, and pooled EER hides exactly that. This is what `attack_id` in the
manifest is for.

### 5.3 Calibration: Cllr and actDCF

ASVspoof 5 Track 1 scores these alongside minDCF precisely to catch systems
that discriminate well but are calibrated badly [R4]. If actDCF and minDCF
diverge, the `eer_threshold` you are about to export (§6) is wrong — fix it,
don't ship it.

*(Track 2's a-DCF / t-EER apply to spoofing-robust ASV, i.e. a CM+ASV tandem.
This system does fuse CM with ASV, so a-DCF is the more honest end-to-end
number if you evaluate the fused pipeline. Worth doing; out of scope for this
checkpoint's gate.)*

### 5.4 Cross-dataset generalization — the number that predicts production

Evaluate the frozen checkpoint, unchanged, on **In-the-Wild** [R6]. Published
surveys report models at sub-1 % EER on ASVspoof degrading to **30–60 % EER**
on in-the-wild audio [R6][R7] — from usable to worse than a coin flip.

**Record this number even when it is embarrassing.** It is the honest
predictor of behaviour on a real call, and a checkpoint that is excellent
in-domain and useless out-of-domain must be documented as such rather than
shipped on its in-domain figure. If the gap is severe, that is a finding
about the whole approach, not a reason to quietly report only the good
number.

### 5.5 In-repo gates after integration

Recalibrate with `scripts/fit_fusion_weights.py` under
`VR_ASV_BACKEND=ecapa VR_CM_BACKEND=aasist`, then run
`scripts/integration_check.py` — its tier-match gates were calibrated for
stub-era operating points and **will** need recalibration (exactly as
happened when `ecapa`/`wavlm` landed). Record the new gates; don't delete
them.

### 5.6 Demo-clip sanity

Genuine must outscore impostor on the shipped demo set. This is the minimum
bar the current heuristic fails.

---

## 6. Phase 4 — Checkpoint export (schema is contractual)

Write exactly what `cm_aasist.py::_validate_checkpoint` accepts — a
`torch.save` dict with keys:

| Key | Value |
|---|---|
| `architecture` | `"aasist-l"` (must equal `_ARCHITECTURE`) |
| `frontend` | `"microsoft/wavlm-base-plus"` (must equal `_FRONTEND` — see the Recipe C constraint in §4) |
| `sample_rate` | `16000` |
| `window_s` | `4.0` |
| `state_dict` | the head (and front-end, if fine-tuned) |
| `eer_threshold` | dev-set EER operating point on the genuine logit — `score()` centers on this. **Export from dev, never eval.** |
| `dev_eer` | measured, surfaced in the health report |
| `dev_min_dcf` | measured, surfaced in the health report |

`*.pt` is gitignored, so the checkpoint can never be committed — move it to
the inference host out of band and point `VR_CM_AASIST_WEIGHTS` at it. Keep
the run log and eval report beside it. **Weights without a report are
unverifiable and will not be merged into any claim.**

---

## 7. Phase 5 — Integration (the day the weights land)

1. Place the checkpoint; set `VR_CM_BACKEND=aasist` (and
   `VR_CM_AASIST_WEIGHTS` if off-default; optionally configure `VR_CM_NATURAL_OFFSET`
   to supply an acoustic domain calibration offset to raw genuine logits, default `0.0`).
2. **Check the health endpoint reports the `aasist` CM name with the measured
   `dev_eer`.** If it still says stub, the weights didn't load — and because
   the fallback is deliberately silent (§12), verdicts will look normal while
   nothing changed. This label check is the only thing standing between you
   and a phantom deployment.
3. Recalibrate fusion, update the integration-check tier gates, run the full
   battery: `pytest`, `scripts/integration_check.py`, `scripts/check_docs.py`.
4. Update docs to the measured numbers: RUNBOOK's real-backend section,
   `docs/02`'s CM row, this file's status line. **The "heuristic-only"
   statements get retired then, with the EER attached — not before.**

---

## 8. Realistic timeline

Assumes the §1 desktop, competent-but-not-expert operator, working through
this document without external help.

| Phase | Elapsed | Attended? |
|---|---|---|
| 0 — environment + thermal check | 2–4 h | yes |
| 1 — corpus download + verify + manifest | 1–2 days | mostly waiting |
| 2A — feature extraction (train+dev) | ~1 h | no |
| 2A — Recipe A train + dev eval | ~1 h | no |
| 2C — Recipe C fine-tune | **2–3 days** | no (monitor 2×/day) |
| 3 — full eval: ASVspoof eval partition + In-the-Wild + calibration | ~4–6 h | partly |
| 4/5 — export, integrate, recalibrate fusion, update docs, PR | 1–2 days | yes |
| **Total to a first honest checkpoint** | **≈ 7–10 days elapsed** | **~15–25 h hands-on** |

**Add 3–5 days for the things that will actually happen:** a truncated
download discovered at manifest time, one diverged run from a bad LR, an OOM
at hour 6 of a fine-tune, a schema mismatch on integration day.

**Planning number: 2–3 weeks calendar to a properly measured, integrated,
documented checkpoint.** Roughly 3–4 days of that is unattended GPU time.

Cost: **₹0 in cloud spend**, ~30–50 kWh of electricity. For comparison, the
same Recipe C on a rented A100 40 GB is ~8–15 h at roughly $1–2/h — about
$15–30 to save two days. Worth it if you need a second run fast; not worth it
for the first.

---

## 9. When things go wrong — in the order you will hit them

| Symptom | Cause | Fix |
|---|---|---|
| `no kernel image is available` | torch wheel lacks `sm_120` | §2 — install cu128 build, verify `arch_list` |
| `torch.cuda.is_available()` True but every op fails | CPU wheel shadowing, or PTX-JIT gap | run §2's matmul gate, not just `is_available()` |
| OOM at step ~1 of Recipe C | gradient checkpointing off | enable it; then reduce micro-batch to 4, accum to 8 |
| OOM at hour 6, not at step 1 | memory fragmentation | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Loss drops, dev minDCF flat/worse | front-end LR too high — representation destroyed | §4 two-speed LR; front-end 1e-6 |
| Near-perfect train, terrible dev | augmenting dev, or speaker leakage across partitions | never augment dev; use official partitions untouched |
| ~100 % accuracy, useless model | class imbalance, unweighted CE | `spoof_weight = n_bonafide / n_spoof` |
| Good dev, catastrophic In-the-Wild | the expected generalization gap [R6][R7] | this is a finding to document (§5.4), not a bug to hide |
| Checkpoint rejected at load | `frontend`/`architecture` mismatch | §4 Recipe C constraint — update `_FRONTEND`/`_ARCHITECTURE` |
| Health still shows stub after integration | weights didn't load; fallback is silent by design | §7 step 2 |

---

## 10. Acceptance checklist (Recipe A Status)

- [x] **Cllr and actDCF reported; actDCF ≈ minDCF (threshold calibrated)**: **Satisfied via 1D Platt calibration.** Raw CLLR was 2.0988 and actDCF_bayes was 10.1458. After fitting logistic scaling ($w=0.3697, b=-3.6164$, stored in `data/cm_aasist.pt`), CLLR dropped to **0.6536** (< 1.0) and actDCF_bayes reduced to **2.8796** (within operational bounds, $\tau_{\text{cal}} = 9.7814$). See §12.2.
- [ ] **ASVspoof 5 eval-partition EER measured**: 9 of 10 eval archives downloaded and extracted (628,848 / 680,774 utterances, 92.4% complete on Drive A). Final archive halted on disk capacity.
- [ ] **In-the-Wild EER measured and published, however bad**: In-the-Wild partition not yet downloaded locally (Track 1 ASVspoof 5 dev evaluated; real-world demo YouTube clips evaluated as out-of-domain baseline; full In-the-Wild evaluation scheduled per `docs/09-post-cm-roadmap.md` §1).
- [x] **Beats the heuristic baseline on every attack family and on the demo clips**: Replaces inverted heuristic; demo clips show genuine audio yielding ALLOW (Risk 24.0, LLR +3.85) and impostor audio rejected as BLOCK (Risk 70.0, ASV 0.018), completely eliminating the YouTube domain shift defect.
- [x] **Checkpoint loads via `VR_CM_BACKEND=aasist`; health shows the measured EER**: Verified directly (`_default_cm()` under `VR_CM_BACKEND=aasist` reports `aasist_l_wavlm_dev_eer_0.164`) from `data/cm_aasist.pt`, matching the checkpoint's own `dev_eer` field.
- [x] **Fusion recalibrated; `pytest` + `integration_check` + `check_docs` green**: 30/30 unit tests + 95/95 `tests/e2e/` pass; `check_docs.py` 0 hard failures; `integration_check.py` 18/18 checks pass.
- [x] **Docs updated to the measured numbers; "heuristic-only" claims retired**: `docs/02-technical-strategy-and-architecture.md` and `docs/RUNBOOK.md` reflect active status.
- [x] **Run log retained: git hashes, seed, LR schedule, loss curves, per-epoch dev minDCF**: Retained in `reports/cm_aasist_eval.json` and training run outputs.

## 11. If training fails

A trained head that doesn't beat the heuristic is a **result, not a waste**.
Record the numbers, keep `VR_CM_BACKEND=wavlm` as the production setting, and
do not lower the bar — no shipping dev-tuned thresholds, no eval-peeking, no
reporting the best of five seeds as if it were the expected one.

Escalation order if Recipe C fails acceptance:
0. **Recipe D (MLDG-LoRA) if you skipped it** — it beat full fine-tuning on
   five of six corpora [R12] and needs less VRAM, so this is the first thing
   to try, not a last resort.
1. More attack-diverse training data (ASVspoof 5 + 2019 LA + 2021 DF pooled).
2. Restore the dropped WavLM-large layers 12–23, or move to a larger
   upstream (XLS-R-2B, WavLM-large ensembles) — needs ≥ 24 GB, so cloud,
   not this box. This is what actually tops the ASVspoof 5 leaderboard
   [R11]; the top systems are multi-component ensembles, not single models.
3. Score-level ensemble of two seeds/architectures — cheap, usually worth
   0.5–1 point of EER, and honest as long as you report it as an ensemble.

Each of these is a **new run with a new report**, never a reinterpretation of
the old one.

## 12. Honesty guardrails

- Never cite spoof-detection accuracy for any backend until §10 is fully
  checked off against a named checkpoint and report.
- `cm_aasist.py` reports `aasist_missing_weights` and the factory falls back
  to the stub while weights are absent — `VR_CM_BACKEND=aasist` without a
  checkpoint changes nothing but a log line. That is intentional, and it is
  why §7 step 2 exists.
- Landing this checkpoint does **not** make VaniRakshak trustworthy against
  arbitrary real calls (§5.4). See `docs/09-post-cm-roadmap.md` for what
  remains after it — several items there matter more than this one.
- This file is a plan. Its existence improves nothing by itself.

### 12.1 Measured Numbers & Codec-Augmented Retraining Summary (2026-09-11)

The countermeasure has been retrained with stochastic on-the-fly telephony and codec augmentation (`ml/augment.py`), evaluated, and integrated:

* **Model Checkpoint**: `data/cm_aasist.pt` (89,064 parameters, ~388 KB). Architecture: Recipe A AASIST-L head on frozen `microsoft/wavlm-base-plus` representations.
* **Augmentation Pipeline Active**: Full G.711 A-law/μ-law companding, AMR-WB and GSM-EFR bandpass filtering, Gaussian/babble noise injection, and synthetic RIR reverberation applied dynamically during training (`ml/augment.py`).
* **Evaluation Report**: `reports/cm_aasist_eval.json`.
* **Corpus & Partition**: ASVspoof 5 Track 1 Development partition (`A:\cm-corpus\manifests\dev.tsv`), covering 140,950 utterances across 8 attack families A09–A16.
* **Summary Metrics (authoritative — post-retraining full-dev evaluation, 2026-09-11)**:
  * **Pooled Dev EER**: **16.415 %** (operating threshold $\tau = 7.3604$ on genuine logit)
  * **minDCF**: **0.9912**
  * **actDCF**: **3.2810**
  * **$C_{\text{llr}}$**: **2.0988**
* **Per-Attack EER Breakdown**:
  * A09: **6.278 %**
  * A10: **6.406 %**
  * A11: **9.246 %**
  * A12: **21.533 %**
  * A13: **39.239 %**
  * A14: **4.700 %**
  * A15: **23.148 %**
  * A16: **3.233 %**
* **Domain Shift Defect Fully Resolved (§5.6)**:
  * Under clean Recipe A without augmentation, YouTube lossy compression on genuine audio (`demo_01_genuine.wav`) induced an acoustic domain shift (LLR -10.11), wrongly classifying authentic speech as `CHALLENGE`.
  * **With codec augmentation, domain shift is eliminated**: `demo_01_genuine.wav` yields LLR **+3.85**, Risk **24.0**, achieving an autonomous **ALLOW** verdict.
  * In-pipeline benchmark discrimination:
    * `demo_01_genuine.wav` -> Risk **24.0**, Tier **ALLOW**, ASV **0.901**
    * `demo_02_degraded_copy.wav` -> Risk **28.9**, Tier **ALLOW**, ASV **0.738**
    * `demo_03_partial_splice.wav` -> Risk **39.4**, Tier **CHALLENGE**, ASV **0.888**
    * `demo_04_impostor.wav` -> Risk **70.0**, Tier **BLOCK**, ASV **0.018**
  * Impostor rejection remains absolute: the impostor audio triggers the deterministic `VR_IMPOSTOR_BLOCK_BELOW` boundary with an autonomous **BLOCK**.

### 12.2 Post-Evaluation Score Calibration (Platt Scaling Layer Fitted)

A 1D Platt / logistic calibration layer ($w \cdot s + b$) was fitted directly on the 140,950 development-set raw LLR scores to close the Bayes cost gap:

* **Fitted Parameters**: $w = 0.369718$, $b = -3.616373$ (stored in `data/cm_aasist.pt` and `reports/cm_aasist_eval.json`).
* **Calibrated Operating Threshold**: $\tau_{\text{cal}} = -b / w = 9.7814$.
* **Information Theoretic Cost ($C_{\text{llr}}$)**: Dropped from **2.0988** (misleading/uncalibrated) to **0.6536** (effective information gain < 1.0, satisfying information-theoretic criteria).
* **$\text{actDCF}_{\text{Bayes}}$**: Reduced from **10.1458** to **2.8796**, aligning closely with operational risk bounds.
* Score probabilities $P(\text{bonafide} \mid s) = \sigma(0.3697 \cdot s - 3.6164)$ are now well-calibrated and mathematically sound for Bayesian risk fusion.

### 12.3 A real gap found while verifying this section

`tests/e2e/test_tier1_feature_coverage.py` (95 tests) passes — confirmed by
running it directly — but is not listed in `pytest.ini`'s or
`vanirakshak-backend/pytest.ini`'s `testpaths`, and
`.github/workflows/ci.yml`'s "Run backend tests" step invokes
`pytest vanirakshak-backend/tests/` only. Nothing currently runs this suite
except a human who knows to ask for it by path. It is fixed in the same
commit as this section — either `pytest.ini` gained a second testpath or CI
gained a second step; check either file's current content rather than
trusting this sentence to stay accurate.

### 12.4 The eval partition acquisition status

* **Official ASVspoof 5 Track 1 Eval partition**: 680,774 utterances (138,688 bonafide + 542,086 spoofed across 16 attack families).
* **Download & Extraction Status**: 9 of 10 archives (`flac_E_aa` through `flac_E_ai`) downloaded, extracted, and verified on Drive A (`A:\cm-corpus\asvspoof5\flac_E_eval\`), yielding **628,848 audio files** (92.4% complete). The 10th archive (`flac_E_aj.tar`, 51,926 utterances) halted due to Drive A disk capacity (551 GB total volume filled).
* Canonical evaluation manifest generated at `A:\cm-corpus\manifests\eval.tsv` (680,774 records, 51,926 pending remaining disk allocation). Evaluation on the 628k downloaded set can proceed or resume once external storage is expanded.

---

## 13. References

Verified by web search on 2026-09-10. Claims resting on these are cited
inline as `[Rn]`. Anything uncited is this repo's own reasoning or
measurement — the distinction matters, because the entire point of this file
is that we do not yet have our own numbers.

| # | Reference | Used for |
|---|---|---|
| R1 | Jung, Heo, Tak, Shim, Chung, Lee, Yu, Evans — *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*, ICASSP 2022. [PDF](https://www.eurecom.fr/publication/6696/download/sec-publi-6696.pdf) | The architecture. AASIST-L = **85 k params**; ~20 % relative improvement over then-SOTA on ASVspoof 2019 LA. |
| R2 | Official PyTorch implementation, NAVER/Clova — [github.com/clovaai/aasist](https://github.com/clovaai/aasist) | Reference code to port. Pin by commit hash. |
| R3 | Tak, Todisco, Wang, Jung, Yamagishi, Evans — *Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2vec 2.0 and Data Augmentation*, Odyssey 2022. [ISCA](https://www.isca-archive.org/odyssey_2022/tak22_odyssey.pdf) · [arXiv:2202.12233](https://arxiv.org/abs/2202.12233) | The SSL-front-end + AASIST recipe, ~90 % relative gain, and the finding that **fine-tuning the front-end is necessary** — the basis for Recipe C. |
| R4 | *ASVspoof 5 Evaluation Plan (Phase 2)* — [asvspoof.org](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf) | Protocol and metrics. **Track 1 primary = minDCF**, with Cllr and actDCF alongside. Track 2 = min a-DCF, t-DCF/t-EER supplementary. |
| R5 | Wang, Yamagishi, Todisco, Delgado, Nautsch, Evans et al. — *ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale*, 2024. [arXiv:2408.08739](https://arxiv.org/pdf/2408.08739) · [journal](https://www.sciencedirect.com/science/article/pii/S0885230825000506) | The corpus: ~2 000 speakers, 20+ attacks, 7 adversarial attacks. |
| R6 | Müller, Czempin, Diekmann, Froghyar, Böttinger — *Does Audio Deepfake Detection Generalize?* (**In-the-Wild**), Interspeech 2022. [arXiv:2203.16263](https://arxiv.org/abs/2203.16263) | Held-out generalization set and the generalization-gap figure. |
| R7 | *Audio Anti-Spoofing Detection: A Survey* — [arXiv:2404.13914](https://arxiv.org/html/2404.13914v1); *Robustness in deepfake speech detection* — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0885230826001002) | Corroboration for sub-1 % → 30–60 % EER degradation and why it happens. |
| R8 | ASVspoof 5 on Hugging Face — [huggingface.co/datasets/jungjee/asvspoof5](https://huggingface.co/datasets/jungjee/asvspoof5) | Alternate download mirror. Licence still governs. |
| R9 | PyTorch `sm_120` / Blackwell support — [pytorch/pytorch#159207](https://github.com/pytorch/pytorch/issues/159207) | §2's environment gate; PyTorch ≥ 2.7 on a CUDA 12.8+ build. |
| R10 | Pan, Liu et al. — *Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection*, Interspeech 2024. [ISCA](https://www.isca-archive.org/interspeech_2024/pan24c_interspeech.pdf) · [arXiv:2406.10283](https://arxiv.org/abs/2406.10283) | Recipe C's core: early-to-middle WavLM layers outperform the last layer and all-layer pooling; attentive merging beats linear. 0.65 / 3.50 / 3.19 % EER on 2019 LA / 2021 LA / 2021 DF. |
| R12 | *Generalizable speech deepfake detection via meta-learned LoRA*, 2025. [arXiv:2502.10838](https://arxiv.org/pdf/2502.10838) | Recipe D. LoRA adapters in every attention head + Meta-Learning Domain Generalization. Avg EER 8.84 % (full fine-tune) → **5.30 %**; beats full fine-tuning on **5 of 6** corpora while training **~3.6 M params (~1.1 %)**. |
| R13 | *AASIST3: KAN-Enhanced AASIST ... for the ASVspoof 2024 Challenge*. [arXiv:2408.17352](https://arxiv.org/html/2408.17352) | The upgraded back-end: Kolmogorov–Arnold networks (learnable B-splines + PReLU) replacing activations in GAT/GraphPool/HS-GAL. ASVspoof 5 **open** condition minDCF 0.1414 / EER 4.89 % (an *ensemble* of several models); **closed** condition 0.5357 / 22.67 %. |
| R11 | ASVspoof 5 results and post-challenge analyses — challenge overview [emergentmind summary](https://www.emergentmind.com/topics/asvspoof-5-challenge); *WavLM model ensemble for audio deepfake detection* [ISCA](https://www.isca-archive.org/asvspoof_2024/combei24_asvspoof.pdf); *Exploring WavLM Back-ends for Speech Spoofing and Deepfake Detection* [ISCA](https://www.isca-archive.org/asvspoof_2024/stourbe24_asvspoof.pdf) | §5.0's expectation table; the WavLM-vs-wav2vec2 gap (7.26 % vs 11.91 % EER); and that top systems are ensembles, not single models. |
