# 19 — Pre-registration: XLS-R frontend ablation

**Status: pre-registered 2026-09-19, not yet run.** Criteria below are fixed
before any result exists, and must not be adjusted afterwards. This follows
this repo's convention (docs/10) of fixing the bar in advance.

Pre-registered *before* the G4 verdict is known, deliberately. If it were
written afterwards it would be fitted to the outcome.

## Hypothesis

Speech DF Arena (docs/17 §0) ranks the XLSR family roughly **twice as good**
as the WavLM family out of domain (XLSR+SLS 13.84 % avg EER vs WavLM-ECAPA
28.74 %). If that ranking transfers to this stack, swapping the frozen
frontend is worth far more than any augmentation lever in docs/10 (~2 pp).

**Null hypothesis:** the Arena ranking is an artefact of *their* training
recipes and does not transfer to ours.

## Protocol — one variable only (ground rule 3)

Identical to the G4/control protocol except the frontend:

| | G4 / control | XLS-R arm |
|---|---|---|
| Frontend | `microsoft/wavlm-base-plus` (95 M, dim 768, 12 layers) | `facebook/wav2vec2-xls-r-300m` (315 M, dim 1024, 24 layers) |
| Head | `AASISTLHead(in_dim=768, merge_layers=12)` | `AASISTLHead(in_dim=1024, merge_layers=24)` |
| Seeds | 42, 43 | 42, 43 |
| Data / epochs / batch / lr / patience | as G4 | identical |
| `--g4-rawboost` | on (G4) / off (control) | **on** — match the G4 arm |
| Encoder | frozen (`--unfreeze-top-n 0`) | frozen |

Two forced deviations, recorded rather than hidden:

1. **`in_dim` must change 768 → 1024.** Verified: XLS-R hidden states are
   `(B, 49, 1024)`. `AASISTLHead(in_dim=768)` will not accept them.
2. **Head init cannot be reused.** `--init-weights data/cm_aasist.pt` carries a
   768-dim head, so it is incompatible. The XLS-R arm trains its head from
   random init, whereas G4/control start from `cm_aasist.pt`. This is a real
   second variable and is the main threat to the comparison being clean.
   Mitigation: run **both** seeds, and if the result is ambiguous, run a
   WavLM-from-random-init arm to isolate it.

`merge_layers=24` (all layers) is the direct analogue of WavLM's merge over
all 12 — not `merge_layers=12`, which on a 24-layer model would mean
"embedding + first 11 layers", a different depth.

## Criteria (fixed in advance)

Evaluated on the **full** 20,000-row eval slice (`clean` — ASVspoof 5 is
already telephone-captured) and on **In-the-Wild** (docs/18):

- [ ] **Primary:** pooled eval EER improves over the best G4-or-control
      checkpoint by **more than the 0.770 pp noise floor** (docs/10 line 168,
      Ground rule 1), on **both** seeds. One seed proves nothing at this
      spread.
- [ ] **Secondary (the one that actually matters):** In-the-Wild EER improves
      by more than the floor. The G4 seed-42 **final** checkpoint (epoch 7),
      re-measured 2026-09-19 07:10 on all 31,779 rows, scores **35.308 %**
      (`clean`) and **44.990 %** (`full_call`) — that is the number to beat,
      not the ~37 %/42.6 % quoted from the epoch-3 checkpoint. If the frontend
      swap only helps in-domain and not out-of-domain, it has not fixed the
      problem that motivated it. Judge against `full_call`, the harder
      condition: `clean` is the one that has been improving while everything
      else regresses (docs/18).
- [ ] **Secondary:** A19 and A20 EER both improve by more than the floor.

**AMENDMENT 2026-09-19 08:00 — the per-attack bar above is underpowered and
is raised here, before any XLS-R result exists.** This is the correct time to
change it; changing it afterwards would be fitting the bar to the outcome.

The 0.770 pp floor (docs/10 Ground rule 1) was measured between two
full-corpus checkpoints on the **pooled** EER. It says nothing about
per-attack precision, and per-attack precision is much worse, because A19 and
A20 have only ~935 and ~920 spoof utterances in the 20k slice. Bootstrapped
(`scripts/eer_bootstrap_ci.py`, 2,000 reps):

| | 95 % CI half-width |
|---|---|
| pooled (n = 15,936 spoof) | **±0.53 pp** |
| A19 (n = 935) | **±1.36 pp** |
| A20 (n = 920) | **±1.36 pp** |

So a 0.770 pp movement on A19 is **well inside** the sampling noise. Keeping
that bar would let the XLS-R arm "PASS" on a difference that is
indistinguishable from resampling luck.

**Revised per-attack criterion:** A19 and A20 must each improve by **more than
1.36 pp**, and the 95 % bootstrap intervals of the XLS-R arm and the
G4/control baseline must **not overlap**. (Equivalently: report the intervals
and require disjointness — do not compare a point estimate to a point
estimate.) The pooled criterion stays at 0.770 pp, which is defensible: it is
larger than the pooled sampling noise of ±0.53 pp.

Open item this creates: docs/10 Ground rule 1 gives a single number for all
metrics. It should probably carry a per-attack floor alongside the pooled
one. **Not patched unilaterally** — recorded here as an open item instead.

**Falsification:** if XLS-R does not beat WavLM beyond the floor on both
seeds, the Arena ranking does not transfer to this stack and the result is
reported as a negative — published as measured, not re-run (ground rule 5).

## De-risked already (2026-09-19)

| Question | Answer |
|---|---|
| Is XLS-R 300M available? | Yes — 710k downloads; staged locally at `A:\hf-local\xls-r-300m` (1.27 GB) |
| Does it load and run? | Yes — 315.4 M params, 25 hidden states, `(B, 49, 1024)`, fp16 on this GPU |
| How much slower? | **1.28× per forward** (measured, 44.0 vs ~34.4 ms/batch @ bs=16). ~45 min/epoch vs ~35 |
| Total GPU cost | ~7.5 h/seed × 2 seeds ≈ **15 h** (vs ~12 h for G4+control) |

### Environment trap: HuggingFace downloads are broken here

`transformers` / `urlretrieve` write **0-byte files** for small assets
(config.json, preprocessor_config.json) — both into the default HF cache and
into `A:\`. This is why `AutoConfig.from_pretrained('facebook/wav2vec2-xls-r-300m')`
fails with *"not a valid JSON file"*.

Working pattern:

```python
b = urllib.request.urlopen(url).read()   # urlretrieve does NOT work
Path(dest).write_bytes(b)
```

`pytorch_model.bin` downloads fine via `urlretrieve`; it is the small JSONs
that come back empty. Manual staging is required for any new HF model.

## When to run

GPU is committed to the G4/control queue. Measured timings as of 2026-09-19
05:32 (seed 42 at epoch 8/10, ~35 min/epoch):

| Stage | ETA |
|---|---|
| G4 seed 42 finishes (epoch 10) | ~06:50 |
| G4 seed 43 | ~12:40 |
| Control seed 42 | ~18:20 |
| Control seed 43 | ~00:10, 09-20 |
| `finalize_calibration.py` + adjudication | ~01:10, 09-20 |

This experiment starts after `finalize_calibration.py` completes, i.e. **not
before ~01:00 on 2026-09-20**. At ~7.5 h/seed it would land around 16:00 on
09-20. Gitignored checkpoints go to `C:\vr-save\ckpt` immediately on write.

**Do not modify `ml/train_cm.py` or `ml/augment.py` before then.** Queued runs
launch new processes that would pick up the edits, silently changing the
control arm and breaking the one-variable rule. The change needed here is
additive (a `--frontend` flag plus `in_dim` derived from the frontend config),
but it must land after the sweep, not during it.
