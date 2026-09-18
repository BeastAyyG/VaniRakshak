# 18 — Where to get audio that genuinely tests this on a call

Compiled 2026-09-19. "Genuinely test on a call" has three separate axes, and
no single dataset covers all three. Knowing which axis a source covers is the
whole point of this document.

| Axis | What it means | Why it matters |
|---|---|---|
| **1. Real telephony capture** | Audio actually traversed a phone network (codecs, packet loss, handset, network noise) | A detector that only ever saw studio audio will not survive a PSTN path |
| **2. Real (not lab-made) fakes** | Deepfakes produced by real generators in the wild, not by a challenge organiser | Lab fakes share artefacts; the generalisation gap is where every system fails |
| **3. Your actual domain** | Indian languages, Indian accents, scam-call register, your own voices | Every public corpus is English-first and studio-first |

---

## Already on this machine

### ASVspoof 5 — `A:\cm-corpus\asvspoof5` ✔ have

**Axis 1 (real telephony), partly axis 2.** This is the important one people
miss: **ASVspoof 5 was captured over real telephone networks.** For this
corpus `clean` *is* the call condition. Running the codec conditions in the
harness on top of it is double-transcoding and will overstate degradation.

- Train manifest: 182,357 utts (A01–A08). Eval slice: `data/eval_benchmark/eval_slice_20k.tsv` — 20,000 utts, bonafide 4,058, attacks **A17–A32** (16 types), including **A19 (935) and A20 (921)** — the two the G4 criterion is written against.

### Your own scam-call samples — `test_audio/` ✔ have

**Axis 3.** `hindi_sample.wav`, `hinglish_sample.wav`, `sample_60s_scam.wav`,
`scam_sample.wav`, plus KYC / FedEx / "bijli" Hinglish scam scripts. Small
(n = 10) but they are the only Indian-language, scam-register audio in the
repo, and the only audio that resembles what you will actually be scored on.

### In-the-Wild — `A:\cm-corpus\external\itw` ✔ downloaded 2026-09-19

**Axis 2.** 37.9 h, **31,779 WAVs**, real deepfakes scraped from online video
of **58 public figures** (17.2 h fake / 20.7 h real). English, online-video
audio — so **not** a phone channel; run it through the `full_call` condition
in the harness to say anything about call performance.

```
python scripts/download_call_testset.py --name itw      # 7.6 GB, ~2 min at 75 MB/s
python scripts/download_call_testset.py --list
```

---

## Worth fetching next

| Source | Axis | Size / access | What it adds that nothing else does |
|---|---|---|---|
| **ASVspoof 2021 DF** | 1 | ~10 GB, open | Explicit codec/transmission conditions C1–C7, **including unknown codecs C5–C7** — the cleanest codec-robustness test that exists |
| **CodecFake** | 1+2 | Open (GitHub) | Deepfakes generated *through* neural codecs — the modern attack shape that RawBoost predates |
| **ADD 2022 / 2023** | 2+3-ish | Registration + agreement | Real-world multilingual, unconstrained conditions, includes telephone channel |
| **ML-ITW** (arXiv 2603.05852) | 2 | New, check release | Multilingual in-the-wild — the first ITW-style set that is not English-only |
| **IndicVoices** (AI4Bharat) | 3 | 23.7K h, 22 Indian languages, HF + aikosh | Indian bonafide **and** cloning source material. Stream a subset — do not pull 23.7K h |
| **Fisher / Switchboard** | 1+3-no | Open, English | Real telephone *conversation*. Bonafide only — this is your **false-positive** test set, and per docs/17 that is the number that matters commercially |

Sources marked open are downloadable with
`scripts/download_call_testset.py` (add the URL to its `SOURCES` table).
ADD requires a signed agreement and cannot be automated.

---

## The only test that is truly genuine: place a real call

Every corpus above is a proxy. If you want to know whether this works on a
call, make a call:

1. **Local SIP**: Asterisk or FreeSWITCH on this box, two softphones, force
   G.711 µ-law or G.722. Free, repeatable, no carrier.
2. **Real PSTN/mobile**: Twilio, or Exotel/Plivo for Indian numbers. Place a
   real call, record both legs. This is the only way to get true carrier
   transcoding, real packet loss and real handset noise.
3. **Red-team your own voice**: enrol a real voice, clone it with a current
   TTS (XTTS-v2 / F5-TTS locally, or ElevenLabs), and play the clone *into*
   the call path. If your own system cannot catch a clone of a voice it just
   enrolled, no benchmark number matters.

---

## The harness

`scripts/eval_call_channel.py` — scores a manifest under explicit call-channel
conditions and reports **per-attack** EER, which is what the G4 criterion is
actually written against (the training log only prints pooled dev EER, which
hides exactly the per-attack failures G4 exists to fix).

```bash
# ASVspoof 5: 'clean' IS the call -- do not add codecs
python scripts/eval_call_channel.py \
  --checkpoint data/cm_g4_rawboost_s42.pt \
  --manifest data/eval_benchmark/eval_slice_20k.tsv \
  --conditions clean

# Non-phone corpora (ITW, IndicVoices): simulate the call path
python scripts/eval_call_channel.py --checkpoint <ckpt> \
  --manifest <itw_manifest.tsv> --conditions clean full_call
```

Conditions: `clean`, `alaw` (G.711 A-law — Europe/India PSTN), `ulaw`
(µ-law — US/Japan), `amr_wb`, `gsm_efr`, `bandpass`, `noise`, `babble`,
`reverb`, `full_call` (bandpass → A-law → 20 dB noise → ±3 dB gain).

---

## First measured run (G4 seed 42, epoch 3 of 10, n = 1500)

| Condition | Pooled EER | minDCF | A19 | A20 |
|---|---|---|---|---|
| `clean` (= the real call) | **10.38 %** | 0.7463 | **11.94 %** | **14.52 %** |
| `full_call` (double-transcoded) | 22.49 % | 0.8427 | 31.34 % | 22.14 % |

Read the `clean` row — that is ASVspoof 5's native telephone condition.
`full_call` is shown only to demonstrate what double-transcoding does; it is
not a realistic condition for this corpus.

Per-attack spread at `clean` is wide: A17 1.19 % → A32 21.54 %. The system is
not uniformly good or bad; it is blind to specific generators.

---

## The generalisation gap — the uncomfortable result

Same checkpoint, same harness, two corpora, ~2 h into training:

Measured on the **full** In-the-Wild corpus (n = 31,779 — 19,963 bonafide /
11,816 spoof), versus a 1,500-row ASVspoof 5 slice:

| Corpus | What it is | n | Pooled EER | minDCF |
|---|---|---|---|---|
| **ASVspoof 5** (`clean` = native telephone) | Lab-generated fakes, real phone channel | 20,000 | **11.24 %** | 0.7071 |
| **In-the-Wild** (`clean`) | **Real** deepfakes scraped from online video | 31,779 | **37.19 %** | 0.9989 |
| **In-the-Wild** (`full_call`) | Real deepfakes + simulated phone path | 31,779 | **42.59 %** | 0.9981 |

**A 3.3× swing depending only on which corpus you score.** The 11 % number is
the one that would end up in a benchmark table; the ~37–43 % numbers describe
what happens on audio a fraudster actually made.

On the same 20k slice, the two attacks the G4 criterion is written against are
**A19 13.70 %** and **A20 16.44 %** — both above the 12 % bar.

> **Sample size changed the verdict once already.** At n = 1,500 this same
> checkpoint read A19 **11.94 %** — apparently *under* the bar. At the full
> 20k it is 13.70 %, over it. Do not adjudicate G4 on a slice.

The 2,000-row ITW pilot earlier gave 36.51 % / 42.22 %, so the pilot was
within 0.7 pp of the full corpus — small pilots are adequate for tracking this
metric, which makes it cheap enough to run every time.

This is not a fluke of this checkpoint — it is the single most consistent
finding in the literature. Speech DF Arena (docs/17 §0) reports exactly this
pattern across 15 systems: near-perfect in-domain scores, EERs of 24–75 %
out-of-domain, and *"not a single model consistently outperforms others
across all datasets."* Its most damaging single observation is that
**almost all open systems are trained on ASVspoof 2019 and then generalise
poorly** — which is what we are reproducing here.

Practical consequences:

- **Never quote an ASVspoof number as a product accuracy figure.** If a buyer
  asks "how accurate is it?", the only defensible answer comes from audio
  resembling their traffic.
- **Track ITW as a standing regression metric**, not a one-off. It is the
  cheapest available proxy for "how blind are we to generators we did not
  train on".
- **This is the argument for IndicVoices and a real-call red-team.** ITW is at
  least English studio-ish audio; Indian telephony is further out of domain
  still, so assume today's real number is worse than 36.5 % until measured.

The `full_call` column is the honest one for ITW — that corpus was never on a
phone, so simulating the path is what makes it a *call* test.

---

## Queued: the G4 control run

`scripts/g4_control_runner.py` is **waiting on the G4 sweep and will take the
GPU automatically when it finishes.** It trains seeds 42 and 43 with
`--g4-rawboost` omitted and nothing else changed: same `--merge-layers 12`,
same `unfreeze_top_n=0`, same data, same init weights.

This is required, not optional. Without it the G4 experiment cannot be read:
any movement in A19/A20 could come from the augmentation or from the seed,
and with a measured seed-to-seed spread of ~3.6 pp those two explanations are
the same size. Ground rule 3 (one variable) is what makes the comparison mean
anything.

Status file: `C:\vr-save\g4-logs\control_status.json`. Kill the process to
cancel; the G4 sweep is unaffected.

---

## Adjudication — `scripts/adjudicate_g4.py`

Applies the pre-registered criterion mechanically so it is not re-litigated by
hand each time. It **never edits or reinterprets the criterion**; where data
conflicts with a pre-registered number it emits an OPEN ITEM instead of
silently moving the threshold.

Criterion (docs/10): *A19 and A20 eval EER both below 12 %, without pooled EER
regressing beyond the noise floor.* The floor governing G3/G4 is **0.770 pp**
(docs/10 line 168, Ground rule 1) — not the 5.95 pp from the 4k/2-epoch probe,
which docs/10 explicitly warns against quoting.

```bash
python scripts/adjudicate_g4.py \
    --g4 data/cm_g4_rawboost_s42.pt data/cm_g4_rawboost_s43.pt \
    --control data/cm_g4_control_s42.pt data/cm_g4_control_s43.pt
```

A missing arm yields `UNRESOLVED`, never `PASS` — an absent control cannot be
scored as a win.

**Current reading** (G4 seed 42, epoch 3 of 10, full 20k):

| Criterion | Value | Status |
|---|---|---|
| A19 EER < 12 % | 13.70 % | **FAIL** |
| A20 EER < 12 % | 16.44 % | **FAIL** |
| Pooled EER no regression | control not trained yet | UNRESOLVED |
| **Overall** | | **FAIL** |

Provisional — 30 %-trained checkpoint. But note it is failing on the *right*
axis: A19 and A20 are the two attacks G4 exists to fix, and they are the two
worst-performing of the 16 in the slice. That is the experiment working even
when the result is negative.

---

## Caveats — do not quote these numbers yet

1. **This checkpoint is mid-training and is the best-so-far, not the final
   model.** Epoch 3 is still the best at the time of writing (epoch 4 came in
   worse: dev EER 14.04 % vs 13.32 %, minDCF 0.9644 vs 0.9467). Dev EER has
   moved 17.87 → 15.88 → 13.32 → 14.04 across epochs. Treat every number here
   as a pipeline check, not a result, and re-run when seed 42 finishes.
2. **No control.** There is no `--merge-layers 12` / non-RawBoost checkpoint to
   compare against yet, so this run cannot tell you whether G4 *helped*. That
   control is required before any claim.
3. **Both corpora are now measured at full width** — ASVspoof 5 at 20,000
   rows, ITW at 31,779. Only the checkpoint is still provisional.
4. **ITW is English, online-video audio** — out of domain on language *and*
   channel *and* generator. It is a stress test, not a deployment
   simulation. It tells you the gap exists; it does not tell you the size of
   the gap on Indian telephony.
5. **Scoring convention is easy to get wrong — I hit it during this run.**
   The contract, from `ml/train_cm.py:919-920`, is:

   ```python
   _, logits = model(feats)                      # returns (embeddings, logits)
   llr = (logits[:, 1] - logits[:, 0])           # bonafide scores higher
   ```

   Taking element 0 as the logits, or scoring `logits[:, 1]` alone, silently
   produces inverted garbage (it looked like a 77 % EER; the true value was
   22 %). `score_pre_aug_eval.py` gets the unpacking and the LLR **right**, so
   it is not affected — but it builds `AASISTLHead(in_dim=768)` with **no
   `merge_layers`**, so it cannot score a 12-layer-merge checkpoint correctly
   at all. `evaluate_3way_benchmark.py` and `evaluate_4way_benchmark.py` build
   most heads the same way; only their `m_attm` passes `merge_layers=12`.
   **Any eval run against a post-G2 checkpoint through those scripts is
   invalid.** The new `eval_call_channel.py` reads `merge_layers`,
   `layer_index` and `freeze_merge` straight from the checkpoint instead of
   assuming, which is the safer pattern.
