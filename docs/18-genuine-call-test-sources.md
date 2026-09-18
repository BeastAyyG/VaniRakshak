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

## Caveats — do not quote these numbers yet

1. **This checkpoint is 30 % trained** (epoch 3 of 10). Treat as a pipeline
   check, not a result.
2. **No control.** There is no `--merge-layers 12` / non-RawBoost checkpoint to
   compare against yet, so this run cannot tell you whether G4 *helped*. That
   control is required before any claim.
3. **n = 1500**, not the full 20k slice.
4. **Scoring convention is easy to get wrong — I hit it during this run.**
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
