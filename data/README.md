# VaniRakshak Data & Benchmark Assets

This directory houses the standardized forensic evaluation datasets, speaker enrollment anchor profiles, and preprocessing scripts for VaniRakshak.

## Structure

```
data/
├── anchor_speaker_embedding.npy  # 192-dim ECAPA-TDNN reference anchor embedding
├── anchor_speaker_profile.json   # Cryptographic DPDP Act 2023 enrollment receipt & metadata
├── consent/                      # Digital consent and statutory recording disclosures
├── manifest.csv                  # (filepath, speaker_id) index of the standardized corpus
├── anchor_speaker/               # Raw target-speaker source recordings
├── anchor_speaker_std/           # Standardized 16 kHz mono target-speaker clips
├── other_speakers/               # Raw non-target source recordings
├── other_speakers_std/           # Standardized 16 kHz mono non-target clips
├── features/                     # Pre-extracted 40-dim MFCC matrices (.npy), per split
├── demo/                         # Standardized public benchmark audio suite
│   ├── demo_01_genuine.wav       # 10s bona fide target speech (Narendra Modi)
│   ├── demo_02_full_clone.wav    # 10s vocoded / neural clone speech
│   ├── demo_03_partial_splice.wav# 10s spliced speech (genuine -> clone -> genuine)
│   ├── demo_04_impostor.wav      # 10s zero-shot impostor speech
│   └── ground_truth.json         # Ground-truth forensic annotations & expected verdicts
├── scripts/                      # Data engineering and evaluation split scripts
│   ├── create_demo_splice.py     # Generates benchmark audio files from source recordings
│   ├── make_splits.py            # Generates train/dev/eval JSON splits and manifest-based train.csv/test.csv
│   ├── standardize.py            # In-memory 16 kHz mono standardization & DSP analysis
│   └── batch_standardize.bat     # Batch wrapper around standardize.py
└── splits/                       # Generated dataset protocol splits (train.csv, test.csv, train/dev/eval.json)
```

## Corpus Composition

The standardized corpus is **two speakers**: 8 target-speaker recordings (`anchor_speaker_std/`)
and 45 non-target clips (`other_speakers_std/`), all chunked from a single non-target recording.
`make_splits.py` therefore holds out whole recordings for the target speaker but can only hold out
a contiguous block for the non-target speaker, and logs a warning when it does. See the note in
[`../ml/README.md`](../ml/README.md) on how that limits the reported metrics.

## Privacy & DPDP Compliance

Per **Section 8(7) of the India Digital Personal Data Protection (DPDP) Act 2023**:
- No raw voice recordings are stored or retained on disk during live inference.
- Speaker profiles are stored exclusively as 192-dimensional non-invertible latent embeddings.
- Every biometric transaction emits a SHA-256 hash-chained cryptographic audit receipt.
