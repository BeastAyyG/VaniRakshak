# VaniRakshak Machine Learning & Forensic Benchmark Suite

Target-conditioned neural models, ASVspoof-5 evaluation metrics, and telephony codec robustness benchmarks.

**The trained checkpoint covers one enrolled identity (Narendra Modi) and has never seen cloned audio — see the note under [Evaluated Metrics](#evaluated-metrics).**

## Architecture & Modules

```
ml/
├── model.py              # AcousticForensicClassifier (BiGRU + Attentive Stats Pooling)
├── loss.py               # ForensicDetectionCostLoss (smooth minDCF) & AdditiveMarginSoftmaxLoss
├── dataset.py            # ForensicAudioDataset with G.711 / AMR-WB / GSM-EFR augmentations
├── train.py              # End-to-end training and checkpoint saving
├── evaluate.py           # Comprehensive ASVspoof-5 and telephony benchmark runner
├── extract_features.py   # Batch 40-dim MFCC extraction; single-file WavLM (768) / ECAPA (192) mode
├── checkpoints/          # Model weights & calibrated thresholds.json
└── requirements.txt      # PyTorch, NumPy, SciPy dependencies
```

## Running Benchmarks

### 1. Unified Forensic Benchmark Battery
Runs ASVspoof-5 metrics, CPU latency SLA, telephony codec robustness, and demo file ground truth:
```bash
python benchmarks/run_benchmarks.py
```

### 2. ML Evaluation & Threshold Calibration
Evaluates a trained checkpoint against the test split:
```bash
python ml/evaluate.py --checkpoint ml/checkpoints/best_model.pt
```
Without `--checkpoint`, this runs the synthetic-stub benchmark battery instead and does not touch the trained weights.

### 3. Model Training
```bash
python ml/train.py --epochs 50 --batch-size 32
```

## Evaluated Metrics

- **minDCF**: Minimum normalized detection cost function under ASVspoof 5 parameters ($C_{\text{miss}}=1, C_{\text{fa}}=1, P_{\text{target}}=0.05$). Measured: `0.0000`.
- **actDCF**: Actual detection cost at theoretical Bayes threshold $\tau_{\text{Bayes}} = -\ln \beta \approx -0.644$.
- **CLLR**: Log-likelihood ratio cost measuring probabilistic calibration. Measured: `0.0049`.
- **EER**: Equal Error Rate operating point ($P_{\text{miss}} = P_{\text{fa}}$). Measured: `0.00%`.

**The shipped checkpoint is fine-tuned for one person.** The enrolled target is a single
identity — **Narendra Modi** — represented by 8 recordings in `data/anchor_speaker_std/`
(see `data/anchor_speaker_profile.json`, `user_id: narendra_modi`). It is a one-person
target-speaker verifier, not a general voice-clone detector. Re-enrolling a different target
requires retraining; the weights here do not transfer.

Measured on the current 11-clip test split (2026-09-08): accuracy `100.00%` (11/11). That
corpus is the one target speaker against a single non-target speaker (45 chunks of one
recording); target clips score LLR ≈ +6.0 against ≈ −5.5 for non-target, a gap of ~11 nats.
`make_splits.py` holds out whole recordings for the target speaker, but the non-target speaker
has only one recording session, so a session-disjoint split is impossible there and a
contiguous block is held out instead. The training set contains **no** cloned or synthetic
audio. These figures therefore measure separation of two unrelated recordings and say nothing
about performance on other speakers or on cloned speech.
