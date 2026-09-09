> **ARCHIVED 2026-09-09.** An early hackathon planning document. Its stated
> scope ("deliberately limited to forensic analysis of uploaded/recorded
> audio files, rather than ... live telephone infrastructure, Android payment
> interception, or emergency-keyword detection") has since been superseded by
> what actually shipped: live WS streaming (`docs/PROTOCOL.md`), Android
> payment-app interception (`vanirakshak-android/`), and keyword-triggered
> detection (`VaniKeywordTrigger.kt`). The file also has a corrupted section
> (a duplicated block starting around "Recommended Project Definition") from
> its original drafting — left as-is rather than silently cleaned up, since
> this is a historical artifact, not living documentation. For the current
> scope and architecture, see `docs/ARCHITECTURE.md`.

# VaniRakshak — Project Context and Practical Build Plan

> **Purpose:** Consolidated project context from this discussion for a hackathon-ready, technically honest, recorded-audio voice-clone detection prototype.

## Problem Focus

**VaniRakshak (वाणी रक्षक)** addresses AI-powered voice-cloning and voice-impersonation attacks.

The final scope is deliberately limited to **forensic analysis of uploaded/recorded audio files**, rather than GPS/SOS features, live telephone infrastructure, Android payment interception, or emergency-keyword detection.

### Core objective

Given a suspicious recording and consented reference recordings of a claimed speaker, the system should:

1. Estimate whether each part of the recording is likely genuine or AI-generated.
2. Compare the recording to the claimed speaker’s enrolled voice profile.
3. Find likely cloned or spliced intervals using segment-level analysis.
4. Present a clear evidence timeline, confidence scores, and an explainable verdict.

## Recommended Project Definition

**VaniRakshak is a target-conditioned recorded-audio fore

> VaniRakshak analyzes recorded speech segment by segment, combining synthetic-audio risk with claimed-speaker similarity to flag likely cloned or spliced voice-impersonation content.

### Honest technical statement

> We build the target-speaker enrollment workflow, dataset pipeline, classifier head, segment-level risk engine, evaluation protocol, FastAPI service, and evidence dashboard. Pretrained speech models are used as foundation feature extractors.

## Feasibility Decision

### Best practical architecture for four people in two days

Use **WavLM-Base+ frozen features + a custom lightweight cnsic prototype.** It uses a person’s consented genuine recordings to train and evaluate a detector against synthetic versions of that same person’s voice.

It is not a universal “100% accurate” detector for every human voice, every voice generator, every language, and every recording condition.

### Suggested one-line pitch

> VaniRakshak analyzes recorded speech segment by segment, combining synthetic-audio risk with claimed-speaker similarity to flag likely cloned or spliced voice-impersonation content.

### Honest technical statement

> We build the target-speaker enrollment workflow, dataset pipeline, classifier head, segment-level risk engine, evaluation protocol, FastAPI service, and evidence dashboard. Pretrained speech models are used as foundation feature extractors.

## Feasibility Decision

### Best practical architecture for four people in two days

Use **WavLM-Base+ frozen features + a custom lightweight classifier + ECAPA-TDNN speaker similarity + sliding-window splice localization**.

```text
Consented genuine target recordings
        |
        +--> ECAPA-TDNN --> average normalized speaker-profile embedding

Suspicious recorded file
        |
        +--> Standardize audio: mono, 16 kHz, normalized amplitude
        |
        +--> Voice activity detection / silence trimming
        |
        +--> 2-second windows with 1-second overlap
        |
        +--> WavLM-Base+ frozen feature extraction
        |
        +--> Custom MLP spoof classifier
        |
        +--> Segment-level spoof probabilities
        |
        +--> Peak + proportion + duration aggregation
        |
        +--> Verdict and highlighted suspicious timestamps

In parallel:
Suspicious file --> ECAPA-TDNN --> cosine similarity to enrolled profile
```

### Why this option is best

- It is achievable in a 48-hour sprint.
- WavLM provides strong pretrained speech features without training a foundation model from scratch.
- Training only a small MLP classifier is fast and manageable on Colab/Kaggle.
- ECAPA-TDNN adds useful claimed-speaker evidence.
- Sliding windows enable partial-splice localization, which makes for a strong visual demo.
- The pipeline is easy to explain, test, deploy, and improve.

## Architectures Considered

| Option | Feasibility in 2 days | Recommendation | Reason |
|---|---:|---|---|
| MFCC + CNN/BiLSTM from scratch | High | Avoid as primary system | Fast baseline but may not generalize well to modern neural-TTS systems |
| AASIST alone | Medium | Optional second-stage experiment | Useful anti-spoof model, but setup and robust evaluation can consume time |
| WavLM-Large + AASIST + ECAPA cross-attention fusion | Low | Avoid for hackathon MVP | Too much model integration, training, calibration, and debugging risk |
| Generic pretrained detector/API only | High | Avoid as primary system | Fast but weak ownership, limited control, and uncertain behavior |
| WavLM-Base+ + MLP + ECAPA + sliding windows | High | **Build this** | Best balance of practicality, explainability, and output quality |

## Important Accuracy Reality

Do not claim complete, universal, or 100% accuracy.

### Controlled target-conditioned setting

If training and test data are properly separated but remain similar in speaker, recording quality, and clone-generator families:

- A reasonable target is approximately **80–90% accuracy** for an initial baseline.
- With better data diversity, augmentation, threshold tuning, and a small ensemble, a **high-80s to mid-90s** result may be possible on the project’s held-out, in-scope test set.

### Worst-case setting

Under unseen generators, unseen languages or accents, heavy edits, strong compression, unusual microphones, noise, or adversarial post-processing, performance may fall close to random:

- Approximate worst-case binary accuracy: **45–60%**.
- A model can miss clones or incorrectly flag genuine speech.

### Why this happens

- The model may learn artifacts from particular clone generators rather than all possible synthetic speech.
- Audio distribution changes across devices, rooms, languages, and codecs.
- A high-quality clone can preserve speaker identity well.
- Short audio and partial splices provide limited evidence.
- Thresholds tuned for one dataset may not transfer to another.

## Correct Use of Speaker Verification

ECAPA-TDNN provides **speaker similarity**, not proof of authenticity.

A high-quality clone can have high similarity to the enrolled target because it is intentionally imitating that person.

### Interpret the two scores together

| Synthetic-risk score | ECAPA similarity | Suggested interpretation |
|---:|---:|---|
| High | High | Likely target-speaker clone / high-risk impersonation |
| High | Low | Likely synthetic speech or wrong claimed ide complexity.
- Train with multiple clone generators.
- Use unseen text prompts in validation and test sets.
- Add hard-negative mining: focus retraining on genuine clips falsely flagged as spoofed and synthetic clips that were missed.

### Audio augmentation

Apply only to training data, not final test data:

- Small pitch changes, approximately plus/minus 1–2 semitones
- Mild speed changes, approximately plus/minus 5–10%ntity |
| Low | High | Likely genuine target recording, subject to uncertainty |
| Low | Low | Different speaker, low-quality recording, or uncertain result |

The dashboard should show these as separate evidence signals rather than claiming ECAPA detects deepfakes.

## Dataset Strategy

Dataset design is more important than adding a very complex architecture.

### Genuine recordings

Collect only with the target speaker’s explicit consent.

Aim for 15–30 minutes of genuine audio, across:

- Multiple recording sessions, not one long session
- Different sentences, topics, and phonetic coverage
- Hindi, English, Hinglish, or relevant local language use
- Neutral, fast, slow, emotional, and conversational speech
- Multiple microphones or distances when possible
- Quiet and mildly noisy/reverberant environments

### Synthetic recordings

Generate clones only with the voice owner’s consent.

Use at least two synthesis families if possible:

- Coqui XTTS-v2
- F5-TTS or E2-TTS
- Optional: an approved commercial generator if permitted and consented

Create diverse text prompts and speaking styles. Do not generate the same paragraph repeatedly.

### Correct train/validation/test split

Avoid leakage. Adjacent clips from the same original recording must not be split across train and test.

| Split | Recommended content |
|---|---|
| Training | Sessions A–C, training prompts, training clone samples |
| Validation | Separate session D and new prompts; used for threshold selection |
| Final test | Separate session E, unseen prompts, and ideally a clone generator not used during training |

The final test set must never be used to choose hyperparameters or thresholds.

## High-Impact Accuracy Improvements

### Data improvements

- Increase genuine-speaker diversity before increasing model complexity.
- Train with multiple clone generators.
- Use unseen text prompts in validation and test sets.
- Add hard-negative mining: focus retraining on genuine clips falsely flagged as spoofed and synthetic clips that were missed.

### Audio augmentation

Apply only to training data, not final test data:

- Small pitch changes, approximately plus/minus 1–2 semitones
- Mild speed changes, approximately plus/minus 5–10%
- Light room reverb
- Moderate background noise
- Loudness variation
- Mild compression or EQ, if relevant to expected recordings

### Model improvements

- Extract WavLM representations from multiple intermediate layers rather than only the final layer.
- Concatenate or pool those features before the classifier.
- Use a lightweight MLP, dropout, weight decay, and early stopping.
- Try focal loss if hard classes are consistently missed.
- Train multiple small heads with different random seeds and average their probabilities.
- Add AASIST only if it is working early and demonstrably improves validation metrics.

### Decision improvements

Do not rely only on `max(segment_probability)`.

For every uploaded file, compute:

```text
peak_spoof_prob      = maximum segment spoof probability
mean_spoof_prob      = average segment spoof probability
high_risk_fraction   = fraction of windows above a chosen threshold
suspicious_duration  = total duration of merged high-risk intervals
speaker_similarity   = ECAPA cosine similarity to enrolled profile
```

Use validation data to tune the thresholds. A small logistic-regression or rule-based meta-decision layer can combine these values.

## Core Technical Stack

| Area | Tool / model | Role |
|---|---|---|
| Audio processing | `torchaudio` | Load files, resample to 16 kHz, convert to mono, crop and pad windows |
| Optional inspection | `soundfile`, `librosa` | File checks, waveform/spectrogram inspection, test-splice generation |
| Voice activity detection | Silero VAD | Remove silence/noise before expensive inference |
| Deepfake features | `microsoft/wavlm-base-plus` | Frozen pretrained acoustic/phonetic representation extractor |
| Speaker verification | `speechbrain/spkrec-ecapa-voxceleb` | Speaker embeddings and cosine-similarity evidence |
| Custom detector | PyTorch MLP | Project-specific genuine-vs-clone decision head |
| Metrics/calibration | scikit-learn | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, threshold tuning |
| API | FastAPI + Uvicorn + `python-multipart` | File-upload analysis endpoint and JSON responses |
| Frontend | React/Vite + Tailwind CSS | Evidence dashboard |
| Waveform UI | Wavesurfer.js | Display audio waveform and highlight suspicious intervals |
| Packaging | Docker | Reproducible local setup and demo |
| Optional detector | AASIST | Add later only after validation proves a real gain |

## Minimal Classifier Design

Use a small classifier head after frozen WavLM embeddings:

```text
WavLM embedding (768 dimensions)
        |
Linear: 768 -> 256
        |
GELU/ReLU + Dropout
        |
Linear: 256 -> 64
        |
GELU/ReLU + Dropout
        |
Linear: 64 -> 2
        |
Bonafide probability / Spoof probability
```

If using multi-layer WavLM features, first concatenate or pool selected layers and update the first input dimension accordingly.

## Sliding-Window Forensics

### Window setup

- Standard audio format: 16 kHz mono PCM
- Window duration: 2.0 seconds
- Stride: 1.0 second
- Adjacent high-risk windows: merge into one suspicious interval

### Why this is important

A full-file average can hide a short synthetic insertion inside an otherwise genuine recording. Segment-level scanning enables localization of suspected splices.

### Example API result

```json
{
  "verdict": "PARTIAL_SYNTHETIC_MANIPULATION",
  "peak_spoof_probability": 0.91,
  "mean_spoof_probability": 0.28,
  "speaker_similarity": 0.88,
  "suspicious_duration_seconds": 3.0,
  "suspicious_intervals": [
    {
      "start": 14.0,
      "end": 17.0,
      "risk": 0.91
    }
  ]
}
```

## Four-Person, 48-Hour Plan

### Developer 1 — Data and adversarial curation

**Deliverables:** `data/train`, `data/val`, `data/test`, consent record, demo files.

- Gather consented genuine recordings from the target speaker.
- Standardize all files to 16 kHz mono WAV.
- Generate synthetic examples using two allowed clone systems.
- Produce properly separated train/validation/test splits.
- Build three final demo files: genuine, fully cloned, partially spliced.

### Developer 2 — ML, features, and evaluation

**Deliverables:** cached embeddings, trained model weights, metrics report.

- Load frozen WavLM-Base+ and ECAPA-TDNN.
- Extract/cache WavLM features for all clips.
- Train the lightweight spoof classifier.
- Calculate accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
- Tune the threshold on validation only.
- Export final model and clearly record the exact evaluation protocol.

### Developer 3 — Backend and forensic engine

**Deliverables:** FastAPI service and JSON segment timeline.

- Implement `POST /api/analyze`.
- Process uploaded recordings in sliding windows.
- Load model weights and return real inference probabilities.
- Compute speaker similarity and segment risk.
- Merge suspicious intervals and return timestamps.
- Ensure no random/mock probabilities remain in the code.

### Developer 4 — Dashboard and demonstration

**Deliverables:** upload UI, waveform evidence view, live demo flow.

- Build a clean file-upload dashboard.
- Display overall verdict, spoof risk, and speaker similarity separately.
- Use Wavesurfer.js to display the timeline.
- Highlight high-risk intervals in red and normal intervals in green/neutral color.
- Create a browser-only simulated transfer-warning panel if required by the problem statement; do not claim real UPI interception unless it has actually been implemented and tested.

## 48-Hour Milestones

### Hours 0–6: Working baseline

- Standardize dataset folders and consented recordings.
- Generate initial synthetic clips.
- Extract WavLM features.
- Train first MLP model.
- Create initial FastAPI endpoint and basic upload UI.

### Hours 6–18: Reliable pipeline

- Add ECAPA enrollment and similarity scoring.
- Add VAD/silence removal.
- Implement clean validation split.
- Add lightweight training augmentation.
- Run initial metric report and threshold tuning.

### Hours 18–32: Splice localization

- Implement 2-second / 1-second sliding windows.
- Return timestamped segment scores.
- Merge overlapping suspicious intervals.
- Integrate backend API with the waveform dashboard.

### Hours 32–48: Validate and rehearse

- Evaluate only once on the held-out final test set.
- Build demo files with known ground-truth timestamps.
- Remove every mock/random score and placeholder.
- Rehearse the demo using local files and explain limitations honestly.
- Add Docker/local setup instructions and screenshots if time permits.

## Suggested Demo Flow

### Test 1: Unseen genuine recording

Upload a genuine recording that was not used in model training.

Expected UI:

- Low spoof risk
- High speaker similarity
- No suspicious intervals
- Verdict: `LIKELY_GENUINE_TARGET_AUDIO`

### Test 2: Full clone

Upload a fully synthetic clone generated with an unseen prompt.

Expected UI:

- High spoof risk
- Often high speaker similarity
- Many or all segments highlighted
- Verdict: `SYNTHETIC_TARGET_IMPERSONATION_SUSPECTED`

### Test 3: Partial splice

Upload a genuine recording where a known synthetic phrase has been inserted.

Expected UI:

- Mostly low-risk timeline
- Clear risk spike at the known splice timestamps
- Red-highlighted suspicious range
- Verdict: `PARTIAL_SYNTHETIC_MANIPULATION`

## What Not to Build Now

Avoid these in the two-day MVP unless already complete and tested:

- Full WavLM-Large + AASIST + ECAPA cross-attention model training
- AASIST training from scratch
- Live telephone/WebRTC integrations
- GPS, SOS, emergency keyword detection
- Android AccessibilityService payment lockout
- Real banking/UPI controls
- Unsupported claims about privacy, compliance, accuracy, latency, or live deployments

## Claims to Avoid

Do not say:

- “100% accurate”
- “Complete accuracy”
- “Detects every AI-generated voice”
- “ECAPA proves a recording is genuine”
- “A custom cross-attention architecture” when only using concatenated embeddings and an MLP
- “Real UPI transaction lockout” if it is only a frontend simulation
- “No data is stored” if uploaded files are temporarily written to disk without secure deletion and documentation

## Strong, Accurate Claims

Use wording such as:

- “A target-conditioned prototype evaluated on a consented, held-out dataset.”
- “Segment-level evidence localization for suspected cloned or spliced audio.”
- “Combines synthetic-audio risk with claimed-speaker similarity.”
- “Designed for recorded-file forensics; robustness to unseen generators and environments remains a future-work area.”
- “Uses pretrained speech representations with a custom classifier, inference pipeline, evaluation protocol, and dashboard.”

## Future Improvements

After the hackathon, improve robustness with:

- AASIST or another raw-waveform spoof detector as a score-level ensemble
- More clone-generator families and voice-conversion models
- More speakers and languages
- Stronger out-of-distribution tests
- Codec/noise/reverb augmentation
- Better calibration for uncertain predictions
- Audio-splice boundary detection
- Model explainability and forensic reporting
- Secure, consent-based speaker enrollment and retention policy

## Final Build Recommendation

Build this first and make it work end to end:

> **WavLM-Base+ frozen features -> custom MLP spoof classifier -> ECAPA target similarity -> 2-second sliding-window risk timeline -> FastAPI -> React/Wavesurfer evidence dashboard.**

Prioritize real inference, clean data splits, validation metrics, and a transparent demo over a complicated architecture with unverified claims.
