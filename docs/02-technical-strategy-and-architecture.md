# Technical Strategy & Architecture

## 1. Executive Summary & Core Philosophy

**VaniRakshak** is a target-conditioned recorded-audio forensic intelligence platform designed for enterprise banking, fintech call centers, and customer-service voice pipelines. Its primary mission is real-time detection and interception of AI-synthesized speech, voice cloning, vocoded replay attacks, and conversational deepfakes.

### The Acoustic-Only Forensic Principle

A fundamental design constraint of VaniRakshak is that it is **strictly an acoustic forensic system**:

| Dimension | VaniRakshak Architectural Reality | Common Industry Misconception |
|---|---|---|
| **Audio Processing** | Pure acoustic waveform and spectral analysis (Welch PSD, sub-band energy ratios, phase jitter, micro-tremors). | ASR / Speech-to-Text (STT) conversion (Whisper, Kaldi). |
| **Analysis Domain** | Physics and DSP characteristics of synthetic vocoders, codec anomalies, and speaker vocal tract resonances. | NLP semantic analysis, grammar inspection, sentiment extraction. |
| **Model Footprint** | Specialized acoustic scoring models running in milliseconds per window. | Large Language Models (LLMs) interpreting call transcripts. |
| **Linguistic Neutrality**| Operates identically across English, Hindi, Hinglish, Marathi, Tamil, Telugu, etc. Language-agnostic by definition. | Degrades on Indian vernaculars, code-switching, and heavy regional accents. |
| **Latency Budget** | $p95 < 200\text{ ms}$ per analysis window on single-threaded CPU. | Multi-second transcription and inference pipelines. |

Deepfake speech models (FastSpeech2, VITS, StyleTTS2, Vall-E, ElevenLabs) generate speech by passing predicted acoustic representations through neural or parametric vocoders (HiFi-GAN, WaveGlow, LPCNet). While these vocoders produce high perceptual naturalness to human ears, they leave distinct mathematical anomalies in the physical audio domain: unnatural phase coherence, lack of physiological micro-tremor, suppressed high-frequency spectral rolloff, and periodic temporal frame energy variance. VaniRakshak exploits these physical DSP artifacts directly.

---

## 2. End-to-End Pipeline Architecture

The end-to-end processing pipeline runs continuously from browser or telephony capture through forensic scoring to hardware transaction interlock:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 1. AUDIO INGESTION (Browser AudioWorklet / Telephony Gateway)                     │
│    • 16 kHz, 16-bit signed mono PCM frames (Float32 in [-1,1] -> Int16)           │
│    • Zero audio playback loopback (muted gain node)                               │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │ WebSocket: /ws/stream (Binary Frames)
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 2. DPDP-COMPLIANT SLIDING BUFFER (volatile RAM only, zero disk writes)            │
│    • Window: 4.04 s (64,640 samples) | Hop: 1.0 s (16,000 samples)                │
│    • Circular FIFO ring buffer in memory (vanirakshak/privacy/buffer.py)          │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │ 4.04 s Audio Window
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 3. MULTI-LAYER FORENSIC DETECTOR STACK (vanirakshak/detectors/)                   │
│  ┌───────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐  │
│  │ Layer 1: CM           │ │ Layer 2: ASV           │ │ Layer 3: Channel/Codec │  │
│  │ AASIST-Shaped Stub    │ │ ECAPA-TDNN Stub        │ │ Multi-Band PSD Welch   │  │
│  │ • Welch PSD HB-ratio  │ │ • 192-dim embedding    │ │ • Energy distribution  │  │
│  │ • Spectral flatness   │ │ • Target user cosine   │ │ • Anomaly score        │  │
│  │ • Energy variance     │ │   similarity           │ │ • Codec categorization │  │
│  │ • ZC Jitter residual  │ │                        │ │   (GSM/AMR/Opus/G.711) │  │
│  │ ➔ LLR & P_genuine     │ │ ➔ ASV Consistency      │ │ ➔ Anomaly & SNR (dB)   │  │
│  └───────────┬───────────┘ └───────────┬────────────┘ └───────────┬────────────┘  │
│              │                         │                          │               │
│              └────────────────────┬────┴──────────────────────────┘               │
│                                   │                                               │
│                      ┌────────────┴───────────┐                                   │
│                      │ Layer 4: Prosody       │                                   │
│                      │ Pitch Bandpass 300-1500│                                   │
│                      │ • Pitch variance       │                                   │
│                      │ • Micro-tremor jitter  │                                   │
│                      │ ➔ Unnaturalness Score  │                                   │
│                      └────────────┬───────────┘                                   │
└───────────────────────────────────┼───────────────────────────────────────────────┘
                                    │ Feature Vector x
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 4. CALIBRATED BAYESIAN RISK FUSION ENGINE (vanirakshak/risk/fusion.py)            │
│    • Logistic regression evidence aggregation: z = β₀ + Σ (wᵢ · xᵢ)               │
│    • Posterior probability: P(spoof|x) = σ(z)                                     │
│    • Raw risk: 100 · P(spoof|x)                                                   │
│    • Fast-Rise Override (if LLR < -2.0) vs. EMA Smoothing (α = 0.65)              │
│    • Dynamic score: 0 to 100                                                      │
└───────────────────────────────────┬───────────────────────────────────────────────┘
                                    │ Risk Score & Tier
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 5. DECISION TIERS & TRANSACTION INTERLOCK (vanirakshak/session.py)                │
│    ├── ALLOW (< 35): Voice verified, transaction authorized                       │
│    ├── CHALLENGE (35-69): Suspicious, trigger dynamic phrase challenge            │
│    └── BLOCK (>= 70): Cloned voice attack, activate transaction interlock latch   │
│         │                                                                         │
│         ├── Server-side Alert Webhook: POST VR_ALERT_WEBHOOK_URL                  │
│         └── SHA-256 Hash-Chained Audit Log Receipt: vanirakshak/privacy/audit.py  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Forensic Detection Layers

### Layer 1: Countermeasure (CM) — Anti-Spoofing
- **Module:** `vanirakshak.detectors.cm.CMModel`
- **Receptive Field:** 4.04 seconds (64,640 samples @ 16 kHz), matching the canonical AASIST input window.
- **Physics Extracted:**
  1. *High-Band Ratio (`hb_ratio`):* $\frac{\sum P_{xx}(4000-8000\text{ Hz})}{\sum P_{xx}(0-4000\text{ Hz})}$. Neural vocoders regularly exhibit steep cutoff or high-frequency attenuation compared to natural vocal tract turbulence.
  2. *Spectral Flatness (`flatness`):* Ratio of geometric mean to arithmetic mean of the power spectrum. Vocoded speech lacks natural formant peaking.
  3. *Frame Energy Variance (`energy_var`):* Temporal dynamics across 400-sample windows. Vocoders produce unnaturally uniform energy envelopes.
  4. *Narrowband Pitch Jitter Residual (`jitter_resid`):* Coefficient of variation (CV) of zero-crossing intervals in the 300–1500 Hz passband. Natural human vocal folds suffer involuntary cycle-to-cycle perturbation ($CV \approx 0.02 - 0.15$), whereas neural speech synthesizers generate overly clean trajectories ($CV < 0.01$).
- **Output:** Log-Likelihood Ratio ($\text{LLR}$):
  $$\text{LLR} = w_0 + w_1 \tanh(5 \cdot \text{hb\_ratio}) + w_2 \tanh(20 \cdot \text{jitter\_resid}) + w_3 (1 - \text{flatness}) + w_4 \tanh(40 \cdot \text{energy\_var})$$
  $$\text{cm\_genuine\_prob} = \sigma(\text{LLR}) = \frac{1}{1 + e^{-\text{LLR}}}$$

### Layer 2: Automatic Speaker Verification (ASV)
- **Module:** `vanirakshak.detectors.asv.ASVModel`
- **Function:** Target-conditioned verification. Determines whether the audio belongs to the claimed target enrolled speaker or is an impersonator.
- **Acoustic Representation:** 192-dimensional vector embedding derived from log-mel spectral filterbanks and deterministic projection.
- **Verification Score:**
  $$\text{Cosine Similarity} = \frac{\mathbf{v}_{\text{enrolled}} \cdot \mathbf{v}_{\text{live}}}{\|\mathbf{v}_{\text{enrolled}}\| \|\mathbf{v}_{\text{live}}\|}$$
  $$\text{asv\_consistency} = \frac{\text{Cosine Similarity} + 1.0}{2.0} \in [0, 1]$$
  *(Un-enrolled sessions return neutral fallback $\text{asv\_consistency} = 0.5$).*

### Layer 3: Channel & Codec Anomaly Detection
- **Module:** `vanirakshak.detectors.channel.detect`
- **Function:** Identifies transmission channel profiles and detects out-of-band energy anomalies characteristic of synthetic generation pipelines.
- **Features:** Energy concentration across sub-bands (<300 Hz, 300–3400 Hz, 4000–8000 Hz).
- **Classification:** Categorizes audio into `gsm_efr_or_amr_wb`, `wideband_opus`, or `g711_or_pstn`.
- **Metrics:** `channel_anomaly` $\in [0, 1]$ (divergence from expected spectral energy distribution) and estimated SNR in dB.

### Layer 4: Prosody & Micro-Tremor Detection
- **Module:** `vanirakshak.detectors.prosody.prosody_score`
- **Function:** Evaluates physiological pitch variations in the fundamental vocal range (300–1500 Hz).
- **Metric:** `prosody_unnatural` $\in [0, 1]$. Synthetic vocoders over-regularize pitch contour and eliminate natural human micro-tremors, yielding high unnaturalness values ($1.0 - \tanh(3 \cdot \text{pitch\_var} + 2 \cdot \text{jitter})$).

---

## 4. Calibrated Bayesian Risk Fusion Engine

The fusion layer (`vanirakshak.risk.fusion`) aggregates the multi-dimensional evidence into a single calibrated risk score.

### Evidence Aggregation & Calibrated Posterior $P(\text{spoof} \mid \mathbf{x})$

For a given analysis window, the evidence vector comprises:
1. Countermeasure spoof probability: $x_{\text{cm}} = 1.0 - \text{cm\_genuine\_prob}$
2. Speaker verification mismatch: $x_{\text{asv}} = 1.0 - \text{asv\_consistency}$
3. Prosody unnaturalness: $x_{\text{prosody}} = \text{prosody\_unnatural}$
4. Channel anomaly: $x_{\text{channel}} = \text{channel\_anomaly}$
5. Contextual risk weight: $x_{\text{context}} \in [0, 1]$ (derived from call metadata: high transaction value $\ge ₹1,00,000$, foreign country code, prior CRM fraud score).

The logit $z$ is computed as the linear combination of weighted forensic evidence plus intercept:

$$z = \beta_0 + w_{\text{cm}} \cdot x_{\text{cm}} + w_{\text{asv}} \cdot x_{\text{asv}} + w_{\text{prosody}} \cdot x_{\text{prosody}} + w_{\text{channel}} \cdot x_{\text{channel}} + w_{\text{context}} \cdot x_{\text{context}}$$

The calibrated posterior probability of the stream being synthetic or an attack is given by the logistic sigmoid:

$$P(\text{spoof} \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

#### Default Calibration Weights (`FusionConfig`)
- Intercept ($\beta_0$ / `VR_FUSION_BIAS`): $-1.60$
- Countermeasure weight ($w_{\text{cm}}$ / `VR_FUSION_W_CM`): $2.40$
- ASV verification weight ($w_{\text{asv}}$ / `VR_FUSION_W_ASV`): $1.60$
- Prosody weight ($w_{\text{prosody}}$ / `VR_FUSION_W_PROSODY`): $0.70$
- Channel anomaly weight ($w_{\text{channel}}$ / `VR_FUSION_W_CHANNEL`): $0.90$

These apply to the default zero-dependency stub detectors and were hand-picked, not fit to any
measured distribution — `scripts/fit_fusion_weights.py` says so in its own header. When real
detector backends are active (`VR_ASV_BACKEND=ecapa`, `VR_CM_BACKEND=wavlm` — see the Production
Upgrade Path table below), `config.py`'s `_REAL_BACKEND_OVERRIDES` auto-applies $w_{\text{cm}}=1.5$,
$w_{\text{asv}}=2.0$, refit against measured real/synthetic score distributions via that script.
$w_{\text{prosody}}$ and $w_{\text{channel}}$ are deliberately left unchanged even then — the
fitting exercise found those two features confounded with recording-source differences unrelated
to genuine-vs-spoofed in the available calibration data, not a signal worth encoding. Full
methodology, the regularization-strength sensitivity found, and the honest partial result (2/4
demo clips match their expected tier with real backends active) are in `archive/AUDIT_REPORT.md` §12.
- Context risk weight ($w_{\text{context}}$ / `VR_FUSION_W_CONTEXT`): $0.50$

The raw window risk score is scaled to the standard percentage range:
$$\text{risk}_{\text{raw}} = 100.0 \times P(\text{spoof} \mid \mathbf{x})$$

---

### Exponential Moving Average (EMA) with Fast-Rise Override

Real-world call audio contains transient glitches, packet drops, or brief coughing fits. An Exponential Moving Average (EMA) smooths scores across sliding windows to prevent spurious alerts during benign conversations.

However, against high-confidence real-time voice cloning attacks, smoothing would introduce hazardous alert delay. To counter this, VaniRakshak implements a **Fast-Rise Override**:

```python
# vanirakshak/risk/fusion.py — RiskFusionEngine.update()
p = calibrate(window, context_risk, self.fusion_cfg, semantic_risk)
if p > self.risk_cfg.fast_rise_threshold:
    # Decisive spoof posterior: bypass EMA smoothing immediately
    risk = 100.0 * p
    self._ema = risk
else:
    # Standard temporal smoothing
    risk = 100.0 * p
    if self._ema is None:
        self._ema = risk
    else:
        self._ema = self.risk_cfg.ema_alpha * risk + (1.0 - self.risk_cfg.ema_alpha) * self._ema
        risk = self._ema

risk = float(max(0.0, min(100.0, risk)))
```

Mathematically:

$$\text{risk}_t = \begin{cases} 
100 \cdot P(\text{spoof} \mid \mathbf{x}_t) & \text{if } P(\text{spoof} \mid \mathbf{x}_t) > \tau_{\text{fast-rise}} \quad (\text{Fast-Rise Override}) \\
\alpha \cdot (100 \cdot P(\text{spoof} \mid \mathbf{x}_t)) + (1 - \alpha) \cdot \text{EMA}_{t-1} & \text{otherwise} 
\end{cases}$$

where default smoothing factor $\alpha = 0.65$ (`VR_EMA_ALPHA`) and
$\tau_{\text{fast-rise}} = 0.85$ (`VR_FAST_RISE_THRESHOLD`) — the trigger is
the **fused posterior probability**, not a raw CM log-likelihood-ratio
threshold on its own; an earlier version of this document (and this section,
until corrected) described the trigger as `LLR_cm < -2.0`, which is not what
the code checks. `docs/07-privacy-compliance-and-apis.md`'s environment
table has always had this right.

**A second, separate override exists that this section did not previously
document:** a deterministic **impostor override**
(`VR_IMPOSTOR_BLOCK_BELOW`, default `0.10`). When `window.asv_score` — the
ASV consistency score — falls below this threshold, the tier is forced to
`BLOCK` regardless of the fused risk score, and both `risk` and the EMA are
floored at the `CHALLENGE`/`BLOCK` boundary so the reported score and tier
never contradict each other. This only engages when ASV actually ran
(`metrics.asv_evaluated`) — an unenrolled/unclaimed caller's neutral `0.5`
score is never below the threshold. See `docs/PROTOCOL.md` §8 for the full
client-facing implication: a client deriving its own tier from `risk_score`
alone can disagree with the server in exactly this case.

---

## 5. Decision Tiers & Transaction Interlock

Every window yields a final risk score $\in [0, 100]$ mapped into three deterministic operational tiers:

| Tier | Risk Threshold | System State & Action | Transaction State |
|---|---|---|---|
| **ALLOW** | $\text{risk} < 35.0$ | Genuine biometric characteristics. Natural vocal tract micro-tremors verified. | **Permitted**. Banking operations and fund transfers execute unimpeded. |
| **CHALLENGE** | $35.0 \le \text{risk} < 70.0$ | Suspicious acoustic characteristics or speaker mismatch. Emits dynamic phonetic challenge phrase. | **Held**. Further transaction progression suspended until challenge verification. |
| **BLOCK** | $\text{risk} \ge 70.0$ | High-confidence voice clone / vocoded replay attack. Latches `interlock_active = True`. | **Terminated**. Banking transaction hard-locked; outbound alert webhook dispatched. |

### The Transaction Interlock Latch

The banking transaction interlock (`vanirakshak.session.CallSession`) operates as a **unidirectional security latch**:
- Once an analysis window triggers the **BLOCK** tier ($\text{risk} \ge 70$), `self._blocked` is set to `True`.
- It remains latched for the remainder of the session; subsequent lower scores cannot release the interlock.
- A failed challenge (`grade_challenge` failure) immediately escalates the session to **BLOCK** and triggers the latch.
- A passed challenge never releases the latch or re-arms a further challenge (one-shot gate: grading is digit-order + latency only, no voice biometrics).
- Outbound alert payload is automatically generated with cryptographic receipt hash and dispatched via `AlertDispatcher` to `VR_ALERT_WEBHOOK_URL`.

---

## 6. Synthetic Stubs vs. Production Deep Learning Models

To ensure technical integrity and honest engineering reporting, the following table clarifies the state of model components:

| Component | Current Implementation in Codebase | Production Upgrade Path (Drop-in Ready) |
|---|---|---|
| **Countermeasure (CM)** | Default: deterministic spectral Welch PSD feature extraction, pitch zero-crossing CV, and linear log-likelihood scoring (`vanirakshak.detectors.cm.CMModel`). | **Implemented.** `VR_CM_BACKEND=aasist` → `detectors.cm_aasist.CMModelAASIST`, real `microsoft/wavlm-base-plus` features feeding a trained AASIST-L graph head (`data/cm_aasist.pt`, 89k params) trained with full stochastic codec/telephony augmentation on ASVspoof 5 Track 1 Train partition (182,357 utterances). Measured ASVspoof 5 Dev partition performance: pooled **EER = 16.415 %**, **minDCF = 0.9912**, **actDCF = 3.2810**, calibrated $C_{\text{llr}} = 0.6536$. Measured ASVspoof 5 Eval partition performance across held-out attacks A17–A32 (`reports/cm_aasist_eval_partition.json`, 20k slice): raw **minDCF = 0.8048** (improves from pre-aug 0.8834), raw **EER = 13.627 %** (pre-aug 12.878%), calibrated **$\text{actDCF}_{\text{Bayes}} = 1.3795$**, **$C_{\text{llr}} = 0.7204$**. Per-attack eval EERs range from 2.93% (A29) to 26.22% (A28). Fully resolves YouTube compression acoustic domain shift: authentic speech (`demo_01_genuine.wav`) is classified as **ALLOW** in the production sliding-window path (Risk **31.6**, mean LLR **+2.55**, 3.4 points below the CHALLENGE threshold; center-cropped 10s yields Risk **24.0**, LLR **+3.85**, ASV 0.901). Impostor speech (`demo_04_impostor.wav`, an authentic human voice) is rejected as **BLOCK** (Risk 70.0) via the hard ASV biometric override (`asv_consistency ~ 0.011 < 0.10`). Falls back to `VR_CM_BACKEND=wavlm` or default DSP stub if checkpoint is absent. |
| **Speaker Verification (ASV)** | Default: 192-dim deterministic random projection from standardized log-mel spectral energy (`vanirakshak.detectors.asv.ASVModel`). | **Implemented.** `VR_ASV_BACKEND=ecapa` → `detectors.asv_ecapa.ASVModelECAPA`, real SpeechBrain `ECAPA-TDNN` (192-dim) via `backend/app/models/ecapa_verifier.py`, same `_embed_raw`/`consistency_score` contract. Real anchor: `data/anchor_asv_embedding_ecapa.json` via `scripts/enroll_anchor.py --backend ecapa` — measured separation 0.845 mean anchor vs. 0.199 mean other-speaker, zero overlap (`archive/AUDIT_REPORT.md` §12). |
| **Channel Detector** | Multi-band Welch power ratio partition (`vanirakshak.detectors.channel`). | Pretrained CNN/ResNet-18 channel codec classifier. |
| **Prosody Extractor** | IIR Butterworth pitch-band filtering and zero-crossing interval variance (`vanirakshak.detectors.prosody`). | PyWorld / CREPE / TorchAudio pitch and shimmer/jitter estimation. |
| **Buffering & Fusion** | Fully functional production-grade DPDP-compliant sliding ring buffer and Bayesian fusion engine. | Direct reuse; zero changes required. |

The contracts, interfaces, data formats, and Bayesian fusion mathematical formulations are identical between stubs and neural models.
