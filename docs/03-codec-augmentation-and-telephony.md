# Codec Augmentation & Telephony Simulation

> **⚠️ DEPRECATED FOR TRAINING — read before using this stack.**
>
> The codec/telephony augmenter is **OFF by default in `ml/train_cm.py`**
> since 2026-09-21, and this document's rationale no longer holds for this
> corpus.
>
> - **docs/20** (matched on/off control): the augmenter is a net loss —
>   **not** a trade-off. Turning it off improved ITW `full_call` by
>   10.56 / 2.38 pp *and* in-domain pooled EER by 6.20 / 2.77 pp.
> - **docs/21** (dose sweep 0.0 → 1.0): **no interior optimum (`AT_0`)**. No
>   partial strength beats "off". The curve is not even monotone — it peaks
>   at dose 0.50 and improves at 1.0, so half-measures are the worst option.
>
> **Why the original rationale failed here:** ASVspoof 5 was *already*
> captured over real telephone networks, so `clean` in this corpus **is** the
> call condition. Simulating codecs on top double-transcodes the audio and
> moves it off-distribution relative to both eval and In-the-Wild data. The
> augmenter teaches invariance to distortions the deployment signal does not
> contain.
>
> Pass `--codec-augment` to re-enable for research. Do not train a shipping
> candidate with it. The rationale below is kept for provenance (ground
> rule 4) and may still apply to corpora that were *not* phone-captured.

## 1. Overview & Context

In voice fraud attacks targeting Indian financial institutions, voice clones rarely arrive as pristine, uncompressed 48 kHz studio master recordings. Attackers route synthetic audio through:
- Cellular carrier networks (2G/3G GSM, 4G VoLTE, 5G VoNR)
- Legacy Public Switched Telephone Networks (PSTN) and PBX switches
- VoIP softphones, SIP trunks, and Session Border Controllers (SBCs)
- OTT VoIP applications (WhatsApp Call, Telegram, Skype)
- Physical room acoustics, speakerphones, and noisy ambient environments (crowded call centers, traffic, cafes)

To guarantee that forensic detectors remain robust under real-world degradation, VaniRakshak incorporates an acoustic channel simulation and data augmentation stack located in [`vanirakshak-backend/vanirakshak/data/augment.py`](file:///d:/VaniRakshak/vanirakshak-backend/vanirakshak/data/augment.py).

---

## 2. Telephony Simulation & Augmentation Primitives

The augmentation engine implements mathematical models of standard telecom compression, filtering, acoustic reverberation, and environmental noise injection.

### 2.1. G.711 A-law Companding (`alaw_encode_decode`)
- **Telecom Standard:** ITU-T Recommendation G.711 A-law (dominant standard throughout India, Europe, and the Commonwealth).
- **Physical Phenomenon:** Converts 16-bit linear PCM into an 8-bit logarithmic representation with parameter $A = 87.6$. This increases the dynamic range of speech over low-bitrate 64 kbps telephone channels at the cost of non-linear quantization distortion and harmonic spreading.
- **Formulation:**
  For normalized input $x \in [-1.0, 1.0]$ with magnitude $|x|$:

  $$F(x) = \operatorname{sgn}(x) \cdot \begin{cases} 
  \dfrac{A |x|}{1 + \ln(A)} & \text{for } |x| < \dfrac{1}{A} \\[8pt]
  \dfrac{1 + \ln(A |x|)}{1 + \ln(A)} & \text{for } \dfrac{1}{A} \le |x| \le 1 
  \end{cases}$$

- **Implementation:** Vectorized piecewise evaluation using `numpy.where`, clipping normalized amplitudes to $[-1.0, 1.0]$.

### 2.2. G.711 μ-law Companding (`ulaw_encode_decode`)
- **Telecom Standard:** ITU-T Recommendation G.711 μ-law (dominant in North America and Japan, frequently encountered on international banking calls and international SIP trunks).
- **Physical Phenomenon:** Non-linear companding with parameter $\mu = 255.0$.
- **Formulation:**

  $$F(x) = \operatorname{sgn}(x) \cdot \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}$$

- **Implementation:** Vectorized evaluation in `augment.py` producing characteristic logarithmic quantization noise.

### 2.3. AMR-WB Simulation (`amr_wb_simulate`)
- **Telecom Standard:** 3GPP TS 26.171 / ITU-T G.722.2 (Adaptive Multi-Rate Wideband / "HD Voice" on 3G/4G networks).
- **Physical Phenomenon:** Expands voice transmission bandwidth beyond narrow PSTN limits to $50\text{ Hz} - 7000\text{ Hz}$, followed by sub-band quantization artifacts.
- **Simulation Strategy:**
  1. **Band-limiting:** 4th-order Butterworth bandpass filter with cutoff frequencies $f_{\text{low}} = 50.0\text{ Hz}$ and $f_{\text{high}} = 7000.0\text{ Hz}$.
  2. **Sub-band Quantization:** Lightweight 4-bit amplitude quantization (16 discrete amplitude levels):
     $$y_{\text{quantized}} = \frac{\operatorname{round}\left(y \cdot \frac{\text{levels}}{2}\right)}{\frac{\text{levels}}{2}}, \quad \text{levels} = 16$$
- **Forensic Impact:** Simulates the harmonic structure and high-frequency rolloff of modern mobile carrier voice codecs without requiring non-free binary codecs.

### 2.4. GSM-EFR Simulation (`gsm_efr_simulate`)
- **Telecom Standard:** GSM 06.60 Enhanced Full Rate (classic PSTN toll-quality mobile cellular channel).
- **Physical Phenomenon:** Strict telephone bandpass limiting to $300\text{ Hz} - 3400\text{ Hz}$.
- **Formulation:** 4th-order Butterworth bandpass IIR filter implemented via second-order sections (`sosfilt`):
  $$H(s) = \prod_{k=1}^2 \frac{b_{0k} s^2 + b_{1k} s + b_{2k}}{s^2 + a_{1k} s + a_{2k}}$$
  with $f_{\text{low}} = 300\text{ Hz}$ and $f_{\text{high}} = 3400\text{ Hz}$.
- **Forensic Impact:** Eliminates chest resonance below 300 Hz and all high-frequency vocal fricatives/harmonics above 3400 Hz. Detectors must distinguish deepfakes solely from narrow-band vocal tract cues.

### 2.5. Additive Gaussian Noise (`add_noise`)
- **Physical Phenomenon:** Thermal line noise, analog transmission interference, and microphone pre-amplifier hiss.
- **Formulation:** Computes signal power $P_{\text{signal}} = \frac{1}{N} \sum_{n=1}^N x[n]^2 + \epsilon$, determines noise power from target Signal-to-Noise Ratio ($\text{SNR}_{\text{dB}}$):
  $$P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{\frac{\text{SNR}_{\text{dB}}}{10}}}$$
  Generates zero-mean Gaussian noise $w[n] \sim \mathcal{N}(0, \sigma^2)$ where $\sigma = \sqrt{P_{\text{noise}}}$, and outputs $y[n] = x[n] + w[n]$.

### 2.6. Babble Noise Simulation (`add_babble_simple`)
- **Physical Phenomenon:** Ambient speech interference from adjacent operators in high-density call centers or crowded public environments.
- **Formulation:** Simulates $N$ independent talkers (default $N=4$). For each talker:
  1. White Gaussian noise is generated.
  2. Bandpassed through a 2nd-order Butterworth filter ($200\text{ Hz} - 3500\text{ Hz}$) matching human speech spectral energy.
  3. The $N$ filtered streams are averaged and scaled to the target SNR (e.g. 5 dB or 15 dB) relative to the primary speaker.

### 2.7. Room Impulse Response (RIR) Convolution (`apply_rir`, `synthetic_rir`)
- **Physical Phenomenon:** Multi-path acoustic reflections occurring when a user or fraudster speaks on a speakerphone or in an untreated room.
- **RIR Generation (`synthetic_rir`):** Simulates an exponentially decaying diffuse reverberation tail parameterized by reverberation time $T_{60} \approx 250 - 280\text{ ms}$:
  $$h[n] = \mathcal{N}(0, 1) \cdot \exp\left(-\frac{n}{f_s \cdot T_{60} / 3}\right), \quad h[0] = 1.0$$
  normalized such that $\max(|h[n]|) = 1.0$.
- **Convolution:** High-speed Fast Fourier Transform (FFT) convolution via `scipy.signal.fftconvolve`:
  $$y[n] = (x * h)[n] = \sum_{k=0}^{L-1} x[k] h[n - k]$$
  truncated to original length $N$ to maintain sample alignment.

### 2.8. Random Gain Jitter (`random_gain`)
- **Physical Phenomenon:** Distance variations from handset microphone, caller movement, and telecom Automatic Gain Control (AGC) hunting.
- **Formulation:** Applies uniform random amplitude scaling $g_{\text{dB}} \sim \mathcal{U}(-6.0, +6.0\text{ dB})$:
  $$y[n] = x[n] \cdot 10^{\frac{g_{\text{dB}}}{20}}$$

---

## 3. Augmentation Registry & Pipeline Execution

Augmentations are registered and dispatched through a unified registry pattern in `vanirakshak/data/augment.py`:

```python
AUGMENT_REGISTRY = {
    "alaw": AugmentSpec("G.711 a-law", alaw_encode_decode),
    "ulaw": AugmentSpec("G.711 μ-law", ulaw_encode_decode),
    "amr_wb": AugmentSpec("AMR-WB (simulated)", None),
    "gsm_efr": AugmentSpec("GSM-EFR (PSTN 300-3400 Hz)", None),
    "babble_5db": AugmentSpec("Babble @ 5 dB SNR", None),
    "babble_15db": AugmentSpec("Babble @ 15 dB SNR", None),
    "rir_office": AugmentSpec("Office RIR", None),
    "gain_jitter": AugmentSpec("Random gain ±6 dB", None),
}
```

The composite evaluation pipeline applies these augmentations deterministically using `apply_random_pipeline`:

```python
y = apply_random_pipeline(
    x,
    sr=16000,
    augments=("alaw", "ulaw", "amr_wb", "gsm_efr", "babble_10db", "rir_office", "gain_jitter"),
    seed=42,
)
```

---

## 4. Pure NumPy / SciPy Architecture: Portability on Stage & Laptop

A conscious design decision in VaniRakshak is that the entire R1 augmentation stack runs in **pure NumPy and SciPy**:

| Benefit | Explanation |
|---|---|
| **Zero GPU Dependency** | Runs on standard developer laptops, edge VMs, and headless CI test runners. |
| **No External C/C++ Binaries** | Bypasses complex compilation requirements for external libraries like `libsox`, `ffmpeg`, or proprietary AMR-WB 3GPP reference C-code. |
| **Deterministic Testing** | Standardized seeds (`numpy.random.default_rng(seed)`) produce bit-for-bit identical waveforms across Windows, macOS, and Linux. |
| **Microsecond Speed** | A 4-second audio chunk passes through companding, filtering, and babble injection in under $2.5\text{ ms}$ on a laptop CPU. |

---

## 5. Release 2 (R2) Roadmap: Bit-Exact Codecs

While the R1 NumPy simulations accurately reflect frequency band-limiting, non-linear quantization noise, and reverberation, Release 2 (R2) will introduce optional high-fidelity bitstream codecs:

1. **Native Opus Encoding / Decoding:**
   - Integration via `torchaudio.io.StreamWriter` / `opusfile`.
   - Variable Bitrate (VBR) compression between 6 kbps and 24 kbps to simulate WebRTC and WhatsApp voice note degradation.
2. **MP3 / AAC Psych-Acoustic Masking:**
   - Integration via `sox` bindings to test forensic resistance against psycho-acoustic perceptual sub-band masking.
3. **Reference 3GPP AMR-NB / AMR-WB Codec:**
   - Exact bit-level compliance with 3GPP TS 26.073 / TS 26.173 fixed-point source code.
