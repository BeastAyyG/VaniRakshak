# 4-Way Side-by-Side Benchmark Report: Attentive Layer Merging vs Single Layer Countermeasures

Evaluation Partition: `data/eval_benchmark/eval_slice_20k.tsv` (20,000 utterances: 4,058 bona fide, 15,942 spoof across all 16 held-out attacks A17–A32).

## 1. Summary Metrics (Raw LLR Basis)

| Model | Layer / Architecture | EER (%) | minDCF | actDCF | actDCF_bayes | CLLR |
|---|---|---|---|---|---|---|
| **Pre-Augmentation Baseline** | Layer 12 (-1) | 12.878% | 0.8834 | 2.5757 | 6.4570 | 1.8171 |
| **Post-Augmentation Model** | Layer 12 (-1) | 13.627% | 0.8048 | 2.7249 | 10.2455 | 1.6238 |
| **Layer 5 (Pan et al. R10)** | Layer 5 | 14.096% | 0.8280 | 2.8190 | 12.1242 | 2.4901 |
| **Attentive Merging (12 Layers)** | **AttentiveMerging [0:12]** | **12.666%** | **0.8804** | **2.5329** | **13.2390** | **3.2391** |

## 2. Attentive Merging Layer Attention Distribution

Learned softmax layer attention weights across `hidden_states[0:12]`:
- **Layer  0**: 8.18%
- **Layer  1**: 9.31%
- **Layer  2**: 9.04%
- **Layer  3**: 9.49%
- **Layer  4**: 9.08%
- **Layer  5**: 7.62%
- **Layer  6**: 7.25%
- **Layer  7**: 6.97%
- **Layer  8**: 7.11%
- **Layer  9**: 7.39%
- **Layer 10**: 8.28%
- **Layer 11**: 10.28%

## 3. Per-Attack Breakdown (EER %) on Held-Out Eval Attacks (A17–A32)

| Attack ID | Pre-Aug L12 (%) | Post-Aug L12 (%) | Layer 5 (%) | Attentive Merge 12L (%) | Best Performing Architecture |
|---|---|---|---|---|---|
| A17 | 5.347% | 3.869% | 1.984% | 1.134% | **Attentive Merge** |
| A18 | 4.707% | 5.347% | 8.034% | 5.002% | Pre-Aug L12 |
| A19 | 9.044% | 21.316% | 30.828% | 20.214% | Pre-Aug L12 |
| A20 | 6.407% | 23.189% | 30.951% | 24.104% | Pre-Aug L12 |
| A21 | 13.085% | 9.177% | 3.524% | 4.929% | Layer 5 |
| A22 | 14.191% | 11.542% | 6.481% | 5.495% | **Attentive Merge** |
| A23 | 8.748% | 6.654% | 9.942% | 8.880% | Post-Aug L12 |
| A24 | 29.333% | 17.846% | 8.034% | 5.150% | **Attentive Merge** |
| A25 | 6.506% | 12.543% | 10.098% | 6.037% | **Attentive Merge** |
| A26 | 4.633% | 7.615% | 7.048% | 6.982% | Pre-Aug L12 |
| A27 | 7.097% | 12.488% | 16.387% | 19.181% | Pre-Aug L12 |
| A28 | 32.257% | 26.220% | 12.945% | 16.008% | Layer 5 |
| A29 | 5.067% | 2.932% | 0.931% | 0.827% | **Attentive Merge** |
| A30 | 6.407% | 9.038% | 17.989% | 14.662% | Pre-Aug L12 |
| A31 | 14.810% | 13.221% | 15.530% | 15.845% | Post-Aug L12 |
| A32 | 11.237% | 17.227% | 20.999% | 24.396% | Pre-Aug L12 |
