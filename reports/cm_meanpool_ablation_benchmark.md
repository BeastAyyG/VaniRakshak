# 5-Way Ablation Benchmark Report: Frozen Meanpool vs Attentive Merging vs Single Layers

Evaluation Partition: `data/eval_benchmark/eval_slice_20k.tsv` (20,000 utterances: 4,058 bona fide, 15,942 spoof across all 16 held-out attacks A17–A32).

## 1. Summary Metrics (Raw LLR Basis)

| Model | Layer / Architecture | EER (%) | minDCF | actDCF | actDCF_bayes | CLLR |
|---|---|---|---|---|---|---|
| **Pre-Aug Baseline** | Layer 12 (-1) | 12.878% | 0.8834 | 2.5757 | 6.4570 | 1.8171 |
| **Post-Aug Model** | Layer 12 (-1) | 13.627% | 0.8048 | 2.7249 | 10.2455 | 1.6238 |
| **Layer 5 (Pan et al. R10)** | Layer 5 | 14.096% | 0.8280 | 2.8190 | 12.1242 | 2.4901 |
| **Attentive Merging (12 Layers)** | AttentiveMerging [0:12] | 12.666% | 0.8804 | 2.5329 | 13.2390 | 3.2391 |
| **Frozen Meanpool (12 Layers)** | **Uniform Mean [0:12]** | **13.504%** | **0.9242** | **2.6998** | **13.6418** | **4.3025** |

## 2. Per-Attack Breakdown (EER %) on Held-Out Eval Attacks (A17–A32)

| Attack ID | Pre-Aug L12 (%) | Post-Aug L12 (%) | Layer 5 (%) | Attentive Merge 12L (%) | Frozen Meanpool 12L (%) |
|---|---|---|---|---|---|
| A17 | 5.347% | 3.869% | 1.984% | 1.134% | 1.257% |
| A18 | 4.707% | 5.347% | 8.034% | 5.002% | 4.876% |
| A19 | 9.044% | 21.316% | 30.828% | 20.214% | 20.823% |
| A20 | 6.407% | 23.189% | 30.951% | 24.104% | 24.321% |
| A21 | 13.085% | 9.177% | 3.524% | 4.929% | 5.988% |
| A22 | 14.191% | 11.542% | 6.481% | 5.495% | 7.001% |
| A23 | 8.748% | 6.654% | 9.942% | 8.880% | 9.170% |
| A24 | 29.333% | 17.846% | 8.034% | 5.150% | 6.051% |
| A25 | 6.506% | 12.543% | 10.098% | 6.037% | 5.784% |
| A26 | 4.633% | 7.615% | 7.048% | 6.982% | 6.580% |
| A27 | 7.097% | 12.488% | 16.387% | 19.181% | 19.147% |
| A28 | 32.257% | 26.220% | 12.945% | 16.008% | 18.950% |
| A29 | 5.067% | 2.932% | 0.931% | 0.827% | 0.827% |
| A30 | 6.407% | 9.038% | 17.989% | 14.662% | 14.327% |
| A31 | 14.810% | 13.221% | 15.530% | 15.845% | 17.743% |
| A32 | 11.237% | 17.227% | 20.999% | 24.396% | 26.737% |
