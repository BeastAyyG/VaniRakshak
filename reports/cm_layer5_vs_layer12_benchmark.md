# Side-by-Side Benchmark Report: WavLM Layer 5 vs Layer 12

Evaluation Partition: data/eval_benchmark/eval_slice_20k.tsv (20,000 utterances: 4,058 bona fide, 15,942 spoof across all 16 held-out attacks A17–A32).

## 1. Summary Metrics (Raw LLR Basis)

| Model | Layer | EER (%) | minDCF | actDCF | actDCF_bayes | CLLR |
|---|---|---|---|---|---|---|
| **Pre-Augmentation Baseline** | 12 (-1) | 12.878% | 0.8834 | 2.5757 | 6.4570 | 1.8171 |
| **Post-Augmentation Model** | 12 (-1) | 13.627% | 0.8048 | 2.7249 | 10.2455 | 1.6238 |
| **Layer 5 (Pan et al. R10)** | **5** | **14.096%** | **0.8280** | **2.8190** | **12.1242** | **2.4901** |

## 2. Per-Attack Breakdown (EER %) on Held-Out Eval Attacks (A17–A32)

| Attack ID | Pre-Aug Layer 12 (%) | Post-Aug Layer 12 (%) | Layer 5 (Pan et al.) (%) | Winner |
|---|---|---|---|---|
| A17 | 5.347% | 3.869% | 1.984% | Layer 5 |
| A18 | 4.707% | 5.347% | 8.034% | Pre-Aug |
| A19 | 9.044% | 21.316% | 30.828% | Pre-Aug |
| A20 | 6.407% | 23.189% | 30.951% | Pre-Aug |
| A21 | 13.085% | 9.177% | 3.524% | Layer 5 |
| A22 | 14.191% | 11.542% | 6.481% | Layer 5 |
| A23 | 8.748% | 6.654% | 9.942% | Post-Aug |
| A24 | 29.333% | 17.846% | 8.034% | Layer 5 |
| A25 | 6.506% | 12.543% | 10.098% | Pre-Aug |
| A26 | 4.633% | 7.615% | 7.048% | Pre-Aug |
| A27 | 7.097% | 12.488% | 16.387% | Pre-Aug |
| A28 | 32.257% | 26.220% | 12.945% | Layer 5 |
| A29 | 5.067% | 2.932% | 0.931% | Layer 5 |
| A30 | 6.407% | 9.038% | 17.989% | Pre-Aug |
| A31 | 14.810% | 13.221% | 15.530% | Post-Aug |
| A32 | 11.237% | 17.227% | 20.999% | Pre-Aug |
