# Project: WavLM Layer 5 AASIST Anti-Spoofing Countermeasure

> **⚠️ STALE — do not use this file for status.** The milestone table below
> still says M1 IN_PROGRESS / M2–M4 PLANNED and describes online codec
> augmentation as part of the training pipeline. Both are out of date.
>
> **Current status lives in `docs/17`–`docs/21`** (in this repo). Summary:
>
> - **Codec/telephony augmentation is OFF** by default in `ml/train_cm.py`
>   and should be removed. docs/20 measured it as a net loss on *both*
>   domains; docs/21 swept the dose axis and found **no interior optimum
>   (`AT_0`)**. See docs/03's deprecation banner.
> - **Ship `data/cm_docs20_armb_s42_cal.pt`** (arm B seed 42, calibrated).
>   It is wired as the backend default and verified by
>   `scripts/verify_cm_deployment.py`. Confirmed best of three arm-B seeds on
>   the deployment distribution (docs/20 §"Which arm B seed to deploy").
> - **Select on the deployment distribution (ITW / telephony), never on
>   in-domain pooled EER.** Three separate experiments have now shown
>   in-domain ranking is actively misleading.
> - **Out-of-domain seed spread is up to 8.26 pp**, so only *paired,
>   within-seed* OOD comparisons are trustworthy (docs/10).
> - Layer 5 vs 12: `--merge-layers 12` is mandatory for comparability
>   (docs/10); `layer_index` is not the operative setting.

## Architecture
- **Corpus & Audio Streaming Layer**: Audio loaded on-the-fly from `A:\cm-corpus\asvspoof5\flac_T\` (train), `flac_D\` (dev), and `flac_E_eval\` (20k eval benchmark slice in `data/eval_benchmark/eval_slice_20k.tsv`). Zero feature arrays written to Drive A to respect the ~7.96 GB free space budget.
- **Frontend Layer**: `microsoft/wavlm-base-plus` extracting intermediate Transformer representation at index 5 (`outputs.hidden_states[5]`), capturing fine spectro-temporal artifacts prior to higher-layer semantic smoothing (Pan et al., Interspeech 2024, R10).
- **Countermeasure Model**: PyTorch `AASISTLHead(in_dim=768)` in `ml/aasist_model.py` (~89k parameters) with graph attention and spectral/temporal pooling.
- **Training Pipeline**: `ml/train_cm.py` with `--layer-index 5`, online codec augmentation (`ml/augment.py`), class-weighted cross-entropy loss, AdamW optimizer with cosine LR scheduling, early stopping on validation minDCF. Export to `data/cm_aasist_layer5.pt`.
- **Evaluation Engine**: `ml/evaluate_cm.py` streaming audio waveforms to compute minDCF ($P_{\text{tar}}=0.05, C_{\text{miss}}=1.0, C_{\text{fa}}=1.0$), actDCF, CLLR, pooled EER, and per-attack breakdown (A17–A32).
- **System Integration**: `vanirakshak-backend/vanirakshak/detectors/cm_aasist.py` supporting `layer_index` from checkpoint dictionary with default `-1` fallback for regression-free backward compatibility.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Configurable Layer Extraction in Feature/Train Pipeline | Support `--layer-index` (default 5 for L5, -1 for L12) in `ml/train_cm.py` and `ml/evaluate_cm.py` | M1 | R1 |
| 2 | Configurable Layer Extraction in Backend Detector | Read `ckpt.get("layer_index", -1)` in `vanirakshak/detectors/cm_aasist.py` | M1 | R1, R4 |
| 3 | Zero-Disk Streaming Validation Strategy | Streaming dev validation in `ml/train_cm.py` respecting Drive A ~8.5 GB budget | M1 | R1 |
| 4 | AASIST Head Training on Layer 5 | Train AASIST-L on Layer 5 on RTX 5060 Ti GPU with class-weighted CE, AdamW, cosine LR, early stopping | M2 | R2 |
| 5 | Checkpoint Schema Validation | Validate `data/cm_aasist_layer5.pt` against required schema (architecture, frontend, state_dict, eer_threshold, dev_eer, dev_min_dcf, layer_index) | M2 | R2 |
| 6 | Unconfounded 20k Eval Benchmark Execution | Benchmark Layer 5 vs Pre-Aug vs Post-Aug on `data/eval_benchmark/eval_slice_20k.tsv` | M3 | R3 |
| 7 | Side-by-Side Metrics & Attack Breakdown | Compute & publish EER, minDCF, actDCF, CLLR, A17–A32 breakdown in `reports/` | M3 | R3 |
| 8 | Pan et al. (R10) & Attack A13 Analysis | Document Attack A13 error rate and minDCF shift in `docs/08-cm-training-plan.md` | M3 | R3 |
| 9 | Four-Gate Verification | Pass pytest backend, pytest e2e (95/95), integration_check (18/18), check_docs (0 hard failures) | M4 | R4 |
| 10 | Backend Layer 5 Detector Integration Verification | Verify detector loads `data/cm_aasist_layer5.pt` and runs integration checks under `VR_CM_BACKEND=aasist` | M4 | R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Layer 5 Extraction & Streaming Pipeline Infrastructure | Features 1, 2, 3 | none | IN_PROGRESS |
| M2 | AASIST Model Training on Layer 5 Representations & Validation | Features 4, 5 | M1 | PLANNED |
| M3 | Unconfounded Evaluation on 20k Eval Benchmark & Analysis | Features 6, 7, 8 | M2 | PLANNED |
| M4 | System Integration & Four-Gate Verification | Features 9, 10 | M3 | PLANNED |

## Interface Contracts
### Checkpoint Dictionary Contract
`data/cm_aasist_layer5.pt` must contain:
- `architecture`: `"aasist-l"`
- `frontend`: `"microsoft/wavlm-base-plus"`
- `sample_rate`: `16000`
- `window_s`: `4.0`
- `layer_index`: `5` (int)
- `state_dict`: Model weights matching `AASISTLHead(in_dim=768)`
- `eer_threshold`: float (decision threshold in logit/LLR space)
- `dev_eer`: float (validation EER)
- `dev_min_dcf`: float (validation minDCF)

### Evaluation TSV Contract (`data/eval_benchmark/eval_slice_20k.tsv`)
- Columns: `path\tlabel\tattack_id`
- Audio format: 16 kHz FLAC in `A:\cm-corpus\asvspoof5\flac_E_eval\`
- Total rows: exactly 20,000 utterances

### Backend Detector Contract (`cm_aasist.py`)
- `VR_CM_AASIST_WEIGHTS`: path to `.pt` file (defaults to `data/cm_aasist.pt` if unset)
- `CMModelAASIST`: extracts `outputs.hidden_states[self.layer_index]` where `self.layer_index = ckpt.get("layer_index", -1)`
- Output: LLR scalar transformed to risk score `risk = 100 / (1 + exp(LLR))`

## Code Layout
- `ml/train_cm.py`: Recipe A training script supporting `--layer-index` and streaming dev validation
- `ml/aasist_model.py`: AASIST-L head architecture
- `ml/evaluate_cm.py`: Benchmark evaluation script supporting `--layer-index`
- `vanirakshak-backend/vanirakshak/detectors/cm_aasist.py`: Production detector with checkpoint `layer_index` support
- `data/eval_benchmark/eval_slice_20k.tsv`: 20,000-utterance evaluation slice
- reports/ directory: Published side-by-side benchmark report (cm_layer5_vs_layer12_benchmark.json)
- `docs/08-cm-training-plan.md`: Documentation of training plan, Pan et al. hypothesis, and A13 analysis
