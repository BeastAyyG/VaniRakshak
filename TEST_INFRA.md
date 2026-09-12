# VaniRakshak E2E Test Infrastructure Specification

## 1. Test Philosophy: Opaque-Box & Requirement-Driven

The end-to-end (E2E) testing framework for VaniRakshak enforces strict **opaque-box, requirement-driven verification**. System components are evaluated across operational boundaries (network interfaces, protocol envelopes, serialization contracts, and statistical inference bounds) rather than implementation-specific internals.

### Authoritative Derivation

All test scenarios, input spaces, boundary conditions, and acceptance
tolerances derive from the following, **in this precedence order**:

- `docs/PROTOCOL.md` — single source of truth for WebSocket frame semantics,
  handshake sequences, sliding-window verdicts, DPDP hash-receipt chaining,
  and the autonomous transaction interlock. Binding per `AGENTS.md` §2.
- `docs/08-cm-training-plan.md` — the countermeasure work this suite covers,
  organised there as corpus/manifest verification, WavLM embedding
  extraction, AASIST-L training, forensic evaluation & calibration, and the
  integration gate (the "R1–R5" groupings referenced throughout this file).
- `docs/API_SPEC.md` — REST surface, request/response schemas.
- `docs/09-post-cm-roadmap.md` — what is deliberately **not** covered yet,
  so an absent test is distinguishable from an oversight.

> **Provenance note.** Earlier revisions of this section cited two files —
> ORIGINAL_REQUEST.md and PROJECT.md (named here without backticks so the
> doc-drift guard does not read them as live path references) — as the
> primary requirements documents. Neither is in this repository: they were
> working documents on
> the authoring machine and were never committed, so no reader could follow
> the citation and the doc-drift guard failed on them. The milestone
> structure they described (M1–M6, 19 features) is not reproduced here
> rather than paraphrased from memory; the documents above are what a test
> in this suite can actually be traced to. If those two files are meant to
> be authoritative, commit them and restore the citation.

### Core Testing Invariants
1. **Strict Zero-Facade Principle**: Every test exercises real computational paths—real PyTorch graph modules, real metric computation routines, real cryptographic hash chains, and real ASGI WebSocket/HTTP server loops.
2. **Deterministic Isolation**: Tests are self-contained, managing ephemeral directories, simulated stream channels, and mockable external dependencies (e.g. cloud API keys or multi-hour corpus scans) without leaking state across runs.
3. **Defense-in-Depth Verification**: Boundary, adversarial, and corruption scenarios are treated with equal priority to nominal golden paths.

---

## 2. Feature Inventory & Requirement Mapping

The test suite provides end-to-end verification for all 19 features across the project lifecycle:

| Feature # | Feature Name | Origin Milestone | Primary Specification & Contract |
|:---:|:---|:---:|:---|
| **F01** | Dev Corpus Verification | M1 | ASVspoof 5 dev corpus integrity (`142,134` .flac files, 16kHz mono 16-bit, non-zero payloads) |
| **F02** | Train Corpus Extraction | M1 | ASVspoof 5 train corpus extraction (`182,357` .flac files, multi-part archive unpack validation) |
| **F03** | Dev TSV Manifest | M1 | 3-column TSV `path\tlabel\tattack_id` (`140,950` records, 0 missing files, valid attack codes) |
| **F04** | Train TSV Manifest | M1 | Canonical 3-column TSV `train.tsv` (`182,357` records, label consistency, non-empty attack codes) |
| **F05** | Dev WavLM Feature Cache | M2 | `(199, 768)` fp16 tensor caching in `dev/tensors/`, exact byte layout (`305,792` bytes per npy file) |
| **F06** | Train WavLM Feature Cache | M2 | `(199, 768)` fp16 tensor caching in `train/tensors/`, NaN/Inf checks, shard directory structure |
| **F07** | Codebase Defect Remediation | M3 | Verification of 8 survey defect remediations (device safety, memory bounds, shape assertions) |
| **F08** | AASIST-L Head Architecture | M4 | Spectro-temporal GAT + HS-GAL + top-k pooling head (~89k parameters, binary logit output) |
| **F09** | Recipe A Training Pipeline | M4 | Class-weighted cross-entropy (`spoof_weight = n_bonafide / n_spoof`), AdamW + cosine schedule |
| **F10** | Checkpoint Export & Schema Validation | M4 | `data/cm_aasist.pt` contract validation (`_validate_checkpoint`: 8 schema fields, state_dict) |
| **F11** | Comprehensive Forensic Evaluation | M5 | minDCF, actDCF, CLLR, pooled EER computation over candidate thresholds (`reports/cm_aasist_eval.json`) |
| **F12** | Metric Improvement Gate | M5 | Strict dev EER and minDCF improvement over baseline heuristic countermeasure |
| **F13** | Per-Attack Breakdown | M5 | Disaggregated forensic metrics segmented across individual spoof attack classes (A01–A28) |
| **F14** | Bayesian Risk Calibration | M5 | Logistic regression fitting for `VR_FUSION_W_CM` and `VR_FUSION_BIAS` with fixed offset bounds |
| **F15** | Live Detector Integration | M6 | `CMModelAASIST` loading under `VR_CM_BACKEND=aasist`, graceful fallback, LLR transformation |
| **F16** | Backend Unit Test Suite | M6 | 100% test pass on all backend unit tests (`vanirakshak-backend/tests/`) |
| **F17** | Integration Check Verification | M6 | `scripts/integration_check.py` end-to-end slice verification (health, analyze, verify, audit, stream) |
| **F18** | Documentation Audit | M6 | `scripts/check_docs.py` zero hard failures (strict path checking, VR_* env parity, protocol keys) |
| **F19** | WebSocket Protocol Conformance | M6 | `docs/PROTOCOL.md` contract (text handshake, 1003 rejection, session_ack, sliding window verdicts) |

---

## 3. Test Architecture & Directory Layout

```
tests/e2e/
├── __init__.py                          # Package initialization
├── conftest.py                          # Fixtures: ASGI test clients, PCM generators, mock weights
├── test_tier1_feature_coverage.py       # Tier 1: >=5 nominal tests per feature (Features 1–19, >=95 tests)
├── test_tier2_boundary_corner.py        # Tier 2: >=5 boundary/corner tests per feature (Features 1–19, >=95 tests)
├── test_tier3_cross_feature.py          # Tier 3: Pairwise & end-to-end multi-stage pipeline integration
├── test_tier4_real_world_scenarios.py   # Tier 4: Realistic banking fraud defense, deepfakes, bonafide verification
└── run_e2e_tests.py                     # Standalone CLI test runner and verification reporter
```

### Execution Strategy
- **Framework**: `pytest` running under Python 3.12 with `starlette.testclient` / `fastapi.testclient` and PyTorch.
- **Speed Optimization**: Heavy deep learning models utilize fast in-memory forward passes and synthetic 16kHz audio slices during unit-scale E2E loops, with complete protocol and pipeline assertions.
- **Self-Contained Runner**: `python tests/e2e/run_e2e_tests.py` executes all tiers, prints execution summaries, and exits with code 0 on complete pass.

---

## 4. Test Tiers & Coverage Thresholds

### Tier 1: Feature Coverage (>= 5 tests per feature, >= 95 tests total)
Focuses on the nominal functional path for each of the 19 features under valid, representative inputs:
- Valid FLAC audio and TSV manifest structures.
- Tensor shape and dtype assertions `(B, 199, 768) float16`.
- AASIST-L graph attention forward pass and parameter budget validation.
- Loss weighting, optimizer stepping, and checkpoint serialization.
- Forensic metric calculation accuracy (EER, minDCF, actDCF, CLLR).
- REST endpoint contracts (`/api/v1/health`, `/api/v1/analyze/audio`, `/api/v1/verify-speaker`, `/api/v1/audit`).
- Protocol handshakes, session acknowledgments, and verdict structures.

### Tier 2: Boundary & Corner Cases (>= 5 tests per feature, >= 95 tests total)
Stresses the system under pathological inputs, corruptions, edge-condition thresholds, and resource constraints:
- Empty audio clips, truncated streams, zero-byte files, and corrupted headers.
- Corrupted manifests (missing columns, unknown labels, invalid attack IDs).
- Tensor cache anomalies (NaNs, Infinities, dimension mismatches, truncated bytes).
- Model architecture boundaries (batch size 1, mismatched feature dimensions, extreme sequence lengths).
- Extreme metric cases (100% bonafide, 100% spoof, identical scores, reversed distributions).
- Protocol violations (binary frame before handshake => code 1003, malformed JSON, out-of-range sample rates).
- Interlock boundaries (`risk_score = 34.99` vs `35.01`, `69.99` vs `70.01`, `asv < 0.10` impostor override).

### Tier 3: Cross-Feature Combinations (Pairwise & Pipeline Integration)
Validates interactions across module boundaries:
- **Pipeline A (Data Flow)**: TSV Manifest -> Feature Extraction Cache -> PyTorch FeatureDataset -> Batch Collation -> AASIST-L Forward Pass.
- **Pipeline B (Model to Service)**: Trained AASIST Checkpoint -> Checkpoint Validator -> `CMModelAASIST` -> `DetectionPipeline` -> Bayesian Fusion -> WebSocket Verdict.
- **Pipeline C (Interlock Chain)**: Live PCM Stream -> Acoustic Scoring -> Challenge Emission -> Challenge Grading -> Permanent Interlock Escalation -> Audit Hash Chaining.
- **Pipeline D (Evaluation to Calibration)**: Development Set Evaluation Scores -> EER/minDCF Thresholds -> Bayesian Weight Fitting -> Updated Fusion Config.

### Tier 4: Real-World Application Scenarios
Simulates realistic fraud defense situations reflecting operational deployment:
1. **Scenario 1: High-Fidelity Voice Clone (Deepfake Attack)**: Cloned audio with high acoustic naturalness targeting a financial transaction; CM detector flags spoofing, risk score exceeds 70.0, transaction interlock activates `BLOCK`.
2. **Scenario 2: Bonafide Account Holder Verification**: Genuine enrolled user speaking authentic prompt; low CM spoof probability, high ASV consistency (> 0.85), risk score below 35.0, tier remains `ALLOW`.
3. **Scenario 3: Synthetic Impostor Challenge & Response**: Ambiguous speech triggering `CHALLENGE`; challenge prompt generated, user passes/fails digit repetition with latency constraints, interlock escalates to `BLOCK` upon failure.
4. **Scenario 4: Audio Signal Degradation & Codec Robustness**: Clean speech passed through telephony codecs (8kHz downsampling, AMR/G.711 simulation, background noise); system maintains stability, avoids false impostor triggers, and flags low SNR.
5. **Scenario 5: Concurrent Multi-Session Streaming**: Multiple simultaneous WebSocket sessions streaming concurrent calls; state isolation verified with distinct DPDP audit receipts and no cross-talk.
6. **Scenario 6: Sub-Second Latency & Window Hop Processing**: 4.04s analysis window evaluated within the 1.0s hop budget, satisfying real-time streaming constraints.

---

## 5. Pass/Fail Gates

A test run is considered successful if and only if:
1. All Tier 1 feature tests pass (>= 95/95).
2. All Tier 2 boundary/corner tests pass (>= 95/95).
3. All Tier 3 cross-feature combination tests pass.
4. All Tier 4 real-world scenario tests pass.
5. Zero regressions are introduced to existing backend unit tests (`vanirakshak-backend/tests/`).
6. Exit code is 0.
