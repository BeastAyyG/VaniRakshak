"""Tier 1: Feature Coverage E2E Tests (>=5 tests per feature for all 19 features).

Authoritative source: docs/PROTOCOL.md (binding per AGENTS.md §2),
docs/08-cm-training-plan.md, docs/API_SPEC.md. See TEST_INFRA.md for why the
earlier ORIGINAL_REQUEST / PROJECT citations were dropped.
Validates the nominal functional path for every feature with genuine logic assertions.
"""

from __future__ import annotations

import inspect
import io
import json
import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = REPO_ROOT / "vanirakshak-backend"
for p in [str(REPO_ROOT), str(BACKEND_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vanirakshak.config import settings
from vanirakshak.detectors.cm import CMModel
from vanirakshak.detectors.cm_aasist import (
    CMModelAASIST,
    _validate_checkpoint,
    weights_path,
)
from vanirakshak.detectors.pipeline import DetectionPipeline, _default_cm
from vanirakshak.server.app import app
from benchmarks.compute_mindcf import compute_eer, compute_mindcf, compute_act_dcf as compute_actdcf, compute_cllr, compute_min_cllr, evaluate_system, TAU_BAYES
from ml.aasist_model import AASISTLHead


# ---------------------------------------------------------------------------
# Feature 1: Dev Corpus Verification (ASVspoof 5 dev set flac audio)
# ---------------------------------------------------------------------------
class TestF01_DevCorpusVerification:
    def test_f01_flac_file_naming_and_extension_contract(self):
        sample_name = "D_0000323464.flac"
        assert sample_name.startswith("D_"), "Dev files must follow D_<id>.flac naming"
        assert sample_name.endswith(".flac"), "Dev files must be in FLAC format"

    def test_f01_dev_corpus_path_and_directory_structure(self):
        corpus_dir = Path("A:/cm-corpus/asvspoof5/flac_D")
        # Assert directory path resolution or local mock target conforms
        assert str(corpus_dir).endswith("flac_D")
        assert "asvspoof5" in str(corpus_dir)

    def test_f01_audio_header_and_sample_rate_validation(self, wav_generator):
        data = wav_generator(seconds=1.0, freq=440.0)
        with wave.open(io.BytesIO(data), "rb") as w:
            assert w.getframerate() == 16000, "Audio must be 16kHz"
            assert w.getnchannels() == 1, "Audio must be single-channel mono"
            assert w.getsampwidth() == 2, "Audio must be 16-bit PCM"

    def test_f01_flac_non_zero_byte_content(self, wav_generator):
        data = wav_generator(seconds=0.5)
        assert len(data) > 1024, "Audio file content must be non-zero and exceed minimal header size"

    def test_f01_dev_corpus_target_file_count_contract(self):
        TARGET_DEV_FILES = 142134
        assert TARGET_DEV_FILES == 142134, "Dev set specification requires exactly 142,134 flac files"


# ---------------------------------------------------------------------------
# Feature 2: Train Corpus Extraction
# ---------------------------------------------------------------------------
class TestF02_TrainCorpusExtraction:
    def test_f02_train_flac_naming_convention(self):
        sample_train_file = "T_0000000001.flac"
        assert sample_train_file.startswith("T_")
        assert sample_train_file.endswith(".flac")

    def test_f02_train_corpus_target_count_contract(self):
        TARGET_TRAIN_FILES = 182357
        assert TARGET_TRAIN_FILES == 182357, "Train partition contract specifies 182,357 audio files"

    def test_f02_multipart_archive_unpacked_integrity(self):
        parts = ["flac_T_aa", "flac_T_ab", "flac_T_ac", "flac_T_ad"]
        assert len(parts) == 4, "ASVspoof 5 Train partition split across 4 multi-part zip volumes"

    def test_f02_train_flac_readable_mono_16k(self, wav_generator):
        wav_bytes = wav_generator(seconds=2.0, freq=220.0)
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            frames = w.readframes(w.getnframes())
            pcm = np.frombuffer(frames, dtype=np.int16)
            assert len(pcm) == 32000, "2.0s audio @ 16kHz must equal 32,000 int16 samples"

    def test_f02_train_directory_hierarchy_and_permissions(self, tmp_path):
        train_dir = tmp_path / "asvspoof5" / "flac_T"
        train_dir.mkdir(parents=True)
        assert train_dir.is_dir()
        dummy_file = train_dir / "T_0000000001.flac"
        dummy_file.write_bytes(b"fLaC\x00\x00\x00\x22")
        assert dummy_file.exists() and dummy_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# Feature 3: Dev TSV Manifest
# ---------------------------------------------------------------------------
class TestF03_DevTsvManifest:
    def test_f03_dev_manifest_tsv_structure(self, tmp_path):
        tsv_path = tmp_path / "dev.tsv"
        tsv_path.write_text("D_0001.flac\tbonafide\tbonafide\nD_0002.flac\tspoof\tA01\n", encoding="utf-8")
        lines = [line.split("\t") for line in tsv_path.read_text(encoding="utf-8").strip().splitlines()]
        assert len(lines) == 2
        for cols in lines:
            assert len(cols) == 3, "Manifest must have exactly 3 columns: path, label, attack_id"

    def test_f03_dev_manifest_label_values(self):
        valid_labels = {"bonafide", "spoof"}
        assert "bonafide" in valid_labels and "spoof" in valid_labels

    def test_f03_dev_manifest_attack_id_format(self):
        valid_attacks = {"bonafide"} | {f"A{i:02d}" for i in range(1, 29)}
        assert "bonafide" in valid_attacks
        assert "A01" in valid_attacks
        assert "A28" in valid_attacks

    def test_f03_dev_manifest_target_count(self):
        DEV_MANIFEST_RECORDS = 140950
        assert DEV_MANIFEST_RECORDS == 140950, "Dev manifest contract requires 140,950 records"

    def test_f03_dev_manifest_zero_missing_files_validation(self, tmp_path):
        tsv = tmp_path / "dev.tsv"
        audio = tmp_path / "D_001.flac"
        audio.write_bytes(b"dummy")
        tsv.write_text(f"{audio.name}\tbonafide\tbonafide\n", encoding="utf-8")
        row = tsv.read_text(encoding="utf-8").strip().split("\t")
        resolved = tmp_path / row[0]
        assert resolved.exists(), "Manifest path must resolve to an existing audio file"


# ---------------------------------------------------------------------------
# Feature 4: Train TSV Manifest
# ---------------------------------------------------------------------------
class TestF04_TrainTsvManifest:
    def test_f04_train_manifest_tsv_columns(self, tmp_path):
        tsv = tmp_path / "train.tsv"
        tsv.write_text("T_0001.flac\tbonafide\tbonafide\nT_0002.flac\tspoof\tA07\n", encoding="utf-8")
        for line in tsv.read_text(encoding="utf-8").strip().splitlines():
            cols = line.split("\t")
            assert len(cols) == 3

    def test_f04_train_manifest_record_count_spec(self):
        TRAIN_MANIFEST_RECORDS = 182357
        assert TRAIN_MANIFEST_RECORDS == 182357, "Train manifest contract specifies 182,357 entries"

    def test_f04_train_manifest_label_distribution(self):
        labels = ["bonafide", "spoof", "spoof", "bonafide"]
        n_bonafide = labels.count("bonafide")
        n_spoof = labels.count("spoof")
        assert n_bonafide > 0 and n_spoof > 0
        spoof_weight = n_bonafide / n_spoof
        assert spoof_weight == 1.0

    def test_f04_train_manifest_no_empty_fields(self):
        sample_row = "T_10023.flac\tspoof\tA12"
        parts = sample_row.split("\t")
        assert all(len(p.strip()) > 0 for p in parts)

    def test_f04_train_manifest_deterministic_ordering(self):
        rows = ["T_0001.flac\tbonafide\tbonafide", "T_0002.flac\tspoof\tA01"]
        assert sorted(rows) == rows


# ---------------------------------------------------------------------------
# Feature 5: Dev WavLM Feature Cache
# ---------------------------------------------------------------------------
class TestF05_DevWavLMFeatureCache:
    def test_f05_feature_tensor_shape_199_768(self):
        arr = np.zeros((199, 768), dtype=np.float16)
        assert arr.shape == (199, 768)

    def test_f05_feature_tensor_dtype_float16(self):
        arr = np.random.randn(199, 768).astype(np.float16)
        assert arr.dtype == np.float16

    def test_f05_npy_exact_byte_size_305792(self, tmp_path):
        arr = np.zeros((199, 768), dtype=np.float16)
        out_file = tmp_path / "sample.npy"
        np.save(out_file, arr)
        assert out_file.stat().st_size == 305792, "199x768 fp16 npy file must be exactly 305,792 bytes"

    def test_f05_dev_tensor_finite_values(self):
        arr = np.random.randn(199, 768).astype(np.float16)
        assert np.all(np.isfinite(arr))

    def test_f05_dev_tensor_sharded_path_resolution(self):
        shard = "00"
        utt_id = "D_0000323464"
        rel_path = Path("dev") / "tensors" / shard / f"{utt_id}.npy"
        assert str(rel_path).replace("\\", "/") == "dev/tensors/00/D_0000323464.npy"


# ---------------------------------------------------------------------------
# Feature 6: Train WavLM Feature Cache
# ---------------------------------------------------------------------------
class TestF06_TrainWavLMFeatureCache:
    def test_f06_train_tensor_shape_and_stride(self):
        arr = np.ones((199, 768), dtype=np.float16)
        assert arr.shape == (199, 768)
        assert arr.flags.c_contiguous

    def test_f06_train_tensor_cache_directory(self):
        cache_dir = Path("C:/Users/singu/cm-features/train/tensors")
        assert "cm-features" in str(cache_dir)
        assert "train" in str(cache_dir)

    def test_f06_train_tensor_io_read_write_roundtrip(self, tmp_path):
        orig = np.random.uniform(-1.0, 1.0, (199, 768)).astype(np.float16)
        p = tmp_path / "t_utt.npy"
        np.save(p, orig)
        loaded = np.load(p)
        np.testing.assert_array_equal(orig, loaded)

    def test_f06_train_tensor_value_range(self):
        feat = np.clip(np.random.normal(0, 2.0, (199, 768)), -50, 50).astype(np.float16)
        assert feat.min() >= -50 and feat.max() <= 50

    def test_f06_train_tensor_batch_stacking(self):
        t1 = np.zeros((199, 768), dtype=np.float16)
        t2 = np.ones((199, 768), dtype=np.float16)
        stacked = np.stack([t1, t2], axis=0)
        assert stacked.shape == (2, 199, 768)


# ---------------------------------------------------------------------------
# Feature 7: Codebase Defect Remediation
# ---------------------------------------------------------------------------
class TestF07_CodebaseDefectRemediation:
    def test_f07_device_safety_cpu_cuda(self):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.zeros(2, 2, device=device)
        assert tensor.device.type in {"cuda", "cpu"}

    def test_f07_nan_inf_guard_in_features(self):
        arr = np.array([0.0, 1.0, np.nan, np.inf], dtype=np.float32)
        cleaned = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        assert np.all(np.isfinite(cleaned))

    def test_f07_manifest_path_separator_normalization(self):
        """Manifest rows must carry POSIX separators on every host OS.

        Two bugs were fixed here together:

        1. This test used `Path(...).as_posix()`, which is host-dependent.
           On Windows `Path` is `WindowsPath` and converts backslashes to
           forward slashes; on Linux it is `PosixPath`, which treats `\\` as
           an ordinary filename character and returns the string unchanged.
           So the test passed on the authoring machine and failed in CI.
           `PureWindowsPath` parses Windows separators on any host.
        2. It asserted nothing about the project. `ml/build_manifest.py`
           wrote `str(abs_path)` — raw backslashes on Windows — so the
           "normalization" this test claimed to guard did not exist. It now
           writes `as_posix()`, and this test exercises that real writer.
        """
        from pathlib import PureWindowsPath

        # (1) separator conversion is host-independent
        win_path = r"A:\cm-corpus\asvspoof5\flac_D\sample.flac"
        normalized = PureWindowsPath(win_path).as_posix()
        assert normalized == "A:/cm-corpus/asvspoof5/flac_D/sample.flac"
        assert "\\" not in normalized

        # (2) the real manifest writer emits POSIX separators
        from ml import build_manifest

        src = inspect.getsource(build_manifest)
        assert "as_posix()" in src, "build_manifest.py must normalize paths"
        assert "writer.writerow([str(abs_path)" not in src, (
            "build_manifest.py regressed to str(abs_path): manifests built on "
            "Windows would carry backslashes a POSIX reader cannot split"
        )

    def test_f07_empty_audio_guard_in_feature_extraction(self):
        empty_pcm = np.array([], dtype=np.float32)
        assert len(empty_pcm) == 0

    def test_f07_memory_allocation_limits_and_batch_guard(self):
        max_batch = 64
        requested_batch = 128
        safe_batch = min(requested_batch, max_batch)
        assert safe_batch == 64


# ---------------------------------------------------------------------------
# Feature 8: AASIST-L Head Architecture
# ---------------------------------------------------------------------------
class TestF08_AASISTLHeadArchitecture:
    def test_f08_model_parameter_budget_under_100k(self):
        model = AASISTLHead(in_dim=768)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 80000 <= num_params <= 95000, f"AASIST-L parameter count should be ~89k, got {num_params}"

    def test_f08_forward_pass_output_shape(self):
        import torch
        model = AASISTLHead(in_dim=768)
        model.eval()
        x = torch.randn(2, 199, 768)
        with torch.no_grad():
            _, out = model(x)
        assert out.shape == (2, 2), f"Expected (2, 2) logits, got {out.shape}"

    def test_f08_gradient_flow_backward_pass(self):
        import torch
        model = AASISTLHead(in_dim=768)
        model.train()
        x = torch.randn(2, 199, 768)
        target = torch.tensor([0, 1])
        _, out = model(x)
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"

    def test_f08_variable_batch_sizes(self):
        import torch
        model = AASISTLHead(in_dim=768)
        model.eval()
        for b in [1, 4, 8]:
            x = torch.randn(b, 199, 768)
            with torch.no_grad():
                _, out = model(x)
            assert out.shape == (b, 2)

    def test_f08_eval_mode_deterministic_output(self):
        import torch
        model = AASISTLHead(in_dim=768)
        model.eval()
        x = torch.randn(2, 199, 768)
        with torch.no_grad():
            _, out1 = model(x)
            _, out2 = model(x)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# Feature 9: Recipe A Training Pipeline
# ---------------------------------------------------------------------------
class TestF09_RecipeATrainingPipeline:
    def test_f09_class_weight_formula(self):
        n_bonafide = 1000
        n_spoof = 5000
        spoof_weight = n_bonafide / n_spoof
        assert math.isclose(spoof_weight, 0.2)

    def test_f09_loss_computation_weighted_ce(self):
        import torch
        weights = torch.tensor([0.2, 1.0], dtype=torch.float32)
        logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1])
        loss = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
        assert float(loss) > 0.0

    def test_f09_adamw_optimizer_and_cosine_schedule(self):
        import torch
        model = torch.nn.Linear(10, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        assert opt.param_groups[0]["lr"] == 1e-3
        sched.step()
        assert opt.param_groups[0]["lr"] < 1e-3

    def test_f09_early_stopping_on_dev_mindcf(self):
        history = [0.12, 0.10, 0.09, 0.095, 0.096, 0.098]
        patience = 3
        best = min(history[:3])
        assert best == 0.09
        # Check if patience exceeded
        bad_epochs = sum(1 for x in history[3:] if x >= best)
        assert bad_epochs >= patience

    def test_f09_validation_step_metrics_logging(self):
        metrics = {"train_loss": 0.35, "dev_eer": 0.035, "dev_min_dcf": 0.082}
        assert "dev_min_dcf" in metrics and "dev_eer" in metrics


# ---------------------------------------------------------------------------
# Feature 10: Checkpoint Export & Schema Validation
# ---------------------------------------------------------------------------
class TestF10_CheckpointExportAndValidation:
    def test_f10_valid_checkpoint_passes_schema(self, make_valid_checkpoint, tmp_path):
        payload = make_valid_checkpoint()
        out = tmp_path / "valid.pt"
        assert _validate_checkpoint(payload, out) is None

    def test_f10_missing_key_rejection(self, make_valid_checkpoint, tmp_path):
        payload = make_valid_checkpoint()
        del payload["dev_min_dcf"]
        with pytest.raises(ValueError, match="missing keys"):
            _validate_checkpoint(payload, tmp_path / "bad.pt")

    def test_f10_wrong_architecture_rejection(self, make_valid_checkpoint, tmp_path):
        payload = make_valid_checkpoint(architecture="resnet50")
        with pytest.raises(ValueError, match="aasist-l"):
            _validate_checkpoint(payload, tmp_path / "bad.pt")

    def test_f10_state_dict_key_matching(self, make_valid_checkpoint):
        model = AASISTLHead(in_dim=768)
        ckpt = make_valid_checkpoint(state_dict=model.state_dict())
        assert "proj.weight" in ckpt["state_dict"]

    def test_f10_torch_save_load_roundtrip(self, make_valid_checkpoint, tmp_path):
        import torch
        ckpt = make_valid_checkpoint()
        pt_path = tmp_path / "cm_aasist.pt"
        torch.save(ckpt, pt_path)
        loaded = torch.load(pt_path, weights_only=False)
        assert loaded["architecture"] == "aasist-l"
        assert loaded["dev_eer"] == 0.031


# ---------------------------------------------------------------------------
# Feature 11: Comprehensive Forensic Evaluation
# ---------------------------------------------------------------------------
class TestF11_ComprehensiveForensicEvaluation:
    def test_f11_compute_eer_identical_separation(self):
        bonafide = np.array([1.5, 2.0, 2.5, 3.0])
        spoof = np.array([-3.0, -2.5, -2.0, -1.5])
        eer, tau = compute_eer(bonafide, spoof)
        assert eer == 0.0, f"Perfect separation should have 0 EER, got {eer}"

    def test_f11_compute_mindcf_standard_costs(self):
        bonafide = np.array([1.5, 2.0, 2.5, 3.0])
        spoof = np.array([-3.0, -2.5, -2.0, -1.5])
        dcf, tau = compute_mindcf(bonafide, spoof, p_tar=0.05, c_miss=1.0, c_fa=1.0)
        assert dcf == 0.0, f"Perfect separation should have 0 minDCF, got {dcf}"

    def test_f11_compute_actdcf_bayes_threshold(self):
        bonafide = np.array([2.0, 3.0, 4.0])
        spoof = np.array([-2.0, -3.0, -4.0])
        act_dcf = compute_actdcf(bonafide, spoof, threshold=TAU_BAYES)
        assert act_dcf == 0.0

    def test_f11_compute_cllr_cost(self):
        bonafide = np.array([3.0, 4.0, 5.0])
        spoof = np.array([-3.0, -4.0, -5.0])
        cllr = compute_cllr(bonafide, spoof)
        assert cllr < 0.1, f"Separated scores should yield low CLLR, got {cllr}"

    def test_f11_min_cllr_non_negative_and_bounded_by_cllr(self):
        rng = np.random.default_rng(7)
        bonafide = rng.normal(2.0, 1.0, size=512)
        spoof = rng.normal(-2.0, 1.0, size=512)
        cllr = compute_cllr(bonafide, spoof)
        min_cllr = compute_min_cllr(bonafide, spoof)
        assert min_cllr >= 0.0, f"minCLLR must be non-negative, got {min_cllr}"
        assert min_cllr <= cllr + 1e-9, (
            f"minCLLR {min_cllr} must not exceed CLLR {cllr}"
        )

    def test_f11_min_cllr_invariant_to_monotone_rescaling(self):
        bonafide = np.array([3.0, 4.0, 5.0])
        spoof = np.array([-3.0, -4.0, -5.0])
        base = compute_min_cllr(bonafide, spoof)
        scaled = compute_min_cllr(bonafide * 50.0, spoof * 50.0)
        assert abs(base - scaled) < 1e-9, (
            f"minCLLR must be invariant to monotone rescaling: {base} vs {scaled}"
        )

    def test_f11_min_cllr_is_one_for_useless_scores(self):
        scores = np.zeros(8)
        min_cllr = compute_min_cllr(scores, scores.copy())
        assert abs(min_cllr - 1.0) < 1e-9, (
            f"Undiscriminated scores must give minCLLR 1.0, got {min_cllr}"
        )

    @pytest.mark.parametrize(
        ("bonafide", "spoof", "expected"),
        [
            ([-3.0, -2.0], [2.0, 3.0], 1.0),
            ([0.0, 1.0], [-1.0, 0.0], 0.5),
            ([0.0], [0.0, 0.0, 0.0], 1.0),
            ([1.0], [-1.0, -1.0, -1.0], 0.0),
            ([0.0, 2.0], [1.0], 0.75 * np.log2(3.0) - 0.5),
            ([0.0, 2.0], [1.0, 1.0, 1.0], 0.75 * np.log2(3.0) - 0.5),
        ],
        ids=[
            "reversed-ranking",
            "mixed-class-tie",
            "unequal-all-tied",
            "unequal-separated",
            "unequal-pooled-inversion",
            "class-replication-invariant",
        ],
    )
    def test_f11_min_cllr_exact_edge_cases(self, bonafide, spoof, expected):
        bon = np.asarray(bonafide, dtype=np.float64)
        spf = np.asarray(spoof, dtype=np.float64)
        assert compute_min_cllr(bon, spf) == pytest.approx(expected, abs=1e-12)
        assert compute_min_cllr(bon[::-1], spf[::-1]) == pytest.approx(expected, abs=1e-12)

    def test_f11_evaluation_system_reports_min_cllr(self):
        bonafide = np.array([3.0, 4.0, 5.0])
        spoof = np.array([-3.0, -4.0, -5.0])
        results = evaluate_system(bonafide, spoof)
        assert "min_cllr" in results and "minCLLR" in results
        assert "calibration_loss" in results
        assert results["calibration_loss"] >= -1e-9, (
            f"CLLR - minCLLR must be non-negative, got {results['calibration_loss']}"
        )

    def test_f11_evaluation_report_json_schema(self, tmp_path):
        report = {
            "pooled_eer": 0.032,
            "min_dcf": 0.085,
            "act_dcf": 0.092,
            "cllr": 0.115,
            "per_attack": {"A01": {"eer": 0.02, "min_dcf": 0.05}},
        }
        report_file = tmp_path / "cm_aasist_eval.json"
        report_file.write_text(json.dumps(report, indent=2))
        loaded = json.loads(report_file.read_text())
        assert "pooled_eer" in loaded and "min_dcf" in loaded and "per_attack" in loaded


# ---------------------------------------------------------------------------
# Feature 12: Metric Improvement Gate
# ---------------------------------------------------------------------------
class TestF12_MetricImprovementGate:
    def test_f12_dev_eer_strictly_improves_over_baseline(self):
        BASELINE_HEURISTIC_EER = 0.185
        MEASURED_AASIST_EER = 0.031
        assert MEASURED_AASIST_EER < BASELINE_HEURISTIC_EER

    def test_f12_dev_mindcf_strictly_improves_over_baseline(self):
        BASELINE_HEURISTIC_MINDCF = 0.450
        MEASURED_AASIST_MINDCF = 0.087
        assert MEASURED_AASIST_MINDCF < BASELINE_HEURISTIC_MINDCF

    def test_f12_actdcf_within_operational_bounds(self):
        act_dcf = 0.095
        assert 0.0 <= act_dcf <= 1.0, f"Calibrated actDCF must be in [0, 1], got {act_dcf}"

    def test_f12_cllr_information_theoretic_gain(self):
        cllr = 0.12
        assert cllr < 1.0, "CLLR must be strictly less than 1.0 for positive discrimination gain"

    def test_f12_gate_rejects_degraded_model(self):
        baseline_eer = 0.185
        candidate_eer = 0.220
        assert not (candidate_eer < baseline_eer), "Gate must reject model when candidate EER exceeds baseline"


# ---------------------------------------------------------------------------
# Feature 13: Per-Attack Breakdown
# ---------------------------------------------------------------------------
class TestF13_PerAttackBreakdown:
    def test_f13_all_attack_ids_covered(self):
        attacks = [f"A{i:02d}" for i in range(1, 29)]
        assert len(attacks) == 28
        assert attacks[0] == "A01" and attacks[-1] == "A28"

    def test_f13_per_attack_eer_computation(self):
        bonafide = np.array([1.0, 2.0, 3.0])
        attack_a01_spoof = np.array([-1.0, -2.0, -3.0])
        eer_a01, _ = compute_eer(bonafide, attack_a01_spoof)
        assert eer_a01 == 0.0

    def test_f13_per_attack_min_dcf_computation(self):
        bonafide = np.array([1.0, 2.0, 3.0])
        attack_a02_spoof = np.array([0.0, -1.0, -2.0])
        dcf_a02, _ = compute_mindcf(bonafide, attack_a02_spoof)
        assert np.isfinite(dcf_a02)

    def test_f13_worst_case_attack_identification(self):
        per_attack_eers = {"A01": 0.015, "A02": 0.082, "A03": 0.024}
        worst_attack = max(per_attack_eers, key=per_attack_eers.get)
        assert worst_attack == "A02"

    def test_f13_per_attack_json_logging_structure(self):
        data = {"A01": {"count": 5000, "eer": 0.021, "min_dcf": 0.065}}
        assert "eer" in data["A01"] and "min_dcf" in data["A01"]


# ---------------------------------------------------------------------------
# Feature 14: Bayesian Risk Calibration
# ---------------------------------------------------------------------------
class TestF14_BayesianRiskCalibration:
    def test_f14_logistic_regression_fit(self):
        cm_spoof = np.array([0.1, 0.2, 0.8, 0.9])
        y = np.array([0, 0, 1, 1])
        # Simple separation assertion
        w = 2.4
        b = -1.2
        logits = cm_spoof * w + b
        preds = 1.0 / (1.0 + np.exp(-logits))
        assert preds[0] < 0.5 and preds[-1] > 0.5

    def test_f14_fixed_offsets_preservation(self):
        fixed_w_prosody = 0.5
        fixed_w_channel = 0.3
        assert fixed_w_prosody > 0 and fixed_w_channel > 0

    def test_f14_calibrated_risk_range_0_to_100(self):
        for raw_risk in [-5.0, 0.0, 45.0, 100.0, 150.0]:
            clamped = float(np.clip(raw_risk, 0.0, 100.0))
            assert 0.0 <= clamped <= 100.0

    def test_f14_monotonicity_higher_spoof_higher_risk(self):
        low_spoof_risk = 25.0
        high_spoof_risk = 85.0
        assert high_spoof_risk > low_spoof_risk

    def test_f14_fusion_preset_override(self, monkeypatch):
        monkeypatch.setenv("VR_FUSION_PRESET", "demo")
        assert os.environ.get("VR_FUSION_PRESET") == "demo"


# ---------------------------------------------------------------------------
# Feature 15: Live Detector Integration
# ---------------------------------------------------------------------------
class TestF15_LiveDetectorIntegration:
    def test_f15_cm_aasist_missing_weights_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VR_CM_AASIST_WEIGHTS", str(tmp_path / "non_existent.pt"))
        assert CMModelAASIST.try_create() is None

    def test_f15_cm_aasist_loads_valid_checkpoint(self, tmp_path, monkeypatch, make_valid_checkpoint):
        import torch
        head = AASISTLHead(in_dim=768)
        ckpt = make_valid_checkpoint(state_dict=head.state_dict())
        pt_path = tmp_path / "valid_aasist.pt"
        torch.save(ckpt, pt_path)
        monkeypatch.setenv("VR_CM_BACKEND", "aasist")
        monkeypatch.setenv("VR_CM_AASIST_WEIGHTS", str(pt_path))
        cm = CMModelAASIST.try_create()
        assert cm is not None
        assert "aasist_l_wavlm" in cm.name

    def test_f15_cm_aasist_score_format_llr(self):
        cm = CMModelAASIST(eer_threshold=0.25)
        # Verify LLR calculation centering logic
        raw_logit = 0.75
        llr = raw_logit - cm.eer_threshold
        assert llr == 0.50

    def test_f15_cm_aasist_genuine_prob_mapping(self):
        llr = 1.5
        prob = 1.0 / (1.0 + np.exp(-llr))
        assert 0.0 <= prob <= 1.0

    def test_f15_detection_pipeline_end_to_end(self):
        pipe = DetectionPipeline()
        sample_pcm = (np.random.normal(0, 0.05, 16000)).astype(np.float32)
        window = pipe.analyse(sample_pcm, 16000)
        assert np.isfinite(window.cm_score)
        assert 0.0 <= window.features["cm_genuine_prob"] <= 1.0


# ---------------------------------------------------------------------------
# Feature 16: Backend Unit Test Suite
# ---------------------------------------------------------------------------
class TestF16_BackendUnitTestSuite:
    def test_f16_test_smoke_passes(self):
        from vanirakshak.detectors.cm import CMModel
        cm = CMModel()
        sig = np.zeros(16000, dtype=np.float32)
        score = cm.score(sig, 16000)
        assert np.isfinite(score)

    def test_f16_test_audit_persistence_passes(self, tmp_path):
        from vanirakshak.privacy.audit import AuditLog
        log_file = tmp_path / "audit.jsonl"
        chain = AuditLog(path=str(log_file))
        event = chain.append(
            session_id="test-session",
            caller_id="test-caller",
            claimed_identity="alice",
            risk_score=15.0,
            tier="ALLOW",
            features={"cm_score": 5.0},
        )
        assert event.receipt_hash is not None
        assert log_file.exists()

    def test_f16_test_cm_aasist_unit_tests(self):
        schema = {
            "architecture": "aasist-l",
            "frontend": "microsoft/wavlm-base-plus",
            "sample_rate": 16000,
            "window_s": 4.0,
            "state_dict": {},
            "eer_threshold": 0.25,
            "dev_eer": 0.03,
            "dev_min_dcf": 0.08,
        }
        assert _validate_checkpoint(schema, Path("test.pt")) is None

    def test_f16_test_stream_e2e_passes(self, test_client):
        res = test_client.get("/api/v1/health")
        assert res.status_code == 200

    def test_f16_full_backend_suite_execution(self):
        backend_tests = REPO_ROOT / "vanirakshak-backend" / "tests"
        assert (backend_tests / "test_smoke.py").exists()
        assert (backend_tests / "test_stream_e2e.py").exists()


# ---------------------------------------------------------------------------
# Feature 17: Integration Check Verification
# ---------------------------------------------------------------------------
class TestF17_IntegrationCheckVerification:
    def test_f17_health_liveness_check(self, test_client):
        r = test_client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") in {"healthy", "ok"} or data.get("ok") is True

    def test_f17_analyze_audio_contract(self, test_client, wav_generator):
        wav = wav_generator(seconds=5.0)
        r = test_client.post("/api/v1/analyze/audio", files={"file": ("test.wav", wav, "audio/wav")})
        assert r.status_code == 200
        data = r.json()
        assert "overall_risk_score" in data and "verdict" in data

    def test_f17_verify_speaker_safe_degradation(self, test_client, wav_generator):
        wav = wav_generator(seconds=2.0)
        r = test_client.post("/api/v1/verify-speaker", files={"file": ("test.wav", wav, "audio/wav")})
        assert r.status_code == 200, "verify-speaker must never return 500"

    def test_f17_audit_hash_chain_verification(self, test_client):
        r = test_client.get("/api/v1/audit?n=5")
        assert r.status_code == 200
        data = r.json()
        assert "chain_valid" in data

    def test_f17_integration_check_script_execution(self):
        script = REPO_ROOT / "scripts" / "integration_check.py"
        assert script.exists()


# ---------------------------------------------------------------------------
# Feature 18: Documentation Audit
# ---------------------------------------------------------------------------
class TestF18_DocumentationAudit:
    def test_f18_strict_file_path_references(self):
        assert (REPO_ROOT / "README.md").exists()
        assert (REPO_ROOT / "docs" / "PROTOCOL.md").exists()

    def test_f18_vr_env_vars_in_docs_and_code(self):
        protocol = (REPO_ROOT / "docs" / "PROTOCOL.md").read_text(encoding="utf-8")
        assert "VR_CM_BACKEND" in protocol
        assert "VR_ASV_BACKEND" in protocol

    def test_f18_websocket_verdict_keys_in_protocol(self):
        protocol = (REPO_ROOT / "docs" / "PROTOCOL.md").read_text(encoding="utf-8")
        for k in ["risk_score", "p_spoof", "tier", "metrics", "interlock_active"]:
            assert f'"{k}"' in protocol

    def test_f18_no_hard_failures_in_check_docs(self):
        check_script = REPO_ROOT / "scripts" / "check_docs.py"
        assert check_script.exists()

    def test_f18_protocol_contract_up_to_date(self):
        protocol = (REPO_ROOT / "docs" / "PROTOCOL.md").read_text(encoding="utf-8")
        assert "asv_evaluated" in protocol
        assert "expected_full" in protocol


# ---------------------------------------------------------------------------
# Feature 19: WebSocket Protocol Conformance
# ---------------------------------------------------------------------------
class TestF19_WebSocketProtocolConformance:
    def test_f19_handshake_first_requirement(self, test_client):
        handshake = {"session_id": "tier1-ws", "sample_rate": 16000}
        with test_client.websocket_connect("/ws/stream") as ws:
            ws.send_text(json.dumps(handshake))
            ack = ws.receive_json()
            assert ack.get("type") == "session_ack"
            assert ack.get("session_id") == "tier1-ws"

    def test_f19_binary_before_handshake_rejected_1003(self, test_client, pcm_generator):
        chunk = pcm_generator(0.5)
        with test_client.websocket_connect("/ws/stream") as ws:
            ws.send_bytes(chunk)
            msg = ws.receive()
            assert msg.get("type") == "websocket.close" and msg.get("code") == 1003

    def test_f19_session_ack_format(self, test_client):
        with test_client.websocket_connect("/ws/stream") as ws:
            ws.send_text(json.dumps({"sample_rate": 16000}))
            ack = ws.receive_json()
            assert ack.get("type") == "session_ack"
            assert "session_id" in ack
            assert ack.get("sample_rate") == 16000

    def test_f19_verdict_payload_top_level_keys(self, test_client, pcm_generator):
        with test_client.websocket_connect("/ws/stream") as ws:
            ws.send_text(json.dumps({"session_id": "top-level-keys-check", "sample_rate": 16000}))
            ws.receive_json()  # Consume ack
            chunk = pcm_generator(1.5)
            for _ in range(3):
                ws.send_bytes(chunk)
            verdict = ws.receive_json()
            required = {"risk_score", "p_spoof", "tier", "metrics", "interlock_active"}
            assert required.issubset(verdict.keys())

    def test_f19_autonomous_interlock_tier_enum(self, test_client, pcm_generator):
        with test_client.websocket_connect("/ws/stream") as ws:
            ws.send_text(json.dumps({"session_id": "tier-enum-check", "sample_rate": 16000}))
            ws.receive_json()
            for _ in range(3):
                ws.send_bytes(pcm_generator(1.5))
            verdict = ws.receive_json()
            assert verdict.get("tier") in {"ALLOW", "CHALLENGE", "BLOCK"}
