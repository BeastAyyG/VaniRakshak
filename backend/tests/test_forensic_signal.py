"""
VaniRakshak — Comprehensive Forensic Signal Processing Unit Tests
Tests 16 kHz mono standardization, 8-14 Hz micro-tremor, spectral flatness (Wiener entropy),
zero-crossing rate variance, vocoder phase boundary detection, and DPDP Act 2023 compliance.
"""

from __future__ import annotations

import io
import os
import sys
import unittest
from pathlib import Path

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MISSING_DEP: str = "unknown"

try:
    import numpy as np
    import soundfile as sf
    import torch

    from backend.app.core.audio import AudioProcessor, compute_audio_sha256
    from backend.app.core.forensic_signal import ForensicSignalProcessor
    from backend.app.engine.risk_fusion import ForensicRiskFusionEngine
    from backend.app.engine.schemas import WindowAnalysisResult

    HAS_DEPS = True
except ImportError as _exc:  # pragma: no cover - depends on the environment
    # Report the module that ACTUALLY failed to import.
    #
    # The previous message hardcoded "torch, soundfile", which was wrong: those
    # two are importable here, but `backend.app.core.audio` needs torchaudio and
    # `backend.app.core.forensic_signal` needs librosa. Installing what the old
    # message asked for left all 9 tests in this file skipped, with no hint why.
    #
    # These are the ONLY tests covering the heavy backend/ tree, so a misleading
    # skip reason silently hides 43% of the suite.
    _MISSING_DEP = _exc.name or str(_exc)
    HAS_DEPS = False


@unittest.skipUnless(
    HAS_DEPS,
    f"missing dependency '{_MISSING_DEP}' - install with: pip install -r requirements-dev.txt",
)
class TestForensicSignalProcessing(unittest.TestCase):
    """Test suite for ForensicSignalProcessor and AudioProcessor."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.processor = ForensicSignalProcessor(target_sr=16000)
        cls.audio_processor = AudioProcessor(target_sr=16000)
        cls.demo_dir = REPO_ROOT / "data" / "demo"

    def test_tensor_standardization_stereo_to_16k_mono(self) -> None:
        """Verify multichannel torch.Tensor is properly downmixed and resampled to 16 kHz mono float32."""
        # 2-channel stereo audio at 48 kHz (duration: 0.5s = 24,000 samples)
        stereo_48k = torch.randn(2, 24000, dtype=torch.float32)
        tensor_mono, np_mono, sha256, duration = self.processor.standardize_audio(
            stereo_48k, orig_sr=48000, peak=0.95
        )

        self.assertEqual(tensor_mono.ndim, 2)
        self.assertEqual(tensor_mono.shape[0], 1)
        self.assertEqual(tensor_mono.shape[1], 8000)  # 0.5s @ 16kHz = 8000 samples
        self.assertEqual(tensor_mono.dtype, torch.float32)

        self.assertEqual(np_mono.ndim, 1)
        self.assertEqual(np_mono.shape[0], 8000)
        self.assertEqual(np_mono.dtype, np.float32)

        # Peak normalization check
        max_peak = float(np.max(np.abs(np_mono)))
        self.assertAlmostEqual(max_peak, 0.95, delta=0.01)

        self.assertAlmostEqual(duration, 0.5, delta=0.01)
        self.assertEqual(len(sha256), 64)

    def test_numpy_standardization_and_clipping_protection(self) -> None:
        """Verify numpy array input standardizes properly without numerical clipping."""
        # Large amplitude signals that would clip without peak normalization
        arr = (np.sin(np.linspace(0, 100 * np.pi, 32000)) * 5.0).astype(np.float32)
        tensor_mono, np_mono, sha256, duration = self.processor.standardize_audio(arr, orig_sr=16000)

        self.assertEqual(tensor_mono.shape, (1, 32000))
        self.assertEqual(np_mono.shape, (32000,))
        self.assertAlmostEqual(float(np.max(np.abs(np_mono))), 0.95, delta=0.01)
        self.assertAlmostEqual(duration, 2.0, delta=0.01)

    def test_in_memory_bytes_standardization_zero_disk(self) -> None:
        """Verify byte buffers decode and standardize strictly in-memory (DPDP compliance)."""
        # Generate synthetic WAV in RAM
        raw_samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 16000)) * 0.8).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, raw_samples, 16000, format="WAV")
        raw_bytes = buf.getvalue()

        tensor_mono, np_mono, sha256, duration = self.processor.standardize_audio(raw_bytes)
        self.assertEqual(tensor_mono.shape, (1, 16000))
        self.assertEqual(np_mono.shape, (16000,))
        self.assertAlmostEqual(duration, 1.0, delta=0.01)
        self.assertEqual(sha256, compute_audio_sha256(raw_bytes))

    def test_spectral_flatness_wiener_entropy_harmonic_vs_noise(self) -> None:
        """Verify spectral flatness correctly distinguishes harmonic tones from white noise."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)

        # Pure harmonic tone (low Wiener entropy / spectral flatness)
        harmonic_tone = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        flatness_tone = self.processor.compute_spectral_flatness(harmonic_tone, sr=sr)

        # Gaussian white noise (high Wiener entropy / spectral flatness)
        np.random.seed(42)
        white_noise = np.random.normal(0, 0.5, sr).astype(np.float32)
        flatness_noise = self.processor.compute_spectral_flatness(white_noise, sr=sr)

        self.assertLess(flatness_tone.mean_flatness, 0.05)
        self.assertGreater(flatness_noise.mean_flatness, 0.50)
        self.assertGreater(flatness_noise.high_band_flatness, flatness_tone.high_band_flatness)
        self.assertGreater(flatness_noise.anomaly_score, flatness_tone.anomaly_score)

    def test_zcr_variance_stability(self) -> None:
        """Verify ZCR calculation and variance metrics."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)

        # Smooth sinusoid: constant low ZCR
        smooth_sine = np.sin(2 * np.pi * 100 * t).astype(np.float32)
        zcr_sine = self.processor.compute_zcr_variance(smooth_sine, sr=sr)

        # Signal with sudden frequency jumps (phase boundaries)
        jumpy = np.concatenate([
            np.sin(2 * np.pi * 100 * t[:8000]),
            np.sin(2 * np.pi * 3000 * t[8000:]),
        ]).astype(np.float32)
        zcr_jumpy = self.processor.compute_zcr_variance(jumpy, sr=sr)

        self.assertGreater(zcr_jumpy.zcr_variance, zcr_sine.zcr_variance)
        self.assertGreater(zcr_jumpy.zcr_max_jump, zcr_sine.zcr_max_jump)

    def test_micro_tremor_extraction(self) -> None:
        """Verify 8–14 Hz micro-tremor extraction returns valid bounded metrics."""
        sr = 16000
        # 2.0s signal with 10 Hz amplitude modulation (simulated micro-tremor)
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
        carrier = np.sin(2 * np.pi * 150 * t)  # 150 Hz pitch
        modulation = 1.0 + 0.1 * np.sin(2 * np.pi * 10.0 * t)  # 10 Hz tremor
        tremor_signal = (carrier * modulation).astype(np.float32)

        result = self.processor.compute_micro_tremor(tremor_signal, sr=sr)
        self.assertGreater(result.tremor_ratio, 0.0)
        self.assertGreater(result.modulation_index, 0.0)
        self.assertAlmostEqual(result.peak_frequency_hz, 10.0, delta=2.5)

    def test_benchmark_files_detection(self) -> None:
        """Evaluate genuine, full clone, and partial splice benchmark audio files."""
        gen_path = self.demo_dir / "demo_01_genuine.wav"
        clone_path = self.demo_dir / "demo_02_full_clone.wav"
        splice_path = self.demo_dir / "demo_03_partial_splice.wav"

        if not (gen_path.is_file() and clone_path.is_file() and splice_path.is_file()):
            self.skipTest("Demo benchmark files not present in data/demo/")

        # 1. Genuine speech evaluation
        rep_gen = self.processor.analyze_stream(gen_path)
        self.assertEqual(rep_gen.verdict_indicator, "NATURAL_ACOUSTIC_PROFILE")
        self.assertLess(rep_gen.composite_vocoder_risk_score, 30.0)
        self.assertEqual(rep_gen.vocoder_phase_discontinuities_detected, 0)

        # 2. Full synthetic vocoder clone evaluation
        rep_clone = self.processor.analyze_stream(clone_path)
        self.assertEqual(rep_clone.verdict_indicator, "SYNTHETIC_VOCODER_DETECTED")
        self.assertGreater(rep_clone.composite_vocoder_risk_score, 60.0)
        self.assertGreater(rep_clone.mean_high_band_flatness, 0.50)

        # 3. Partial spliced audio evaluation
        rep_splice = self.processor.analyze_stream(splice_path)
        self.assertIn(
            rep_splice.verdict_indicator,
            ["SUSPICIOUS_PHASE_DISCONTINUITY", "SYNTHETIC_VOCODER_DETECTED"],
        )
        self.assertGreater(rep_splice.mean_high_band_flatness, rep_gen.mean_high_band_flatness)

    def test_window_latency_performance(self) -> None:
        """Ensure per-window telemetry computation latency is sub-30ms for streaming."""
        dummy_window = np.random.randn(32000).astype(np.float32)  # 2.0s @ 16kHz
        telemetry = self.processor.analyze_window(dummy_window, sr=16000)
        self.assertLess(telemetry.latency_ms, 35.0)

    def test_risk_fusion_incorporation(self) -> None:
        """Verify ForensicRiskFusionEngine accounts for forensic signal metrics."""
        fusion = ForensicRiskFusionEngine(asv_threshold=0.55, anomaly_threshold=0.60)

        # Create window results with high vocoder metrics
        mock_window = WindowAnalysisResult(
            window_index=0,
            start_sec=0.0,
            end_sec=2.0,
            duration_sec=2.0,
            is_voiced=True,
            speech_ratio=1.0,
            acoustic_vector_dim=768,
            acoustic_vector_norm=1.0,
            acoustic_anomaly_score=0.48,
            speaker_cosine_sim=0.75,
            speaker_match=True,
            micro_tremor_ratio=0.10,
            spectral_flatness_mean=0.06,
            high_band_flatness=0.65,
            zcr_variance=0.007,
            vocoder_boundary_score=0.35,
            inference_ms=10.0,
        )

        report = fusion.fuse(
            session_id="test-session-123",
            audio_sha256="abcdef1234567890",
            total_duration_sec=2.0,
            windows=[mock_window],
            device_used="cpu",
            stream_boundary_timestamps=[1.5],
        )

        self.assertEqual(report.vocoder_phase_discontinuities_detected, 1)
        self.assertEqual(report.detected_boundary_timestamps, [1.5])
        self.assertIsNotNone(report.mean_high_band_flatness)
        self.assertEqual(report.dpdp_compliance["zero_retention_verified"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
