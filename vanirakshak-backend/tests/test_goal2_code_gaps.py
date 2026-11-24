"""Goal 2 code gaps (muse-inbox issue 2): RawBoost, neural codecs, L6-L11
layer-list pooling, and the training checkpoint calibration block.

These tests run without training data, without a GPU, and without torch:
the new modules (``ml/rawboost.py``, ``ml/neural_codecs.py``,
``ml/layer_pooling.py``, ``ml/calibration.py``) have no torch dependency,
and ``ml/train_cm.py`` / ``ml/evaluate_cm.py`` are checked by parsing
their source, the same way ``test_ml_train_defaults.py`` does, because
importing them pulls in torch and transformers.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.calibration import calibration_block, compute_cllr, fit_platt  # noqa: E402
from ml.layer_pooling import (  # noqa: E402
    hidden_state_indices,
    parse_merge_layer_list,
    resolve_merge_config,
)
from ml.neural_codecs import (  # noqa: E402
    NEURAL_CODECS,
    NeuralCodecAugmenter,
    apply_neural_codec,
    available_codecs,
)
from ml.rawboost import RawBoostAugmenter, apply_rawboost  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CM = REPO_ROOT / "ml" / "train_cm.py"
EVALUATE_CM = REPO_ROOT / "ml" / "evaluate_cm.py"
SR = 16000


def _speechlike(seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# RawBoost
# ---------------------------------------------------------------------------

def test_rawboost_off_is_identity_and_on_changes_signal():
    x = _speechlike()
    rng = np.random.default_rng(0)
    assert RawBoostAugmenter(0.0)(x, rng) is x
    y = RawBoostAugmenter(1.0)(x, np.random.default_rng(0))
    assert y is not x and y.shape == x.shape
    assert np.isfinite(y).all() and np.abs(y).max() <= 1.0
    assert not np.array_equal(y, x)


def test_rawboost_deterministic_for_fixed_seed():
    x = _speechlike()
    aug = RawBoostAugmenter(1.0)
    a = aug(x, np.random.default_rng(42))
    b = aug(x, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_rawboost_each_component_keeps_shape_and_range():
    x = _speechlike()
    rng = np.random.default_rng(1)
    y = apply_rawboost(x, rng, p_linear=1.0, p_nonlinear=0.0, p_impulsive=0.0, p_coloured=0.0)
    assert y.shape == x.shape and np.isfinite(y).all()
    y = apply_rawboost(x, rng, p_linear=0.0, p_nonlinear=1.0, p_impulsive=0.0, p_coloured=0.0)
    assert y.shape == x.shape and np.isfinite(y).all()
    y = apply_rawboost(x, rng, p_linear=0.0, p_nonlinear=0.0, p_impulsive=1.0, p_coloured=0.0)
    assert y.shape == x.shape and np.isfinite(y).all()
    y = apply_rawboost(x, rng, p_linear=0.0, p_nonlinear=0.0, p_impulsive=0.0, p_coloured=1.0)
    assert y.shape == x.shape and np.isfinite(y).all()


def test_rawboost_describe_records_parameters():
    d = RawBoostAugmenter(0.5).describe()
    assert d["p"] == 0.5
    assert d["applied"] == "raw audio, before prepare_audio_window"
    assert set(d) >= {"p_linear", "p_nonlinear", "p_impulsive", "p_coloured", "snr_range_db"}


# ---------------------------------------------------------------------------
# Neural codecs
# ---------------------------------------------------------------------------

def test_neural_codec_list_covers_encodec_opus_mp3():
    assert set(NEURAL_CODECS) >= {
        "encodec_24k", "encodec_12k", "encodec_6k", "encodec_3k", "encodec_1_5k",
        "opus_24k", "opus_12k", "opus_6k", "mp3_32k", "mp3_16k",
    }


@pytest.mark.parametrize("codec", ["opus_24k", "opus_12k", "opus_6k", "mp3_32k", "mp3_16k"])
def test_ffmpeg_neural_codec_round_trip(codec):
    ok, reason = available_codecs()[codec]
    if not ok:
        pytest.skip(f"{codec} unavailable: {reason}")
    x = _speechlike()
    y = apply_neural_codec(x, codec)
    assert y.dtype == np.float32 and y.size == x.size
    assert np.isfinite(y).all() and np.abs(y).max() <= 1.0


def test_encodec_unavailable_is_reported_not_claimed():
    avail = available_codecs()
    ok, reason = avail["encodec_24k"]
    if ok:
        pytest.skip("encodec is installed on this machine")
    assert reason, "an unavailable codec must say why"
    with pytest.raises(RuntimeError, match="unavailable"):
        apply_neural_codec(_speechlike(), "encodec_24k")


def test_neural_codec_augmenter_off_is_identity():
    x = _speechlike()
    rng = np.random.default_rng(0)
    assert NeuralCodecAugmenter(0.0)(x, rng) is x


def test_neural_codec_augmenter_drops_unavailable_and_records_them():
    ok, reason = available_codecs()["opus_24k"]
    if not ok:
        pytest.skip(f"opus_24k unavailable: {reason}")
    aug = NeuralCodecAugmenter(0.5, ["encodec_24k", "opus_24k"])
    d = aug.describe()
    assert "encodec_24k" in d["unavailable"]
    assert "opus_24k" in d["codecs"]
    assert d["requested"] == ["encodec_24k", "opus_24k"]
    y = aug(_speechlike(), np.random.default_rng(0))
    assert np.isfinite(y).all()


def test_neural_codec_augmenter_strict_raises_on_unavailable():
    with pytest.raises(RuntimeError, match="encodec_24k"):
        NeuralCodecAugmenter(0.5, ["encodec_24k"], strict=True)


def test_neural_codec_augmenter_raises_when_nothing_is_available():
    with pytest.raises(RuntimeError, match="no requested neural codec is available"):
        NeuralCodecAugmenter(0.5, ["encodec_24k", "encodec_12k"])


def test_neural_codec_unknown_is_rejected():
    with pytest.raises(ValueError):
        NeuralCodecAugmenter(0.5, ["flac"])
    with pytest.raises(ValueError):
        apply_neural_codec(_speechlike(), "flac")


# ---------------------------------------------------------------------------
# Layer-list pooling
# ---------------------------------------------------------------------------

def test_parse_merge_layer_list():
    assert parse_merge_layer_list("6,7,8,9,10,11") == [6, 7, 8, 9, 10, 11]
    assert parse_merge_layer_list(None) is None
    assert parse_merge_layer_list("") is None
    with pytest.raises(ValueError):
        parse_merge_layer_list("6,12")
    with pytest.raises(ValueError):
        parse_merge_layer_list("6,6")
    with pytest.raises(ValueError):
        parse_merge_layer_list("six")


def test_hidden_state_indices_add_one_for_embedding():
    # Transformer layers 6-11 are hidden states 7-12: hidden_states[0] is
    # the embedding output, hidden_states[k] is transformer layer k-1.
    assert hidden_state_indices([6, 7, 8, 9, 10, 11]) == [7, 8, 9, 10, 11, 12]


def test_resolve_merge_config_layer_list():
    cfg = resolve_merge_config(None, [6, 7, 8, 9, 10, 11])
    assert cfg["merge_layers"] == 6
    assert cfg["merge_layer_list"] == [6, 7, 8, 9, 10, 11]
    assert cfg["last_used_layer"] == 11


def test_resolve_merge_config_prefix_unchanged():
    cfg = resolve_merge_config(12, None)
    assert cfg["merge_layers"] == 12
    assert cfg["merge_layer_list"] is None
    assert cfg["last_used_layer"] == 10


def test_resolve_merge_config_rejects_both_options():
    with pytest.raises(ValueError, match="only one of"):
        resolve_merge_config(12, [6, 7, 8, 9, 10, 11])


def test_resolve_merge_config_neither_is_single_layer():
    cfg = resolve_merge_config(None, None)
    assert cfg == {"merge_layers": None, "merge_layer_list": None, "last_used_layer": None}


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
    raise AssertionError(f"{name} not found in {path}")


def test_select_wavlm_features_supports_layer_list():
    src = _function_source(TRAIN_CM, "select_wavlm_features")
    assert "merge_layer_list" in src
    assert "hidden_state_indices" in src
    assert "merge_layers" in src


def test_evaluate_cm_audio_path_honours_layer_list():
    src = _function_source(EVALUATE_CM, "evaluate_checkpoint_on_audio")
    assert "merge_layer_list" in src
    assert "select_wavlm_features" in src


# ---------------------------------------------------------------------------
# Calibration block
# ---------------------------------------------------------------------------

def _separable_scores(seed: int = 0):
    rng = np.random.default_rng(seed)
    bon = rng.normal(5.0, 1.0, 500)
    spf = rng.normal(-5.0, 1.0, 500)
    return bon, spf


def test_fit_platt_returns_positive_scale():
    bon, spf = _separable_scores()
    w, b = fit_platt(bon, spf)
    assert w > 0 and np.isfinite(w) and np.isfinite(b)


def test_fit_platt_rejects_degenerate_input():
    with pytest.raises(ValueError):
        fit_platt([], [1.0, 2.0])
    with pytest.raises(ValueError):
        fit_platt([1.0, 1.0], [1.0, 1.0])


def test_calibration_block_has_threshold_llr_params_and_data_reference():
    bon, spf = _separable_scores()
    block = calibration_block(bon, spf, threshold=0.5, data_reference="dev.tsv")
    assert block["method"] == "platt"
    assert block["scale_w"] > 0 and np.isfinite(block["bias_b"])
    assert block["threshold"] == 0.5
    assert block["data_reference"] == "dev.tsv"
    assert block["fitted_on"] == "dev"
    assert block["n_bonafide"] == 500 and block["n_spoof"] == 500
    assert "calibrated_cllr" in block


def test_calibration_improves_cllr_on_separable_scores():
    bon, spf = _separable_scores()
    w, b = fit_platt(bon, spf)
    assert compute_cllr(w * bon + b, w * spf + b) < compute_cllr(bon, spf)


def test_calibration_block_records_failed_fit_instead_of_crashing():
    block = calibration_block([1.0, 1.0], [1.0, 1.0], 0.0, "dev.tsv")
    assert block["method"] == "none"
    assert "reason" in block and block["data_reference"] == "dev.tsv"


# ---------------------------------------------------------------------------
# train_cm.py CLI defaults and checkpoint payload (parsed, not imported)
# ---------------------------------------------------------------------------

def _cli_defaults(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                default = None
                for kw in node.keywords:
                    if kw.arg == "default":
                        default = kw.value
                found.setdefault(arg.value, default)
    return found


@pytest.mark.parametrize("flag,expected", [
    ("--merge-layer-list", None),
    ("--rawboost-p", 0.0),
    ("--neural-codec-p", 0.0),
    ("--real-codec-p", 0.0),
    ("--codec-aug-p", 0.0),
    ("--p-codec", 0.5),
    ("--p-noise", 0.3),
    ("--p-bandpass", 0.3),
    ("--p-rir", 0.2),
    ("--p-gain", 0.3),
    ("--init-weights", None),
])
def test_goal2_cli_defaults_do_not_change_default_behaviour(flag, expected):
    defaults = _cli_defaults(TRAIN_CM)
    assert flag in defaults, f"ml/train_cm.py no longer defines {flag}"
    node = defaults[flag]
    if expected is None:
        assert node is None or (isinstance(node, ast.Constant) and node.value is None), (
            f"{flag} must default to None, got {ast.dump(node) if node else None}"
        )
    else:
        assert isinstance(node, ast.Constant), f"{flag} has a non-literal default"
        assert float(node.value) == pytest.approx(expected)


def test_train_cm_checkpoint_payload_has_calibration_and_layer_list():
    src = TRAIN_CM.read_text(encoding="utf-8")
    assert '"calibration":' in src
    assert '"merge_layer_list":' in src
    assert '"rawboost":' in src
    assert '"neural_codec":' in src
    assert "calibration_block(" in src
