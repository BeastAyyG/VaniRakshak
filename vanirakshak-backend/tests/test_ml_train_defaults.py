"""Guard the training script's CLI defaults.

These are checked by parsing ``ml/train_cm.py`` rather than importing it: the
training module pulls in ``transformers`` and ``torch``, and the point of these
tests is to hold a line that must not move even on a box that cannot run
training at all.

Background (docs/10, 2026-09-13): ``--init-weights`` used to default to
``data/cm_aasist.pt``. Every run was therefore a fine-tune of that one head
while being written up as an independent experiment, and ``strict=False``
accepted the partial match without a word. The default is now ``None`` and
must stay that way.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Optional

import pytest

TRAIN_CM = Path(__file__).resolve().parents[2] / "ml" / "train_cm.py"

# The augmenter probabilities as they were hard-coded before they became flags.
# They are an experiment variable now, so drifting them silently would change
# what every future "same recipe" claim means.
EXPECTED_AUG_DEFAULTS = {
    "--p-codec": 0.5,
    "--p-noise": 0.3,
    "--p-bandpass": 0.3,
    "--p-rir": 0.2,
    "--p-gain": 0.3,
}


def _parse() -> ast.Module:
    assert TRAIN_CM.is_file(), f"expected the training script at {TRAIN_CM}"
    return ast.parse(TRAIN_CM.read_text(encoding="utf-8"))


def _argparse_defaults(tree: ast.Module) -> Dict[str, Optional[ast.expr]]:
    """Map every ``--flag`` in a ``parser.add_argument`` call to its ``default=`` node.

    A flag with no explicit ``default=`` maps to ``None``, matching argparse's
    own behaviour.
    """
    found: Dict[str, Optional[ast.expr]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                default: Optional[ast.expr] = None
                for kw in node.keywords:
                    if kw.arg == "default":
                        default = kw.value
                found.setdefault(arg.value, default)
    return found


@pytest.fixture(scope="module")
def cli_defaults() -> Dict[str, Optional[ast.expr]]:
    return _argparse_defaults(_parse())


def test_init_weights_has_no_warm_start_default(cli_defaults):
    """A warm start must be explicit; it must never be the silent default."""
    assert "--init-weights" in cli_defaults, (
        "ml/train_cm.py no longer defines --init-weights. If the flag was "
        "removed, delete this test deliberately rather than leaving a gap."
    )
    node = cli_defaults["--init-weights"]
    if node is None:
        return  # no default= at all -> argparse gives None, which is correct
    assert isinstance(node, ast.Constant), (
        "--init-weights has a non-literal default. Only a literal None is "
        "acceptable; anything else silently warm-starts every run."
    )
    assert node.value is None, (
        f"--init-weights defaults to {node.value!r}. It defaulted to "
        "data/cm_aasist.pt once, which made every run a fine-tune of one head "
        "while looking like an independent experiment (docs/10, 2026-09-13). "
        "Callers that want a warm start must pass it explicitly."
    )


@pytest.mark.parametrize("flag,expected", sorted(EXPECTED_AUG_DEFAULTS.items()))
def test_augmenter_probability_defaults_are_unchanged(cli_defaults, flag, expected):
    """The five augmenter probabilities must default to today's hard-coded values."""
    assert flag in cli_defaults, f"ml/train_cm.py no longer defines {flag}"
    node = cli_defaults[flag]
    assert isinstance(node, ast.Constant), f"{flag} has a non-literal default"
    assert float(node.value) == pytest.approx(expected), (
        f"{flag} now defaults to {node.value!r}, was {expected!r}. Changing an "
        "augmenter probability changes the recipe; do it as an explicit "
        "experiment with a pre-registered criterion, not as a new default."
    )
