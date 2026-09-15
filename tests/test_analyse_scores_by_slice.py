"""Tests for the score-slicing tool.

Pins the three things that would silently corrupt a slice analysis:
the join key, the space-separated protocol parser, and the normalisation map.
A wrong join key or a wrong column index produces plausible-looking numbers for
the wrong utterances, which is the worst possible failure mode for an analysis
tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "analyse_scores_by_slice", REPO_ROOT / "scripts" / "analyse_scores_by_slice.py"
)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


# --------------------------------------------------------------------------
# The join key
# --------------------------------------------------------------------------
def test_utt_id_is_the_filename_stem():
    assert mod.utt_id_from_path("A:/cm-corpus/asvspoof5/flac_E_eval/E_0004993854.flac") == "E_0004993854"


def test_utt_id_handles_windows_separators():
    # Manifests are written with forward slashes, but a Windows-joined path
    # must not silently produce a key with a backslash in it.
    assert mod.utt_id_from_path("A:\\cm-corpus\\flac_E_eval\\E_0006624752.flac") == "E_0006624752"


def test_utt_id_strips_only_the_extension_not_dots_in_the_name():
    assert mod.utt_id_from_path("x/E_1.2.3.flac") == "E_1.2.3"


# --------------------------------------------------------------------------
# The protocol parser: space-separated, no header
# --------------------------------------------------------------------------
def test_load_protocol_parses_space_separated_rows(tmp_path):
    p = tmp_path / "proto.tsv"
    p.write_text(
        "E_1607 E_0009538969 M C05 2 E_0009486171 AC1 A26 spoof -\n"
        "E_2192 E_0004993854 F - 0 - - bonafide bonafide -\n",
        encoding="utf-8",
    )
    proto = mod.load_protocol(p)
    assert len(proto) == 2
    assert proto["E_0009538969"][0] == "E_1607"       # speaker
    assert proto["E_0009538969"][5] == "E_0009486171"  # source utterance
    assert proto["E_0009538969"][6] == "AC1"           # AC
    assert proto["E_0009538969"][7] == "A26"           # attack
    assert proto["E_0004993854"][7] == "bonafide"


def test_load_protocol_skips_short_rows(tmp_path):
    p = tmp_path / "proto.tsv"
    p.write_text("too few fields\nE_1 E_2 F - 0 - - A17 spoof -\n", encoding="utf-8")
    proto = mod.load_protocol(p)
    assert list(proto) == ["E_2"]


def test_column_indices_match_the_documented_schema():
    # The docstring promises: speaker utt_id gender codec codec_q source_utt AC attack key -
    assert mod.PROTOCOL_FIELDS == {
        "speaker": 0, "utt_id": 1, "gender": 2, "codec": 3,
        "codec_q": 4, "source_utt": 5, "ac": 6, "attack": 7,
    }


# --------------------------------------------------------------------------
# Normalisation -- a wrong mapping reports the wrong slice as worst
# --------------------------------------------------------------------------
def test_codec_dash_becomes_uncoded():
    assert mod.normalise("codec", "-") == "uncoded"
    assert mod.normalise("codec", "C05") == "C05"


def test_source_utt_becomes_presence_not_an_id():
    # Slicing on raw source IDs would give ~136k single-utterance groups.
    assert mod.normalise("source_utt", "-") == "no-source"
    assert mod.normalise("source_utt", "E_0009486171") == "has-source"


def test_codec_q_zero_is_labelled_uncoded_not_q0():
    assert mod.normalise("codec_q", "0") == "q0-uncoded"
    assert mod.normalise("codec_q", "3") == "q3"


def test_ac_bonafide_placeholder_is_labelled():
    assert mod.normalise("ac", "-") == "bonafide"
    assert mod.normalise("ac", "AC2") == "AC2"


def test_unknown_fields_pass_through_unchanged():
    assert mod.normalise("attack", "A28") == "A28"
    assert mod.normalise("speaker", "E_1607") == "E_1607"
    assert mod.normalise("gender", "M") == "M"


# --------------------------------------------------------------------------
# End-to-end on a synthetic dump, including the join
# --------------------------------------------------------------------------
def test_analyse_groups_by_attack_and_uses_all_bonafide(tmp_path):
    proto = tmp_path / "proto.tsv"
    proto.write_text(
        # 4 bonafide (no attack), 2 A17 spoofs, 2 A28 spoofs
        "S1 B1 F - 0 - - bonafide bonafide -\n"
        "S1 B2 F - 0 - - bonafide bonafide -\n"
        "S1 B3 F - 0 - - bonafide bonafide -\n"
        "S1 B4 F - 0 - - bonafide bonafide -\n"
        "S2 U1 F - 0 - AC1 A17 spoof -\n"
        "S2 U2 F - 0 - AC1 A17 spoof -\n"
        "S3 U3 F - 0 - AC1 A28 spoof -\n"
        "S3 U4 F - 0 - AC1 A28 spoof -\n",
        encoding="utf-8",
    )
    scores = tmp_path / "scores.tsv"
    lines = ["path\tlabel\tattack\tM1"]
    # compute_eer's convention is BONAFIDE HIGHER (bonafide < tau is a miss,
    # spoof >= tau is a false alarm). So bonafide sits at +5, the easy attack
    # well below it, and the hard attack well above it.
    for b in ("B1", "B2", "B3", "B4"):
        lines.append(f"/x/{b}.flac\tbonafide\t-\t5.0")
    lines.append("/x/U1.flac\tspoof\tA17\t-5.0")
    lines.append("/x/U2.flac\tspoof\tA17\t-5.0")
    lines.append("/x/U3.flac\tspoof\tA28\t6.0")
    lines.append("/x/U4.flac\tspoof\tA28\t6.0")
    scores.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows, models = mod.load_scores(scores)
    assert models == ["M1"]
    proto_map = mod.load_protocol(proto)
    pooled, groups = mod.analyse(rows, proto_map, "M1", "attack")

    # A17 spoofs (-5.0) are below every bonafide (+5.0): EER 0.
    assert groups["A17"]["n_bon"] == 0
    eer_a17, _ = mod.compute_eer(
        [float(r["M1"]) for r in rows if r["label"] == "bonafide"],
        groups["A17"]["spf"],
    )
    assert eer_a17 == 0.0

    # A28 spoofs (+6.0) are above every bonafide (+5.0): EER 1.0.
    eer_a28, _ = mod.compute_eer(
        [float(r["M1"]) for r in rows if r["label"] == "bonafide"],
        groups["A28"]["spf"],
    )
    assert eer_a28 == 1.0

    # The bonafide rows land in their own slice, keyed by the protocol's
    # placeholder, and carry no spoofs.
    assert groups["bonafide"]["n_bon"] == 4
    assert groups["bonafide"]["spf"] == []

    # Pooled EER sits between the two extremes.
    assert 0.0 < pooled < 1.0


def test_analyse_reports_utterances_missing_from_the_protocol(tmp_path, capsys):
    proto = tmp_path / "proto.tsv"
    proto.write_text("S1 B1 F - 0 - - bonafide bonafide -\n", encoding="utf-8")
    scores = tmp_path / "scores.tsv"
    scores.write_text(
        "path\tlabel\tattack\tM1\n"
        "/x/B1.flac\tbonafide\t-\t-1.0\n"
        "/x/NOPE.flac\tspoof\tA17\t1.0\n",
        encoding="utf-8",
    )
    rows, _ = mod.load_scores(scores)
    _, groups = mod.analyse(rows, mod.load_protocol(proto), "M1", "attack")
    out = capsys.readouterr().out
    assert "no protocol row" in out
    assert "NOPE" not in groups  # the unjoinable row was excluded, not mislabelled
