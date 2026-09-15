#!/usr/bin/env python
"""Slice CM scores by protocol metadata, without re-scoring.

Why this exists
---------------
`docs/13` establishes that A28 is the dominant error but is **not**
metadata-distinctive at the whole-attack level: same codec mix, same
source-presence, and all sixteen eval attacks share the same 367 speakers. That
rules out the coarse explanations but cannot answer "does A28's error
*concentrate* on a subset of speakers / a codec / an AC group?" -- for that you
need per-utterance scores joined against the protocol.

`scripts/score_checkpoint_eval.py --dump-scores` produces exactly that TSV, so
the question becomes a join instead of another GPU pass.

Protocol schema (verified 2026-09-15)
-------------------------------------
The protocols are **space-separated with no header row**, 10 fields:

    speaker  utt_id  gender  codec  codec_q  source_utt  AC  attack  key  -

The manifests are tab-separated with a header and carry only
``path, label, attack_id``, discarding speaker, source utterance and AC.

Usage
-----
    # per-codec EER for one model
    python scripts/analyse_scores_by_slice.py \
        --scores handoff/scores/eval_20k_S42.tsv \
        --protocol A:/cm-corpus/protocols/ASVspoof5.eval.track_1.tsv \
        --model S42 --by codec

    # which speakers contribute most to the error (spoof side)
    python scripts/analyse_scores_by_slice.py ... --by speaker --top 20 --min-count 20

Honest caveats, printed with the output
---------------------------------------
Per-slice EER on small counts is noisy: an EER estimated from 30 spoofs has a
standard error of several points. `--min-count` exists so that a slice is not
reported as "worst" on 5 utterances, and every table prints its counts next to
its EER for exactly this reason.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compute_mindcf import compute_eer  # noqa: E402

# Which protocol column backs each sliceable field, and how to normalise it.
PROTOCOL_FIELDS = {
    "speaker": 0,
    "utt_id": 1,
    "gender": 2,
    "codec": 3,
    "codec_q": 4,
    "source_utt": 5,
    "ac": 6,
    "attack": 7,
}


def utt_id_from_path(p: str) -> str:
    """Manifest paths are absolute .flac paths; the protocol keys on the stem."""
    return Path(p.replace("\\", "/")).stem


def load_protocol(path: Path) -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, Tuple[str, ...]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            out[parts[1]] = tuple(parts)
    return out


def load_scores(path: Path) -> Tuple[List[dict], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    models = [c for c in (rows[0].keys() if rows else []) if c not in ("path", "label", "attack")]
    return rows, models


def normalise(field: str, value: str) -> str:
    """Make slices readable and comparable."""
    if field == "codec":
        return "uncoded" if value == "-" else value
    if field == "source_utt":
        return "no-source" if value == "-" else "has-source"
    if field == "codec_q":
        return "q0-uncoded" if value == "0" else f"q{value}"
    if field == "ac":
        return "bonafide" if value == "-" else value
    return value


def analyse(rows: List[dict], proto: Dict[str, Tuple[str, ...]], model: str,
            field: str) -> Tuple[float, Dict[str, dict]]:
    idx = PROTOCOL_FIELDS[field]
    y = np.array([1 if r["label"] == "bonafide" else 0 for r in rows])
    s = np.array([float(r[model]) for r in rows])
    bon = s[y == 1]
    pooled_eer, _ = compute_eer(bon, s[y == 0])

    groups: Dict[str, dict] = defaultdict(lambda: {"n_bon": 0, "spf": []})
    missing = 0
    for r, score, label in zip(rows, s, y):
        meta = proto.get(utt_id_from_path(r["path"]))
        if meta is None:
            missing += 1
            continue
        key = normalise(field, meta[idx])
        if label == 1:
            groups[key]["n_bon"] += 1
        else:
            groups[key]["spf"].append(score)
    if missing:
        print(f"  NOTE: {missing}/{len(rows)} utterances had no protocol row "
              f"and were excluded from slicing.")
    return pooled_eer, groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True,
                    help="TSV from score_checkpoint_eval.py --dump-scores")
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--model", type=str, default=None,
                    help="Column to analyse. Default: the first model column.")
    ap.add_argument("--by", type=str, default="codec",
                    choices=sorted(PROTOCOL_FIELDS), help="Slice axis.")
    ap.add_argument("--top", type=int, default=0,
                    help="Show only the N worst slices by EER (0 = all).")
    ap.add_argument("--min-count", type=int, default=10,
                    help="Suppress slices with fewer spoof utterances than this.")
    args = ap.parse_args()

    rows, models = load_scores(args.scores)
    if not rows:
        raise SystemExit(f"{args.scores}: no rows")
    model = args.model or models[0]
    if model not in models:
        raise SystemExit(f"model {model!r} not in {models}")

    proto = load_protocol(args.protocol)
    print(f"Scores   : {args.scores}  ({len(rows)} utterances)")
    print(f"Protocol : {args.protocol}  ({len(proto)} rows)")
    print(f"Model    : {model}")
    print(f"Slice    : --by {args.by}")

    pooled, groups = analyse(rows, proto, model, args.by)
    print(f"\nPooled EER (all slices) : {pooled * 100:.3f} %")

    # All spoofs are scored against ALL bonafide, so per-slice EER is the
    # standard "this attack vs the bonafide cohort" figure -- comparable across
    # slices, and comparable to the per-attack EER in the reports.
    all_bon = np.array([float(r[model]) for r in rows if r["label"] == "bonafide"])
    table = []
    for key, g in groups.items():
        n_spf = len(g["spf"])
        if n_spf < args.min_count:
            continue
        eer, _ = compute_eer(all_bon, np.array(g["spf"]))
        table.append((key, n_spf, g["n_bon"], eer))

    table.sort(key=lambda t: -t[3])
    if args.top:
        table = table[: args.top]

    print(f"\n{'slice':<24} {'spoof':>7} {'bonafide':>9} {'EER %':>8} {'vs pooled':>10}")
    print("-" * 62)
    for key, n_spf, n_bon, eer in table:
        print(f"{key:<24} {n_spf:>7} {n_bon:>9} {eer * 100:>8.3f} "
              f"{(eer - pooled) * 100:>+9.3f}")

    suppressed = sum(1 for g in groups.values() if len(g["spf"]) < args.min_count)
    print(f"\n({suppressed} slice(s) suppressed below --min-count {args.min_count}; "
          f"{len(groups)} total)")
    print("Per-slice EER is noisy at small counts -- read the counts column "
          "before believing an ordering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
