"""Build an In-the-Wild manifest in this repo's eval TSV contract.

Contract (PROJECT.md): path <TAB> label <TAB> attack_id, with label in
{bonafide, spoof} and attack_id "-" for bonafide.

In-the-Wild labels use "bona-fide" (hyphenated), which ml/train_cm.py accepts
but the eval contract does not -- normalise it here. ITW has no attack
taxonomy, so every spoof row is tagged "ITW"; per-attack breakdowns collapse
to a single group, which is correct: ITW is one condition, not 16.

Usage:
    python scripts/build_itw_manifest.py [--limit N] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ITW_DIR = Path(r"A:\cm-corpus\external\itw\release_in_the_wild")
DEFAULT_OUT = Path(r"A:\cm-corpus\manifests\itw_manifest.tsv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--itw-dir", default=str(ITW_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    itw = Path(args.itw_dir)
    meta = itw / "meta.csv"
    if not meta.exists():
        print(f"missing {meta} -- download/extract first", file=sys.stderr)
        return 2

    rows = []
    missing = 0
    with open(meta, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            raw = (r.get("label") or "").strip().lower()
            if raw in ("bona-fide", "bonafide", "real", "genuine"):
                label, atk = "bonafide", "-"
            elif raw == "spoof":
                label, atk = "spoof", "ITW"
            else:
                continue
            p = itw / r["file"].strip()
            if not p.exists():
                missing += 1
                continue
            rows.append((str(p), label, atk))
            if args.limit and len(rows) >= args.limit:
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write("path\tlabel\tattack_id\n")
        for p, lab, atk in rows:
            fh.write(f"{p}\t{lab}\t{atk}\n")

    c = Counter(lab for _, lab, _ in rows)
    print(f"wrote {out}")
    print(f"  rows {len(rows)}  bonafide {c['bonafide']}  spoof {c['spoof']}")
    print(f"  skipped {missing} rows whose audio is not extracted yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
