"""Re-encrypt the enrolment store under a new key (docs/14 W8).

Keys come from the environment, never the command line (argv is visible to
other processes):

    VR_ENROL_STORE_KEY      current key
    VR_ENROL_STORE_NEW_KEY  new key (32 bytes, base64 or hex)

Stop the server first. Afterwards, restart it with VR_ENROL_STORE_KEY set to
the new key. Safe to re-run if interrupted.

Usage:
    python scripts/rotate_enrolment_key.py [--store DIR]   # default: VR_ENROL_STORE_PATH
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "vanirakshak-backend"))

from vanirakshak.privacy.enrolment_store import parse_key, rotate_key  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default=os.environ.get("VR_ENROL_STORE_PATH", ""))
    args = ap.parse_args()
    old, new = os.environ.get("VR_ENROL_STORE_KEY", ""), os.environ.get("VR_ENROL_STORE_NEW_KEY", "")
    if not args.store or not old or not new:
        print("need --store (or VR_ENROL_STORE_PATH), VR_ENROL_STORE_KEY and VR_ENROL_STORE_NEW_KEY",
              file=sys.stderr)
        return 2
    report = rotate_key(Path(args.store), parse_key(old), parse_key(new))
    print(json.dumps(report, indent=2))
    # Unreadable records open with neither key: leave them for a human.
    return 1 if report["unreadable"] else 0


if __name__ == "__main__":
    sys.exit(main())
