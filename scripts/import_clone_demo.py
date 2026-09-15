#!/usr/bin/env python
"""Import a REAL neural voice-clone attack into the demo benchmark suite.

Why this exists
---------------
Every clip in the shipped demo suite has ``is_real_clone: false``
(see ``data/demo/ground_truth.json``). demo_02 is a *signal-degraded copy*,
not a clone, and demo_04 is a different human — so the countermeasure (CM)
path has never actually been exercised by anything in the suite. A
voice-clone detector whose benchmark contains no clones cannot demonstrate
that it detects clones.

This script imports a genuine ASVspoof5 attack, together with the bonafide
utterance it was cloned from, so the pair isolates the CM: same speaker, one
real recording and one AI-generated clone of it.

Selection rules
---------------
The pair is a **spoof utterance and a bonafide utterance from the same
speaker**, both uncoded (``codec == '-'``) and both present on disk.

Do NOT try to derive the pair from the protocol's ``source_utt`` column.
It is a trap: in ASVspoof5 eval track_1 that column holds the *pre-codec*
version of the same utterance, not the bonafide victim the attack was cloned
from. Verified directly — 405,633 of 405,633 spoof rows with a non-``-``
``source_utt`` resolve to another **spoof** row, so taking that file on faith
imports a second spoof; the pair then appears to "prove" the countermeasure
is broken when in fact the selection was. Same-speaker matching avoids the
column entirely.

Licensing
---------
ASVspoof5 corpus audio is not redistributed here: this script **copies from
your local corpus on demand** and the resulting WAVs are gitignored. Run it
once after cloning; the demo suite works without it, minus the clone pair.

Usage
-----
    python scripts/import_clone_demo.py                 # dry run: show the plan
    python scripts/import_clone_demo.py --apply         # write wavs + ground truth
    python scripts/import_clone_demo.py --apply --score # also score with the live CM
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIRS = [REPO_ROOT / "data" / "demo", REPO_ROOT / "frontend" / "public" / "demo"]

DEFAULT_CORPUS = Path(r"A:\cm-corpus")
PROTOCOL_REL = Path("protocols") / "ASVspoof5.eval.track_1.tsv"
FLAC_REL = Path("asvspoof5") / "flac_E_eval"

TARGET_SR = 16000
MIN_SECONDS = 4.5          # must exceed the 4.04 s analysis window
CLONE_NAME = "demo_05_voice_clone.wav"
SOURCE_NAME = "demo_06_same_speaker_bonafide.wav"


# --------------------------------------------------------------------------
# protocol / audio helpers
# --------------------------------------------------------------------------
def load_protocol(path: Path) -> Dict[str, dict]:
    """utt_id -> row. Space-separated, no header (see docs/13)."""
    rows: Dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 9:
                continue
            rows[parts[1]] = {
                "speaker": parts[0],
                "gender": parts[2],
                "codec": parts[3],
                "source_utt": parts[5],
                "attack": parts[7],
                "key": parts[8],
            }
    return rows


def read_audio(path: Path):
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32")
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    return np.asarray(x, dtype=np.float32), int(sr)


def resample(x, sr: int, target: int):
    import numpy as np
    from scipy.signal import resample_poly
    from math import gcd

    if sr == target:
        return x
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def write_wav(path: Path, x, sr: int) -> None:
    import numpy as np

    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def find_pairs(corpus: Path, attack: Optional[str], limit: int) -> List[dict]:
    """Same-speaker (bonafide, spoof) pairs, both uncoded and both on disk."""
    proto_path = corpus / PROTOCOL_REL
    flac_dir = corpus / FLAC_REL
    if not proto_path.is_file():
        raise SystemExit(f"protocol not found: {proto_path}")
    if not flac_dir.is_dir():
        raise SystemExit(f"flac dir not found: {flac_dir}")

    by_speaker: Dict[str, List[Tuple[str, str, str]]] = {}
    with proto_path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 9 or parts[3] != "-":
                continue  # uncoded only: keep codec out of the comparison
            by_speaker.setdefault(parts[0], []).append(
                (parts[1], parts[7], parts[8])  # utt, attack, key
            )

    out: List[dict] = []
    for speaker, items in by_speaker.items():
        bona = [u for u, a, k in items if k == "bonafide" and (flac_dir / f"{u}.flac").is_file()]
        spfs = [u for u, a, k in items if k == "spoof" and (flac_dir / f"{u}.flac").is_file()]
        if not bona or not spfs:
            continue
        utt_to_attack = {u: a for u, a, k in items}
        for spf in spfs:
            if attack and utt_to_attack[spf] != attack:
                continue
            out.append({
                "speaker": speaker,
                "attack": utt_to_attack[spf],
                "spoof_utt": spf,
                "bonafide_utt": bona[0],
            })
            break
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_entry(label: str, is_clone: bool, provenance: str, wav: Path,
                sr: int, n_samples: int, expected_verdict: str,
                expected_tier: str, note: Optional[str] = None,
                measured_llr: Optional[float] = None) -> dict:
    e = {
        "label": label,
        "target_speaker": "non-target speaker (ASVspoof5)",
        "is_real_clone": is_clone,
        "is_spliced": False,
        "provenance": provenance,
        "duration_sec": round(n_samples / float(sr), 3),
        "sha256": _sha256(wav),
        "expected_verdict": expected_verdict,
        "expected_tier": expected_tier,
    }
    if measured_llr is not None:
        e["measured_cm_llr_aasist"] = round(float(measured_llr), 3)
    if note:
        e["expectation_note"] = note
    return e


def update_ground_truth(pair: dict, sr_clone: int, n_clone: int,
                        sr_src: int, n_src: int,
                        llr_clone: Optional[float], llr_src: Optional[float],
                        wav_dir: Path) -> None:
    gt_path = wav_dir / "ground_truth.json"
    if not gt_path.is_file():
        print(f"  [skip] no ground_truth.json at {gt_path}")
        return
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    clone_note = (
        "First clip in the suite with is_real_clone: true. It is a real "
        f"ASVspoof5 attack ({pair['attack']}) generated from the bonafide "
        f"utterance in {SOURCE_NAME}, and it is what actually exercises the "
        "countermeasure. Imported, not authored: see scripts/import_clone_demo.py. "
        "The speaker is NOT the enrolled anchor, so the deterministic impostor "
        "override also fires — the tier below is driven by identity as well as "
        "by the CM. To isolate the CM, enrol the source clip and re-run with "
        "that claimed identity; the CM term alone still flips."
    )
    gt["files"][CLONE_NAME] = build_entry(
        "voice_clone", True,
        f"ASVspoof5 eval {pair['spoof_utt']} — {pair['attack']} attack, "
        f"same speaker ({pair['speaker']}) as {SOURCE_NAME}, uncoded",
        wav_dir / CLONE_NAME, sr_clone, n_clone,
        "SYNTHETIC_VOICE_CLONE", "BLOCK", clone_note, llr_clone,
    )
    gt["files"][SOURCE_NAME] = build_entry(
        "clone_source", False,
        f"ASVspoof5 eval {pair['bonafide_utt']} — a genuine (uncoded) "
        f"recording of the same speaker the {pair['attack']} attack imitates",
        wav_dir / SOURCE_NAME, sr_src, n_src,
        "AUTHENTIC_TARGET", "ALLOW",
        "The real human whose voice the attack imitates — same speaker, "
        "not necessarily the exact source utterance (the eval protocol does "
        "not expose that link; see the docstring in this file). Bonafide, hence "
        "AUTHENTIC_TARGET as far as the countermeasure is concerned — but a "
        "different person from the enrolled anchor, so under "
        "claimed_identity=<anchor> the impostor override still returns BLOCK. "
        "Enrol this clip as its own identity to compare like with like.",
        llr_src,
    )
    gt["provenance_note"] = (
        gt.get("provenance_note", "")
        + f" {CLONE_NAME}/{SOURCE_NAME} were imported from the local ASVspoof5 "
        "corpus by scripts/import_clone_demo.py (not redistributed; rerun that "
        "script after a fresh clone)."
    ).strip()
    gt_path.write_text(json.dumps(gt, indent=2) + "\n", encoding="utf-8")
    print(f"  [ok] updated {gt_path}")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--attack", default=None, help="e.g. A17; default: any")
    ap.add_argument("--index", type=int, default=0, help="which candidate pair")
    ap.add_argument("--apply", action="store_true", help="write files")
    ap.add_argument("--score", action="store_true", help="score with the live CM")
    ap.add_argument("--cm-backend", default="aasist")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    pairs = find_pairs(corpus, args.attack, args.index + 1)
    if not pairs:
        raise SystemExit("no (spoof, bonafide-source) pair found — try another --attack")
    pair = pairs[min(args.index, len(pairs) - 1)]
    print(f"[plan] attack={pair['attack']} speaker={pair['speaker']} (uncoded)")
    print(f"       clone  {pair['spoof_utt']}")
    print(f"       source {pair['bonafide_utt']}")

    flac = corpus / FLAC_REL
    xc, src_sr = read_audio(flac / f"{pair['spoof_utt']}.flac")
    xs, ssr_sr = read_audio(flac / f"{pair['bonafide_utt']}.flac")
    xc, xs = resample(xc, src_sr, TARGET_SR), resample(xs, ssr_sr, TARGET_SR)
    print(f"       durations: clone {xc.size/TARGET_SR:.2f}s  source {xs.size/TARGET_SR:.2f}s")
    if min(xc.size, xs.size) < MIN_SECONDS * TARGET_SR:
        raise SystemExit(f"clip shorter than {MIN_SECONDS}s; pick another --index")

    if not args.apply:
        print("\n[dry run] nothing written. Re-run with --apply.")
        return 0

    llr_c = llr_s = None
    if args.score:
        os.environ.setdefault("VR_CM_BACKEND", args.cm_backend)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        sys.path.insert(0, str(REPO_ROOT / "vanirakshak-backend"))
        from vanirakshak.detectors.pipeline import DetectionPipeline  # noqa: E402

        cm = DetectionPipeline().cm
        print(f"       scoring with {getattr(cm, 'name', '?')}")
        llr_c, llr_s = cm.score(xc, TARGET_SR), cm.score(xs, TARGET_SR)
        print(f"       CM LLR  clone={llr_c:+.3f}  source={llr_s:+.3f}")

    primary = OUT_DIRS[0]
    write_wav(primary / CLONE_NAME, xc, TARGET_SR)
    write_wav(primary / SOURCE_NAME, xs, TARGET_SR)
    print(f"  [ok] wrote {primary / CLONE_NAME}")
    print(f"  [ok] wrote {primary / SOURCE_NAME}")

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        if d != primary:
            shutil.copy2(primary / CLONE_NAME, d / CLONE_NAME)
            shutil.copy2(primary / SOURCE_NAME, d / SOURCE_NAME)
        update_ground_truth(pair, TARGET_SR, xc.size, TARGET_SR, xs.size,
                            llr_c, llr_s, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
