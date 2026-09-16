#!/usr/bin/env python
"""VaniRakshak doc/code drift check.

Docs drift from code silently -- this repo has already been through one
audit-and-remediate cycle where fixes in `docs/archive/AUDIT_REPORT.md` section 12
addressed findings that its own section 5/section 7/section 9 still describe as broken. This
script mechanically catches the three patterns that caused most of the
drift this branch fixed, so it doesn't recur unnoticed:

  1. File paths.   Every backtick-quoted path a doc mentions must exist on
     disk. Would have caught the `anchor_speaker_profile.json` (deleted)
     and `backend/app/core/...` (deleted) references that survived in three
     and four files respectively after those trees were removed.
  2. VR_* env vars. Every `VR_*` token in code (anywhere under
     vanirakshak-backend/vanirakshak/) should appear in at least one doc;
     every `VR_*` token in a doc should exist in code. Would have caught
     `VR_IMPOSTOR_BLOCK_BELOW` being undocumented everywhere except a JSON
     fixture's prose field, and would catch a typo'd or renamed var
     instantly instead of after someone tries it and it silently no-ops.
  3. WS verdict keys. The keys `docs/PROTOCOL.md`'s canonical verdict JSON
     example carries must be a superset of what
     `vanirakshak-backend/tests/test_stream_e2e.py` actually asserts --
     the two describing different contracts is worse than either alone.

Exit code is 0 iff there are no HARD failures. WARN-level findings print but
never fail the run -- see WARNING POLICY below for exactly what's soft vs.
hard and why, since that line is a judgment call worth being explicit about.

Usage:
    python scripts/check_docs.py
    python scripts/check_docs.py --verbose   # also print every clean check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "vanirakshak-backend"
FRONTEND_ROOT = REPO_ROOT / "frontend"
VANIRAKSHAK_PKG = BACKEND_ROOT / "vanirakshak"

# Docs this check reads. Deliberately excludes docs/archive/ (superseded by
# definition -- flagging its dead references would be noise, not a finding)
# and node_modules/.venv (not this repo's content).
DOC_GLOBS = [
    "*.md",
    "docs/*.md",
    "frontend/*.md",
    "vanirakshak-backend/*.md",
    "vanirakshak-android/*.md",
    "ml/*.md",
    "data/*.md",
    "backend/*.md",
    "ayush_agnihotri/*.md",
    "test_audio/*.md",
]

# ---------------------------------------------------------------------------
# WARNING POLICY -- read this before changing HARD/WARN below.
#
# HARD (fails the run):
#   - A path that looks unambiguously like a real repo file reference (has a
#     recognized extension: .py .ts .tsx .md .json .txt .bat .toml .yaml) and
#     resolves nowhere, either from repo root or the citing doc's own
#     directory. False positives here are rare because the extension filter
#     already excludes prose fragments and code identifiers.
#   - A VR_* var named in a doc that does not exist anywhere in
#     vanirakshak-backend/vanirakshak/. This is almost always a typo, a var
#     that was renamed without updating the doc, or a var that never
#     existed -- all three are worth a red exit code.
#   - A WS verdict key `test_stream_e2e.py` asserts that PROTOCOL.md's
#     canonical example doesn't carry. The test is the operational contract;
#     a doc silently missing a key it should show is a real mismatch.
#
# WARN (prints, does not fail):
#   - A VR_* var that exists in code but appears in no doc. Undocumented is
#     a real gap, but a brand-new experimental var mid-development
#     shouldn't block an unrelated commit from passing this check -- surface
#     it, don't gate on it.
#   - A path-shaped token without a recognized extension (e.g. a directory
#     reference like `vanirakshak/detectors/`) that doesn't resolve. These
#     have a much higher false-positive rate (trailing slashes, glob-like
#     prose, code identifiers that happen to contain a slash in a docstring
#     quote) so treating every miss as fatal would make the check too noisy
#     to trust, which defeats the point.
#
# This split is itself a judgment call, not a settled fact about the repo --
# if HARD failures start feeling like noise, or WARN findings start hiding
# real breakage, move the line rather than silencing the check.
# ---------------------------------------------------------------------------

# Extensions confident enough to hard-fail on a missing path.
STRICT_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".json", ".txt", ".bat", ".toml", ".yaml", ".yml"}

# Path-shaped tokens that are never real repo files -- skip rather than flag.
PATH_IGNORE_PATTERNS = [
    re.compile(r"^https?://"),
    re.compile(r"^ws://"),
    re.compile(r"^wss://"),
    re.compile(r"^\$\{"),
    re.compile(r"^<"),
    re.compile(r"\*"),
    re.compile(r"^\."),  # relative dotfiles/dot-paths read case-by-case, not worth the false-positive rate
    re.compile(r"^node_modules/"),
    # Absolute Windows drive paths (A:\cm-corpus\..., C:\VaniRakshak\...).
    # These are deliberately NOT repo-relative: the whole point of citing them
    # is that the thing lives outside the checkout (corpus on Drive A, sibling
    # working copies). Resolving them against REPO_ROOT would be meaningless.
    re.compile(r"^[A-Za-z]:[\\/]"),
]


# `docs/NN` shorthand -- this repo's numbered design documents are cited by
# number ("see docs/10", "per docs/13") rather than by full filename. The
# convention predates the NN-<slug>.md rename and is used ~30 times across the
# docs, so it is real vocabulary, not a typo. Resolve it instead of flagging it.
DOCS_SHORTHAND = re.compile(r"^docs/(\d{1,3})/?$")


def _resolve_docs_shorthand(candidate: str) -> bool:
    """True if `candidate` is `docs/NN` and a file named NN-*.md exists."""
    m = DOCS_SHORTHAND.match(candidate)
    if not m:
        return False
    return any(p.name.startswith(f"{int(m.group(1)):02d}-") for p in (REPO_ROOT / "docs").glob("*.md"))

# A path candidate must contain at least one of these to look like a repo
# reference rather than a stray identifier or math expression.
PATH_LIKE = re.compile(
    r"`([A-Za-z0-9_][A-Za-z0-9_\-./]*\.(?:py|ts|tsx|jsx?|md|json|txt|bat|toml|ya?ml))`"
)
DIR_LIKE = re.compile(r"`((?:[A-Za-z0-9_][A-Za-z0-9_\-.]*/)+[A-Za-z0-9_][A-Za-z0-9_\-.]*/?)`")

VR_VAR = re.compile(r"\bVR_[A-Z0-9_]+\b")

# Filenames that are deliberately cited in "this used to exist, here's why
# it's gone" corrective prose (data/README.md's, ml/README.md's, and
# backend/README.md's own retraction notes) -- these are not stale
# references to fix, they're the documented record of a deletion. A hard
# failure here would mean either never being able to explain a removal by
# name, or the check permanently flagging text that's correct as written.
# Add to this set only for that exact pattern -- a name that genuinely
# should still resolve belongs in the resolution logic above, not here.
KNOWN_HISTORICAL_REFERENCES = {
    "anchor_speaker_profile.json",  # deleted; data/README.md + ml/README.md explain why
    "data/anchor_speaker_profile.json",  # same file, cited with its data/ prefix elsewhere
    "app/main.py",  # backend/README.md's retirement note for the standalone server
}

# Paths that are REAL and correct to cite, but generated locally and
# deliberately gitignored -- so they exist on a developer's disk and never in
# a fresh CI checkout. Distinct from KNOWN_HISTORICAL_REFERENCES above: those
# name things that are gone, these name things that are absent *here* by
# design. Hard-failing on them would force docs to stop telling developers
# about tooling they are expected to run.
#
# Add a path here only if it is genuinely generated and genuinely gitignored
# -- verify with `git check-ignore -v <path>` before adding, so this does not
# become a place to silence real misses.
GENERATED_NOT_COMMITTED = {
    "graft/INDEX.md",  # graft's local graph cache; .gitignore `/graft/`, rebuilt by `graft build`
}


class Findings:
    def __init__(self) -> None:
        self.hard: List[str] = []
        self.warn: List[str] = []

    def hard_fail(self, msg: str) -> None:
        self.hard.append(msg)

    def warn_msg(self, msg: str) -> None:
        self.warn.append(msg)


def _iter_docs() -> List[Path]:
    seen: Set[Path] = set()
    out: List[Path] = []
    for pattern in DOC_GLOBS:
        for p in REPO_ROOT.glob(pattern):
            if "node_modules" in p.parts or ".venv" in p.parts:
                continue
            if "docs/archive" in str(p.relative_to(REPO_ROOT)).replace("\\", "/"):
                continue
            if p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


_IGNORED_DIR_NAMES = {"node_modules", ".venv", ".git", "__pycache__", ".next", ".pytest_cache"}


def _build_basename_index() -> Dict[str, List[Path]]:
    """basename -> every matching file anywhere in the repo (minus noise dirs).

    Docs constantly cite files as `vanirakshak/session.py` (shorthand that
    omits the vanirakshak-backend/ package root), or as a bare `config.py`
    once the surrounding prose has already established which module it's in.
    Trying to enumerate every valid prefix convention is a losing game --
    a basename index sidesteps it: if a file with that exact name exists
    ANYWHERE sane in the repo, the reference isn't stale. This does mean a
    basename that exists in two unrelated places could mask a wrong specific
    path, which is an acceptable trade for a doc-drift check, not a full
    reference validator.
    """
    index: Dict[str, List[Path]] = {}
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _IGNORED_DIR_NAMES for part in p.parts):
            continue
        index.setdefault(p.name, []).append(p)
    return index


def _resolve(candidate: str, doc: Path, basename_index: Dict[str, List[Path]]) -> bool:
    for base in (REPO_ROOT, doc.parent, BACKEND_ROOT, FRONTEND_ROOT):
        if (base / candidate).exists():
            return True
    return Path(candidate).name in basename_index


def check_paths(docs: List[Path], findings: Findings, verbose: bool) -> None:
    checked = 0
    basename_index = _build_basename_index()
    if verbose:
        print(f"  basename index: {len(basename_index)} distinct filenames across the repo")

    for doc in docs:
        text = doc.read_text(encoding="utf-8", errors="replace")
        rel_doc = doc.relative_to(REPO_ROOT)

        for m in PATH_LIKE.finditer(text):
            candidate = m.group(1)
            if any(pat.search(candidate) for pat in PATH_IGNORE_PATTERNS):
                continue
            checked += 1
            if _resolve(candidate, doc, basename_index):
                continue
            if candidate in KNOWN_HISTORICAL_REFERENCES:
                findings.warn_msg(
                    f"{rel_doc}: references deleted `{candidate}` -- allowlisted as an intentional historical/retraction reference"
                )
                continue
            if candidate in GENERATED_NOT_COMMITTED:
                findings.warn_msg(
                    f"{rel_doc}: references `{candidate}` -- allowlisted as generated-and-gitignored (absent in a fresh checkout by design)"
                )
                continue
            findings.hard_fail(
                f"{rel_doc}: references `{candidate}` -- not found anywhere in the repo (checked repo-root-relative, this doc's own directory, vanirakshak-backend/-relative, and by exact filename)"
            )

        for m in DIR_LIKE.finditer(text):
            candidate = m.group(1)
            if any(pat.search(candidate) for pat in PATH_IGNORE_PATTERNS):
                continue
            if "." in Path(candidate).name and Path(candidate).suffix.lower() in STRICT_EXTENSIONS:
                continue  # already covered by PATH_LIKE above
            if any((base / candidate).exists() for base in (REPO_ROOT, doc.parent, BACKEND_ROOT, FRONTEND_ROOT)):
                continue
            if _resolve_docs_shorthand(candidate):
                continue  # `docs/10` -> `docs/10-quality-and-hardening-plan.md`
            findings.warn_msg(
                f"{rel_doc}: references `{candidate}` -- not found at repo root or relative to this doc (dir/loosely-typed path, WARN only)"
            )

    if verbose:
        print(f"  checked {checked} strict-extension path references across {len(docs)} docs")


def _vr_vars_in_code() -> Set[str]:
    out: Set[str] = set()
    for py in VANIRAKSHAK_PKG.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        out.update(VR_VAR.findall(text))
    # run.py / cli entry point can also read env vars directly.
    for py in (BACKEND_ROOT / "run.py",):
        if py.is_file():
            out.update(VR_VAR.findall(py.read_text(encoding="utf-8", errors="replace")))
    # A var name immediately followed by an f-string `{` (e.g.
    # f"VR_FUSION_W_{k.upper()}") always matches as a truncated prefix ending
    # in "_" -- every real var in this codebase ends in a letter or digit, so
    # this filter has no false negatives against the actual naming convention.
    return {v for v in out if not v.endswith("_")}


def _vr_vars_in_docs(docs: List[Path]) -> Dict[str, List[str]]:
    """var -> list of docs (relative path strings) that mention it."""
    out: Dict[str, List[str]] = {}
    for doc in docs:
        text = doc.read_text(encoding="utf-8", errors="replace")
        rel = str(doc.relative_to(REPO_ROOT)).replace("\\", "/")
        for var in set(VR_VAR.findall(text)):
            out.setdefault(var, []).append(rel)
    return out


def check_env_vars(docs: List[Path], findings: Findings, verbose: bool) -> None:
    in_code = _vr_vars_in_code()
    in_docs = _vr_vars_in_docs(docs)

    undocumented = sorted(in_code - in_docs.keys())
    unknown_in_docs = sorted(in_docs.keys() - in_code)

    for var in undocumented:
        findings.warn_msg(f"{var}: exists in code, not documented anywhere")

    for var in unknown_in_docs:
        cited_in = ", ".join(in_docs[var])
        findings.hard_fail(f"{var}: cited in {cited_in} but does not exist anywhere in vanirakshak-backend/vanirakshak/")

    if verbose:
        print(f"  {len(in_code)} VR_* vars in code, {len(in_docs)} distinct VR_* vars cited across docs")


def check_ws_verdict_keys(findings: Findings, verbose: bool) -> None:
    test_file = BACKEND_ROOT / "tests" / "test_stream_e2e.py"
    protocol_doc = REPO_ROOT / "docs" / "PROTOCOL.md"

    if not test_file.is_file():
        findings.hard_fail(f"expected {test_file.relative_to(REPO_ROOT)} to exist -- cannot cross-check WS verdict keys")
        return
    if not protocol_doc.is_file():
        findings.hard_fail("docs/PROTOCOL.md does not exist -- cannot cross-check WS verdict keys")
        return

    test_src = test_file.read_text(encoding="utf-8")
    top_match = re.search(r"REQUIRED_TOP_LEVEL\s*=\s*\{([^}]*)\}", test_src, re.DOTALL)
    metrics_match = re.search(r"REQUIRED_METRICS\s*=\s*\{([^}]*)\}", test_src, re.DOTALL)
    if not top_match or not metrics_match:
        findings.hard_fail("could not find REQUIRED_TOP_LEVEL / REQUIRED_METRICS in test_stream_e2e.py -- script needs updating to match")
        return

    required_top = set(re.findall(r'"([a-z_]+)"', top_match.group(1)))
    required_metrics = set(re.findall(r'"([a-z_]+)"', metrics_match.group(1)))

    protocol_src = protocol_doc.read_text(encoding="utf-8")
    # Pull the first fenced ```json verdict block in section 3 (the canonical example).
    json_blocks = re.findall(r"```json\n(.*?)\n```", protocol_src, re.DOTALL)
    if not json_blocks:
        findings.hard_fail("docs/PROTOCOL.md has no fenced ```json blocks to check verdict keys against")
        return
    # The canonical verdict example is the first block containing "risk_score".
    verdict_block = next((b for b in json_blocks if '"risk_score"' in b), None)
    if verdict_block is None:
        findings.hard_fail("docs/PROTOCOL.md: no ```json block containing \"risk_score\" found -- is the canonical verdict example still there?")
        return

    doc_top_keys = set(re.findall(r'"([a-z_]+)":', verdict_block))
    missing_top = required_top - doc_top_keys
    for key in sorted(missing_top):
        findings.hard_fail(f"docs/PROTOCOL.md's canonical verdict example is missing top-level key '{key}', which test_stream_e2e.py asserts")

    metrics_match_doc = re.search(r'"metrics":\s*\{([^}]*)\}', verdict_block, re.DOTALL)
    doc_metrics_keys = set(re.findall(r'"([a-z_]+)":', metrics_match_doc.group(1))) if metrics_match_doc else set()
    missing_metrics = required_metrics - doc_metrics_keys
    for key in sorted(missing_metrics):
        findings.hard_fail(f"docs/PROTOCOL.md's canonical verdict example's metrics block is missing '{key}', which test_stream_e2e.py asserts")

    if verbose:
        print(f"  test asserts top-level={sorted(required_top)} metrics={sorted(required_metrics)}")
        print(f"  doc example carries top-level={sorted(doc_top_keys)} metrics={sorted(doc_metrics_keys)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="VaniRakshak doc/code drift check")
    ap.add_argument("--verbose", action="store_true", help="print per-check detail even when clean")
    args = ap.parse_args()

    print("=" * 78)
    print("VaniRakshak doc/code drift check")
    print("=" * 78)

    docs = _iter_docs()
    findings = Findings()

    print(f"\n  [1/3] File path references ({len(docs)} docs scanned)")
    check_paths(docs, findings, args.verbose)

    print("\n  [2/3] VR_* environment variable coverage")
    check_env_vars(docs, findings, args.verbose)

    print("\n  [3/3] WS verdict key cross-check (PROTOCOL.md vs test_stream_e2e.py)")
    check_ws_verdict_keys(findings, args.verbose)

    print("\n" + "=" * 78)
    if findings.warn:
        print(f"{len(findings.warn)} WARNING(s):")
        for w in findings.warn:
            print(f"  ~ {w}")
    if findings.hard:
        print(f"\n{len(findings.hard)} FAILURE(s):")
        for h in findings.hard:
            print(f"  [FAIL] {h}")
        print("=" * 78)
        return 1

    print("No hard failures." + (f" ({len(findings.warn)} warnings above.)" if findings.warn else ""))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
