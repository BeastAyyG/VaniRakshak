#!/usr/bin/env python
"""Safe git plumbing for the VaniRakshak workspace.

Why this module exists
----------------------
On this machine (git 2.55.0.windows.3) several ref operations report success
while writing nothing. All of these were observed directly, not inferred:

* ``git checkout`` / ``git reset --hard`` remove files from the working tree and
  then fail to write the replacements back. Observed twice: 143 files (the whole
  of ``data/``, ``docs/`` and ``tests/``) and, separately, 24 ``docs/`` files.
  They were recoverable only because every one of them was tracked.
* ``git update-ref <ref> <sha>`` exits 0 and writes no ref.
* ``git branch <name> <start-point>`` exits 0 and writes no ref.
* ``git fetch`` downloads objects but writes no remote-tracking refs.

Writing the ref file directly works, and plumbing commands (``merge-tree``,
``commit-tree``, ``hash-object``, ``rev-parse``) are reliable because they only
touch the object store. ``git push`` also works normally.

So this module enforces one rule: **never touch the working tree.** Build
objects and refs directly, then verify every write by reading it back.

The second rule is that **gitignored files are outside git's protection**. The
checkpoints in ``data/`` are not tracked, so a wiped tree cannot be repaired by
git alone. Nothing here should ever create a situation where the tree needs
repair.

Usage
-----
    python scripts/vr_git.py merge-baseline --base main --work from-scratch-results \\
        --branch integration-2026-09-17 [--push]
    python scripts/vr_git.py write-ref refs/heads/<name> <sha> [--force]
    python scripts/vr_git.py verify-ref refs/heads/<name> <sha>
    python scripts/vr_git.py ls-remote

Exit code is 0 on success, 1 on a handled error, 2 on bad usage.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REMOTE = "origin"
BRANCH_REF_PREFIX = "refs/heads/"

# Branches this module refuses to create, overwrite or push to directly.
# `main` is a squash spine: work reaches it through a PR, not a push.
PROTECTED_BRANCHES = frozenset({"main", "master"})

# These break ref advertisement with "expected flush after ref listing".
PROXY_ENV_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")

# Commit identity for agent-authored commits (runbook docs/14 section 3).
AGENT_NAME = "VaniRakshak Agent"
AGENT_EMAIL = "vanirakshak@beastayyg.dev"

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ABBREV_SHA_RE = re.compile(r"^[0-9a-f]{4,40}$")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class VrGitError(Exception):
    """Base class for every error raised by this module."""


class GitCommandError(VrGitError):
    """A git invocation failed unexpectedly."""

    def __init__(self, args: list[str], returncode: int, stderr: str):
        self.args_ran = args
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"git {' '.join(args)} failed (exit {returncode}): {stderr.strip()}"
        )


class InvalidRefError(VrGitError):
    """A ref name or revision could not be resolved."""


class ProtectedBranchError(VrGitError):
    """Attempted to write a branch that must only change through a PR."""


class RefWriteError(VrGitError):
    """A ref could not be written (exists without --force, or bad layout)."""


class VerificationError(VrGitError):
    """A write was reported by git but could not be read back."""


class MergeConflictError(VrGitError):
    """Two refs do not merge cleanly.

    Carries the conflicting paths so a caller can report them without
    re-running the merge.
    """

    def __init__(self, base: str, work: str, conflicts: list[str]):
        self.base = base
        self.work = work
        self.conflicts = tuple(conflicts)
        listed = "\n".join(f"  - {c}" for c in self.conflicts) or "  (none reported)"
        super().__init__(
            f"{base} and {work} do not merge cleanly. Conflicting paths:\n{listed}"
        )


# ---------------------------------------------------------------------------
# Low level
# ---------------------------------------------------------------------------
def git_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for git, with the sandbox proxy variables removed."""
    env = dict(os.environ if base is None else base)
    for name in PROXY_ENV_VARS:
        env.pop(name, None)
    return env


def run_git(
    args: list[str],
    repo: Path = REPO_ROOT,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run git in ``repo`` and return the completed process.

    Raises GitCommandError only when ``check`` is true; callers that need to
    interpret a non-zero exit (merge-tree uses 1 for "conflicted") pass
    ``check=False``.
    """
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=git_env(env),
    )
    if check and proc.returncode != 0:
        raise GitCommandError(args, proc.returncode, proc.stderr)
    return proc


def git_dir(repo: Path = REPO_ROOT) -> Path:
    """Absolute path to the git directory.

    Resolved through git rather than assumed to be ``<repo>/.git`` so that
    linked worktrees, where ``.git`` is a file, resolve to their own private
    ref store.
    """
    proc = run_git(["rev-parse", "--absolute-git-dir"], repo=repo)
    path = Path(proc.stdout.strip())
    if not path.is_dir():
        raise InvalidRefError(f"git reported a non-directory git dir: {path}")
    return path


def rev_parse(rev: str, repo: Path = REPO_ROOT) -> str:
    """Resolve ``rev`` to a full 40-char SHA, or raise InvalidRefError."""
    if not rev or not rev.strip():
        raise InvalidRefError("empty revision")
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{rev}^{{commit}}"],
        capture_output=True,
        text=True,
        env=git_env(),
    )
    if proc.returncode != 0:
        raise InvalidRefError(f"cannot resolve {rev!r}: {proc.stderr.strip()}")
    sha = proc.stdout.strip()
    if not FULL_SHA_RE.match(sha):
        raise InvalidRefError(f"rev-parse returned a non-SHA for {rev!r}: {sha!r}")
    return sha


def normalise_branch_ref(name: str) -> str:
    """Turn a branch name or full ref into a full ``refs/heads/...`` ref."""
    if not name or not name.strip():
        raise InvalidRefError("empty ref name")
    name = name.strip()
    if name.startswith(BRANCH_REF_PREFIX):
        return name
    if name.startswith("refs/"):
        raise InvalidRefError(
            f"{name!r} is not a branch ref; this module only manages refs/heads/"
        )
    return BRANCH_REF_PREFIX + name


def branch_name(ref: str) -> str:
    """Branch name for a ref, validated as a branch."""
    ref = normalise_branch_ref(ref)
    return ref[len(BRANCH_REF_PREFIX):]


def _assert_not_protected(ref: str) -> None:
    if branch_name(ref) in PROTECTED_BRANCHES:
        raise ProtectedBranchError(
            f"{branch_name(ref)} is protected: it changes through a PR, not a "
            f"direct ref write. Use --push to a non-protected branch and open a PR."
        )


# ---------------------------------------------------------------------------
# Refs
# ---------------------------------------------------------------------------
def ref_path(ref: str, repo: Path = REPO_ROOT) -> Path:
    """Filesystem path for ``ref`` inside the resolved git directory.

    Normalised first, so ``"feature"`` and ``"refs/heads/feature"`` resolve to
    the same file. Without that, a caller could write through one spelling and
    read back through the other and silently miss.
    """
    full_ref = normalise_branch_ref(ref)
    return git_dir(repo) / Path(full_ref)


def _read_packed_ref(ref: str, repo: Path) -> str | None:
    """Look ``ref`` up in packed-refs.

    Needed because ``git pack-refs`` / ``git gc`` move refs out of
    ``refs/...`` files. A reader that only checks the loose file would report
    an existing ref as absent the first time the repo is packed.
    """
    packed = git_dir(repo) / "packed-refs"
    if not packed.exists():
        return None
    for line in packed.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            return parts[0].strip()
    return None


def read_ref(ref: str, repo: Path = REPO_ROOT) -> str | None:
    """Read a ref directly from disk, bypassing git's ref cache.

    Checks the loose ref file first, then packed-refs. Returns None when the ref
    exists in neither. Raises on a malformed entry rather than returning a
    truncated SHA, because callers compare this for equality.
    """
    full_ref = normalise_branch_ref(ref)
    path = ref_path(full_ref, repo)
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        if content.startswith("ref: "):
            raise InvalidRefError(f"{full_ref} is a symbolic ref, not a direct ref")
        if not FULL_SHA_RE.match(content):
            raise InvalidRefError(
                f"{full_ref} holds {content!r}, which is not a full SHA"
            )
        return content
    return _read_packed_ref(full_ref, repo)


def write_ref(
    ref: str,
    sha: str,
    repo: Path = REPO_ROOT,
    force: bool = False,
) -> str:
    """Write ``ref`` to point at ``sha``, then read it back.

    Written as a file because git's own ref writers silently no-op on this
    machine. The read-back is the point: a write that does not stick is
    otherwise indistinguishable from success.
    """
    full_ref = normalise_branch_ref(ref)
    _assert_not_protected(full_ref)
    resolved = rev_parse(sha, repo=repo)

    existing = read_ref(full_ref, repo=repo)
    if existing is not None and not force:
        raise RefWriteError(
            f"{full_ref} already points at {existing[:12]}; pass force=True to move it"
        )

    path = ref_path(full_ref, repo)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resolved + "\n", encoding="utf-8")
    except OSError as exc:
        raise RefWriteError(f"could not write {path}: {exc}") from exc

    written = read_ref(full_ref, repo=repo)
    if written != resolved:
        raise VerificationError(
            f"wrote {full_ref} as {resolved[:12]} but read back {written!r}"
        )
    return resolved


def verify_ref(ref: str, sha: str, repo: Path = REPO_ROOT) -> None:
    """Assert that ``ref`` resolves to ``sha``; raise VerificationError if not."""
    full_ref = normalise_branch_ref(ref)
    expected = rev_parse(sha, repo=repo)
    actual = read_ref(full_ref, repo=repo)
    if actual is None:
        raise VerificationError(f"{full_ref} does not exist")
    if actual != expected:
        raise VerificationError(
            f"{full_ref} is {actual[:12]}, expected {expected[:12]}"
        )


# ---------------------------------------------------------------------------
# Object construction
# ---------------------------------------------------------------------------
def _unquote_path(raw: str) -> str:
    """Undo git's C-style quoting of a path.

    ``merge-tree --name-only`` quotes any path containing a byte git considers
    special (including a stray carriage return), so comparing a raw line against
    a real filename fails on exactly the paths most worth reporting.
    """
    if len(raw) < 2 or not (raw.startswith('"') and raw.endswith('"')):
        return raw
    body = raw[1:-1]
    simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"',
              "a": "\a", "b": "\b", "f": "\f", "v": "\v"}
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in simple:
                out.append(simple[nxt])
                i += 2
                continue
            if nxt in "01234567":  # octal escape, up to three digits
                digits = ""
                j = i + 1
                while j < len(body) and len(digits) < 3 and body[j] in "01234567":
                    digits += body[j]
                    j += 1
                out.append(chr(int(digits, 8)))
                i = j
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def merge_trees(base: str, work: str, repo: Path = REPO_ROOT) -> tuple[str, list[str]]:
    """Merge two commits out-of-tree, returning ``(tree_sha, conflicts)``.

    Uses ``git merge-tree --write-tree``, which touches neither the working
    tree nor the index, so a bad merge costs nothing to undo.

    Raises MergeConflictError when the merge is not clean. git signals
    "conflicted" with exit code 1, which is a normal outcome here, not a
    GitCommandError.
    """
    base_sha = rev_parse(base, repo=repo)
    work_sha = rev_parse(work, repo=repo)

    proc = run_git(
        ["merge-tree", "--write-tree", "--name-only", base_sha, work_sha],
        repo=repo,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise GitCommandError(
            ["merge-tree", "--write-tree", base_sha, work_sha],
            proc.returncode,
            proc.stderr,
        )

    lines = proc.stdout.splitlines()
    tree = lines[0].strip() if lines else ""
    if not FULL_SHA_RE.match(tree):
        raise GitCommandError(
            ["merge-tree", "--write-tree", base_sha, work_sha],
            proc.returncode,
            f"expected a tree SHA on the first line, got {tree!r}",
        )

    # --name-only lists conflicted paths between the tree SHA and a blank line.
    conflicts: list[str] = []
    for line in lines[1:]:
        if not line.strip():
            break
        conflicts.append(_unquote_path(line.strip()))

    if conflicts:
        raise MergeConflictError(base, work, conflicts)
    return tree, []


def commit_tree(
    tree: str,
    parents: list[str],
    message: str,
    repo: Path = REPO_ROOT,
) -> str:
    """Create a commit for ``tree`` with the given parents.

    Pass two parents for a merge commit, one for a normal commit, none for a
    root commit.
    """
    if not FULL_SHA_RE.match(tree):
        raise InvalidRefError(f"{tree!r} is not a full tree SHA")
    if not message.strip():
        raise InvalidRefError("empty commit message")

    args = ["commit-tree", tree]
    for parent in parents:
        args += ["-p", rev_parse(parent, repo=repo)]
    args += ["-m", message]

    env = git_env()
    env["GIT_AUTHOR_NAME"] = AGENT_NAME
    env["GIT_AUTHOR_EMAIL"] = AGENT_EMAIL
    env["GIT_COMMITTER_NAME"] = AGENT_NAME
    env["GIT_COMMITTER_EMAIL"] = AGENT_EMAIL

    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise GitCommandError(args, proc.returncode, proc.stderr)
    sha = proc.stdout.strip()
    if not FULL_SHA_RE.match(sha):
        raise InvalidRefError(f"commit-tree returned {sha!r}")
    return sha


def build_merge_commit(
    base: str,
    work: str,
    message: str,
    repo: Path = REPO_ROOT,
) -> str:
    """Merge ``work`` into ``base`` and return the new commit SHA.

    Nothing is written to the working tree and no ref is created; the caller
    decides where to point a branch.
    """
    tree, _ = merge_trees(base, work, repo=repo)
    return commit_tree(tree, [base, work], message, repo=repo)


# ---------------------------------------------------------------------------
# Remote
# ---------------------------------------------------------------------------
def remote_refs(remote: str = DEFAULT_REMOTE, repo: Path = REPO_ROOT) -> dict[str, str]:
    """Branch heads on ``remote`` as ``{branch: sha}``.

    Queried with ls-remote because this repo does not keep usable
    remote-tracking refs.
    """
    proc = run_git(["ls-remote", "--heads", remote], repo=repo)
    refs: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        sha, ref = parts[0].strip(), parts[1].strip()
        if ref.startswith(BRANCH_REF_PREFIX):
            refs[ref[len(BRANCH_REF_PREFIX):]] = sha
    return refs


def push(
    sha: str,
    branch: str,
    remote: str = DEFAULT_REMOTE,
    repo: Path = REPO_ROOT,
    force: bool = False,
) -> None:
    """Push ``sha`` to ``branch`` on ``remote`` and confirm it landed.

    Verification is against ls-remote, not git's own success message, because
    a push that reports success and writes nothing is the failure mode this
    module exists to defend against.

    ``force`` uses ``--force-with-lease`` pinned to the SHA we can currently
    see on the remote, never bare ``--force``. A lease refuses the push if
    anyone moved the branch since we last looked, which is the only thing
    making a forced update of a shared branch defensible. If the branch does
    not exist upstream there is nothing to lease against and a plain push is
    used, since creating a branch cannot clobber anything.
    """
    _assert_not_protected(normalise_branch_ref(branch))
    resolved = rev_parse(sha, repo=repo)
    target = BRANCH_REF_PREFIX + branch

    args = ["push"]
    if force:
        current = remote_refs(remote, repo=repo).get(branch)
        if current is None:
            pass  # brand-new branch; nothing to overwrite
        else:
            args.append(f"--force-with-lease={target}:{current}")
    args += [remote, f"{resolved}:{target}"]

    run_git(args, repo=repo)

    landed = remote_refs(remote, repo=repo).get(branch)
    if landed != resolved:
        raise VerificationError(
            f"push reported success but {branch} is {landed!r}, expected {resolved[:12]}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_merge_baseline(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    try:
        sha = build_merge_commit(args.base, args.work, args.message, repo=repo)
    except MergeConflictError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"merge commit: {sha}")
    write_ref(args.branch, sha, repo=repo, force=args.force)
    print(f"branch:       {args.branch} -> {sha[:12]}")
    if args.push:
        push(sha, args.branch, remote=args.remote, repo=repo)
        print(f"pushed:       {args.remote}/{args.branch}")
    return 0


def _cmd_write_ref(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    sha = write_ref(args.ref, args.sha, repo=repo, force=args.force)
    print(f"{normalise_branch_ref(args.ref)} -> {sha}")
    return 0


def _cmd_verify_ref(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    verify_ref(args.ref, args.sha, repo=repo)
    print(f"ok: {normalise_branch_ref(args.ref)} == {rev_parse(args.sha, repo=repo)[:12]}")
    return 0


def _cmd_ls_remote(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    refs = remote_refs(args.remote, repo=repo)
    width = max((len(b) for b in refs), default=10)
    for branch in sorted(refs):
        print(f"  {branch:<{width}}  {refs[branch][:12]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safe git plumbing that never touches the working tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        default=str(REPO_ROOT),
        help="repository to operate on (default: this repo)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    merge = sub.add_parser("merge-baseline", help="merge two refs into a new commit")
    merge.add_argument("--base", required=True)
    merge.add_argument("--work", required=True)
    merge.add_argument("--branch", required=True, help="local branch to create")
    merge.add_argument("--message", required=True)
    merge.add_argument("--remote", default=DEFAULT_REMOTE)
    merge.add_argument("--push", action="store_true")
    merge.add_argument("--force", action="store_true", help="move --branch if it exists")
    merge.set_defaults(func=_cmd_merge_baseline)

    write = sub.add_parser("write-ref", help="point a branch at a commit")
    write.add_argument("ref")
    write.add_argument("sha")
    write.add_argument("--force", action="store_true")
    write.set_defaults(func=_cmd_write_ref)

    verify = sub.add_parser("verify-ref", help="assert a branch points at a commit")
    verify.add_argument("ref")
    verify.add_argument("sha")
    verify.set_defaults(func=_cmd_verify_ref)

    ls = sub.add_parser("ls-remote", help="list remote branch heads")
    ls.add_argument("--remote", default=DEFAULT_REMOTE)
    ls.set_defaults(func=_cmd_ls_remote)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except VrGitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
