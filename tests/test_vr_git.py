#!/usr/bin/env python
"""Tests for scripts/vr_git.py -- the safe git plumbing wrapper.

These run against throwaway repositories created with plumbing only, so no test
ever touches the real working tree. That is deliberate: the failures this module
defends against are working-tree failures, and a test suite that damages the
tree to test the guard would be worse than no suite.

Run with:  python tests/run_all.py test_vr_git.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402  (provided by tests/run_all.py)

import vr_git  # noqa: E402
from vr_git import (  # noqa: E402
    GitCommandError,
    InvalidRefError,
    MergeConflictError,
    ProtectedBranchError,
    RefWriteError,
    VerificationError,
    VrGitError,
)


# ---------------------------------------------------------------------------
# Helpers -- everything below uses plumbing, never checkout
# ---------------------------------------------------------------------------
def _git(repo: Path, *args: str, stdin: str | None = None, check: bool = True):
    """Run git with byte pipes.

    Deliberately not ``text=True``: on Windows that translates ``\\n`` to
    ``\\r\\n`` on stdin, so ``git mktree`` receives a filename with a trailing
    carriage return and the resulting tree contains a path nobody asked for.
    """
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=stdin.encode("utf-8") if stdin is not None else None,
        capture_output=True,
        env=vr_git.git_env(),
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"git {args} failed: {proc.stderr.decode('utf-8', 'replace')}"
        )
    return proc.stdout.decode("utf-8", "replace").strip()


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test Runner")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


def make_commit(
    repo: Path, content: str, message: str, parents=(), filename: str = "file.txt"
) -> str:
    """Create a commit containing one file, without touching the worktree.

    ``filename`` is a parameter because a clean merge needs the two sides to
    touch *different* paths; two commits editing the same file always conflict.
    """
    blob = _git(repo, "hash-object", "-w", "--stdin", stdin=content)
    tree = _git(
        repo,
        "mktree",
        stdin=f"100644 blob {blob}\t{filename}\n",
    )
    args = ["commit-tree", tree]
    for parent in parents:
        args += ["-p", parent]
    args += ["-m", message]
    return _git(repo, *args)


# ---------------------------------------------------------------------------
# Ref name handling
# ---------------------------------------------------------------------------
def test_normalise_branch_ref_accepts_bare_name():
    assert vr_git.normalise_branch_ref("feature") == "refs/heads/feature"


def test_normalise_branch_ref_accepts_full_ref():
    assert vr_git.normalise_branch_ref("refs/heads/feature") == "refs/heads/feature"


def test_normalise_branch_ref_handles_nested_name():
    assert vr_git.normalise_branch_ref("exp/cm") == "refs/heads/exp/cm"


def test_normalise_branch_ref_rejects_non_branch_ref():
    with pytest.raises(InvalidRefError):
        vr_git.normalise_branch_ref("refs/tags/v1")


def test_normalise_branch_ref_rejects_empty():
    for bad in ("", "   "):
        with pytest.raises(InvalidRefError):
            vr_git.normalise_branch_ref(bad)


def test_branch_name_round_trips():
    assert vr_git.branch_name("refs/heads/exp/cm") == "exp/cm"
    assert vr_git.branch_name("exp/cm") == "exp/cm"


# ---------------------------------------------------------------------------
# Protected branches
# ---------------------------------------------------------------------------
def test_writing_main_is_refused(tmp_path):
    repo = make_repo(tmp_path)
    sha = make_commit(repo, "a\n", "first")
    with pytest.raises(ProtectedBranchError):
        vr_git.write_ref("main", sha, repo=repo)


def test_protection_survives_full_ref_form(tmp_path):
    repo = make_repo(tmp_path)
    sha = make_commit(repo, "a\n", "first")
    with pytest.raises(ProtectedBranchError):
        vr_git.write_ref("refs/heads/master", sha, repo=repo)


def test_pushing_main_is_refused(tmp_path):
    repo = make_repo(tmp_path)
    sha = make_commit(repo, "a\n", "first")
    with pytest.raises(ProtectedBranchError):
        vr_git.push(sha, "main", repo=repo)


def test_non_protected_branch_is_allowed(tmp_path):
    repo = make_repo(tmp_path)
    sha = make_commit(repo, "a\n", "first")
    assert vr_git.write_ref("feature/x", sha, repo=repo) == sha


# ---------------------------------------------------------------------------
# Writing and reading refs
# ---------------------------------------------------------------------------
def test_write_ref_sticks_and_verifies(tmp_path):
    repo = make_repo(tmp_path)
    sha = make_commit(repo, "a\n", "first")
    vr_git.write_ref("feature", sha, repo=repo)
    assert vr_git.read_ref("feature", repo=repo) == sha


def test_write_ref_resolves_short_sha(tmp_path):
    repo = make_repo(tmp_path)
    sha = make_commit(repo, "a\n", "first")
    vr_git.write_ref("feature", sha[:10], repo=repo)
    assert vr_git.read_ref("feature", repo=repo) == sha


def test_write_ref_creates_nested_dirs(tmp_path):
    repo = make_repo(tmp_path)
    sha = make_commit(repo, "a\n", "first")
    vr_git.write_ref("exp/nested/name", sha, repo=repo)
    assert vr_git.read_ref("exp/nested/name", repo=repo) == sha


def test_existing_ref_requires_force(tmp_path):
    repo = make_repo(tmp_path)
    first = make_commit(repo, "a\n", "first")
    second = make_commit(repo, "b\n", "second")
    vr_git.write_ref("feature", first, repo=repo)
    with pytest.raises(RefWriteError):
        vr_git.write_ref("feature", second, repo=repo)
    # and the original is untouched
    assert vr_git.read_ref("feature", repo=repo) == first


def test_force_moves_existing_ref(tmp_path):
    repo = make_repo(tmp_path)
    first = make_commit(repo, "a\n", "first")
    second = make_commit(repo, "b\n", "second")
    vr_git.write_ref("feature", first, repo=repo)
    vr_git.write_ref("feature", second, repo=repo, force=True)
    assert vr_git.read_ref("feature", repo=repo) == second


def test_read_ref_returns_none_when_absent(tmp_path):
    repo = make_repo(tmp_path)
    assert vr_git.read_ref("nope", repo=repo) is None


def test_read_ref_finds_packed_ref(tmp_path):
    repo = make_repo(tmp_path)
    sha = make_commit(repo, "a\n", "first")
    vr_git.write_ref("feature", sha, repo=repo)
    _git(repo, "pack-refs", "--all")
    assert not vr_git.ref_path("feature", repo).exists(), "ref should now be packed"
    assert vr_git.read_ref("feature", repo=repo) == sha


def test_packed_ref_still_counts_as_existing(tmp_path):
    repo = make_repo(tmp_path)
    first = make_commit(repo, "a\n", "first")
    second = make_commit(repo, "b\n", "second")
    vr_git.write_ref("feature", first, repo=repo)
    _git(repo, "pack-refs", "--all")
    with pytest.raises(RefWriteError):
        vr_git.write_ref("feature", second, repo=repo)


def test_read_ref_rejects_symbolic_ref(tmp_path):
    repo = make_repo(tmp_path)
    sha = make_commit(repo, "a\n", "first")
    vr_git.write_ref("feature", sha, repo=repo)
    vr_git.ref_path("alias", repo).write_text("ref: refs/heads/feature\n", encoding="utf-8")
    with pytest.raises(InvalidRefError):
        vr_git.read_ref("alias", repo=repo)


def test_unquote_path_handles_plain_and_quoted():
    assert vr_git._unquote_path("file.txt") == "file.txt"
    assert vr_git._unquote_path('"file.txt\\r"') == "file.txt\r"
    assert vr_git._unquote_path('"a\\tb"') == "a\tb"
    assert vr_git._unquote_path('"a\\303\\251"') == "a\xc3\xa9"


def test_verify_ref_accepts_match(tmp_path):
    repo = make_repo(tmp_path)
    sha = make_commit(repo, "a\n", "first")
    vr_git.write_ref("feature", sha, repo=repo)
    vr_git.verify_ref("feature", sha, repo=repo)  # must not raise


def test_verify_ref_rejects_mismatch(tmp_path):
    repo = make_repo(tmp_path)
    first = make_commit(repo, "a\n", "first")
    second = make_commit(repo, "b\n", "second")
    vr_git.write_ref("feature", first, repo=repo)
    with pytest.raises(VerificationError):
        vr_git.verify_ref("feature", second, repo=repo)


def test_verify_ref_rejects_missing_ref(tmp_path):
    repo = make_repo(tmp_path)
    sha = make_commit(repo, "a\n", "first")
    with pytest.raises(VerificationError):
        vr_git.verify_ref("feature", sha, repo=repo)


# ---------------------------------------------------------------------------
# Resolution errors
# ---------------------------------------------------------------------------
def test_rev_parse_rejects_unknown_revision(tmp_path):
    repo = make_repo(tmp_path)
    with pytest.raises(InvalidRefError):
        vr_git.rev_parse("does-not-exist", repo=repo)


def test_rev_parse_rejects_empty(tmp_path):
    repo = make_repo(tmp_path)
    with pytest.raises(InvalidRefError):
        vr_git.rev_parse("  ", repo=repo)


def test_write_ref_rejects_unresolvable_sha(tmp_path):
    repo = make_repo(tmp_path)
    with pytest.raises(InvalidRefError):
        vr_git.write_ref("feature", "deadbeef" * 5, repo=repo)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------
def test_merge_trees_clean_merge(tmp_path):
    repo = make_repo(tmp_path)
    base = make_commit(repo, "a\n", "base")
    left = make_commit(repo, "b\n", "left", parents=(base,), filename="left.txt")
    right = make_commit(repo, "c\n", "right", parents=(base,), filename="right.txt")
    tree, conflicts = vr_git.merge_trees(left, right, repo=repo)
    assert conflicts == []
    assert len(tree) == 40


def test_merge_trees_conflict_names_the_file(tmp_path):
    repo = make_repo(tmp_path)
    base = make_commit(repo, "a\n", "base")
    left = make_commit(repo, "left\n", "left", parents=(base,))
    right = make_commit(repo, "right\n", "right", parents=(base,))
    try:
        vr_git.merge_trees(left, right, repo=repo)
    except MergeConflictError as exc:
        assert "file.txt" in exc.conflicts
    else:
        raise AssertionError("expected MergeConflictError")


def test_build_merge_commit_has_both_parents(tmp_path):
    repo = make_repo(tmp_path)
    base = make_commit(repo, "a\n", "base")
    left = make_commit(repo, "b\n", "left", parents=(base,), filename="left.txt")
    right = make_commit(repo, "c\n", "right", parents=(base,), filename="right.txt")
    merge = vr_git.build_merge_commit(left, right, "merge: test", repo=repo)
    parents = _git(repo, "rev-list", "--parents", "-n", "1", merge).split()
    assert parents[0] == merge
    assert set(parents[1:]) == {left, right}


def test_build_merge_commit_refuses_conflicted_merge(tmp_path):
    repo = make_repo(tmp_path)
    base = make_commit(repo, "a\n", "base")
    left = make_commit(repo, "left\n", "left", parents=(base,))
    right = make_commit(repo, "right\n", "right", parents=(base,))
    with pytest.raises(MergeConflictError):
        vr_git.build_merge_commit(left, right, "merge: test", repo=repo)


def test_commit_tree_rejects_empty_message(tmp_path):
    repo = make_repo(tmp_path)
    base = make_commit(repo, "a\n", "base")
    with pytest.raises(InvalidRefError):
        vr_git.commit_tree(
            _git(repo, "rev-parse", f"{base}^{{tree}}"), [base], "  ", repo=repo
        )


def test_commit_tree_uses_agent_identity(tmp_path):
    repo = make_repo(tmp_path)
    base = make_commit(repo, "a\n", "base")
    tree = _git(repo, "rev-parse", f"{base}^{{tree}}")
    sha = vr_git.commit_tree(tree, [base], "second", repo=repo)
    author = _git(repo, "log", "-1", "--format=%an <%ae>", sha)
    assert "VaniRakshak Agent" in author


# ---------------------------------------------------------------------------
# Remote
# ---------------------------------------------------------------------------
def add_bare_remote(repo: Path, tmp_path: Path, name: str = "origin") -> Path:
    """A local bare repo standing in for a real remote."""
    bare = tmp_path / f"{name}.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", str(bare)],
        check=True,
        capture_output=True,
        env=vr_git.git_env(),
    )
    _git(repo, "remote", "add", name, str(bare))
    return bare


def test_push_lands_and_is_verified(tmp_path):
    repo = make_repo(tmp_path)
    add_bare_remote(repo, tmp_path)
    sha = make_commit(repo, "a\n", "first")
    vr_git.write_ref("feature", sha, repo=repo)
    vr_git.push(sha, "feature", remote="origin", repo=repo)
    assert vr_git.remote_refs("origin", repo=repo)["feature"] == sha


def test_remote_refs_returns_empty_for_unknown_remote_branch(tmp_path):
    repo = make_repo(tmp_path)
    add_bare_remote(repo, tmp_path)
    assert vr_git.remote_refs("origin", repo=repo) == {}


def test_force_push_moves_branch_that_already_exists(tmp_path):
    repo = make_repo(tmp_path)
    add_bare_remote(repo, tmp_path)
    first = make_commit(repo, "a\n", "first")
    second = make_commit(repo, "b\n", "second")
    vr_git.push(first, "feature", remote="origin", repo=repo)
    vr_git.push(second, "feature", remote="origin", repo=repo, force=True)
    assert vr_git.remote_refs("origin", repo=repo)["feature"] == second


def test_push_without_force_refuses_non_fast_forward(tmp_path):
    repo = make_repo(tmp_path)
    add_bare_remote(repo, tmp_path)
    first = make_commit(repo, "a\n", "first")
    second = make_commit(repo, "b\n", "second")
    vr_git.push(first, "feature", remote="origin", repo=repo)
    with pytest.raises(GitCommandError):
        vr_git.push(second, "feature", remote="origin", repo=repo)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def test_git_env_strips_proxy_variables():
    # Not using monkeypatch: tests/run_all.py's shim implements setattr only.
    saved = {n: os.environ.get(n) for n in vr_git.PROXY_ENV_VARS}
    try:
        for name in vr_git.PROXY_ENV_VARS:
            os.environ[name] = "http://127.0.0.1:58583"
        env = vr_git.git_env()
        for name in vr_git.PROXY_ENV_VARS:
            assert name not in env, f"{name} survived git_env()"
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_git_env_preserves_unrelated_variables():
    os.environ["VR_KEEP_ME"] = "1"
    try:
        assert vr_git.git_env().get("VR_KEEP_ME") == "1"
    finally:
        os.environ.pop("VR_KEEP_ME", None)


def test_git_env_does_not_mutate_the_process_environment():
    os.environ["http_proxy"] = "http://127.0.0.1:58583"
    try:
        vr_git.git_env()
        assert os.environ["http_proxy"] == "http://127.0.0.1:58583"
    finally:
        os.environ.pop("http_proxy", None)


def test_run_git_raises_on_failure(tmp_path):
    repo = make_repo(tmp_path)
    with pytest.raises(GitCommandError):
        vr_git.run_git(["rev-parse", "nope"], repo=repo)


def test_git_dir_points_at_git_directory(tmp_path):
    repo = make_repo(tmp_path)
    assert (vr_git.git_dir(repo) / "HEAD").exists()


def test_all_errors_share_a_base():
    for exc in (
        GitCommandError,
        InvalidRefError,
        ProtectedBranchError,
        RefWriteError,
        VerificationError,
        MergeConflictError,
    ):
        assert issubclass(exc, VrGitError)
