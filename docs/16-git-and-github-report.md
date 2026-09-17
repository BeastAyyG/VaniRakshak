# 16 — Git and GitHub: what happened, and why it matters here

Written 2026-09-17. A record of every git/GitHub incident in this project, the
root causes that were actually found (several earlier diagnoses were wrong and
are listed as such), the current state of the remote, and the operating rules
that follow.

---

## 1. Why git matters more than usual in this repo

Most projects use git to track code. Here the **record** is the deliverable, and
that changes what git is for:

- **`reports/*.json` are immutable evidence.** docs/10 ground rule 6: *"Never
  edit an existing `reports/` JSON. They are the record."* A report that can be
  edited after the fact proves nothing.
- **`docs/10` is a pre-registered protocol.** Ground rule 4: *"State the success
  criterion before the run, in this file, and keep it."* The only thing that
  makes that credible is a commit timestamp showing the criterion existed
  before the numbers did. Without git, "we pre-registered this" is an assertion.
- **The headline claim is a reproducibility claim.** "From-scratch initialisation
  beats the warm-started lineage on 5 seeds" is exactly the kind of claim that
  must be re-derivable by someone else from the tree plus the checkpoints.
- **Checkpoints live outside git.** `data/*.pt` are gitignored (binary, and the
  ft4 encoder alone is 57 MB). So the single most valuable and least protected
  artifacts are the ones git cannot restore. This is the root of every loss
  described below.
- **Three worktrees share one object store** (`C:\VaniRakshak`,
  `C:\VaniRakshak-main`, `C:\vr-results`), so git is also the coordination
  layer between them.
- Push target: **`https://github.com/BeastAyyG/VaniRakshak`** (private).

---

## 2. What actually happened — in order

### 2.1 The original damage: "vanishing refs"

The shared `.git` object store started showing residue:

```
warning: garbage found: C:/VaniRakshak/.git/worktrees/vr-results/refs
```

Branches appeared to be created and then did not exist. The first hypothesis —
Syncthing damaging the repositories — was **investigated and withdrawn**:
Syncthing's config defines exactly two shares, `C:\VaniRakshak\handoff` and
`A:\cm-corpus\manifests`, and neither `C:\VaniRakshak-main` nor `C:\vr-results`
is among them. What Syncthing actually threatens is *run reproducibility*: the
manifests share is the direct input of every training run, so if `train.tsv`
changes mid-run the inputs move underneath it.

### 2.2 Root cause found (2026-09-15): nested branch names silently fail

**`git version 2.55.0.windows.3` reports success and writes nothing for nested
branch names.**

```
$ git checkout -b exp/cm-from-scratch-results
Switched to a new branch 'exp/cm-from-scratch-results'
$ git branch
  results                 # <- the new branch does not exist
$ git rev-parse HEAD
fatal: ambiguous argument 'HEAD': unknown revision
```

The same with `git update-ref refs/heads/exp/<name>`: exit code 0, no ref
written, and the now-empty `refs/heads/exp/` directory is pruned. A **flat**
name (`refs/heads/results`) writes and survives normally, and a manual nested
file write also survives — so the filesystem is fine and the defect is in git's
nested-ref handling on this build.

**This explains the original damage.** The empty `refs/` directory left inside a
worktree's private ref store is exactly the residue this failure produces. Every
worktree whose branch name contained a `/` was exposed to it.

Consequences, still in force:

- **Use flat local branch names.** The working branch is `from-scratch-results`,
  not `exp/cm-from-scratch-results`.
- **Push still uses the nested name**, because only *local* ref creation is
  broken: `git push origin from-scratch-results:refs/heads/exp/cm-from-scratch-results`.
- **Never trust a `checkout -b` or `update-ref` exit code on this box.** Verify
  with `git rev-parse HEAD` or `git branch`.

### 2.3 The 13-file rescue

`C:\VaniRakshak` held **13 uncommitted files that existed nowhere else** — 838
insertions, three of them previously untracked
(`scripts/check_ft4_train.py`, `vanirakshak-backend/tests/test_auth.py`,
`vanirakshak-backend/vanirakshak/server/auth.py`). Committed as `7e05188` on
`rescue-vr-2026-09-16` off `0ff355e`, pushed, and verified with `ls-remote`.
Explicit paths only — never `git add -A`, because `.headrun/` is not gitignored
and would be swept in.

### 2.4 The stash that the rescue bundle missed

A git **stash** (`4ec84778743da042d44de4ec824b1a5f91625f91`) had been pushed as a
branch. It was absent from the rescue bundle — verified by cloning the bundle in
isolation and failing to resolve the commit — so it had exactly one copy, inside
the very object store the rescue existed to protect. Preserved two ways: a
bundle (30,688,581 bytes) and a patch (9,053 bytes), both verified. Later
recovered as a real commit `0a5a3e8` on `feat/cm-train-unfreeze-top-n`.

**Correction to the original assessment:** the stash turned out to be a
*superseded* early draft of a feature that `main` already implements
(`--unfreeze-top-n`, `--encoder-lr`). Preserving it was still correct practice;
it was not a novel lead. Do not build on that branch.

### 2.5 Why every push died silently — the hanging credential helper

Three earlier diagnoses were **wrong** and are recorded here so they are not
re-tried:

| earlier claim | verdict |
|---|---|
| "the proxy truncates the connection" | **wrong** — proxy env vars break the *ref listing* (`fatal: expected flush after ref listing`), but with them unset `git ls-remote` returns refs. Not the cause of the hang. |
| "there is no GitHub token" | **wrong** — a token exists and is valid. |
| "the stored `gho_` token is rejected" | **wrong** — an artifact of a test where the username was mis-supplied. |

**Actual cause:** the system-scope config
`.../PortableGit/versions/1.2.0/etc/gitconfig` sets
`credential.helper=helper-selector`, and that binary **hangs forever**:

```
printf "protocol=https\nhost=github.com\n\n" | timeout 12 git-credential-helper-selector.exe get
  -> exit 124, 13 s, 0 bytes
```

Because git consults helpers in order (system → global → local) and never
reached the working `wincred` entry, every push died silently — matching the
240 s / 300 s SIGTERM with zero output seen earlier.

**Fix (applied, verified).** An empty `credential.helper=` value *resets the
accumulated list*, so placing one **before** the wincred entry discards the
system-scope selector:

```bash
git config --global --replace-all credential.helper ""
git config --global --add credential.helper '!".../git-credential-wincred.exe"'
```

Order matters: `--add` alone would append the empty entry *after* wincred and
wipe the working helper too. Backup at `C:\Users\singu\.gitconfig.bak-vr`.
Result: `git credential fill` returns in 1–2 s (was: hang).

**Consequence:** choose `wincred`. Do **not** use `manager` as the standing
helper — its OAuth/device flow needs a real terminal and hangs identically. Any
git *network* operation must run with the proxy env vars unset.

### 2.6 The 143-file deletion (2026-09-17) — the dangerous one

```
git checkout -b integration-2026-09-17 ecb48b4
```

**deleted 143 files** — the entire `data/` directory, `docs/`, and `tests/` —
and then silently failed to write the replacements back. It did the same again
on the way back: 24 `docs/` files deleted, not restored.

Recovery, all verified:

- tracked files: `git checkout -- .` (twice) — tree back to dirty=0
- **gitignored checkpoints, which git cannot restore**: copied back from
  `C:\vr-save\ckpt` — 4 × `cm_scratch_{S42,B43,S43,B42}.pt` and
  `cm_aasist_ft4_s42.pt` (57 MB)

Nothing was permanently lost **only because `C:\vr-save\ckpt` existed**.

> **Do not run `git checkout` or `git reset --hard` in this workspace.**
> Build commits with plumbing so the working tree is never touched.

### 2.7 Branch cleanup — 10 remote branches to 7

Audited all 10 remote branches against `main` (`ecb48b4`) using unique-commit
counts, then cleaned up with approval.

**Deleted (7):** `VaniRaksha-App-v1`, `VaniRaksha-App-v2` (0 unique commits
each, both fully contained in `v3APP`); `exp/cm-from-scratch` (PR #7 head,
already squash-merged as `e661b9f`); `rescue-wip-train-cm-2026-09-16` (a git
*stash* pushed as a branch — work extracted first); `v3APP`, `cm-app-integration`,
`ayush` (deleted after rename).

**Renamed (3)**, to match CONTRIBUTING.md's `feat/` `fix/` `docs/` convention:

| from | to | sha |
|---|---|---|
| `v3APP` | `feat/android-v3-live-voice` | `50beddc` |
| `cm-app-integration` | `feat/cm-app-integration` | `ce3bf1d` |
| `ayush` | `docs/readme-kavach-rebrand` | `552b26d` |

**Work recovered:** the stash branch held a version of `ml/train_cm.py`
(blob `c9181fb3`, +83/−16) that existed on no other branch — bfloat16 WavLM
frontend loading, top-N transformer layer unfreezing, discriminative learning
rates, persisted fine-tuned frontend weights, 2 new CLI args. Preserved as
`0a5a3e8`, verified blob-identical before deleting the stash branch.

**Result:** 10 branches → 7, all correctly tracked. Local hygiene: created the
missing local `main`, removed a stale `[branch "master"]` config, removed
leftover bundle remotes `rescue` and `stashsrc`.

### 2.8 The 2026-09-17 data loss — three checkpoints destroyed

At **01:32–01:34** a stray destructive git operation from another session
(`.git/index` and `ORIG_HEAD` rewritten; `docs/` and `tests/` re-checked-out)
deleted:

- `data/cm_scratch_S44.pt`
- `data/cm_scratch_S45.pt`
- `data/cm_scratch_S46.pt`

**Unrecoverable.** Not in `C:\vr-save\ckpt` (which holds only S42/S43/B42/B43),
not in the Recycle Bin. **S45 was the best seed of the five**
(EER 10.221 %, minDCF 0.6832) — so the best weights this project has ever
produced have no copy on disk.

The *scores* are unaffected: the claim report was written from the real files
before the deletion. But S45 cannot be shipped without a retrain, and S42 —
which survives — is the swap candidate for that reason.

### 2.9 Sandbox quirks in this shell

Several git ref operations **exit 0 but silently do nothing**:

| operation | behaviour |
|---|---|
| `git fetch origin` | downloads objects, prints `* [new branch]`, writes **no** `refs/remotes/origin/*` |
| `git update-ref <ref> <sha>` | no-op |
| `git branch <name> <start-point>` | no-op |
| `git branch` (no start point) | works |
| `git branch -D`, `git config`, `git ls-remote` | work |
| **`git push`** (incl. `--delete`) | works |

**Workaround:** write ref files directly, e.g.
`echo <sha> > .git/refs/remotes/origin/main` (mkdir -p nested dirs first). Build
commits without touching the worktree using a temp index:
`GIT_INDEX_FILE=.git/tmp-index git read-tree <base>`, then
`update-index --cacheinfo`, `write-tree`, `commit-tree`. Push with raw SHA
refspecs: `git push origin <sha>:refs/heads/<name>`.

**Gotcha:** `git remote remove <name>` prunes unrelated `refs/remotes/origin/*`
entries. Recreate tracking refs by hand afterwards.

**Windows trap:** `subprocess(text=True)` translates `\n` to `\r\n` on stdin, so
`git mktree` receives a filename with a trailing CR. Use byte pipes whenever git
reads stdin.

---

## 3. What we do about it — `scripts/vr_git.py`

The ad-hoc ref hacks were turned into a tested module. It **never touches the
working tree**: `merge-tree` → `commit-tree` → write the ref file → read it back.

- Typed exceptions: `VrGitError` base; `GitCommandError`, `InvalidRefError`,
  `ProtectedBranchError`, `RefWriteError`, `VerificationError`,
  `MergeConflictError` (carries `.conflicts`).
- `PROTECTED_BRANCHES = {main, master}` — refuses to write or push those.
- No hard-coded SHAs; every ref/rev is a parameter.
- **Every write is verified by reading it back.** `push()` verifies via
  `ls-remote` and uses `--force-with-lease=<ref>:<sha>` pinned to the SHA last
  seen on the remote — never bare `--force`. Falls back to a plain push when the
  branch does not exist upstream (nothing to clobber).
- CLI: `merge-baseline`, `write-ref`, `verify-ref`, `ls-remote`.
- `tests/test_vr_git.py`: **34 tests**, green against throwaway repos built with
  plumbing only.

Safe pattern, for reference:

```bash
TREE=$(git merge-tree --write-tree <base> <work>)   # prints tree oid; --name-only shows conflicts
COMMIT=$(git commit-tree $TREE -p <base> -p <work> -m "msg")
echo $COMMIT > .git/refs/heads/<name>               # write the ref directly
git push origin $COMMIT:refs/heads/<name>
```

`git merge-tree --write-tree` is the way to *measure* merge conflicts before
committing to anything — it touches nothing.

### Operating rules, in force

1. **Never `git checkout` / `git reset --hard` in `C:\vr-work\repo`.**
2. **Flat local branch names only.**
3. **Explicit paths in `git add` — never `git add -A`** (`.headrun/` is not
   gitignored).
4. **Never push to `main`.** Branch off `main`, prefixed `feat/` `fix/` `docs/`
   `refactor/` `test/`.
5. **Copy finished checkpoints to `C:\vr-save\ckpt` immediately.** Any untracked
   file under `data/` can vanish without warning.
6. **`main` is a squash spine** — PRs #1–#9 all landed as single squashed
   commits. Do not put a two-parent merge commit on `main`.
7. Before deleting a branch, check for unique commits:
   `git rev-list --count <branch> ^<every other ref>`. Zero unique commits is
   safe; anything else needs the work extracted first.
8. `ls-remote refs/pull/*` sorts **lexicographically**, so `10` appears before
   `4`. Do not use `tail` to find the newest PR.

---

## 4. Current state — 2026-09-17 12:32

### Remote branches (9)

| branch | sha | note |
|---|---|---|
| `main` | `ecb48b4` | the squash spine |
| `integration-2026-09-17` | `e5723c3` | **PR #10 head**, clean baseline |
| `from-scratch-results` | `8893590` | the experiment line |
| `fix/protocol-interlock-reconciliation` | `ed15033` | 7 conflicts vs `main` |
| `feat/android-v3-live-voice` | `50beddc` | renamed from `v3APP` |
| `feat/cm-app-integration` | `ce3bf1d` | renamed from `cm-app-integration` |
| `feat/cm-train-unfreeze-top-n` | `0a5a3e8` | recovered stash; **superseded, do not build on** |
| `docs/readme-kavach-rebrand` | `552b26d` | renamed from `ayush` |
| `rescue-vr-2026-09-16` | `7e05188` | the 13-file rescue |

### Local

```
  feat/cm-train-unfreeze-top-n 0a5a3e8 [gone]
* from-scratch-results         0376c48 [ahead 1]
  integration-2026-09-17       e5723c3
  main                         ecb48b4 [gone]
```

- **Unpushed: exactly one commit** — `0376c48` *"docs(10): correct criterion 3
  to an eval-vs-eval comparison"*.
- Working tree clean.

### Pull requests

- **PR #10 is open**, head `e5723c39b98f88af5a70df4ba01a79ac19002aa3`, target
  `main`, zero conflicts. `main` is an ancestor of the head, so it is a
  fast-forward in graph terms.
- **It must be merged with "Squash and merge"** in the GitHub UI. Pushing the
  merge commit would put a two-parent merge on `main` and break the squash
  spine.
- `gh` is **not installed**, so this cannot be done from this machine.

---

## 5. Open items

| # | item | blocked on |
|---|---|---|
| 1 | Squash-merge **PR #10** | a human clicking the button; no `gh` here |
| 2 | Push `0376c48` (one commit) | nothing — just needs doing |
| 3 | `fix/protocol-interlock-reconciliation`: **7 conflicts** vs `main` (`README.md`, `data/demo/ground_truth.json`, `docs/10` add/add, `docs/11` add/add, `docs/RUNBOOK.md`, `ml/train_cm.py` add/add, `vanirakshak-backend/.../server/app.py`) | runbook §8.3 — the G3 narrative stance is a human decision |
| 4 | `from-scratch-results` has no type prefix | CONTRIBUTING.md convention; deliberately left alone as the user's active branch |
| 5 | `rescue-vr-2026-09-16` and `feat/cm-train-unfreeze-top-n` are candidates for retirement | verification that nothing unique remains |
| 6 | USB copy of `C:\vr-save` | no removable drive attached |

---

## 6. The one-paragraph version

Git is load-bearing here because the record *is* the product: the pre-registered
criteria, the immutable `reports/` JSON, and a claim that must be re-derivable.
This machine's git build has a defect that silently drops nested branch names —
which is what damaged the ref store in the first place — and every push failed
for a week because a system-scope credential helper hangs forever, not because
of the proxy or the token. A careless `git checkout -b` twice deleted large
parts of the tree. The tracked files were recoverable every time; the
**gitignored checkpoints were not**, and that is where the real loss happened:
`cm_scratch_S44/S45/S46.pt`, including the best seed the project has produced.
The response is `scripts/vr_git.py` — plumbing-only ref writes, protected
`main`, read-back verification, leased force-pushes — plus a hard rule never to
run `git checkout` or `git reset --hard` in this workspace, and to copy every
finished checkpoint to `C:\vr-save\ckpt` the moment it exists.
