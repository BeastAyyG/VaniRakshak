# 14 — Repository integration runbook

**Dated, 2026-09-16.** This is the ordered, verifiable sequence for folding the
four unintegrated work lines into `main`. Every step has a gate; **stop at the
first gate that fails** and fix it before continuing.

Decisions taken up front:

- **Spine = `main` (`ecb48b4`).** Every other line forks from a commit already
  inside `main`, and `main` carries docs/13 — the pre-registration that the later
  results must be consistent with.
- **Order of work = rescue → consolidate → merge → verify → push → retire.**
- Nothing here edits a pre-registered criterion or overwrites a
  `reports/*.json`. See §8.

## 1. Scope

In scope: the four work lines, the 21 uncommitted files, the `C:\vr-save`
archive, and the missing remote branches.

Out of scope, deliberately:

- `VaniRaksha-App-v1` / `-v2` — separate Android/iOS product line.
- `ayush` — already merged at `63c547d`.
- **Model weights.** `.gitignore` has `*.pt` / `*.pth` with only
  `!data/cm_aasist.pt` and `!data/demo/*.wav` whitelisted. A merge carries code,
  docs and reports. Checkpoints stay on disk and are reconciled by hand.

## 2. Topology (verified 2026-09-16)

Remote `origin` = `https://github.com/BeastAyyG/VaniRakshak.git`.

`main` is the squash spine: `63c547d` (merge `ayush`) → PR #1…#7 (`e661b9f`) →
`b3ff560` (#8, docs/13) → `ecb48b4` (#9, docs/13 amendment 1).

| line | head | where | unique content |
|---|---|---|---|
| protocol interlock | `ed15033` | remote + `C:\VaniRakshak` | G5 auth, G6/G7 honesty, G8 threshold, docs 09/11, G3 benchmark artifact — 11 commits |
| local twin | `0ff355e` | **`C:\VaniRakshak` only** | `ed15033` + 1 commit: seed-42 fine-tune eval |
| pre-registration | `a74f719` | `C:\vr-work\repo`, already pushed | docs/10 gate pre-registration, full-partition re-score |
| from-scratch results | `f374a05` | `C:\vr-results`, **not on remote** | S42/B42/S43/B43 results + criteria adjudication |

Useful facts, all measured:

- `base(0ff355e, ed15033)` = `ba0fdc7`. `bf53963` and `ba0fdc7` are **already
  ancestors of `ed15033`** — the local branch is `ed15033` plus exactly one
  commit. Do **not** merge the branch; cherry-pick the single commit (§6.2).
- `base(0ff355e, f374a05)` = `63c547d`.
- Non-repo archive: `C:\vr-save` (bundles, patches, `ckpt`,
  `C:\vr-save\RESCUE-REPORT.txt`).

## 3. Preconditions

- `C:\vr-work\repo` is the integration workspace — it is the only copy holding
  `main`, `ed15033`, `f374a05` and `a74f719` at once.
- Python for tests: `C:/VaniRakshak/.venv/Scripts/python.exe`. No `pip`, no
  `pytest` in that venv.
- Commit identity for agent work: `VaniRakshak Agent <vanirakshak@beastayyg.dev>`.

## 4. Step 0 — Freeze

```
cmd //c "taskkill /IM syncthing.exe /F"
git -C C:/VaniRakshak status --porcelain | wc -l
```

Syncthing owns `A:\cm-corpus\manifests`, which feeds every training run. It did
not damage the repos, but a run started mid-sync is not reproducible. Turn it
back on only after §10.

Unset the sandbox proxy before any network git operation — it is what produces
`send-pack: unexpected disconnect while reading sideband packet`:

```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

## 5. Step 1 — Rescue the uncommitted work first

These 21 files are the **only** copies of real work. Nothing else in this
runbook is allowed to run until they are committed somewhere.

**Never `git add -A`.** `.headrun/` is not gitignored and must not be swept in.

### 5.1 `C:\VaniRakshak` (13 files)

```
git -C C:/VaniRakshak checkout -b rescue-vr-2026-09-16
git -C C:/VaniRakshak rev-parse HEAD          # gate: must print 0ff355ee...
git -C C:/VaniRakshak add docs/API_SPEC.md docs/PROTOCOL.md \
  frontend/app/components/EnrolledSpeakersPanel.tsx frontend/app/page.tsx \
  scripts/evaluate_finetune_benchmark.py scripts/integration_check.py scripts/check_ft4_train.py \
  vanirakshak-backend/tests/test_smoke.py vanirakshak-backend/tests/test_stream_e2e.py \
  vanirakshak-backend/tests/test_auth.py \
  vanirakshak-backend/vanirakshak/config.py vanirakshak-backend/vanirakshak/server/app.py \
  vanirakshak-backend/vanirakshak/server/auth.py
git -C C:/VaniRakshak -c user.name="VaniRakshak Agent" \
  -c user.email="vanirakshak@beastayyg.dev" \
  commit -m "wip: rescue uncommitted interlock, auth and eval-harness work (2026-09-16)"
```

### 5.2 `C:\VaniRakshak-main` (7 entries — the irreplaceable ones)

Detached at `e661b9f`. This tree holds the experiment infra that exists
nowhere else: `scripts/experiment_{control,progress,state}.py`,
`tests/test_experiment_progress.py`, `reports/cm_from_scratch_2026-09-14.json`.

```
git -C C:/VaniRakshak-main checkout -b rescue-main-2026-09-16
git -C C:/VaniRakshak-main rev-parse HEAD     # gate: must print e661b9f...
git -C C:/VaniRakshak-main add scripts/experiment_control.py scripts/experiment_progress.py \
  scripts/experiment_state.py scripts/run_from_scratch_experiment.py \
  tests/test_experiment_progress.py reports/cm_from_scratch_2026-09-14.json
git -C C:/VaniRakshak-main -c user.name="VaniRakshak Agent" \
  -c user.email="vanirakshak@beastayyg.dev" \
  commit -m "wip: rescue uncommitted experiment-control infra and its report (2026-09-16)"
```

`.headrun/` stays untracked. Add it to `.gitignore` in a separate commit if you
want a clean `status`.

### 5.3 `C:\vr-results` (1 file)

```
git -C C:/vr-results checkout -b rescue-results-2026-09-16
git -C C:/vr-results add docs/10-quality-and-hardening-plan.md
git -C C:/vr-results -c user.name="VaniRakshak Agent" \
  -c user.email="vanirakshak@beastayyg.dev" \
  commit -m "wip: rescue uncommitted docs/10 edits (2026-09-16)"
```

## 6. Step 2 — Consolidate objects into `C:\vr-work\repo`

`0ff355e` exists only in `C:\VaniRakshak`. Move it by bundle — the path already
proven in `C:\vr-save`.

```
git -C C:/VaniRakshak bundle create C:/vr-save/xfer-0ff355e.bundle 0ff355e
git -C C:/vr-work/repo fetch --no-tags C:/vr-save/xfer-0ff355e.bundle 0ff355e
git -C C:/vr-work/repo tag keep-0ff355e FETCH_HEAD
git -C C:/vr-work/repo rev-parse keep-0ff355e    # gate: 0ff355eeed778ff1baee44368dbeae3bc87115af
```

Pull in the three rescue branches the same way. **Do not rely on
remote-tracking refs** — this repo stores them unreliably (§12), so always fetch
to `FETCH_HEAD` and pin a flat tag:

```
git -C C:/vr-work/repo fetch --no-tags C:/VaniRakshak        rescue-vr-2026-09-16
git -C C:/vr-work/repo tag keep-rescue-vr FETCH_HEAD
git -C C:/vr-work/repo fetch --no-tags C:/VaniRakshak-main   rescue-main-2026-09-16
git -C C:/vr-work/repo tag keep-rescue-main FETCH_HEAD
git -C C:/vr-work/repo fetch --no-tags C:/vr-results         rescue-results-2026-09-16
git -C C:/vr-work/repo tag keep-rescue-results FETCH_HEAD
git -C C:/vr-work/repo rev-parse keep-rescue-vr keep-rescue-main keep-rescue-results
```

## 7. Step 3 — Merge, in dependency order

```
git -C C:/vr-work/repo checkout -b integration-2026-09-16 ecb48b4
git -C C:/vr-work/repo rev-parse HEAD          # gate: ecb48b4...
```

### 7.1 `ed15033` — foundation (G5 auth, G6/G7 honesty, G8, docs 09/11)

```
git -C C:/vr-work/repo merge --no-ff ed15033 -m "merge: protocol interlock reconciliation (G5-G8)"
```

Expected conflicts: `docs/RUNBOOK.md`, `docs/09-*`, `docs/11-*`, backend
`config.py` / `app.py`.

### 7.2 `0ff355e` — one cherry-pick, not a branch merge

```
git -C C:/vr-work/repo cherry-pick 0ff355e
```

Cherry-picking instead of merging avoids resurrecting the 1-ahead/11-behind
divergence, which no longer exists once `ed15033` is in.

### 7.3 `a74f719` — pre-registration and full-partition re-score

```
git -C C:/vr-work/repo merge --no-ff a74f719 -m "merge: from-scratch pre-registration and re-score"
```

### 7.4 `f374a05` — results and adjudication

```
git -C C:/vr-work/repo merge --no-ff f374a05 -m "merge: from-scratch results and criteria adjudication"
```

Expect `reports/` and `docs/10` conflicts against 7.3. Resolve per §8.

## 8. Conflict rules — the ones git cannot decide

1. **Never overwrite an existing `reports/*.json`.** New run, new filename.
2. **Never silently edit a criterion.** Add a dated amendment note and leave the
   original visible.
3. **The G3 narrative needs a human decision.** `ed15033` contains
   `cd42c43 docs(G3): record the miss; G4's premise is backwards, stop CM work
   here`, while `0ff355e` and the `f374a05`/`a74f719` lines continue CM
   measurement regardless. Integrated docs must say which stance holds — do not
   let both survive unresolved.
4. **Dev ≠ eval.** Dev is A09–A16 with no codec column; eval is A17–A32 with
   codecs on 74.7 % of rows. The dev-derived `eer_threshold` in each checkpoint
   is **0.85–1.98 nats too high** for eval (4/4 runs). Never let a
   dev-selected threshold be presented as an eval number.
5. **State the baseline for any A28 claim.** Layer 5 = 12.945 %, shipped
   `cm_aasist.pt` = 26.220 %, from-scratch L12 = 27.772–31.863 %. "Improvement"
   is meaningless until the comparison point is named.
6. **Calibration cannot move EER or minDCF** — both are threshold-optimised and
   therefore invariant under any monotone transform. It is a reporting win
   (actDCF 2.27 → ~0.65), not a route to the G3 bar.

## 9. Step 5 — Verification gates

```
C:/VaniRakshak/.venv/Scripts/python.exe tests/run_all.py
git -C C:/vr-work/repo fsck --no-progress 2>&1 | grep -v dangling
git -C C:/vr-work/repo log --oneline --graph -20
```

After **every** ref-creating command in this runbook, re-run
`git rev-parse HEAD` and confirm it changed to what you asked for. On this box
`checkout -b` and `update-ref` report success while writing nothing.

Also confirm the four lines are truly merged:

```
for c in ed15033 0ff355e a74f719 f374a05; do
  git -C C:/vr-work/repo merge-base --is-ancestor $c HEAD && echo "$c IN" || echo "$c MISSING"
done
```

All four must print `IN`.

## 10. Step 6 — Push

```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
git -C C:/vr-work/repo push origin integration-2026-09-16
git -C C:/vr-work/repo push origin keep-rescue-vr:refs/heads/rescue-vr-2026-09-16
git -C C:/vr-work/repo push origin keep-rescue-main:refs/heads/rescue-main-2026-09-16
git -C C:/vr-work/repo push origin keep-rescue-results:refs/heads/rescue-results-2026-09-16
git -C C:/vr-work/repo push origin f374a05:refs/heads/exp/cm-from-scratch-results
git ls-remote --heads origin        # gate: confirm, do not trust git status
```

Then open the PR `integration-2026-09-16` → `main`, matching the squash-merge
convention of #1…#9. Direct push to `main` is available
(`... :refs/heads/main`) but breaks the repo's own history convention.

## 11. Step 7 — Retire the copies

Only after the push **and** a fresh-clone build test. Then remove
`C:\VaniRakshak-main` and `C:\vr-results`, keep `C:\VaniRakshak` (it holds the
only venv and the checkpoints) and keep `C:\vr-save` for two weeks. Restart
Syncthing last.

## 12. Trap list for this machine

- **Flat local branch names only.** Nested local names (`exp/foo`) silently fail
  on git 2.55.0.windows.3 — `checkout -b` and `update-ref` return exit 0 and
  write nothing. Push to the nested name anyway:
  `git push origin flat-name:refs/heads/exp/foo`.
- **No ref deeper than three components is stored locally**, which is why
  `git branch -r` is always empty and `git status` shows a bogus `[gone]`. Use
  `git ls-remote` for remote truth.
- `git -C C:/path` works; `git -C /c/path` does not — git is a native Windows
  binary.
- `nohup cmd &` inside a tool call does not survive the call returning. Use the
  runner's own background mechanism.
- `nvidia-smi` fails with `Failed to initialize NVML` while torch works fine.
- A full eval re-score is **~2.4 h** for 628 k utterances at ~74 utts/s. Do not
  re-run a condition set because you dislike its numbers.

## 13. Rollback

Every step before §10 is local and reversible: `C:\vr-work\repo` starts clean,
the three rescue branches are tagged `keep-*`, and `0ff355e` is preserved as a
bundle in `C:\vr-save`. To abandon the attempt, delete
`integration-2026-09-16` and restart from §7. Nothing on the remote changes
until §10.
