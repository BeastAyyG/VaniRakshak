# 23 — Small tasks for any agent

Self-contained cards from `docs/22-goals.md`. Each fits in one PR and needs no
design decision. Before starting: read `AGENTS.md` and `CONTRIBUTING.md`;
branch from `main` as `fix/…`, `feat/…` or `docs/…`; one card per PR.

**Every card's definition of done also includes:**
`pytest vanirakshak-backend/tests` green, `python scripts/check_docs.py` with
no hard failures, and, if `frontend/` changed, `npm run lint && npm run build`
in `frontend/`. Never overwrite a file in `reports/`. Wire-format changes go in
`docs/PROTOCOL.md` first.

Mark a card taken by opening a draft PR titled with its ID.

---

### A1 — Tests must not write the repo's audit log
`vanirakshak-backend/vanirakshak/server/app.py` builds `AUDIT` at import with
`settings.audit.path` (default `data/audit_log.jsonl`), so every test run
appends to the real file. In `vanirakshak-backend/tests/conftest.py`, point
`VR_AUDIT_PATH` at a temp file **before** the app is imported (a
`pytest_configure` hook or env in `pytest.ini`), or monkeypatch
`app_module.AUDIT` to `AuditLog(path=None)`.
**Done:** a test run leaves `data/audit_log.jsonl` unchanged (compare its
mtime/size before and after).

### A2 — Enrolment sessions leak into `SESSIONS`
`POST /v1/enroll` in `server/app.py` creates a `CallSession` `enroll-xxxx` and
stores it in `SESSIONS` forever. It is only needed for `sess.enrol(...)`.
Enrol through `PIPELINE.enrol(user_id, x, sr)` (or drop the session after
use) so nothing is added to `SESSIONS`.
**Done:** a test enrols 3 users and asserts `len(SESSIONS)` is unchanged;
`GET /v1/health` `active_sessions` does not grow on enrolment.

### A3 — Replace `audioop` before Python 3.13
`scripts/gpu_pipeline.py::_g711u` uses `audioop.lin2ulaw/ulaw2lin`
(deprecated, removed in 3.13). Write a bit-exact numpy G.711 mu-law
encode/decode (ITU-T G.711 segment table) in the same file.
**Done:** a new test in `vanirakshak-backend/tests/` asserts the numpy codec
equals `audioop` on all 65,536 int16 values (skip the test when `audioop` is
unavailable); no `import audioop` remains outside that test.

### A4 — Frontend drill words still contain "seven"
`frontend/app/page.tsx` (~line 39) keeps its own challenge word list with
"seven". The backend removed it (`vanirakshak/challenge/engine.py`
`ENGLISH_WORDS`) because the grader reads number words. Mirror the backend
list exactly.
**Done:** the two lists match; lint and build pass.

### A5 — Dashboard shows the new verdict fields
In `frontend/app/page.tsx`, show `metrics.tier_override` on each verdict
("BLOCK — voice matches but sounds synthetic" for `clone`, "not the enrolled
voice" for `impostor`). After a REST challenge grade, show `grade_source`,
`latency_ms`, `acoustic.verdict` and `transcript` (fields in
`docs/API_SPEC.md`). Display only; no logic change.
**Done:** fields render when present and nothing breaks when absent; lint and
build pass.

### A6 — Integration check: WS challenge fixture under fitted settings
`scripts/integration_check.py` "a challenge fired on the WS path" fails when
`VR_CHALLENGE_TRIGGER_CONSECUTIVE=3` (see `reports/t2_p1t_integration.json`):
the 6-window demo_01 fixture never reaches 3 consecutive CHALLENGE windows.
Stream the fixture enough times (or a longer high-risk fixture) that the
configured trigger can be met; read the value from `settings.challenge`.
**Done:** with `vanirakshak-backend/examples/decision_layer_fitted.env`
exported and `VR_ASV_BACKEND=ecapa VR_CM_BACKEND=aasist`, the check passes; the
stub run stays 25/25.

### A7 — Unit tests for the GPU pipeline's pure parts
`scripts/gpu_pipeline.py` has no tests. Add a new test module for it under
`vanirakshak-backend/tests/`, covering: `t1_grade` on a
small hand-made calibration JSON (P1-S/P1-F pass and fail cases);
`_d2_summarise` on fake shards including `unreadable` rows (they must be
excluded); `_g711u` length and dtype. No GPU, no corpus: build inputs in
`tmp_path` and monkeypatch the module's `T1_DIR` / `D2_DIR`.
**Done:** tests run in < 5 s without CUDA.

### A8 — Stale numbers in CONTRIBUTING.md
`CONTRIBUTING.md` says `make test` → "15 passed" and the integration check has
"17 checks". Today they are 159 and 25. Replace hard counts with "all pass"
and point to the commands, so they cannot go stale again.
**Done:** no hard-coded test counts remain in CONTRIBUTING.md.

### A9 — RUNBOOK section for the GPU box
Add to `docs/RUNBOOK.md`: how to run `scripts/gpu_pipeline.py` (stages,
`--only`, `--redo`, `--state-name`, resume behaviour, outputs under
`C:\vr-out`), how to start `scripts/gpu_dashboard.py` (127.0.0.1 by default,
`--host <tailscale-ip>` for your own devices; read-only), the ffmpeg AMR
requirement, and the Windows cuBLAS note in `challenge/transcribe.py`.
Take facts from the module docstrings; do not invent flags.
**Done:** every flag mentioned exists in `--help`.

### A10 — Clear the docs-check warnings
`python scripts/check_docs.py` prints ~60 WARN lines for loose paths such as
`docs/10` or `origin/main`. Either teach the checker to accept `docs/NN`
shorthand when `docs/NN-*.md` exists, or fix the references. Do not weaken a
hard check.
**Done:** warnings drop below 10; hard failures stay at 0.

### A11 — Startup prints become logging
`server/app.py` uses `print(...)` for start-up and runtime messages
(enrolment store, WebSocket errors). Switch to the module `logger`
at the right level; keep the messages.
**Done:** no `print(` left in `vanirakshak-backend/vanirakshak/` outside
`cli/`; tests pass.

### A12 — Voiceprint retention purge (docs/22 G6)
Add a retention setting in days (unset = keep forever; prefix it like
`VR_ENROL_STORE_PATH`) to `EnrolmentConfig` in `config.py`. At start-up, and from
a new purge script under `scripts/`, erase store records whose `created_at` is older than that, through
`EnrolmentStore.erase`, with an `ERASED` audit event per record (same features
as `DELETE /v1/enroll`). Document the variable in PROTOCOL.md §8.
**Done:** a test with a fake clock purges an old record and keeps a new one.

### A13 — Label the Android heuristic honestly (docs/22 G7)
In `vanirakshak-android`, the on-device `VaniLiveVoiceAnalyzer` produces an
untrained "fake" score. Wherever it is shown to the user, label it
"on-device heuristic — not a trained detector", and prefer the backend verdict
when one is available. No model or threshold changes.
**Done:** the label appears in the UI string resources; the app builds.

---

## Not for a small agent

These need full context or an owner decision (see `docs/22`): G1 step-up and
shadow mode, `docs/16` and any CM training (G3), data collection and consent
design (G2, G6), fusion refit and speaker-grouped simulation (G5), threshold
changes of any kind.
