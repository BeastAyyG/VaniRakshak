# docs/ — index

What each document is and where to find it. Paths are relative to the repo root
(`C:\vr-work\repo`). Last verified against the working tree **2026-09-16**.

> **Looking for something by file, tool or dependency?** Use
> [`PROJECT_INVENTORY.md`](PROJECT_INVENTORY.md) — it is the file/tool map, this
> page is the document map.

---

## 1. Start here

| If you are… | Read |
|---|---|
| New to the project | [`11-handoff-checklist.md`](11-handoff-checklist.md), then [`PROJECT_INVENTORY.md`](PROJECT_INVENTORY.md) |
| Running it locally | [`../README.md`](../README.md) → [`RUNBOOK.md`](RUNBOOK.md) |
| Touching the WebSocket/API contract | [`PROTOCOL.md`](PROTOCOL.md) + [`API_SPEC.md`](API_SPEC.md) |
| Judging whether a result is honest | [`10-quality-and-hardening-plan.md`](10-quality-and-hardening-plan.md) |
| Folding branches together / pushing | [`14-repository-integration-runbook.md`](14-repository-integration-runbook.md) |

---

## 2. Numbered design documents

Numbered `NN-<slug>.md`. They are referenced everywhere by their **number**
(`docs/10`, `docs/13`, …) — that shorthand means "the file whose name starts
with that number", not a directory.

| # | File | Subject |
|---|---|---|
| 02 | [`02-technical-strategy-and-architecture.md`](02-technical-strategy-and-architecture.md) | Technical strategy & architecture |
| 03 | [`03-codec-augmentation-and-telephony.md`](03-codec-augmentation-and-telephony.md) | Codec augmentation & telephony simulation |
| 04 | [`04-metrics-and-honest-evaluation.md`](04-metrics-and-honest-evaluation.md) | Metrics & honest evaluation protocol |
| 07 | [`07-privacy-compliance-and-apis.md`](07-privacy-compliance-and-apis.md) | Privacy compliance & API reference |
| 08 | [`08-cm-training-plan.md`](08-cm-training-plan.md) | Trained countermeasure build runbook (AASIST) |
| 09 | [`09-post-cm-roadmap.md`](09-post-cm-roadmap.md) | What remains after the trained countermeasure |
| 10 | [`10-quality-and-hardening-plan.md`](10-quality-and-hardening-plan.md) | **Ground rules** + the pre-registered headline claim; `scripts/check_claim.py` adjudicates against this |
| 11 | [`11-handoff-checklist.md`](11-handoff-checklist.md) | Handoff checklist — contributor entry point |
| 12 | [`12-product-vision.md`](12-product-vision.md) | Product vision (internal) |
| 13 | [`13-calibration-decomposition-and-dev-eval-transfer.md`](13-calibration-decomposition-and-dev-eval-transfer.md) | Calibration decomposition & the dev→eval transfer gap |
| 14 | [`14-repository-integration-runbook.md`](14-repository-integration-runbook.md) | Repository/branch integration runbook |

**Numbering gaps are not missing files.** There is no `01`, `05` or `06` — the
sequence was renumbered during an earlier reorganisation. Don't go looking for
them.

---

## 3. Reference documents

| File | Subject |
|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | **Binding** frontend ↔ backend contract: WebSocket handshake, binary PCM framing, verdict JSON. `frontend/app/page.tsx` and `vanirakshak-backend/vanirakshak/server/app.py` must both match it. |
| [`API_SPEC.md`](API_SPEC.md) | HTTP API specification (REST endpoints, schemas). |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Architecture overview / narrative. |
| [`RUNBOOK.md`](RUNBOOK.md) | Operations runbook — what `dev:all` vs `dev:all:real` changes. |
| [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) | Third-party integration guide. |
| [`DESIGN.md`](DESIGN.md) | UI design system. |
| [`FLOWCHARTS_CORRECTED.md`](FLOWCHARTS_CORRECTED.md) | Corrected architecture flowcharts (rendered version: [`flowcharts.html`](flowcharts.html)). |
| [`PROJECT_INVENTORY.md`](PROJECT_INVENTORY.md) | **File, tool & dependency inventory** — the authoritative "what lives where" map. |

---

## 4. Archive

[`archive/`](archive/) holds superseded material — audit and integration reports
from a previous remediation cycle, plus the original project context. **Read for
history, not for current truth.** It is deliberately excluded from
`scripts/check_docs.py`, so its stale paths will not be flagged.

---

## 5. Keeping this index honest

`scripts/check_docs.py` is the drift gate (`make check-docs`). It fails the run
on three things only:

1. A backtick-quoted repo path with a real file extension (`.py`, `.ts`, `.tsx`,
   `.md`, `.json`, …) that resolves nowhere.
2. A `VR_*` env var named in a doc that does not exist in
   `vanirakshak-backend/vanirakshak/`.
3. A WebSocket verdict key asserted by `vanirakshak-backend/tests/test_stream_e2e.py`
   that `PROTOCOL.md`'s canonical example omits.

Everything else prints as a warning. Windows paths (`A:\cm-corpus\...`) and the
`docs/NN` shorthand are recognised and not flagged as missing.
