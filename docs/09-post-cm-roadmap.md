# 09 — What remains after the trained countermeasure

`docs/08-cm-training-plan.md` ends the moment a measured AASIST checkpoint is
loaded and the docs carry its real numbers. This document answers the next
question: **that landed — now what?**

It exists because "train the CM" is the last item on the current backlog, and
finishing a backlog is not the same as being deployable. Everything below is
a real gap in this repo as of 2026-09-10, verified against the code rather
than assumed. Items are ordered by *what would hurt first in production*, not
by effort.

---

## P0 — Blocking any real deployment

### 1. There is no authentication on any endpoint

Every REST route and the WebSocket accept unauthenticated connections. Anyone
who can reach the port can enroll a voiceprint, stream audio, read the audit
chain (`GET /v1/audit`), or exhaust the box. `docs/INTEGRATION_GUIDE.md` §1
states this plainly and tells integrators to put it behind their own gateway,
which is honest but is not a solution — it just relocates the problem to
whoever deploys it.

What "done" looks like: per-client credentials on both the WS handshake and
REST; authorization on `/v1/enroll` specifically (enrolling a voiceprint for
someone else's `user_id` is an identity attack, not a config mistake); rate
limits on `/analyze` and `/enroll`; and `GET /v1/audit` restricted, since it
is a log of who was scored and when.

### 2. The semantic layer contradicts the zero-retention claim

`vanirakshak/semantic/signal.py` uploads the caller's 4.04 s audio window to
**Groq**, a third-party US cloud, for transcription. Meanwhile
`GET /v1/health` reports `zero_retention: true` and every `/analyze` response
carries `dpdp_compliance.zero_retention_verified: true` citing DPDP Act 2023
§8(7).

Both statements are individually defensible — the *server* does not persist
audio — but read together by an auditor or a user, "zero retention" implies
the audio does not leave. It does. Under DPDP this is a cross-border transfer
of biometric personal data to a processor, and the person on the call has not
consented to it in any flow this repo implements.

Options, in descending order of honesty:
- Run ASR on-device/on-prem (the OpenVINO Whisper path in
  **VaniRakshak SACH** already demonstrates local transcription) and drop the
  cloud dependency for the semantic layer entirely.
- Keep Groq but make it opt-in per session, disclosed in the UI, off by
  default, and reword the health/DPDP fields to say exactly what is and is
  not retained *and where it is sent*.

Doing nothing is the one option that isn't available, because the current
wording will be read as a guarantee it doesn't make.

### 3. Consent and lawful-basis for call recording is unimplemented

The system listens to a live call. India's DPDP Act requires notice and
consent from the data principal; the *other* party to the call is a data
principal too, and there is no flow anywhere in this repo that obtains,
records, or evidences their consent — `data/consent/` holds disclosures but
nothing enforces them at runtime. The Android guard
(**VaniRakshak Kavach**) makes this sharper, since it captures during real
calls, and app-store policy on call recording is its own separate constraint.

This is a legal-review item, not an engineering one. It should be answered
before a pilot with real callers, not after.

---

## P1 — The detection story is still one speaker deep

### 4. ASV ships as a random projection by default

`VR_ASV_BACKEND=stub` produces a deterministic random projection, not
biometrics. Real ECAPA-TDNN exists behind `VR_ASV_BACKEND=ecapa` and measured
cleanly on the demo corpus (genuine 0.854 vs impostor 0.044 — RUNBOOK), but
the *default* configuration a new operator boots is the stub, and
`EnrolledSpeakersPanel` will happily "enroll" voiceprints into it.

Make `ecapa` the default once the dependency cost is acceptable, or refuse to
enroll under the stub backend rather than storing a projection that looks
like a voiceprint in the UI.

### 5. One enrolled anchor, one corpus, two speakers

`ml/README.md` is explicit: the identity model is fine-tuned for a single
person and the corpus is two speakers (8 target clips vs 45 chunks of one
non-target). Every identity number this repo can currently produce is drawn
from that. Multi-speaker enrollment works mechanically; it has never been
*evaluated* multi-speaker.

Needed: an enrollment corpus with enough distinct speakers to compute a real
ASV EER, and per-speaker calibration rather than one global threshold.

### 6. Nothing re-evaluates the CM as attacks evolve

Even after `docs/08` lands, a countermeasure is a perishable good — new TTS
and voice-conversion systems appear continuously, and §4.4 of that document
records that in-domain excellence routinely does not survive contact with
in-the-wild audio. There is currently no scheduled re-evaluation, no drift
monitoring on score distributions, and no process for retiring a checkpoint.

Minimum viable version: a quarterly re-run of the eval battery against a
refreshed attack set, with the operating threshold re-derived and the health
endpoint surfacing the checkpoint's age.

---

## P2 — Operational maturity

### 7. Sessions and rate limits

`SESSIONS` is an in-process dict with no eviction beyond socket close; a
crash or restart drops every live call's state, and nothing bounds how many
sessions one client may open. Horizontal scaling is impossible as written —
two backend replicas do not share session or anchor state.

### 8. Audit chain operations

The chain now persists to JSONL (`VR_AUDIT_PATH`) and re-verifies on restart,
which closed the previous in-memory-only gap. What is still missing:
rotation (the process never truncates), retention policy, and any tamper
*response* — `verify_chain()` reports a broken chain but nothing acts on it.

### 9. Observability

There is no metrics endpoint, no structured logging, no alerting on the
system's own health (as distinct from `VR_ALERT_WEBHOOK_URL`, which alerts on
*call* risk). In production you want tier-distribution drift, ASR failure
rate, WS reconnect rate, and per-window latency percentiles — the latency
figures in `docs/04` are benchmark numbers, not production telemetry.

### 10. ~~No CI~~ — resolved

`.github/workflows/ci.yml` runs on every PR to `main`: backend unit tests,
`tests/e2e/`, `scripts/integration_check.py`, `scripts/check_docs.py`, and
frontend lint + build. All five passed on PR #1's head `0a34961`
(2026-09-13).

**Still not covered:** the Android `testDebugUnitTest` suite, which needs the
Android SDK and JDK 17 and is run by hand only. And CI triggers on pull
requests to `main`, not on direct pushes to feature branches, so work pushed
without an open PR is unverified until one exists.

---

## P3 — Product-level questions this repo cannot answer alone

- **What is the actual false-positive cost?** A `BLOCK` on a legitimate
  customer call is a real harm. There is no measured false-positive rate on
  genuine Indian telephony audio, and the tier thresholds (35/70) were set by
  judgement, not by a cost model.
- **Who is accountable for a wrong verdict?** The system engages a
  transaction interlock. If it blocks wrongly, or fails to block, the
  liability question needs an owner before a bank pilot.
- **Accessibility of the challenge.** The CHALLENGE tier asks the caller to
  repeat digits under a 400–4000 ms latency window. Speech impairments,
  elderly callers, poor lines, and non-Hindi/English speakers will fail this
  disproportionately. That is a fairness problem with a measurable
  disparate-impact question attached.

---

## Sequencing suggestion

1. **CI first** (§10) — cheapest, and protects everything after it.
2. **Auth + the Groq/DPDP disclosure** (§1, §2) — these are the two that make
   the difference between "demo" and "may be pointed at a real caller".
3. **Legal review of consent** (§3) in parallel, since it has lead time.
4. **Default to real ECAPA and build a multi-speaker eval set** (§4, §5).
5. Then the CM training run (`docs/08`), because by that point you'll have CI
   to verify it and a real eval set to measure it against.

That ordering deliberately puts the trained countermeasure **after** several
cheaper items. It is the most technically interesting piece of work left, and
also not the thing most likely to hurt someone if it stays undone.
