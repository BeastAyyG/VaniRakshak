# VaniRakshak — Android call-guard client

On-device half of the VaniRakshak system. It owns the microphone during a live
call and enforces the autonomous interlock on the handset. See `DESIGN.md` for
architecture.

**Correction (this line previously said "does not compute forensics locally" —
that was wrong and is fixed here):** this app runs its OWN independent
real-time forensic heuristic, `VaniLiveVoiceAnalyzer` — pure Kotlin, on-device,
computing zero-crossing-rate variance, energy-plateau flatness, and spectral
diff ratio every ~1s of speech to produce a synthetic-voice score, a speaker-
match score, and a dynamic risk level (SAFE/WARNING/DANGER), entirely without
calling the backend. It does not share code, weights, or calibration with the
Python backend's `vanirakshak/detectors/{cm,cm_wavlm}.py` — two separate,
unreconciled "is this fake" heuristics exist in the system. Like the Python
CM stub, this Kotlin heuristic is hand-tuned thresholds, never trained or
validated against real synthetic/cloned audio — see docs/archive/AUDIT_REPORT.md §12.3.

The backend is used for exactly one thing: `POST /v1/verify-speaker`, the
identity check below. The server owns *that* verdict; the app renders it.

## Build

Requires **JDK 17+** and the **Android SDK** (Gradle 8.7 comes from the wrapper).

```bash
# local.properties must point at your SDK (it currently says C:\Android\Sdk)
./gradlew :app:assembleDebug
./gradlew :app:test                       # JVM unit tests
```

## Run against the backend

The backend listens on the dev machine at `:8000`. `127.0.0.1` on the phone is
the phone itself, so bridge the port:

```bash
adb reverse tcp:8000 tcp:8000
adb install app/build/outputs/apk/debug/app-debug.apk
```

Then start the backend from the repo root:

```bash
npm run dev:all        # or: uvicorn vanirakshak.server.app:app --port 8000
```

## Permissions

On first run grant: **Microphone**, **Phone**, **Contacts**, **Notifications**.
For the full Scam Shield also grant **Display over other apps**
(`SYSTEM_ALERT_WINDOW`) and **Usage access** (`PACKAGE_USAGE_STATS`) — both are
manual Settings toggles, not runtime dialogs.

## Backend integration — current surface

Exactly one endpoint is used:

| Endpoint | Purpose |
|---|---|
| `POST /v1/verify-speaker` | **Identity** check — "is this the enrolled target voice?" |

> **Honest scope:** this is an identity check, **not** a clone/synthetic
> verdict. The model was trained on one target vs one non-target speaker and has
> never seen cloned audio. When the ML stack is missing it returns
> `available=false` with a reason; the app degrades to "unavailable" and never
> fakes a result.

The fused risk score, `ALLOW / CHALLENGE / BLOCK` tiers, the WebSocket stream
(`ws://host:8000/ws/stream`) and the dynamic-phrase challenge are **intentionally
not wired into this client** — see "Known gaps" in `DESIGN.md`. The web console
(`frontend/`) exercises the full flowchart.

## Tests

```bash
./gradlew :app:test
```

JVM unit tests cover the risk bus, call router, call receiver, acoustic
classification, incident logger, and the scam-shield controller.

## Privacy

Call audio is processed in RAM only and never persisted, matching the DPDP Act
2023 zero-retention posture described in the backend's privacy layer.
