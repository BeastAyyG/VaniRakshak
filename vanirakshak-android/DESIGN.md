# VaniRakshak Android — Design & Architecture

> Replaces a stray "Pinterest design-system analysis" that was previously
> committed under this filename and had nothing to do with VaniRakshak.

**Package:** `com.vericall`
**Module dir:** `vanirakshak-android/`
**Role in the system:** the on-device call-guard client. It owns the microphone
during a live call, ships audio to the VaniRakshak forensic backend, and
enforces the autonomous interlock (ALLOW / CHALLENGE / BLOCK) on the handset.

The web console (`frontend/`) and the FastAPI engine (`vanirakshak-backend/`)
are the rest of the system. **The server owns the verdict** — the phone renders
and enforces it, it does not compute it locally.

---

## 1. Runtime flow (mirrors `docs/FLOWCHARTS_CORRECTED.md`)

```
Incoming/outgoing call
      │
      ▼
VaniCallReceiver  (PHONE_STATE)  ──or──  VaniCallScreeningService (Android 7+)
      │
      ▼
VaniCallGuardService  — foreground service, owns the mic (16 kHz mono PCM)
      │
      ├──► VaniKeywordTrigger      — cheap on-device cue to decide when to sample
      ├──► VaniLiveVoiceAnalyzer   — windowing + acoustic features
      └──► VaniSpeakerVerifier     — POST /v1/verify-speaker  (identity only)
      │
      ▼
VaniRiskBus  — single source of truth for the live risk state
      │
      ▼
VaniScamShieldController — maps tier → user-facing defence
      │
      ├── ALLOW      → nothing (green)
      ├── CHALLENGE  → surface a live challenge, OTP shield ON
      └── BLOCK      → coercion overlay, OTP suppression, incident logged
```

## 2. Package map

| Package | Responsibility |
|---|---|
| `audio` | `VaniKeywordTrigger`, `VaniLiveVoiceAnalyzer`, `VaniSpeakerVerifier` |
| `receiver` | `VaniCallReceiver` (PHONE_STATE), `VaniNotificationActionReceiver` |
| `service` | `VaniCallGuardService` (mic + risk publisher), `VaniCallScreeningService` |
| `shield` | Coercion overlay/activity, OTP suppression, incident logger, watcher |
| `state` | `CallStateStore` — persisted per-call state |
| `temp` | UI: temp-contacts list/detail/bottom sheet |
| `util` | `VaniCallRouter`, `VaniNotificationHelper`, `VaniRiskBus` |
| `model` / `data` | `VaniContact`, `VaniSampleData` |

## 3. Permissions (see `AndroidManifest.xml`)

| Permission | Why |
|---|---|
| `RECORD_AUDIO` | capture call audio for forensic analysis |
| `READ_PHONE_STATE`, `READ_CALL_LOG` | detect real incoming calls |
| `READ_CONTACTS` | distinguish known contacts from unknown callers |
| `POST_NOTIFICATIONS` | risk alerts + interlock notices |
| `FOREGROUND_SERVICE_MICROPHONE` | required from Android 14 or `startForeground()` throws |
| `SYSTEM_ALERT_WINDOW` | coercion warning overlay above banking apps |
| `PACKAGE_USAGE_STATS` | detect which app is foreground during a high-risk call |
| `INTERNET` | reach the backend (currently via `adb reverse`) |

## 4. Backend integration (current state)

**Implemented:** `VaniSpeakerVerifier` encodes a PCM window as WAV and POSTs it
to `http://127.0.0.1:8000/v1/verify-speaker`, reached over
`adb reverse tcp:8000 tcp:8000`.

**Honest scope of that endpoint:** it answers *identity* only — "is this the
enrolled target voice?" It is **not** a clone/synthetic verdict. The model was
trained on one target vs one non-target speaker and has never seen cloned audio.
It returns `available=false` with a reason when the ML stack is absent; the app
must degrade to "unavailable" rather than faking a result.

**Not yet implemented (gap vs the flowchart):**
- No consumption of the fused **risk score 0–100** or the
  `ALLOW / CHALLENGE / BLOCK` tier (bands `<35 / 35–69 / ≥70`).
- No WebSocket streaming client for `ws://host:8000/ws/stream`
  (JSON handshake → binary 16-bit PCM @16 kHz → verdict frames).
- No **CHALLENGE** loop on-device: server issues a dynamic phrase; a replay
  cannot answer it and a live TTS is caught by response latency
  (400–4000 ms human window).
- No grading call to `POST /api/v1/sessions/{sid}/challenge`.
- No local rendering of the SHA-256 receipt / audit chain.

## 5. Honest scope & limitations

- On-device classes are **heuristic triggers and feature extraction**, not
  forensics. The forensic decision is the backend's.
- The backend's CM/ASV detectors are stubs pending real AASIST/ECAPA weights;
  numbers must not be presented as validated detection performance.
- Requires `adb reverse` (or a real network address) to reach the backend —
  `127.0.0.1` on the phone is the phone itself, not the dev machine.
- Call audio capture is legally sensitive: audio is processed in RAM only and
  never persisted, matching the DPDP Act 2023 zero-retention posture.

## 6. Build & run

Requires **JDK 17+**, the **Android SDK**, and Gradle 9.6 (wrapper provided).

```bash
# point local.properties at your SDK (currently C:\Android\Sdk)
./gradlew :app:assembleDebug
./gradlew :app:test                     # JVM unit tests
adb reverse tcp:8000 tcp:8000           # expose the backend to the phone
adb install app/build/outputs/apk/debug/app-debug.apk
```

Unit tests live in `app/src/test/kotlin/com/vericall/` and cover the risk bus,
call router, receiver, acoustic classification, incident logger and shield
controller.

## 7. Known gaps / TODO

### Deferred by decision (integration stays as-is)

These are *deliberate* scope decisions, not oversights. The client keeps the
single `/v1/verify-speaker` call; the full flowchart is exercised by the web
console instead.

- [ ] Consume fused risk + tier from the backend (WS streaming client).
- [ ] Implement the on-device CHALLENGE prompt + response grading
      (`POST /api/v1/sessions/{sid}/challenge`).
- [ ] Surface the SHA-256 audit receipt in the incident log.

### Open

- [ ] Replace the hardcoded `127.0.0.1` + `adb reverse` with a configurable
      backend host (a debug settings screen would remove the adb dependency).
- [ ] `Payment_Source_Code.zip` and `VaniRakshak_app-debug.apk` sit untracked in
      the working tree; `*.apk` is now gitignored. Consider deleting them — they
      are regenerable local artifacts, not source.
- [ ] This module currently cannot be compiled in the author's automation
      environment (no Android SDK, no Gradle, JDK 8 only — AGP needs 17+).
      All Kotlin changes must be built and tested in Android Studio.
