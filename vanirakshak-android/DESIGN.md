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
      ├──► VaniKeywordTrigger      — on-device SpeechRecognizer, fires on bank/OTP/payment cues
      ├──► VaniLiveVoiceAnalyzer   — independent on-device heuristic: ZCR variance, energy-
      │                              plateau flatness, spectral diff ratio → its OWN synthetic-
      │                              score/speaker-match/risk-level, published to VaniRiskBus
      │                              WITHOUT calling the backend (see §5 — this is a full local
      │                              verdict engine, not just feature extraction)
      └──► VaniSpeakerVerifier     — POST /v1/verify-speaker  (identity only, the ONE
                                     backend call this app makes)
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

**Implemented:**
- `VaniSpeakerVerifier` encodes a PCM window as WAV and POSTs it to
  `{base}/v1/verify-speaker` (identity only), where the base host/port comes
  from `VaniBackendConfig` (Settings UI, default `127.0.0.1:8000`).
- `VaniStreamingClient` (OkHttp) opens `ws://host:8000/ws/stream`, sends a
  JSON handshake first (`session_id`, device, sample_rate…), then streams
  binary 16-bit PCM @16 kHz frames. It trusts the server verdict frames
  (`tier`, `risk`, `interlock_active`, `challenge`, `receipt_hash`)
  authoritatively; `VaniCallGuardService` tries this server path first and
  falls back to the on-device `VaniLiveVoiceAnalyzer` when unreachable.
- Challenge grading via `POST /api/v1/sessions/{sid}/challenge`
  (`VaniChallengeGrader`, advisory only — a pass never clears a server BLOCK).
- SHA-256 audit receipt (`receipt_hash`) rendered in the detail UI and
  persisted in the incident log (`receipt_hash` + `server_tier` columns).

**Honest scope of the identity endpoint:** it answers *identity* only — "is this the
enrolled target voice?" It is **not** a clone/synthetic verdict. The model was
trained on one target vs one non-target speaker and has never seen cloned audio.
It returns `available=false` with a reason when the ML stack is absent; the app
must degrade to "unavailable" rather than faking a result.

**Not yet implemented (gap vs the flowchart):**
- No consumption of the fused **risk score 0–100** beyond the WS verdict
  frames (bands `<35 / 35–69 / ≥70` are applied server-side).
- Response-latency grading (400–4000 ms human window) is partially wired
  (issued-at timestamp → grader) but not validated against real calls.
- No local rendering of the full SHA-256 audit chain (only the latest
  `receipt_hash` is shown/stored).

## 5. Honest scope & limitations

- **Correction (this bullet previously said on-device code does not compute
  forensics — that was wrong):** `VaniLiveVoiceAnalyzer` computes a full local
  forensic verdict — a synthetic-voice score, a speaker-match score, and a
  SAFE/WARNING/DANGER risk level with messages like *"neural voice clone
  confirmed"* — entirely on-device, without calling the backend. It is hand-
  tuned Kotlin thresholds (pitch-jitter/ZCR variance, energy-plateau flatness,
  spectral diff ratio), never trained or validated against real synthetic/
  cloned audio — the same category of limitation as the Python `cm.py` stub,
  but a **separate, unreconciled implementation**: it shares no code, weights,
  or calibration with `vanirakshak-backend/vanirakshak/detectors/`. The phone
  and the backend can disagree about the same audio and nothing reconciles it.
  Full writeup: `docs/archive/AUDIT_REPORT.md` §12.3.
- The one thing that *is* backend-authoritative is `VaniSpeakerVerifier`'s
  identity check (`/v1/verify-speaker`) — the server owns that verdict.
- The backend's CM/ASV detectors default to stubs; `VR_ASV_BACKEND=ecapa` /
  `VR_CM_BACKEND=wavlm` swap in real pretrained models (not wired into this
  Android client either way — it only ever calls `/v1/verify-speaker`).
  Numbers from either path must not be presented as validated anti-spoofing
  performance — see `docs/archive/AUDIT_REPORT.md` §12.
- Requires `adb reverse` (or a real network address) to reach the backend —
  `127.0.0.1` on the phone is the phone itself, not the dev machine.
- Call audio capture is legally sensitive: audio is processed in RAM only and
  never persisted, matching the DPDP Act 2023 zero-retention posture.

## 6. Build & run

Requires **JDK 17+**, the **Android SDK**, and Gradle 8.7 (wrapper provided).

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

### Done (was deferred, now implemented)

- [x] Consume fused risk + tier from the backend (WS streaming client).
- [x] On-device CHALLENGE prompt + response grading
      (`POST /api/v1/sessions/{sid}/challenge`, advisory only).
- [x] Surface the SHA-256 audit receipt in the incident log.
- [x] Configurable backend host (Settings screen; removes the hard
      `adb reverse` dependency — `adb reverse` remains the default path).

### Open
- [ ] `Payment_Source_Code.zip` and `VaniRakshak_app-debug.apk` sit untracked in
      the working tree; `*.apk` is now gitignored. Consider deleting them — they
      are regenerable local artifacts, not source.
- [ ] This module's build files were fixed without a compiling environment:
      the `kotlin-android` plugin is now applied (all sources are Kotlin) and
      versions are pinned to a real set (AGP 8.6.1 + Gradle 8.7 + Kotlin
      2.0.20 + JDK 17; the previous AGP 9.4.0 was never published). A static
      audit found Kotlin sources, resources, and the manifest internally
      consistent (every `R.*` reference resolves, all manifest components
      exist). Still required: one real `./gradlew :app:assembleDebug` +
      `:app:test` run with JDK 17 and the Android SDK — unavailable in the
      author's automation environment (no SDK, JDK 8 only). Build and test
      in Android Studio before trusting this module.
