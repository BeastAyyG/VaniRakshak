<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

# VaniRakshak Agent Rules & Architecture Directives

## 1. Core Mandate
VaniRakshak is a target-conditioned recorded-audio forensic system that streams 16-bit mono PCM audio at 16 kHz over WebSockets to a FastAPI forensic backend. The backend scores risk, detects synthetic audio/cloning, and enforces autonomous transaction interlocks.

## 2. Separation of Concerns
- **Backend owns the verdict & interlocks**: `vanirakshak-backend` computes the Bayesian fused risk score and determines whether an interlock is `ALLOW`, `CHALLENGE`, or `BLOCK`. The Next.js frontend only displays verdicts and locks controls accordingly.
- **WebSocket Protocol contract**: Any changes to message fields or formats MUST match `docs/PROTOCOL.md`. Both `frontend/app/page.tsx` and `vanirakshak-backend/vanirakshak/server/app.py` must stay in sync with this protocol.
- **Server-Side Alerting**: Outbound security webhooks (`vanirakshak/alerts/`) must execute server-side. Never dispatch alert webhooks or store webhook secrets in client code.

## 3. Git & Hygiene
- Keep `.opencode/`, `.workbuddy-ai/`, `.venv/`, and audio datasets out of git.
- Adhere to `CONTRIBUTING.md` for branch naming, PR syncs, and verification steps.
- **Push Target**: When requested to push code, always push to `https://github.com/BeastAyyG/VaniRakshak` (remote `origin`).

## 4. Android Development Workflow (`vanirakshak-android`)

### Architecture & Module
- **Module Location:** `vanirakshak-android/app`
- **Application ID / Package:** `com.vericall.temp` (namespace: `com.vericall`)
- **Launcher Activity:** `com.vericall.SplashActivity` -> `com.vericall.VaniMainActivity`
- **Minimum SDK:** 24 (Android 7.0)
- **Target & Compile SDK:** 35 (Android 15)
- **Java / JDK:** Java 17+ (built with Gradle 9.6 wrapper + Android Gradle Plugin 9.4.0)

### Key Source Directories
- `vanirakshak-android/app/src/main/kotlin/com/vericall/`:
  - `config/`: `VaniConfig.kt` (dynamic backend endpoint resolution & SharedPreferences)
  - `contacts/`: `VaniContactAdapter.kt`, `VaniEnrollmentSheet.kt`, `VaniForensicDetailActivity.kt`, `VaniContactsRosterActivity.kt`
  - `audio/`: `VaniLiveVoiceAnalyzer.kt` (mic capture & 16kHz PCM streaming), `VaniSpeakerVerifier.kt` (`POST /v1/verify-speaker`), `VaniKeywordTrigger.kt`
  - `service/`: `VaniCallGuardService.kt` (foreground mic service during live call), `VaniCallScreeningService.kt` (call screening role)
  - `shield/`: `VaniOtpShieldService.kt` (OTP suppression), `VaniCoercionOverlay.kt` (overlay defense over banking apps), `VaniScamShieldController.kt`
  - `receiver/`: `VaniCallReceiver.kt` (telephony state detection)

### Backend Integration
The Android app communicates with the VaniRakshak forensic server at `http://127.0.0.1:8000/v1/verify-speaker`.
Since `127.0.0.1` on the Android phone references the handset itself, reverse port forwarding must be active:
```powershell
adb reverse tcp:8000 tcp:8000
```

### Standard Commands Flow
1. **Build Debug APK:**
   ```powershell
   cd vanirakshak-android
   .\gradlew.bat :app:assembleDebug
   ```
   Generated APK: `vanirakshak-android/app/build/outputs/apk/debug/app-debug.apk`

2. **Install on Physical Handset:**
   ```powershell
   adb install -r app\build\outputs\apk\debug\app-debug.apk
   ```

3. **Launch Application:**
   ```powershell
   adb shell am start -n com.vericall.temp/.temp.TempContactsActivity
   ```

4. **Monitor Logcat:**
   ```powershell
   adb logcat -v time | Select-String "com.vericall|Vani|AndroidRuntime"
   ```

5. **Run JVM Unit Tests:**
   ```powershell
   .\gradlew.bat :app:test
   ```
