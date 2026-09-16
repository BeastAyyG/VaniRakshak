package com.vericall.service

import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.content.ContextCompat
import com.vericall.config.VaniConfig
import com.vericall.audio.VaniContinuousAudioEngine
import com.vericall.audio.VaniKeywordTrigger
import com.vericall.audio.VaniLiveVoiceAnalyzer
import com.vericall.audio.VaniSpeakerVerifier
import com.vericall.audio.VaniStreamingOrchestrator
import com.vericall.model.RiskLevel
import com.vericall.model.VaniContact
import com.vericall.shield.VaniCoercionOverlay
import com.vericall.shield.VaniForegroundWatcher
import com.vericall.shield.VaniScamShieldController
import com.vericall.util.VaniNotificationHelper
import com.vericall.util.VaniRiskBus

/**
 * Sole owner of the microphone during a call (via VaniContinuousAudioEngine).
 *
 * ARCHITECTURE (v2):
 *   Creates a single [VaniContinuousAudioEngine] instance that:
 *   - Opens AudioRecord ONCE at call start
 *   - Distributes PCM to STT, acoustic analysis, streaming, and WAV recording
 *   - Closes AudioRecord ONCE at call end
 *
 *   No more dual mic ownership. No more mic churn.
 *
 * Anything that wants live results subscribes to [VaniRiskBus].
 */
class VaniCallGuardService : Service() {

    /** The single continuous audio engine — sole mic owner for the entire call */
    private var audioEngine: VaniContinuousAudioEngine? = null
    private var foregroundWatcher: VaniForegroundWatcher? = null
    private var callerName: String = "Caller"
    private var activePhone: String? = null
    private var activeCallId: String? = null
    private var keywordTrigger: VaniKeywordTrigger? = null

    /** Sent by the UI "Verify Speaker" button (or a keyword trigger) */
    companion object {
        const val ACTION_VERIFY_SPEAKER = "com.vericall.action.VERIFY_SPEAKER"
        const val EXTRA_KEYWORD_TRIGGER = "extra_keyword_trigger"
        const val EXTRA_CALL_ID = "extra_call_id"
        val TRIGGER_KEYWORDS = setOf(
            "bank", "banking", "rupee", "rupees", "rupay", "rs ", "rs.", "payment",
            "transaction", "transfer", "account", "otp", "पैसा", "बैंक", "रुपया", "भुगतान"
        )
    }

    private val verifyReceiver = object : android.content.BroadcastReceiver() {
        override fun onReceive(context: android.content.Context?, intent: android.content.Intent?) {
            if (intent?.action == ACTION_VERIFY_SPEAKER) triggerSpeakerVerify()
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val name = intent?.getStringExtra(VaniNotificationHelper.EXTRA_CALLER_NAME) ?: "Caller"
        val phone = intent?.getStringExtra(VaniNotificationHelper.EXTRA_CALLER_PHONE) ?: ""
        val enrolled = intent?.getBooleanExtra(VaniNotificationHelper.EXTRA_CALLER_ENROLLED, false) ?: false
        val incomingCallId = intent?.getStringExtra(EXTRA_CALL_ID) ?: VaniRiskBus.activeCallId ?: "call_${System.currentTimeMillis()}"

        callerName = name

        // Initialize Scam-Shield session orchestrator
        VaniScamShieldController.onCallStarted(name, phone)
        if (foregroundWatcher == null) {
            foregroundWatcher = VaniForegroundWatcher(this).apply { start() }
        }

        // Avoid restarting if already actively capturing this call
        if (audioEngine?.isEngineRunning == true && (activePhone == phone || activeCallId == incomingCallId)) {
            android.util.Log.d("VaniCallGuardService", "Audio engine already active for $phone ($incomingCallId), skipping restart.")
            return START_NOT_STICKY
        }

        VaniNotificationHelper.createChannels(this)
        val notif = VaniNotificationHelper.buildAnalyzingNotification(this, name, phone, enrolled)

        // Robust FGS start: handles Android 14+ background restriction exemptions
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                startForeground(
                    VaniNotificationHelper.ID_LIVE,
                    notif,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
                )
            } catch (e: SecurityException) {
                android.util.Log.w("VaniCallGuardService", "Microphone FGS restricted, attempting phoneCall FGS: ${e.message}")
                try {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                        startForeground(
                            VaniNotificationHelper.ID_LIVE,
                            notif,
                            ServiceInfo.FOREGROUND_SERVICE_TYPE_PHONE_CALL
                        )
                    } else {
                        startForeground(VaniNotificationHelper.ID_LIVE, notif)
                    }
                } catch (e2: Exception) {
                    android.util.Log.e("VaniCallGuardService", "Fallback FGS start failed: ${e2.message}")
                }
            } catch (e: Exception) {
                android.util.Log.e("VaniCallGuardService", "FGS start failed: ${e.message}")
                runCatching { startForeground(VaniNotificationHelper.ID_LIVE, notif) }
            }
        } else {
            startForeground(VaniNotificationHelper.ID_LIVE, notif)
        }

        // Stop any existing engine before starting fresh
        audioEngine?.stop()
        audioEngine = null
        activePhone = phone
        activeCallId = incomingCallId
        VaniRiskBus.activeCallId = incomingCallId

        // Listen for the manual "Verify Speaker" broadcast
        registerReceiver(
            verifyReceiver,
            android.content.IntentFilter(ACTION_VERIFY_SPEAKER),
            android.content.Context.RECEIVER_NOT_EXPORTED
        )

        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            VaniRiskBus.publish(
                VaniRiskBus.Snapshot(
                    0, 0, 0, RiskLevel.WARNING,
                    getString(com.vericall.R.string.analyzer_no_permission)
                )
            )
            VaniNotificationHelper.showLiveRiskNotification(
                this, name, phone, 0, 0, 0, RiskLevel.WARNING,
                getString(com.vericall.R.string.analyzer_no_permission),
                enrolled
            )
            return START_NOT_STICKY
        }

        val contact = VaniContact(
            id = "live_call",
            name = name,
            phone = phone,
            relation = "Live incoming call",
            isVoiceEnrolled = enrolled
        )

        // ═══════════════════════════════════════════════════════════════════
        // NEW v2 ARCHITECTURE:
        // Single VaniContinuousAudioEngine owns the mic for the ENTIRE call.
        // It distributes PCM to: STT, acoustic analysis, streaming, WAV.
        // ═══════════════════════════════════════════════════════════════════

        val engine = VaniContinuousAudioEngine(this@VaniCallGuardService)

        // Set up acoustic analyzer as a consumer (no mic ownership)
        val analyzer = VaniLiveVoiceAnalyzer(this@VaniCallGuardService).apply {
            startAnalysis(contact)
            onRiskUpdated = { risk, synthetic, speakerMatch, level, message ->
                val prev = VaniRiskBus.last
                val snapshot = prev?.copy(
                    risk = maxOf(risk, prev.threatScore),
                    synthetic = synthetic,
                    speakerMatch = speakerMatch,
                    level = level,
                    message = message
                ) ?: VaniRiskBus.Snapshot(
                    risk = risk,
                    synthetic = synthetic,
                    speakerMatch = speakerMatch,
                    level = level,
                    message = message
                )
                VaniRiskBus.publish(snapshot)
                VaniScamShieldController.onRiskScoreUpdated(risk)
                VaniNotificationHelper.showLiveRiskNotification(
                    this@VaniCallGuardService, name, phone,
                    risk, synthetic, speakerMatch, level, message, enrolled
                )
            }
        }
        engine.acousticAnalyzer = analyzer

        // Set up remote streaming orchestrator (no-op if no backend configured)
        if (VaniConfig.hasRemoteBackend(this)) {
            engine.streamingOrchestrator = VaniStreamingOrchestrator(this@VaniCallGuardService)
        }

        // Set up transcript callback: feed utterances directly to VaniRiskBus so UI updates immediately
        engine.onTranscriptUpdated = { utterances ->
            val prev = VaniRiskBus.last
            val fullText = utterances.joinToString(" ") { it.text }
            val baseSnapshot = prev ?: VaniRiskBus.Snapshot(
                risk = 0, synthetic = 0, speakerMatch = 50,
                level = RiskLevel.SAFE, message = "🛡 Protection Active"
            )
            VaniRiskBus.publish(
                baseSnapshot.copy(
                    transcript = fullText,
                    utterances = utterances
                )
            )
        }

        // Set up scam alert callback
        engine.onScamAlert = { detection ->
            val prev = VaniRiskBus.last
            val alertLevel = when {
                detection.threatScore >= 70 -> RiskLevel.DANGER
                detection.threatScore >= 35 -> RiskLevel.WARNING
                else -> RiskLevel.SAFE
            }
            VaniNotificationHelper.showLiveRiskNotification(
                this@VaniCallGuardService, name, phone,
                detection.threatScore, prev?.synthetic ?: 0, prev?.speakerMatch ?: 50,
                alertLevel, "⚠️ ${detection.redFlags.firstOrNull() ?: "Suspicious pattern"}", enrolled
            )
        }

        // Start the continuous engine (opens mic ONCE)
        engine.start(incomingCallId, callerName = name, callerPhone = phone)
        audioEngine = engine

        android.util.Log.i("VaniCallGuardService",
            "═══ v2 Architecture: VaniContinuousAudioEngine is sole mic owner ═══")

        return START_NOT_STICKY
    }

    /**
     * Grab the last ~2s of call audio for speaker verification.
     */
    private fun triggerSpeakerVerify() {
        val pcm = audioEngine?.getRecentPcm(2.0f) ?: ShortArray(0)
        if (pcm.isEmpty()) {
            VaniRiskBus.publish(
                VaniRiskBus.Snapshot(
                    0, 0, 0, RiskLevel.WARNING,
                    "Speaker verify: not enough live audio captured yet."
                )
            )
            return
        }
        VaniSpeakerVerifier.verify(pcm, 16000, this) { verdict ->
            val (level, message) = when {
                !verdict.available ->
                    RiskLevel.WARNING to "Speaker verify unavailable: ${verdict.reason ?: "backend not reachable"}"
                verdict.isTarget ->
                    RiskLevel.SAFE to "Speaker verified — this is the enrolled voice (Modi ji)."
                else ->
                    RiskLevel.DANGER to "Speaker is NOT the enrolled Modi voice. Verify caller identity before any bank/transaction step."
            }
            VaniRiskBus.publish(
                VaniRiskBus.Snapshot(
                    risk = if (verdict.isTarget) 6 else 92,
                    synthetic = if (verdict.isTarget) 4 else 95,
                    speakerMatch = if (verdict.isTarget) 98 else 8,
                    level = level,
                    message = message
                )
            )
        }
    }

    override fun onDestroy() {
        val lastUtterances = audioEngine?.getTranscript() ?: emptyList()
        val lastSnapshot = VaniRiskBus.last
        val callId = activeCallId ?: VaniRiskBus.activeCallId

        // Stop the continuous audio engine (closes mic ONCE)
        audioEngine?.stop()
        audioEngine = null

        // Safety fallback: ensure any captured speech is preserved in call history
        if (!callId.isNullOrBlank()) {
            val existing = com.vericall.data.VaniCallHistoryRepository.getCallById(this, callId)
            val utterances = when {
                lastUtterances.isNotEmpty() -> lastUtterances
                !lastSnapshot?.utterances.isNullOrEmpty() -> lastSnapshot.utterances
                !lastSnapshot?.transcript.isNullOrBlank() -> listOf(
                    com.vericall.model.TranscriptUtterance(speaker = "You", text = lastSnapshot.transcript)
                )
                else -> emptyList()
            }
            if (existing != null && !existing.hasTranscript && utterances.isNotEmpty()) {
                val updated = existing.copy(
                    hasTranscript = true,
                    transcript = utterances
                )
                com.vericall.data.VaniCallHistoryRepository.saveCallRecord(this, updated)
            } else if (existing == null && utterances.isNotEmpty()) {
                val newRecord = com.vericall.model.VaniCallRecord(
                    id = callId,
                    number = activePhone ?: "",
                    callerName = callerName,
                    direction = com.vericall.model.CallDirection.INCOMING,
                    timestamp = System.currentTimeMillis(),
                    durationSec = 0,
                    hasTranscript = true,
                    transcript = utterances,
                    riskScore = lastSnapshot?.risk ?: 0,
                    riskLevel = lastSnapshot?.level?.name ?: "SAFE",
                    isVoiceVerified = false,
                    isProtected = (lastSnapshot?.risk ?: 0) >= 50
                )
                com.vericall.data.VaniCallHistoryRepository.saveCallRecord(this, newRecord)
            }
        }

        activePhone = null
        keywordTrigger?.stop()
        keywordTrigger = null
        runCatching { unregisterReceiver(verifyReceiver) }

        foregroundWatcher?.stop()
        foregroundWatcher = null
        VaniCoercionOverlay.hide(this)

        VaniScamShieldController.onCallEnded(this, lastSnapshot)

        lastSnapshot?.let { snapshot ->
            VaniNotificationHelper.showCallSummary(this, callerName, snapshot)
        }
        VaniRiskBus.reset()
        VaniNotificationHelper.cancelLive(this)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
        super.onDestroy()
    }
}
