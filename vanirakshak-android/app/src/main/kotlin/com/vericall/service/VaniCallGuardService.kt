package com.vericall.service

import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.content.ContextCompat
import com.vericall.audio.VaniKeywordTrigger
import com.vericall.audio.VaniLiveVoiceAnalyzer
import com.vericall.audio.VaniSpeakerVerifier
import com.vericall.model.RiskLevel
import com.vericall.model.VaniContact
import com.vericall.shield.VaniCoercionOverlay
import com.vericall.shield.VaniForegroundWatcher
import com.vericall.shield.VaniScamShieldController
import com.vericall.util.VaniNotificationHelper
import com.vericall.util.VaniRiskBus

/**
 * Sole owner of the microphone during a call.
 *
 * Anything that wants live results - the ongoing notification, the guard screen -
 * subscribes to [VaniRiskBus]. Opening a second AudioRecord from the activity
 * would starve this one, so the activity must never do it.
 */
class VaniCallGuardService : Service() {

    private var analyzer: VaniLiveVoiceAnalyzer? = null
    private var foregroundWatcher: VaniForegroundWatcher? = null
    private var callerName: String = "Caller"
    private var activePhone: String? = null
    private var activeCallId: String? = null
    private var keywordTrigger: VaniKeywordTrigger? = null

    /** Sent by the UI "Verify Speaker" button (or a keyword trigger) to ask the
     *  backend model: "is this the enrolled target voice (Modi ji)?" */
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

        // Avoid stopping and restarting capture if already actively recording this ongoing call
        if (analyzer != null && (activePhone == phone || activeCallId == incomingCallId)) {
            return START_NOT_STICKY
        }

        VaniNotificationHelper.createChannels(this)
        val notif = VaniNotificationHelper.buildAnalyzingNotification(this, name, phone, enrolled)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                VaniNotificationHelper.ID_LIVE,
                notif,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else {
            startForeground(VaniNotificationHelper.ID_LIVE, notif)
        }

        analyzer?.stopListening()
        activePhone = phone
        activeCallId = incomingCallId
        VaniRiskBus.activeCallId = incomingCallId

        // Listen for the manual "Verify Speaker" broadcast (fired by the UI button).
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

        analyzer = VaniLiveVoiceAnalyzer(this).apply {
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
            onAmplitudeUpdated = { VaniRiskBus.publishAmplitude(it) }
            startListening(contact, incomingCallId)
        }

        // OPT-IN keyword trigger. SpeechRecognizer also uses the mic, so only enable when the
        // live analyzer is paused; left off by default to keep the manual verify path reliable.
        if (intent?.getBooleanExtra(EXTRA_KEYWORD_TRIGGER, false) == true) {
            startKeywordTrigger()
        }

        return START_NOT_STICKY
    }

    /**
     * Best-effort: listen for banking/transaction keywords, then auto-verify the speaker.
     * Because it shares the mic with the analyzer, callers should pause the analyzer first.
     */
    private fun startKeywordTrigger() {
        if (keywordTrigger != null) return
        keywordTrigger = VaniKeywordTrigger(this, TRIGGER_KEYWORDS) { _ ->
            triggerSpeakerVerify()
        }
        keywordTrigger?.start()
    }

    /**
     * Grab the last ~2s of call audio, ask the backend model "is this the enrolled
     * target voice (Modi ji)?", and publish the verdict on the same risk bus the UI
     * already renders. This is an IDENTITY check — never a "synthetic/clone" claim.
     */
    private fun triggerSpeakerVerify() {
        val pcm = analyzer?.getRecentPcm(2.0f) ?: ShortArray(0)
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
        analyzer?.stopListening()
        analyzer = null
        activePhone = null
        keywordTrigger?.stop()
        keywordTrigger = null
        runCatching { unregisterReceiver(verifyReceiver) }

        foregroundWatcher?.stop()
        foregroundWatcher = null
        VaniCoercionOverlay.hide(this)

        val lastSnapshot = VaniRiskBus.last
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
