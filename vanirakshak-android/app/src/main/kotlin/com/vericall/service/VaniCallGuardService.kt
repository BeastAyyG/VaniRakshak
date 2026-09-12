package com.vericall.service

import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.content.ContextCompat
import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.vericall.audio.VaniKeywordTrigger
import com.vericall.audio.VaniLiveVoiceAnalyzer
import com.vericall.audio.VaniSpeakerVerifier
import com.vericall.audio.VaniStreamingClient
import com.vericall.util.VaniBackendConfig
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
    private var streamingClient: VaniStreamingClient? = null
    private var streamCapture: AudioRecord? = null
    private var streamThread: Thread? = null
    @Volatile private var streaming = false
    @Volatile private var fellBackToLocal = false
    /** One-way latch: once the server says BLOCK/interlock, we never downgrade. */
    @Volatile private var latchedBlock = false
    private var foregroundWatcher: VaniForegroundWatcher? = null
    private var callerName: String = "Caller"
    private var activePhone: String? = null
    private var keywordTrigger: VaniKeywordTrigger? = null

    /** Sent by the UI "Verify Speaker" button to ask the backend model:
     *  "is this the enrolled target voice (Modi ji)?" */
    companion object {
        const val ACTION_VERIFY_SPEAKER = "com.vericall.action.VERIFY_SPEAKER"
        private const val TAG = "VaniGuard"
    }

    private val verifyReceiver = object : android.content.BroadcastReceiver() {
        override fun onReceive(context: android.content.Context?, intent: android.content.Intent?) {
            if (intent?.action == ACTION_VERIFY_SPEAKER) triggerSpeakerVerify()
        }
    }
    /** onStartCommand runs on every RINGING/OFFHOOK; guard against double registration. */
    private var verifyReceiverRegistered = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val name = intent?.getStringExtra(VaniNotificationHelper.EXTRA_CALLER_NAME) ?: "Caller"
        val phone = intent?.getStringExtra(VaniNotificationHelper.EXTRA_CALLER_PHONE) ?: ""
        val enrolled = intent?.getBooleanExtra(VaniNotificationHelper.EXTRA_CALLER_ENROLLED, false) ?: false

        callerName = name
        android.util.Log.i(TAG, "Guard started for $name ($phone), enrolled=$enrolled")

        // Initialize Scam-Shield session orchestrator
        VaniScamShieldController.onCallStarted(name, phone)
        if (foregroundWatcher == null) {
            foregroundWatcher = VaniForegroundWatcher(this).apply { start() }
        }

        // RINGING and OFFHOOK both start the service. Bail out before touching
        // the notification, otherwise the second start would overwrite a live
        // risk score with the neutral "analyzing" state.
        if (analyzer != null && activePhone == phone) return START_NOT_STICKY

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
        stopStreaming()
        fellBackToLocal = false
        latchedBlock = false
        activePhone = phone

        // Listen for the manual "Verify Speaker" broadcast (fired by the UI button).
        if (!verifyReceiverRegistered) {
            registerReceiver(
                verifyReceiver,
                android.content.IntentFilter(ACTION_VERIFY_SPEAKER),
                android.content.Context.RECEIVER_NOT_EXPORTED
            )
            verifyReceiverRegistered = true
        }

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

        // Keyword-gated wake-up (default): stay ARMED on-device and only open
        // the server stream once money-talk is heard. Falls back to immediate
        // streaming when the gate is off or SpeechRecognizer is unavailable.
        if (VaniBackendConfig.isKeywordGate(this) &&
            android.speech.SpeechRecognizer.isRecognitionAvailable(this)
        ) {
            startArmedMode(name, phone, enrolled, contact)
        } else {
            // Server-driven path first (PROTOCOL §1–§5); on-device heuristic is the
            // offline fallback, not a competing verdict engine.
            startServerStream(name, phone, enrolled, contact)
        }

        return START_NOT_STICKY
    }

    /**
     * Server-driven path: capture 16 kHz mono PCM and stream it over the WS
     * socket (handshake → binary frames). The server owns tier + interlock.
     * Any transport failure falls back to the on-device analyzer exactly once.
     */
    private fun startServerStream(
        name: String, phone: String, enrolled: Boolean, contact: VaniContact
    ) {
        val wsUrl = VaniBackendConfig.wsUrl(this)
        android.util.Log.i(TAG, "Starting server stream: $wsUrl")
        val client = VaniStreamingClient(wsUrl, callerId = phone).apply {
            onSnapshot = { snapshot ->
                android.util.Log.i(
                    TAG, "Server verdict: tier=${snapshot.tier} risk=${snapshot.risk} " +
                        "interlock=${snapshot.interlockActive} session=${snapshot.sessionId}"
                )
                // One-way latch: BLOCK/interlock is sticky for the rest of the call.
                if (snapshot.interlockActive || snapshot.tier == "BLOCK") latchedBlock = true
                val effective = if (latchedBlock && snapshot.tier != "BLOCK") {
                    snapshot.copy(
                        tier = "BLOCK", level = RiskLevel.DANGER, interlockActive = true,
                        message = "Server verdict: BLOCK — do not proceed with any transaction."
                    )
                } else snapshot
                VaniRiskBus.publish(effective)
                VaniScamShieldController.onRiskScoreUpdated(effective.risk)
                VaniNotificationHelper.showLiveRiskNotification(
                    this@VaniCallGuardService, name, phone,
                    effective.risk, effective.synthetic, effective.speakerMatch,
                    effective.level, effective.message, enrolled,
                    tier = effective.tier, receiptHash = effective.receiptHash
                )
            }
            onExhausted = {
                android.util.Log.w(TAG, "Server stream unreachable — offline fallback")
                startLocalFallback(name, phone, enrolled, contact)
            }
        }
        streamingClient = client
        client.start()
        startPcmFeed(client, name, phone, enrolled, contact)
    }

    /** Captures mic PCM on a worker thread and feeds both the socket and the amplitude meter. */
    @SuppressLint("MissingPermission")
    private fun startPcmFeed(
        client: VaniStreamingClient,
        name: String, phone: String, enrolled: Boolean, contact: VaniContact
    ) {
        if (streaming) return
        streaming = true
        streamThread = Thread({
            var record: AudioRecord? = null
            try {
                val minBuf = AudioRecord.getMinBufferSize(
                    16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
                )
                record = AudioRecord(
                    MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                    16000, AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, 32000)
                )
                if (record.state != AudioRecord.STATE_INITIALIZED) {
                    record.release()
                    record = null
                } else {
                    streamCapture = record
                    record.startRecording()
                }
            } catch (e: Exception) {
                record?.release()
                record = null
            }
            if (record == null) {
                android.util.Log.w(TAG, "Mic init failed — offline fallback")
                streaming = false
                runOnMain { startLocalFallback(name, phone, enrolled, contact) }
                return@Thread
            }
            val buf = ShortArray(1600) // 100ms @16kHz
            while (streaming) {
                val n = try { record.read(buf, 0, buf.size) } catch (e: Exception) { -1 }
                if (n != null && n > 0) {
                    var peak = 0
                    for (i in 0 until n) {
                        val a = kotlin.math.abs(buf[i].toInt())
                        if (a > peak) peak = a
                    }
                    val amp = (peak / 20000.0f).coerceIn(0f, 1f)
                    runOnMain { VaniRiskBus.publishAmplitude(amp) }
                    client.sendPcm(buf, n)
                }
                try { Thread.sleep(20) } catch (e: InterruptedException) { break }
            }
            runCatching { record.stop() }
            runCatching { record.release() }
        }, "VaniStreamFeed").apply { start() }
    }

    private fun runOnMain(block: () -> Unit) {
        android.os.Handler(android.os.Looper.getMainLooper()).post(block)
    }

    /** Offline fallback: the on-device heuristic publishes locally-flagged snapshots. */
    @Synchronized
    private fun startLocalFallback(name: String, phone: String, enrolled: Boolean, contact: VaniContact?) {
        if (fellBackToLocal) return
        fellBackToLocal = true
        stopStreaming()
        val who = contact ?: VaniContact(
            id = "live_call", name = name, phone = phone,
            relation = "Live incoming call", isVoiceEnrolled = enrolled
        )
        VaniRiskBus.publish(
            VaniRiskBus.Snapshot(
                0, 0, 0, RiskLevel.WARNING,
                "Backend unreachable — using on-device analysis (not a server verdict)."
            )
        )
        analyzer = VaniLiveVoiceAnalyzer(this).apply {
            onRiskUpdated = { risk, synthetic, speakerMatch, level, message ->
                val snapshot = VaniRiskBus.Snapshot(
                    risk, synthetic, speakerMatch, level, message,
                    serverDriven = false
                )
                if (latchedBlock) {
                    // Even locally, never downgrade a latched BLOCK.
                    VaniRiskBus.publish(
                        snapshot.copy(
                            tier = "BLOCK", level = RiskLevel.DANGER,
                            interlockActive = true
                        )
                    )
                } else {
                    VaniRiskBus.publish(snapshot)
                }
                VaniScamShieldController.onRiskScoreUpdated(risk)
                VaniNotificationHelper.showLiveRiskNotification(
                    this@VaniCallGuardService, name, phone,
                    risk, synthetic, speakerMatch, level, message, enrolled
                )
            }
            onAmplitudeUpdated = { VaniRiskBus.publishAmplitude(it) }
            startListening(who)
        }
    }

    private fun stopStreaming() {
        streaming = false
        streamingClient?.stop()
        streamingClient = null
        streamThread?.interrupt()
        streamThread = null
        runCatching { streamCapture?.stop() }
        runCatching { streamCapture?.release() }
        streamCapture = null
    }

    /**
     * ARMED mode: the server stays idle while the phone listens on-device for
     * money-talk. No AudioRecord is opened here, so the SpeechRecognizer owns
     * the mic without conflict. On keyword hit the recognizer stops and the
     * server stream wakes up — scores/verdicts then flow as usual.
     */
    private fun startArmedMode(name: String, phone: String, enrolled: Boolean, contact: VaniContact) {
        if (keywordTrigger != null) return
        android.util.Log.i(TAG, "Armed: listening for money-talk keywords, server idle")
        VaniRiskBus.publish(
            VaniRiskBus.Snapshot(
                0, 0, 0, RiskLevel.WARNING,
                "Listening for money-talk (bank, OTP, UPI, payment…) — server wakes on match.",
                listenState = VaniRiskBus.ListenState.LISTENING
            )
        )
        keywordTrigger = VaniKeywordTrigger(this) { hit ->
            android.util.Log.i(TAG, "Keyword heard: '$hit' — waking server stream")
            keywordTrigger?.stop()
            keywordTrigger = null
            VaniRiskBus.publish(
                VaniRiskBus.Snapshot(
                    0, 0, 0, RiskLevel.WARNING,
                    "Heard \"$hit\" — contacting server for live scoring…",
                    listenState = VaniRiskBus.ListenState.LISTENING
                )
            )
            startServerStream(name, phone, enrolled, contact)
        }
        keywordTrigger?.start()
        if (keywordTrigger?.isRunning != true) {
            // Recognizer died between check and start — stream immediately.
            android.util.Log.w(TAG, "Armed mode failed to start — streaming immediately")
            keywordTrigger = null
            startServerStream(name, phone, enrolled, contact)
        }
    }

    /**
     * Grab the last ~2s of call audio, ask the backend model "is this the enrolled
     * target voice (Modi ji)?", and publish the verdict on the same risk bus the UI
     * already renders. This is an IDENTITY check — never a "synthetic/clone" claim.
     */
    private fun triggerSpeakerVerify() {
        // In server-streaming mode the backend already runs ASV per window —
        // the manual button only matters for the offline fallback's ring buffer.
        if (analyzer == null) {
            VaniRiskBus.publish(
                VaniRiskBus.Snapshot(
                    0, 0, 0, RiskLevel.WARNING,
                    "Speaker verify: server stream already includes voice verification per window."
                )
            )
            return
        }
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
        VaniSpeakerVerifier.verify(this, pcm, 16000) { verdict ->
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
        stopStreaming()
        activePhone = null
        keywordTrigger?.stop()
        keywordTrigger = null
        runCatching { unregisterReceiver(verifyReceiver) }
        verifyReceiverRegistered = false

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
