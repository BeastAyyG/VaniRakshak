package com.vericall.audio

import android.app.AlertDialog
import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.widget.ScrollView
import android.widget.TextView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Real-time developer diagnostics monitor and state tracker for telephony call audio.
 *
 * Tracks hardware AudioRecord metrics, sample throughput, voice presence,
 * STT events, and risk engine state.
 */
object VaniAudioDiagnosticsHud {

    data class LiveAudioMetrics(
        var deviceModel: String = "${Build.MANUFACTURER} ${Build.MODEL}",
        var androidVersion: String = "Android ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})",
        var callId: String = "IDLE",
        var callState: String = "DISCONNECTED",
        var isSpeakerphoneOn: Boolean = false,
        var activeAudioSource: String = "NONE",
        var directTelephonyAvailable: Boolean = false,
        var uplinkAvailable: Boolean = false,
        var downlinkAvailable: Boolean = false,
        var twoWayAvailable: Boolean = false,
        var audioFramesReceived: Long = 0L,
        var nonZeroAudioFrames: Long = 0L,
        var currentRmsEnergy: Double = 0.0,
        var peakAmplitude: Int = 0,
        var userVoiceDetected: Boolean = false,
        var callerVoiceDetected: Boolean = false,
        var sttProvider: String = "Android On-Device SpeechRecognizer",
        var isSttActive: Boolean = false,
        var lastSttEvent: String = "None",
        var lastSttEventTime: Long = 0L,
        var lastError: String = "None",
        var lastErrorTime: Long = 0L,
        var level1Matches: Int = 0,
        var level2ThreatScore: Float = 0.0f,
        var level3WebStatus: String = "IDLE"
    )

    private val metrics = LiveAudioMetrics()
    private val mainHandler = Handler(Looper.getMainLooper())

    fun updateCallState(state: String, callId: String = "") {
        metrics.callState = state
        if (callId.isNotBlank()) metrics.callId = callId
    }

    fun updateSpeakerphone(isSpeakerOn: Boolean) {
        metrics.isSpeakerphoneOn = isSpeakerOn
    }

    fun updateAudioSource(sourceName: String, isTwoWay: Boolean, uplink: Boolean, downlink: Boolean) {
        metrics.activeAudioSource = sourceName
        metrics.twoWayAvailable = isTwoWay
        metrics.uplinkAvailable = uplink
        metrics.downlinkAvailable = downlink
        metrics.directTelephonyAvailable = isTwoWay || (uplink && downlink)
    }

    fun recordAudioFrame(samplesCount: Int, nonZeroCount: Int, rms: Double, peak: Int) {
        metrics.audioFramesReceived++
        if (nonZeroCount > 0) {
            metrics.nonZeroAudioFrames++
        }
        metrics.currentRmsEnergy = rms
        metrics.peakAmplitude = peak
        if (rms > 12.0) {
            metrics.userVoiceDetected = true
        }
    }

    fun updateSttState(isActive: Boolean, provider: String = "") {
        metrics.isSttActive = isActive
        if (provider.isNotBlank()) metrics.sttProvider = provider
    }

    fun recordSttEvent(event: String) {
        metrics.lastSttEvent = event
        metrics.lastSttEventTime = System.currentTimeMillis()
    }

    fun recordError(error: String) {
        metrics.lastError = error
        metrics.lastErrorTime = System.currentTimeMillis()
    }

    fun recordRiskMetrics(l1Matches: Int, l2Score: Float, l3Status: String) {
        metrics.level1Matches = l1Matches
        metrics.level2ThreatScore = l2Score
        metrics.level3WebStatus = l3Status
    }

    fun getFormattedSummary(): String {
        val timeFmt = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault())
        val lastSttStr = if (metrics.lastSttEventTime > 0) {
            "${timeFmt.format(Date(metrics.lastSttEventTime))} — ${metrics.lastSttEvent}"
        } else "None"

        val lastErrStr = if (metrics.lastErrorTime > 0) {
            "${timeFmt.format(Date(metrics.lastErrorTime))} — ${metrics.lastError}"
        } else "None"

        return buildString {
            append("Device: ${metrics.deviceModel}\n")
            append("Android version: ${metrics.androidVersion}\n")
            append("Call ID: ${metrics.callId}\n")
            append("Call state: ${metrics.callState}\n")
            append("───────────────────────────────────────────\n")
            append("AUDIO CAPTURE DIAGNOSTICS\n")
            append("Direct telephony audio: ${if (metrics.directTelephonyAvailable) "YES" else "NO"}\n")
            append("Uplink (User): ${if (metrics.uplinkAvailable) "YES" else "NO"}\n")
            append("Downlink (Caller): ${if (metrics.downlinkAvailable) "YES" else "NO"}\n")
            append("Two-way: ${if (metrics.twoWayAvailable) "YES" else "NO"}\n")
            append("Audio source: ${metrics.activeAudioSource}\n")
            append("Speakerphone: ${if (metrics.isSpeakerphoneOn) "ON" else "OFF (Earpiece Mode)"}\n")
            append("Audio frames received: ${metrics.audioFramesReceived}\n")
            append("Non-zero audio frames: ${metrics.nonZeroAudioFrames}\n")
            append("Current RMS: ${"%.1f".format(metrics.currentRmsEnergy)} | Peak: ${metrics.peakAmplitude}\n")
            append("User voice detected: ${if (metrics.userVoiceDetected) "YES" else "NO"}\n")
            append("Caller voice detected: ${if (metrics.callerVoiceDetected) "YES" else "NO"}\n")
            append("───────────────────────────────────────────\n")
            append("STT & SPEECH ENGINE\n")
            append("STT: ${if (metrics.isSttActive) "ON" else "OFF"}\n")
            append("STT provider: ${metrics.sttProvider}\n")
            append("Last STT event: $lastSttStr\n")
            append("Last error: $lastErrStr\n")
            append("───────────────────────────────────────────\n")
            append("FRAUD RISK PIPELINE\n")
            append("Level 1 Keywords: ${metrics.level1Matches} hits\n")
            append("Level 2 Threat Score: ${"%.2f".format(metrics.level2ThreatScore)}\n")
            append("Level 3 Web Intelligence: ${metrics.level3WebStatus}\n")
        }
    }

    /**
     * Displays an auto-refreshing Developer Diagnostics Dialog on top of the given context.
     */
    fun showDiagnosticsDialog(context: Context) {
        val textView = TextView(context).apply {
            setPadding(40, 30, 40, 30)
            textSize = 13f
            typeface = android.graphics.Typeface.MONOSPACE
            setTextColor(0xFFE2E8F0.toInt())
            text = getFormattedSummary()
        }

        val scrollView = ScrollView(context).apply {
            setBackgroundColor(0xFF0F172A.toInt())
            addView(textView)
        }

        val dialog = AlertDialog.Builder(context)
            .setTitle("🔬 Live Telephony & Audio Diagnostics")
            .setView(scrollView)
            .setPositiveButton("Close", null)
            .setNeutralButton("Run Hardware Audit") { _, _ ->
                val caps = VaniCallAudioCapability.getCapabilities(context, forceRefresh = true)
                AlertDialog.Builder(context)
                    .setTitle("Hardware Audit Results")
                    .setMessage(caps.diagnosticSummary)
                    .setPositiveButton("OK", null)
                    .show()
            }
            .create()

        dialog.show()

        // Auto-refresh the dialog while visible
        val refreshRunnable = object : Runnable {
            override fun run() {
                if (dialog.isShowing) {
                    textView.text = getFormattedSummary()
                    mainHandler.postDelayed(this, 1000)
                }
            }
        }
        mainHandler.postDelayed(refreshRunnable, 1000)
    }
}
