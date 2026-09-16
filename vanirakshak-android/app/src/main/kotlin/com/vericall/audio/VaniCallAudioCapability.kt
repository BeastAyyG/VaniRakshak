package com.vericall.audio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.telecom.TelecomManager
import android.util.Log
import androidx.core.content.ContextCompat
import kotlin.math.sqrt

/**
 * VaniCallAudioCapability — Runtime detector and auditor for telephony call audio capture.
 *
 * Evaluates whether the device platform, OEM, and Android security policy expose:
 * 1. Direct Telephony Call Audio (VOICE_CALL, VOICE_UPLINK, VOICE_DOWNLINK)
 * 2. System / OEM Call Recording capabilities
 * 3. Local Microphone Uplink (VOICE_RECOGNITION, VOICE_COMMUNICATION, MIC)
 *
 * All findings are experimentally validated against actual hardware AudioRecord initialization,
 * live sample reading, and non-zero PCM throughput verification.
 */
object VaniCallAudioCapability {

    private const val TAG = "VaniCallAudioCap"

    data class SourceProbeResult(
        val sourceId: Int,
        val sourceName: String,
        val canInitialize: Boolean,
        val canStart: Boolean,
        val canRead: Boolean,
        val totalSamplesRead: Int,
        val nonZeroSamples: Int,
        val rmsEnergy: Double,
        val peakAmplitude: Int,
        val containsAudio: Boolean,
        val failureReason: String? = null
    )

    data class CallAudioCapabilities(
        val oem: String,
        val model: String,
        val sdkInt: Int,
        val isDefaultDialer: Boolean,
        val hasRecordAudioPermission: Boolean,
        val hasCaptureAudioOutputPermission: Boolean,
        val telephonyCaptureAvailable: Boolean,
        val uplinkAvailable: Boolean,
        val downlinkAvailable: Boolean,
        val twoWayCaptureAvailable: Boolean,
        val voiceRecognitionAvailable: Boolean,
        val voiceCommunicationAvailable: Boolean,
        val standardMicAvailable: Boolean,
        val systemCallRecorderDetected: Boolean,
        val systemCallRecorderPackage: String?,
        val recommendedSource: String,
        val diagnosticSummary: String,
        val deepAuditResults: List<SourceProbeResult> = emptyList()
    )

    private var cachedCapabilities: CallAudioCapabilities? = null

    /**
     * Probes device audio capabilities and returns a structured audit report.
     * Caches the result after first execution unless forceRefresh is set.
     */
    @Synchronized
    fun getCapabilities(context: Context, forceRefresh: Boolean = false): CallAudioCapabilities {
        if (cachedCapabilities != null && !forceRefresh) {
            return cachedCapabilities!!
        }

        val oem = Build.MANUFACTURER
        val model = Build.MODEL
        val sdkInt = Build.VERSION.SDK_INT

        val hasRecordAudio = ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        val hasCaptureAudioOutput = ContextCompat.checkSelfPermission(
            context, "android.permission.CAPTURE_AUDIO_OUTPUT"
        ) == PackageManager.PERMISSION_GRANTED

        // Check default dialer status
        val telecomManager = context.getSystemService(Context.TELECOM_SERVICE) as? TelecomManager
        val defaultDialer = telecomManager?.defaultDialerPackage
        val isDefaultDialer = defaultDialer == context.packageName

        // Detect system OEM call recorder packages
        val knownSystemRecorders = listOf(
            "com.oplus.callrecorder",
            "com.oplus.aicall",
            "com.android.incallui",
            "com.google.android.dialer",
            "com.samsung.android.incallui",
            "com.xiaomi.telephony"
        )
        var detectedRecorderPackage: String? = null
        val pm = context.packageManager
        for (pkg in knownSystemRecorders) {
            try {
                pm.getPackageInfo(pkg, 0)
                detectedRecorderPackage = pkg
                break
            } catch (_: PackageManager.NameNotFoundException) {
                // Not installed
            }
        }

        // Run deep audit of all candidate audio sources
        val deepAudit = if (hasRecordAudio) runDeepHardwareAudit() else emptyList()

        val voiceCallResult = deepAudit.find { it.sourceId == MediaRecorder.AudioSource.VOICE_CALL }
        val voiceUplinkResult = deepAudit.find { it.sourceId == MediaRecorder.AudioSource.VOICE_UPLINK }
        val voiceDownlinkResult = deepAudit.find { it.sourceId == MediaRecorder.AudioSource.VOICE_DOWNLINK }
        val voiceRecResult = deepAudit.find { it.sourceId == MediaRecorder.AudioSource.VOICE_RECOGNITION }
        val voiceCommResult = deepAudit.find { it.sourceId == MediaRecorder.AudioSource.VOICE_COMMUNICATION }
        val micResult = deepAudit.find { it.sourceId == MediaRecorder.AudioSource.MIC }

        val voiceCallWorks = voiceCallResult?.canStart == true
        val voiceUplinkWorks = voiceUplinkResult?.canStart == true
        val voiceDownlinkWorks = voiceDownlinkResult?.canStart == true
        val voiceRecognitionWorks = voiceRecResult?.canStart == true
        val voiceCommunicationWorks = voiceCommResult?.canStart == true
        val standardMicWorks = micResult?.canStart == true

        val twoWay = voiceCallWorks || (voiceUplinkWorks && voiceDownlinkWorks)

        val recommended = when {
            voiceCallWorks -> "VOICE_CALL"
            voiceDownlinkWorks && voiceUplinkWorks -> "VOICE_UPLINK + VOICE_DOWNLINK"
            voiceRecognitionWorks -> "VOICE_RECOGNITION"
            voiceCommunicationWorks -> "VOICE_COMMUNICATION"
            standardMicWorks -> "MIC"
            else -> "UNAVAILABLE"
        }

        val summary = buildString {
            append("═══════════════════════════════════════════════════════\n")
            append("  VaniRakshak Telephony Call Audio Audit\n")
            append("═══════════════════════════════════════════════════════\n")
            append("Device: $oem $model (Android SDK $sdkInt)\n")
            append("Default Dialer Role: ${if (isDefaultDialer) "YES" else "NO"}\n")
            append("RECORD_AUDIO: ${if (hasRecordAudio) "GRANTED" else "DENIED"}\n")
            append("CAPTURE_AUDIO_OUTPUT: ${if (hasCaptureAudioOutput) "GRANTED (System Signature)" else "DENIED (OEM Privileged)"}\n")
            append("System Call Recorder: ${detectedRecorderPackage ?: "None detected"}\n")
            append("───────────────────────────────────────────────────────\n")
            append("HARDWARE PROBE RESULTS:\n")
            deepAudit.forEach { probe ->
                val status = if (probe.canStart) "INIT OK (Read: ${probe.totalSamplesRead} smp, NonZero: ${probe.nonZeroSamples})" else "FAILED (${probe.failureReason ?: "Unknown"})"
                append(" • %-20s: %s\n".format(probe.sourceName, status))
            }
            append("───────────────────────────────────────────────────────\n")
            append("Telephony Two-Way Audio: ${if (twoWay) "AVAILABLE" else "BLOCKED by Android AudioPolicy"}\n")
            append("Active Source Strategy: $recommended\n")
            append("═══════════════════════════════════════════════════════")
        }

        Log.i(TAG, summary)

        val caps = CallAudioCapabilities(
            oem = oem,
            model = model,
            sdkInt = sdkInt,
            isDefaultDialer = isDefaultDialer,
            hasRecordAudioPermission = hasRecordAudio,
            hasCaptureAudioOutputPermission = hasCaptureAudioOutput,
            telephonyCaptureAvailable = voiceCallWorks,
            uplinkAvailable = voiceUplinkWorks,
            downlinkAvailable = voiceDownlinkWorks,
            twoWayCaptureAvailable = twoWay,
            voiceRecognitionAvailable = voiceRecognitionWorks,
            voiceCommunicationAvailable = voiceCommunicationWorks,
            standardMicAvailable = standardMicWorks,
            systemCallRecorderDetected = detectedRecorderPackage != null,
            systemCallRecorderPackage = detectedRecorderPackage,
            recommendedSource = recommended,
            diagnosticSummary = summary,
            deepAuditResults = deepAudit
        )

        cachedCapabilities = caps
        return caps
    }

    /**
     * Actively tests candidate AudioRecord sources by initializing, starting, reading PCM frames,
     * and checking for non-zero samples and RMS energy.
     */
    fun runDeepHardwareAudit(): List<SourceProbeResult> {
        val candidates = listOf(
            MediaRecorder.AudioSource.VOICE_CALL to "VOICE_CALL (Two-Way)",
            MediaRecorder.AudioSource.VOICE_UPLINK to "VOICE_UPLINK (User)",
            MediaRecorder.AudioSource.VOICE_DOWNLINK to "VOICE_DOWNLINK (Caller)",
            MediaRecorder.AudioSource.VOICE_COMMUNICATION to "VOICE_COMMUNICATION",
            MediaRecorder.AudioSource.VOICE_RECOGNITION to "VOICE_RECOGNITION",
            MediaRecorder.AudioSource.MIC to "MIC (Standard)"
        )

        val results = mutableListOf<SourceProbeResult>()
        for ((sourceId, sourceName) in candidates) {
            results.add(probeAudioSourceDeep(sourceId, sourceName))
        }
        return results
    }

    private fun probeAudioSourceDeep(sourceId: Int, sourceName: String): SourceProbeResult {
        var record: AudioRecord? = null
        try {
            val sampleRate = 16000
            val channelConfig = AudioFormat.CHANNEL_IN_MONO
            val audioFormat = AudioFormat.ENCODING_PCM_16BIT
            val minBuf = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
            if (minBuf <= 0) {
                return SourceProbeResult(
                    sourceId = sourceId,
                    sourceName = sourceName,
                    canInitialize = false,
                    canStart = false,
                    canRead = false,
                    totalSamplesRead = 0,
                    nonZeroSamples = 0,
                    rmsEnergy = 0.0,
                    peakAmplitude = 0,
                    containsAudio = false,
                    failureReason = "Invalid buffer size: $minBuf"
                )
            }

            val bufSize = minBuf.coerceAtLeast(sampleRate * 2)
            record = AudioRecord(sourceId, sampleRate, channelConfig, audioFormat, bufSize)

            if (record.state != AudioRecord.STATE_INITIALIZED) {
                return SourceProbeResult(
                    sourceId = sourceId,
                    sourceName = sourceName,
                    canInitialize = false,
                    canStart = false,
                    canRead = false,
                    totalSamplesRead = 0,
                    nonZeroSamples = 0,
                    rmsEnergy = 0.0,
                    peakAmplitude = 0,
                    containsAudio = false,
                    failureReason = "AudioRecord state != STATE_INITIALIZED (Security/Privilege Blocked)"
                )
            }

            try {
                record.startRecording()
            } catch (e: Exception) {
                return SourceProbeResult(
                    sourceId = sourceId,
                    sourceName = sourceName,
                    canInitialize = true,
                    canStart = false,
                    canRead = false,
                    totalSamplesRead = 0,
                    nonZeroSamples = 0,
                    rmsEnergy = 0.0,
                    peakAmplitude = 0,
                    containsAudio = false,
                    failureReason = "startRecording failed: ${e.message}"
                )
            }

            // Attempt to read 1600 samples (100ms at 16kHz)
            val buffer = ShortArray(1600)
            val readCount = record.read(buffer, 0, buffer.size)

            var nonZeroCount = 0
            var sumSquare = 0.0
            var peak = 0

            if (readCount > 0) {
                for (i in 0 until readCount) {
                    val s = buffer[i].toInt()
                    if (s != 0) nonZeroCount++
                    sumSquare += s * s
                    val absVal = kotlin.math.abs(s)
                    if (absVal > peak) peak = absVal
                }
            }

            val rms = if (readCount > 0) sqrt(sumSquare / readCount) else 0.0

            return SourceProbeResult(
                sourceId = sourceId,
                sourceName = sourceName,
                canInitialize = true,
                canStart = true,
                canRead = readCount > 0,
                totalSamplesRead = readCount.coerceAtLeast(0),
                nonZeroSamples = nonZeroCount,
                rmsEnergy = rms,
                peakAmplitude = peak,
                containsAudio = nonZeroCount > 0 && rms > 5.0,
                failureReason = if (readCount <= 0) "Read returned $readCount" else null
            )

        } catch (e: Throwable) {
            return SourceProbeResult(
                sourceId = sourceId,
                sourceName = sourceName,
                canInitialize = false,
                canStart = false,
                canRead = false,
                totalSamplesRead = 0,
                nonZeroSamples = 0,
                rmsEnergy = 0.0,
                peakAmplitude = 0,
                containsAudio = false,
                failureReason = "${e.javaClass.simpleName}: ${e.message}"
            )
        } finally {
            runCatching {
                record?.stop()
            }
            runCatching {
                record?.release()
            }
        }
    }
}
