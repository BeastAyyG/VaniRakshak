package com.vericall.audio

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.vericall.model.RiskLevel
import com.vericall.model.TranscriptUtterance
import com.vericall.util.VaniRiskBus
import kotlin.math.max

/**
 * VaniContinuousAudioEngine — Coordinates telephony capture, STT, and the 3-tier risk pipeline.
 *
 * ARCHITECTURE PRINCIPLES:
 * 1. Probes hardware capabilities to determine whether direct two-way telephony audio is accessible.
 * 2. If direct telephony audio is available, captures raw PCM via [VaniTelephonyAudioSource] (VOICE_CALL).
 * 3. If direct telephony audio is restricted by Android AudioPolicy, uses the honest microphone fallback.
 * 4. Strictly avoids concurrent microphone ownership conflicts (never runs a raw mic AudioRecord
 *    while SpeechRecognizer is actively listening).
 * 5. Audio hardware opens ONCE at call start and closes ONCE at call end. No churn, no chimes.
 */
class VaniContinuousAudioEngine(private val context: Context) {

    companion object {
        private const val TAG = "VaniContinuousAudio"
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val CHUNK_SIZE_SAMPLES = 1600  // 100ms at 16kHz
    }

    // ─── State ──────────────────────────────────────────────────────────────────

    private var callAudioSource: VaniCallAudioSource? = null
    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null

    @Volatile
    private var isRunning = false
    val isEngineRunning: Boolean get() = isRunning

    private var activeAudioSourceName: String = "NONE"
    private var callId: String = ""
    private val mainHandler = Handler(Looper.getMainLooper())

    // ─── Consumers ──────────────────────────────────────────────────────────────

    var acousticAnalyzer: VaniLiveVoiceAnalyzer? = null
    var streamingOrchestrator: VaniStreamingOrchestrator? = null

    var onAmplitudeUpdated: ((Float) -> Unit)? = null
    var onTranscriptUpdated: ((List<TranscriptUtterance>) -> Unit)? = null
    var onScamAlert: ((VaniScamDetector.DetectionResult) -> Unit)? = null
    var onClaimExtracted: ((String) -> Unit)? = null

    // ─── Ring buffer for speaker verification ───────────────────────────────────

    private val recentCapacity = SAMPLE_RATE * 4  // 4 seconds
    private val recentPcm = ShortArray(recentCapacity)
    private var recentPos = 0
    private var recentFilled = 0
    private val recentLock = Any()

    fun getRecentPcm(seconds: Float = 2.0f): ShortArray {
        val want = (SAMPLE_RATE * seconds).toInt().coerceAtMost(recentCapacity)
        synchronized(recentLock) {
            if (recentFilled < want) return ShortArray(0)
            val out = ShortArray(want)
            var idx = (recentPos - want)
            if (idx < 0) idx += recentCapacity
            for (i in 0 until want) {
                out[i] = recentPcm[idx]
                idx = (idx + 1) % recentCapacity
            }
            return out
        }
    }

    private fun pushRecent(buf: ShortArray, count: Int) {
        synchronized(recentLock) {
            for (i in 0 until count) {
                recentPcm[recentPos] = buf[i]
                recentPos = (recentPos + 1) % recentCapacity
                if (recentFilled < recentCapacity) recentFilled++
            }
        }
    }

    // ─── WAV recording ──────────────────────────────────────────────────────────

    private var wavOutputStream: java.io.FileOutputStream? = null
    private var wavFile: java.io.File? = null
    private var totalAudioBytesWritten: Long = 0L

    // ─── Speech Engine ──────────────────────────────────────────────────────────

    private var speechEngine: VaniLiveSpeechEngine? = null

    // ─── Risk Pipeline ──────────────────────────────────────────────────────────

    private var contextualRiskEngine: VaniContextualRiskEngine? = null
    private var webIntelligenceEngine: VaniWebIntelligenceEngine? = null
    private var riskFusionEngine: VaniRiskFusionEngine? = null

    // ─── Public API ─────────────────────────────────────────────────────────────

    /**
     * Starts the continuous audio protection pipeline for an active cellular call.
     */
    @SuppressLint("MissingPermission")
    fun start(sessionCallId: String, callerName: String = "", callerPhone: String = "") {
        if (isRunning) {
            Log.d(TAG, "start() called while already running, ignoring")
            return
        }

        callId = sessionCallId
        isRunning = true
        totalAudioBytesWritten = 0L

        Log.i(TAG, "═══════════════════════════════════════════════════════")
        Log.i(TAG, "  VaniContinuousAudioEngine STARTING")
        Log.i(TAG, "  Session: $callId")
        Log.i(TAG, "═══════════════════════════════════════════════════════")

        VaniAudioDiagnosticsHud.updateCallState("ACTIVE", callId)

        // 1. Audit capabilities and select the audio capture strategy
        val caps = VaniCallAudioCapability.getCapabilities(context)
        val hasTelephonyTwoWay = caps.telephonyCaptureAvailable

        activeAudioSourceName = if (hasTelephonyTwoWay) {
            "Telephony Two-Way (VOICE_CALL)"
        } else {
            "Microphone Uplink (${caps.recommendedSource})"
        }

        Log.i(TAG, "Selected audio strategy: $activeAudioSourceName (Telephony Available: $hasTelephonyTwoWay)")

        // 2. Initialize WAV recording file
        initWavRecording()

        // 3. Initialize the 3-tier risk pipeline
        initRiskPipeline()

        val speakerLabel = if (hasTelephonyTwoWay) "You" else "You (Microphone)"

        // 4. Audio capture & ASR strategy:
        // When direct two-way telephony audio is available, capture via VOICE_CALL.
        // Otherwise, prioritize high-accuracy remote Faster-Whisper streaming when connected,
        // with instant automatic fail-safe fallback to on-device SpeechRecognizer when untethered.
        if (hasTelephonyTwoWay) {
            val telephonySource = VaniTelephonyAudioSource()
            callAudioSource = telephonySource
            telephonySource.start { pcmBytes, readBytes ->
                handleTelephonyPcm(pcmBytes, readBytes)
            }
            if (streamingOrchestrator != null) {
                streamingOrchestrator?.onTranscriptUpdated = { utterances ->
                    onTranscriptUpdated?.invoke(utterances)
                    contextualRiskEngine?.onUtterancesUpdated(utterances)
                }
                streamingOrchestrator?.startStreaming(
                    callerName = callerName,
                    callerPhone = callerPhone,
                    claimedIdentity = callerName,
                    callId = callId
                )
            } else {
                startSpeechEngine(speakerLabel)
            }
        } else {
            // Microphone mode
            if (streamingOrchestrator != null) {
                Log.i(TAG, "✓ Remote streaming orchestrator configured: wiring dual ASR fail-safe")
                streamingOrchestrator?.onTranscriptUpdated = { utterances ->
                    onTranscriptUpdated?.invoke(utterances)
                    contextualRiskEngine?.onUtterancesUpdated(utterances)
                }
                streamingOrchestrator?.onConnectionStateChanged = { connected ->
                    if (connected) {
                        Log.i(TAG, "✓ Gateway connected: Faster-Whisper active, stopping local SpeechRecognizer")
                        stopSpeechEngine()
                        if (audioRecord == null && isRunning) {
                            startMicrophoneCaptureThread()
                        }
                    } else {
                        Log.w(TAG, "⚠️ Gateway disconnected / unreachable: activating on-device SpeechRecognizer")
                        stopMicrophoneCapture()
                        startSpeechEngine(speakerLabel)
                    }
                }

                // Initially start mic capture for remote streaming
                startMicrophoneCaptureThread()
                streamingOrchestrator?.startStreaming(
                    callerName = callerName,
                    callerPhone = callerPhone,
                    claimedIdentity = callerName,
                    callId = callId
                )

                // Safety fallback timer: if gateway does not establish connection in 2.5s, engage local SpeechRecognizer
                mainHandler.postDelayed({
                    if (isRunning && streamingOrchestrator?.isGatewayConnected != true && speechEngine == null) {
                        Log.w(TAG, "⚠️ Gateway not connected within 2.5s — activating on-device SpeechRecognizer")
                        stopMicrophoneCapture()
                        startSpeechEngine(speakerLabel)
                    }
                }, 2500)
            } else {
                startSpeechEngine(speakerLabel)
            }
        }

        val statusMsg = if (hasTelephonyTwoWay) {
            "🛡 Two-Way Telephony Protection Active"
        } else {
            "🛡 Monitoring Your Voice • Android Caller Audio Restricted"
        }
        publishStatus(statusMsg)
    }

    private fun startSpeechEngine(speakerLabel: String) {
        if (speechEngine != null) return
        mainHandler.post {
            if (!isRunning || speechEngine != null) return@post
            val engine = VaniLiveSpeechEngine(context)
            engine.onTranscriptUpdated = { utterances ->
                onTranscriptUpdated?.invoke(utterances)
                contextualRiskEngine?.onUtterancesUpdated(utterances)
            }
            engine.onScamAlert = { detection ->
                onScamAlert?.invoke(detection)
                riskFusionEngine?.onLevel1Alert(detection)
            }
            engine.onClaimExtracted = { claim ->
                onClaimExtracted?.invoke(claim)
            }
            engine.start(callId, label = speakerLabel)
            speechEngine = engine
            Log.i(TAG, "✓ On-device SpeechRecognizer started as active ASR")
        }
    }

    private fun stopSpeechEngine() {
        mainHandler.post {
            speechEngine?.stop()
            speechEngine = null
            Log.i(TAG, "On-device SpeechRecognizer stopped")
        }
    }

    private fun stopMicrophoneCapture() {
        try {
            audioRecord?.stop()
        } catch (_: Exception) {}
        try {
            audioRecord?.release()
        } catch (_: Exception) {}
        audioRecord = null
        recordingThread = null
    }

    /**
     * Shuts down all capture sources, speech recognizers, and risk engines at call termination.
     */
    fun stop() {
        if (!isRunning) return
        isRunning = false

        Log.i(TAG, "═══════════════════════════════════════════════════════")
        Log.i(TAG, "  VaniContinuousAudioEngine STOPPING")
        Log.i(TAG, "═══════════════════════════════════════════════════════")

        VaniAudioDiagnosticsHud.updateCallState("TERMINATED", callId)
        mainHandler.removeCallbacksAndMessages(null)

        // Stop speech engine on main thread
        stopSpeechEngine()

        // Stop direct telephony capture if active
        callAudioSource?.stop()
        callAudioSource?.release()
        callAudioSource = null

        // Stop remote streaming
        streamingOrchestrator?.stopStreaming()

        // Stop AudioRecord if open
        stopMicrophoneCapture()

        // Finalize WAV file
        finalizeWavRecording()

        // Shutdown risk pipeline
        contextualRiskEngine = null
        webIntelligenceEngine?.shutdown()
        webIntelligenceEngine = null
        riskFusionEngine = null

        Log.i(TAG, "✓ All pipelines stopped cleanly")
    }

    fun getTranscript(): List<TranscriptUtterance> {
        val backendTranscript = streamingOrchestrator?.getTranscript()
        if (!backendTranscript.isNullOrEmpty()) {
            return backendTranscript
        }
        return speechEngine?.getTranscript() ?: emptyList()
    }

    @SuppressLint("MissingPermission")
    private fun startMicrophoneCaptureThread() {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        val bufSize = max(minBuf, CHUNK_SIZE_SAMPLES * 2 * 4)
        val candidateSources = intArrayOf(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            MediaRecorder.AudioSource.MIC,
            MediaRecorder.AudioSource.VOICE_COMMUNICATION
        )

        var record: AudioRecord? = null
        for (src in candidateSources) {
            try {
                val candidate = AudioRecord(src, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufSize)
                if (candidate.state == AudioRecord.STATE_INITIALIZED) {
                    record = candidate
                    Log.i(TAG, "✓ Microphone AudioRecord initialized on source: $src")
                    break
                } else {
                    candidate.release()
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed initializing candidate audio source $src: ${e.message}")
            }
        }

        if (record == null) {
            Log.e(TAG, "AudioRecord could not be initialized on any microphone source")
            VaniAudioDiagnosticsHud.recordError("AudioRecord init failed on all microphone sources")
            return
        }

        audioRecord = record
        try {
            record.startRecording()
            Log.i(TAG, "✓ Microphone AudioRecord recording started successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Microphone AudioRecord startRecording failed: ${e.message}")
            VaniAudioDiagnosticsHud.recordError("AudioRecord startRecording failed: ${e.message}")
            return
        }

        recordingThread = Thread({
            val readBuffer = ByteArray(CHUNK_SIZE_SAMPLES * 2) // 1600 samples = 3200 bytes (100ms)
            while (isRunning && audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                val bytesRead = audioRecord?.read(readBuffer, 0, readBuffer.size) ?: -1
                if (bytesRead > 0) {
                    handleTelephonyPcm(readBuffer, bytesRead)
                }
            }
        }, "VaniContinuousMicThread").apply {
            priority = Thread.MAX_PRIORITY
            start()
        }
    }

    private fun handleTelephonyPcm(pcmBytes: ByteArray, readBytes: Int) {
        val sampleCount = readBytes / 2
        val shortBuffer = ShortArray(sampleCount)
        var maxAmp = 0
        var sumSquares = 0.0
        for (i in 0 until sampleCount) {
            val low = pcmBytes[i * 2].toInt() and 0xFF
            val high = pcmBytes[i * 2 + 1].toInt()
            val sample = ((high shl 8) or low).toShort()
            shortBuffer[i] = sample
            val absSample = kotlin.math.abs(sample.toInt())
            if (absSample > maxAmp) maxAmp = absSample
            sumSquares += (sample * sample)
        }

        val rms = kotlin.math.sqrt(sumSquares / sampleCount).toFloat()
        val normalizedAmp = (maxAmp / 20000.0f).coerceIn(0f, 1f)
        VaniRiskBus.publishAmplitude(normalizedAmp)

        pushRecent(shortBuffer, sampleCount)

        streamingOrchestrator?.sendAudioChunk(pcmBytes)

        try {
            wavOutputStream?.write(pcmBytes, 0, readBytes)
            totalAudioBytesWritten += readBytes
        } catch (_: Exception) {}

        acousticAnalyzer?.analyzeChunk(shortBuffer, sampleCount)
        VaniAudioDiagnosticsHud.recordAudioFrame(sampleCount, sampleCount, rms.toDouble(), maxAmp)
    }

    // ─── WAV Recording Management ───────────────────────────────────────────────

    private fun initWavRecording() {
        try {
            val recordDir = java.io.File(context.filesDir, "call_records").apply { mkdirs() }
            val safeId = if (callId.isNotBlank()) callId else "call_${System.currentTimeMillis()}"
            wavFile = java.io.File(recordDir, "$safeId.wav")
            wavOutputStream = java.io.FileOutputStream(wavFile)
            writeWavHeader(wavOutputStream!!, 0)
            Log.i(TAG, "WAV recording initialized: ${wavFile?.absolutePath}")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to initialize WAV recording: ${e.message}")
        }
    }

    private fun finalizeWavRecording() {
        try {
            wavOutputStream?.let { stream ->
                stream.channel.position(0)
                writeWavHeader(stream, totalAudioBytesWritten)
                stream.flush()
                stream.close()
                Log.i(TAG, "WAV recording finalized: ${wavFile?.absolutePath} ($totalAudioBytesWritten bytes)")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to finalize WAV header: ${e.message}")
        } finally {
            wavOutputStream = null
        }
    }

    private fun writeWavHeader(out: java.io.FileOutputStream, totalAudioLen: Long) {
        val totalDataLen = totalAudioLen + 36
        val byteRate = (SAMPLE_RATE * 1 * 16 / 8).toLong()
        val header = ByteArray(44)
        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1
        header[21] = 0
        header[22] = 1
        header[23] = 0
        header[24] = (SAMPLE_RATE and 0xff).toByte()
        header[25] = ((SAMPLE_RATE shr 8) and 0xff).toByte()
        header[26] = ((SAMPLE_RATE shr 16) and 0xff).toByte()
        header[27] = ((SAMPLE_RATE shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = 2
        header[33] = 0
        header[34] = 16
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
        header[43] = ((totalAudioLen shr 24) and 0xff).toByte()
        out.write(header, 0, 44)
    }

    // ─── Risk Pipeline Management ───────────────────────────────────────────────

    private fun initRiskPipeline() {
        val fusion = VaniRiskFusionEngine()
        riskFusionEngine = fusion

        val web = VaniWebIntelligenceEngine(context)
        web.onEvidenceCollected = { evidence ->
            fusion.onLevel3Evidence(evidence)
            VaniAudioDiagnosticsHud.recordRiskMetrics(0, 0f, "Web evidence: ${evidence.summary}")
        }
        webIntelligenceEngine = web

        val contextual = VaniContextualRiskEngine()
        contextual.onRiskUpdated = { result ->
            fusion.onLevel2Update(result)
            VaniAudioDiagnosticsHud.recordRiskMetrics(
                result.activeCategories.size,
                result.contextScore.toFloat(),
                "Context score: ${result.contextScore}"
            )
        }
        contextual.onWebSearchTriggered = { query ->
            web.search(query)
            VaniAudioDiagnosticsHud.recordRiskMetrics(0, 0f, "Searching web: $query")
        }
        contextualRiskEngine = contextual
    }

    private fun publishStatus(message: String) {
        val prev = VaniRiskBus.last
        val baseSnapshot = prev ?: VaniRiskBus.Snapshot(
            risk = 0, synthetic = 0, speakerMatch = 50,
            level = RiskLevel.SAFE, message = message
        )
        VaniRiskBus.publish(baseSnapshot.copy(message = message))
    }
}
