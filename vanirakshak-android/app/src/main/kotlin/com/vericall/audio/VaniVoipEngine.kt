package com.vericall.audio

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.NoiseSuppressor
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.vericall.config.VaniConfig
import com.vericall.data.VaniCallHistoryRepository
import com.vericall.model.CallDirection
import com.vericall.model.RiskLevel
import com.vericall.model.TranscriptUtterance
import com.vericall.model.VaniCallRecord
import com.vericall.model.VaniContact
import com.vericall.util.VaniRiskBus
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * VaniVoipEngine — Workaround 2: VaniRakshak-Controlled VoIP Calls (100% Digital Two-Way Stream).
 *
 * ARCHITECTURAL ADVANTAGES OVER CELLULAR SIM CALLS:
 * 1. Cellular SIM calls are routed through the modem DSP and isolated by Android AudioPolicy.
 * 2. Vani VoIP calls use [AudioManager.MODE_IN_COMMUNICATION] where the app owns both the
 *    microphone capture and the digital incoming network audio buffer.
 * 3. Incoming caller PCM frames are received directly in app memory before playing to [AudioTrack],
 *    granting 100% unrestricted digital access to the caller's voice for real-time transcription,
 *    synthetic voice detection, and threat analysis.
 * 4. Local microphone audio is captured with hardware Acoustic Echo Cancellation (AEC).
 */
class VaniVoipEngine(private val context: Context) {

    companion object {
        private const val TAG = "VaniVoipEngine"
        const val SAMPLE_RATE = 16000
        const val CHANNEL_IN = AudioFormat.CHANNEL_IN_MONO
        const val CHANNEL_OUT = AudioFormat.CHANNEL_OUT_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        const val CHUNK_SAMPLES = 1600 // 100ms at 16kHz
    }

    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile
    private var isCallActive = false
    val isRunning: Boolean get() = isCallActive

    private var activeCallId: String = ""
    private var peerName: String = "VoIP Caller"
    private var peerNumber: String = ""
    private var callStartTime: Long = 0L

    // Hardware Audio Pipeline
    private var audioRecord: AudioRecord? = null
    private var audioTrack: AudioTrack? = null
    private var micThread: Thread? = null
    private var aec: AcousticEchoCanceler? = null
    private var ns: NoiseSuppressor? = null

    // Streaming & Forensics
    var streamingOrchestrator: VaniStreamingOrchestrator? = null
    var acousticAnalyzer: VaniLiveVoiceAnalyzer? = null

    private val transcriptUtterances = mutableListOf<TranscriptUtterance>()
    private var isMuted = false
    private var isSpeakerOn = false

    // WAV Recording
    private var wavFile: File? = null
    private var wavOutputStream: FileOutputStream? = null
    private var totalBytesWritten = 0L

    // Callbacks
    var onTranscriptUpdated: ((List<TranscriptUtterance>) -> Unit)? = null
    var onAmplitudeUpdated: ((Float) -> Unit)? = null
    var onCallStateChanged: ((String) -> Unit)? = null

    /**
     * Initiates a Vani Secure VoIP session.
     */
    @SuppressLint("MissingPermission")
    fun startCall(
        callId: String = "voip_${UUID.randomUUID().toString().take(8)}",
        callerName: String = "VoIP Caller",
        callerPhone: String = "",
        isSimulatedTest: Boolean = false
    ) {
        if (isCallActive) {
            Log.w(TAG, "VoIP session already active: $activeCallId")
            return
        }

        isCallActive = true
        activeCallId = callId
        peerName = callerName
        peerNumber = callerPhone
        callStartTime = System.currentTimeMillis()
        VaniRiskBus.activeCallId = activeCallId
        transcriptUtterances.clear()

        Log.i(TAG, "════════════════════════════════════════════════════════════")
        Log.i(TAG, "  VaniVoipEngine: STARTING SECURE VOIP CALL")
        Log.i(TAG, "  Session: $activeCallId | Peer: $peerName ($peerNumber)")
        Log.i(TAG, "  Mode: 100% Digital Two-Way Stream (Zero OS Restrictions)")
        Log.i(TAG, "════════════════════════════════════════════════════════════")

        // 1. Configure Android Audio System for VoIP Communication
        setupAudioMode()

        // 2. Initialize AudioTrack for playback of digital incoming caller audio
        setupAudioTrack()

        // 3. Initialize AudioRecord with Hardware AEC for outgoing user audio
        setupAudioRecord()

        // 4. Initialize Local WAV file for two-way recording
        initWavRecording()

        // 5. Initialize Forensics & Streaming Orchestrator (handles single authoritative WebSocket)
        setupForensics()

        if (isSimulatedTest) {
            runInteractiveVoipSimulation()
        }

        publishVoipStatus("🌐 Secure VoIP Connected • 100% Digital Two-Way Stream Active")
        onCallStateChanged?.invoke("CONNECTED")
    }

    /**
     * Terminates the VoIP session and finalizes call history.
     */
    fun endCall() {
        if (!isCallActive) return
        isCallActive = false

        Log.i(TAG, "VaniVoipEngine: Ending VoIP call session: $activeCallId")
        onCallStateChanged?.invoke("DISCONNECTED")

        // Stop microphone capture
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (_: Exception) {}
        audioRecord = null
        micThread?.interrupt()
        micThread = null

        // Teardown effects
        runCatching { aec?.release() }
        runCatching { ns?.release() }
        aec = null
        ns = null

        // Stop playback
        try {
            audioTrack?.stop()
            audioTrack?.release()
        } catch (_: Exception) {}
        audioTrack = null

        // Stop streaming and acoustic analyzers
        streamingOrchestrator?.stopStreaming()
        acousticAnalyzer?.stopAnalysis()

        // Reset audio manager
        audioManager.mode = AudioManager.MODE_NORMAL
        audioManager.isSpeakerphoneOn = false

        // Finalize WAV recording
        finalizeWavRecording()

        // Save Call Record to History
        val durationSec = ((System.currentTimeMillis() - callStartTime) / 1000).toInt().coerceAtLeast(1)
        val lastSnap = VaniRiskBus.last
        val record = VaniCallRecord(
            id = activeCallId,
            number = peerNumber.ifBlank { "Vani Secure VoIP" },
            callerName = peerName,
            direction = CallDirection.OUTGOING,
            timestamp = callStartTime,
            durationSec = durationSec,
            hasTranscript = transcriptUtterances.isNotEmpty(),
            transcript = transcriptUtterances.toList(),
            riskScore = lastSnap?.risk ?: 0,
            riskLevel = lastSnap?.level?.name ?: "SAFE",
            isVoiceVerified = (lastSnap?.speakerMatch ?: 0) >= 80,
            isProtected = true,
            audioFilePath = wavFile?.absolutePath
        )
        VaniCallHistoryRepository.saveCallRecord(context, record)

        Log.i(TAG, "✓ VoIP session finalized and persisted to history. Duration: ${durationSec}s")
    }

    private fun setupAudioMode() {
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        audioManager.isSpeakerphoneOn = isSpeakerOn
        Log.i(TAG, "Audio mode set to MODE_IN_COMMUNICATION (Speaker: $isSpeakerOn)")
    }

    private fun setupAudioTrack() {
        val minBuf = AudioTrack.getMinBufferSize(SAMPLE_RATE, CHANNEL_OUT, AUDIO_FORMAT)
        val bufSize = max(minBuf, CHUNK_SAMPLES * 2 * 4)

        val attributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()

        val format = AudioFormat.Builder()
            .setSampleRate(SAMPLE_RATE)
            .setChannelMask(CHANNEL_OUT)
            .setEncoding(AUDIO_FORMAT)
            .build()

        audioTrack = AudioTrack(
            attributes,
            format,
            bufSize,
            AudioTrack.MODE_STREAM,
            AudioManager.AUDIO_SESSION_ID_GENERATE
        )
        audioTrack?.play()
        Log.i(TAG, "✓ AudioTrack initialized for low-latency VoIP playback")
    }

    @SuppressLint("MissingPermission")
    private fun setupAudioRecord() {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_IN, AUDIO_FORMAT)
        val bufSize = max(minBuf, CHUNK_SAMPLES * 2 * 4)

        val candidateSources = intArrayOf(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            MediaRecorder.AudioSource.MIC,
            MediaRecorder.AudioSource.VOICE_COMMUNICATION
        )

        var record: AudioRecord? = null
        for (src in candidateSources) {
            try {
                val candidate = AudioRecord(src, SAMPLE_RATE, CHANNEL_IN, AUDIO_FORMAT, bufSize)
                if (candidate.state == AudioRecord.STATE_INITIALIZED) {
                    record = candidate
                    Log.i(TAG, "✓ VoIP AudioRecord initialized on source: $src")
                    break
                } else {
                    candidate.release()
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed initializing candidate audio source $src: ${e.message}")
            }
        }

        if (record != null) {
            audioRecord = record

            // Attach hardware echo cancellation and noise suppression
            val sessionId = record.audioSessionId
            if (AcousticEchoCanceler.isAvailable()) {
                aec = AcousticEchoCanceler.create(sessionId)?.apply { enabled = true }
                Log.i(TAG, "✓ AcousticEchoCanceler engaged for VoIP session")
            }
            if (NoiseSuppressor.isAvailable()) {
                ns = NoiseSuppressor.create(sessionId)?.apply { enabled = true }
                Log.i(TAG, "✓ NoiseSuppressor engaged for VoIP session")
            }

            record.startRecording()
            startMicCaptureLoop()
        } else {
            Log.e(TAG, "Failed to initialize AudioRecord for VoIP on any source")
        }
    }

    private fun startMicCaptureLoop() {
        micThread = Thread({
            val buffer = ByteArray(CHUNK_SAMPLES * 2) // 100ms PCM chunk
            while (isCallActive && audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                val read = audioRecord?.read(buffer, 0, buffer.size) ?: -1
                if (read > 0) {
                    handleOutgoingUserAudio(buffer, read)
                }
            }
        }, "VaniVoipMicThread").apply {
            priority = Thread.MAX_PRIORITY
            start()
        }
    }

    /**
     * Handles outgoing audio recorded from the user's microphone.
     */
    private fun handleOutgoingUserAudio(pcmBytes: ByteArray, readBytes: Int) {
        if (isMuted) return

        // 1. Calculate live RMS energy for UI waveform
        var maxAmp = 0
        var sumSquare = 0.0
        val sampleCount = readBytes / 2
        for (i in 0 until sampleCount) {
            val low = pcmBytes[i * 2].toInt() and 0xFF
            val high = pcmBytes[i * 2 + 1].toInt()
            val sample = ((high shl 8) or low).toShort()
            val a = abs(sample.toInt())
            if (a > maxAmp) maxAmp = a
            sumSquare += (sample * sample)
        }
        val normalizedAmp = (maxAmp / 20000.0f).coerceIn(0f, 1f)
        VaniRiskBus.publishAmplitude(normalizedAmp)
        mainHandler.post { onAmplitudeUpdated?.invoke(normalizedAmp) }

        // 2. Stream out to remote Faster-Whisper gateway via streaming orchestrator
        streamingOrchestrator?.sendAudioChunk(pcmBytes.copyOfRange(0, readBytes))

        // 3. Write user voice to session WAV file
        try {
            wavOutputStream?.write(pcmBytes, 0, readBytes)
            totalBytesWritten += readBytes
        } catch (_: Exception) {}
    }

    /**
     * Receives incoming digital audio packets from the remote caller.
     * ZERO RESTRICTIONS — VaniRakshak has direct digital access to this PCM!
     */
    fun handleIncomingCallerAudio(pcmBytes: ByteArray, readBytes: Int) {
        if (!isCallActive || readBytes <= 0) return

        // 1. Play caller's voice through handset speaker/earpiece
        audioTrack?.write(pcmBytes, 0, readBytes)

        // 2. Feed caller PCM into Acoustic Voice Spoofing Analyzer (Synthetic Clone Detection)
        val sampleCount = readBytes / 2
        val shortBuffer = ShortArray(sampleCount)
        for (i in 0 until sampleCount) {
            val low = pcmBytes[i * 2].toInt() and 0xFF
            val high = pcmBytes[i * 2 + 1].toInt()
            shortBuffer[i] = ((high shl 8) or low).toShort()
        }
        acousticAnalyzer?.analyzeChunk(shortBuffer, sampleCount)

        // 3. Stream caller PCM chunk to Live ASR Gateway for real-time transcription
        streamingOrchestrator?.sendAudioChunk(pcmBytes.copyOfRange(0, readBytes))

        // 4. Record to session WAV file
        try {
            wavOutputStream?.write(pcmBytes, 0, readBytes)
            totalBytesWritten += readBytes
        } catch (_: Exception) {}
    }

    private fun setupForensics() {
        // Acoustic analyzer for voice clone detection
        val analyzer = VaniLiveVoiceAnalyzer(context)
        val contact = VaniContact(
            id = activeCallId,
            name = peerName,
            phone = peerNumber,
            relation = "VoIP Caller",
            isVoiceEnrolled = false
        )
        analyzer.startAnalysis(contact)
        analyzer.onRiskUpdated = { risk, synthetic, match, level, msg ->
            val prev = VaniRiskBus.last
            VaniRiskBus.publish(
                (prev ?: VaniRiskBus.Snapshot(0, 0, 0, RiskLevel.SAFE, msg)).copy(
                    risk = maxOf(risk, prev?.threatScore ?: 0),
                    synthetic = synthetic,
                    speakerMatch = match,
                    level = level,
                    message = msg
                )
            )
        }
        acousticAnalyzer = analyzer

        // Live dual streaming orchestrator for real-time transcription & Crawl4AI
        val orchestrator = VaniStreamingOrchestrator(context)
        orchestrator.onTranscriptUpdated = { utterances ->
            transcriptUtterances.clear()
            transcriptUtterances.addAll(utterances)
            val fullText = utterances.joinToString(" ") { it.text }
            val prev = VaniRiskBus.last
            VaniRiskBus.publish(
                (prev ?: VaniRiskBus.Snapshot(0, 0, 0, RiskLevel.SAFE, "🛡 VoIP Call Protected")).copy(
                    transcript = fullText,
                    utterances = utterances
                )
            )
            mainHandler.post { onTranscriptUpdated?.invoke(transcriptUtterances.toList()) }
        }
        orchestrator.startStreaming(
            callerName = peerName,
            callerPhone = peerNumber,
            claimedIdentity = peerName,
            callId = activeCallId
        )
        streamingOrchestrator = orchestrator
    }

    /**
     * Built-in interactive VoIP simulation for instant demonstration and testing
     * without needing two simultaneous physical devices.
     */
    private fun runInteractiveVoipSimulation() {
        mainHandler.postDelayed({
            if (!isCallActive) return@postDelayed
            simulateCallerUtterance(
                "Hello, I am calling from Mumbai Cyber Crime Branch. Your bank account is flagged for digital arrest.",
                threatScore = 75,
                scamType = "Digital Arrest Scam"
            )
        }, 2000)

        mainHandler.postDelayed({
            if (!isCallActive) return@postDelayed
            simulateCallerUtterance(
                "Transfer 25000 rupees immediately to verify your identity or your Aadhaar and accounts will be seized.",
                threatScore = 95,
                scamType = "Financial Coercion"
            )
        }, 6000)
    }

    private fun simulateCallerUtterance(text: String, threatScore: Int, scamType: String) {
        val utterance = TranscriptUtterance(
            speaker = "Caller",
            text = text,
            timestamp = System.currentTimeMillis(),
            isFinal = true
        )
        transcriptUtterances.add(utterance)
        onTranscriptUpdated?.invoke(transcriptUtterances.toList())

        // Synthesize acoustic audio chunk for AudioTrack & Analyzer
        val testTone = generateSpeechAcousticPcm(text.length * 40)
        handleIncomingCallerAudio(testTone, testTone.size)

        val prev = VaniRiskBus.last
        val baseSnapshot = prev ?: VaniRiskBus.Snapshot(
            risk = 0, synthetic = 0, speakerMatch = 50,
            level = RiskLevel.SAFE, message = "🌐 VoIP Active"
        )
        VaniRiskBus.publish(
            baseSnapshot.copy(
                risk = threatScore,
                synthetic = 92,
                speakerMatch = 15,
                level = if (threatScore >= 70) RiskLevel.DANGER else RiskLevel.WARNING,
                message = "🚨 $scamType Detected over Digital VoIP Stream",
                transcript = transcriptUtterances.joinToString(" ") { it.text },
                utterances = transcriptUtterances.toList(),
                threatScore = threatScore,
                scamType = scamType,
                redFlags = listOf("Digital Arrest Threat", "Aadhaar Seizure Claim", "Urgent Fund Transfer")
            )
        )
    }

    /**
     * Generates a calibrated multi-harmonic acoustic signal matching human formant frequencies.
     */
    private fun generateSpeechAcousticPcm(sampleDurationMs: Int): ByteArray {
        val totalSamples = (SAMPLE_RATE * (sampleDurationMs / 1000.0)).toInt().coerceAtLeast(1600)
        val pcm = ByteArray(totalSamples * 2)
        val f0 = 180.0 // Fundamental frequency (Hz)
        val f1 = 750.0 // First formant
        val f2 = 1750.0 // Second formant

        for (i in 0 until totalSamples) {
            val t = i.toDouble() / SAMPLE_RATE
            val s0 = Math.sin(2.0 * Math.PI * f0 * t) * 6000.0
            val s1 = Math.sin(2.0 * Math.PI * f1 * t) * 4000.0
            val s2 = Math.sin(2.0 * Math.PI * f2 * t) * 2000.0
            val combined = (s0 + s1 + s2).toInt().coerceIn(-32767, 32767).toShort()

            pcm[i * 2] = (combined.toInt() and 0xFF).toByte()
            pcm[i * 2 + 1] = ((combined.toInt() shr 8) and 0xFF).toByte()
        }
        return pcm
    }

    // ─── Public Control APIs ───────────────────────────────────────────────────

    fun toggleMute(): Boolean {
        isMuted = !isMuted
        return isMuted
    }

    fun toggleSpeaker(): Boolean {
        isSpeakerOn = !isSpeakerOn
        audioManager.isSpeakerphoneOn = isSpeakerOn
        return isSpeakerOn
    }

    // ─── WAV Recording Management ───────────────────────────────────────────────

    private fun initWavRecording() {
        try {
            val dir = File(context.filesDir, "call_records").apply { mkdirs() }
            wavFile = File(dir, "$activeCallId.wav")
            wavOutputStream = FileOutputStream(wavFile)
            writeWavHeader(wavOutputStream!!, 0)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to init VoIP WAV: ${e.message}")
        }
    }

    private fun finalizeWavRecording() {
        try {
            wavOutputStream?.let { stream ->
                stream.channel.position(0)
                writeWavHeader(stream, totalBytesWritten)
                stream.flush()
                stream.close()
            }
        } catch (_: Exception) {}
        wavOutputStream = null
    }

    private fun writeWavHeader(out: FileOutputStream, totalAudioLen: Long) {
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

    private fun publishVoipStatus(msg: String) {
        val prev = VaniRiskBus.last
        val baseSnapshot = prev ?: VaniRiskBus.Snapshot(
            risk = 0, synthetic = 0, speakerMatch = 50,
            level = RiskLevel.SAFE, message = msg
        )
        VaniRiskBus.publish(baseSnapshot.copy(message = msg))
    }
}
