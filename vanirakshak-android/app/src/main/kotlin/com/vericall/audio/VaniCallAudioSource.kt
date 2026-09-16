package com.vericall.audio

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Clean abstraction for call audio capture sources in VaniRakshak.
 *
 * Strictly distinguishes direct telephony call capture from local microphone capture.
 * Never misrepresents microphone audio as two-way call recording.
 */
enum class AudioSourceType(val displayName: String, val isTwoWay: Boolean) {
    TELEPHONY_CALL_AUDIO("Telephony Call Audio (Two-Way)", true),
    TELEPHONY_DOWNLINK("Telephony Downlink (Caller Voice)", false),
    TELEPHONY_UPLINK("Telephony Uplink (User Voice)", false),
    MICROPHONE_UPLINK("Microphone (Local User Voice Only)", false),
    UNAVAILABLE("Audio Capture Unavailable", false)
}

interface VaniCallAudioSource {
    val sourceType: AudioSourceType
    val audioSourceId: Int
    fun isAvailable(): Boolean
    fun start(listener: (pcm: ByteArray, readBytes: Int) -> Unit)
    fun stop()
    fun release()

    companion object {
        private const val TAG = "VaniCallAudioSource"

        /**
         * Resolves the best legitimately available audio capture source on the current device.
         */
        fun createBestAvailable(context: Context): VaniCallAudioSource {
            val caps = VaniCallAudioCapability.getCapabilities(context)
            return when {
                caps.telephonyCaptureAvailable -> {
                    Log.i(TAG, "Selected Telephony Audio Source (Two-Way VOICE_CALL)")
                    VaniAudioDiagnosticsHud.updateAudioSource("VOICE_CALL (Two-Way)", isTwoWay = true, uplink = true, downlink = true)
                    VaniTelephonyAudioSource()
                }
                caps.voiceRecognitionAvailable -> {
                    Log.i(TAG, "Selected Microphone Audio Source (VOICE_RECOGNITION)")
                    VaniAudioDiagnosticsHud.updateAudioSource("VOICE_RECOGNITION (Mic Uplink)", isTwoWay = false, uplink = true, downlink = false)
                    VaniMicrophoneAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                }
                caps.voiceCommunicationAvailable -> {
                    Log.i(TAG, "Selected Microphone Audio Source (VOICE_COMMUNICATION)")
                    VaniAudioDiagnosticsHud.updateAudioSource("VOICE_COMMUNICATION (Mic Uplink)", isTwoWay = false, uplink = true, downlink = false)
                    VaniMicrophoneAudioSource(MediaRecorder.AudioSource.VOICE_COMMUNICATION)
                }
                caps.standardMicAvailable -> {
                    Log.i(TAG, "Selected Microphone Audio Source (MIC)")
                    VaniAudioDiagnosticsHud.updateAudioSource("MIC (Standard)", isTwoWay = false, uplink = true, downlink = false)
                    VaniMicrophoneAudioSource(MediaRecorder.AudioSource.MIC)
                }
                else -> {
                    Log.w(TAG, "No audio capture source available")
                    VaniAudioDiagnosticsHud.updateAudioSource("UNAVAILABLE", isTwoWay = false, uplink = false, downlink = false)
                    VaniUnavailableAudioSource
                }
            }
        }
    }
}

/**
 * Captures direct telephony call audio if granted by system policy (VOICE_CALL).
 */
class VaniTelephonyAudioSource : VaniCallAudioSource {

    override val sourceType: AudioSourceType = AudioSourceType.TELEPHONY_CALL_AUDIO
    override val audioSourceId: Int = MediaRecorder.AudioSource.VOICE_CALL

    private var audioRecord: AudioRecord? = null
    @Volatile private var isRecording = false
    private var workerThread: Thread? = null

    override fun isAvailable(): Boolean {
        var testRecord: AudioRecord? = null
        return try {
            val minBuf = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            if (minBuf <= 0) return false
            testRecord = AudioRecord(MediaRecorder.AudioSource.VOICE_CALL, 16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, minBuf)
            testRecord.state == AudioRecord.STATE_INITIALIZED
        } catch (_: Throwable) {
            false
        } finally {
            runCatching { testRecord?.release() }
        }
    }

    @SuppressLint("MissingPermission")
    override fun start(listener: (pcm: ByteArray, readBytes: Int) -> Unit) {
        if (isRecording) return
        try {
            val sampleRate = 16000
            val channelConfig = AudioFormat.CHANNEL_IN_MONO
            val audioFormat = AudioFormat.ENCODING_PCM_16BIT
            val minBuf = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat).coerceAtLeast(sampleRate * 2)

            audioRecord = AudioRecord(MediaRecorder.AudioSource.VOICE_CALL, sampleRate, channelConfig, audioFormat, minBuf)
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                val err = "VOICE_CALL state != STATE_INITIALIZED. Privileged telephony permission missing."
                Log.w("VaniTelephonyAudioSource", err)
                VaniAudioDiagnosticsHud.recordError(err)
                return
            }

            audioRecord?.startRecording()
            isRecording = true
            VaniAudioDiagnosticsHud.updateAudioSource("VOICE_CALL (Active)", isTwoWay = true, uplink = true, downlink = true)

            workerThread = Thread({
                val buffer = ByteArray(3200) // 100ms at 16kHz 16-bit mono
                while (isRecording) {
                    val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                    if (read > 0) {
                        measureAndRecordDiagnostics(buffer, read)
                        listener(buffer, read)
                    }
                }
            }, "VaniTelephonyAudioThread").apply {
                priority = Thread.MAX_PRIORITY
                start()
            }

        } catch (e: Exception) {
            Log.e("VaniTelephonyAudioSource", "Failed to start telephony audio: ${e.message}")
            VaniAudioDiagnosticsHud.recordError("Telephony start error: ${e.message}")
        }
    }

    override fun stop() {
        isRecording = false
        runCatching { audioRecord?.stop() }
    }

    override fun release() {
        stop()
        runCatching { audioRecord?.release() }
        audioRecord = null
        workerThread = null
    }

    private fun measureAndRecordDiagnostics(buffer: ByteArray, bytesRead: Int) {
        val samples = bytesRead / 2
        var nonZero = 0
        var sumSquare = 0.0
        var peak = 0

        for (i in 0 until samples) {
            val low = buffer[i * 2].toInt() and 0xFF
            val high = buffer[i * 2 + 1].toInt()
            val sample = (high shl 8) or low
            if (sample != 0) nonZero++
            sumSquare += sample * sample
            val absVal = abs(sample)
            if (absVal > peak) peak = absVal
        }

        val rms = if (samples > 0) sqrt(sumSquare / samples) else 0.0
        VaniAudioDiagnosticsHud.recordAudioFrame(samples, nonZero, rms, peak)
    }
}

/**
 * Captures local microphone audio (Uplink).
 */
class VaniMicrophoneAudioSource(
    private val audioSource: Int = MediaRecorder.AudioSource.VOICE_RECOGNITION
) : VaniCallAudioSource {

    override val sourceType: AudioSourceType = AudioSourceType.MICROPHONE_UPLINK
    override val audioSourceId: Int = audioSource

    private var audioRecord: AudioRecord? = null
    @Volatile private var isRecording = false
    private var workerThread: Thread? = null

    override fun isAvailable(): Boolean {
        var testRecord: AudioRecord? = null
        return try {
            val minBuf = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            if (minBuf <= 0) return false
            testRecord = AudioRecord(audioSource, 16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, minBuf)
            testRecord.state == AudioRecord.STATE_INITIALIZED
        } catch (_: Throwable) {
            false
        } finally {
            runCatching { testRecord?.release() }
        }
    }

    @SuppressLint("MissingPermission")
    override fun start(listener: (pcm: ByteArray, readBytes: Int) -> Unit) {
        if (isRecording) return
        try {
            val sampleRate = 16000
            val channelConfig = AudioFormat.CHANNEL_IN_MONO
            val audioFormat = AudioFormat.ENCODING_PCM_16BIT
            val minBuf = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat).coerceAtLeast(sampleRate * 2)

            audioRecord = AudioRecord(audioSource, sampleRate, channelConfig, audioFormat, minBuf)
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                val err = "AudioRecord state != STATE_INITIALIZED for source $audioSource"
                Log.w("VaniMicrophoneAudioSource", err)
                VaniAudioDiagnosticsHud.recordError(err)
                return
            }

            audioRecord?.startRecording()
            isRecording = true
            VaniAudioDiagnosticsHud.updateAudioSource("Source $audioSource (Active)", isTwoWay = false, uplink = true, downlink = false)

            workerThread = Thread({
                val buffer = ByteArray(3200) // 100ms at 16kHz 16-bit mono
                while (isRecording) {
                    val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                    if (read > 0) {
                        measureAndRecordDiagnostics(buffer, read)
                        listener(buffer, read)
                    }
                }
            }, "VaniMicAudioThread").apply {
                priority = Thread.MAX_PRIORITY
                start()
            }

        } catch (e: Exception) {
            Log.e("VaniMicrophoneAudioSource", "Failed to start microphone audio: ${e.message}")
            VaniAudioDiagnosticsHud.recordError("Microphone start error: ${e.message}")
        }
    }

    override fun stop() {
        isRecording = false
        runCatching { audioRecord?.stop() }
    }

    override fun release() {
        stop()
        runCatching { audioRecord?.release() }
        audioRecord = null
        workerThread = null
    }

    private fun measureAndRecordDiagnostics(buffer: ByteArray, bytesRead: Int) {
        val samples = bytesRead / 2
        var nonZero = 0
        var sumSquare = 0.0
        var peak = 0

        for (i in 0 until samples) {
            val low = buffer[i * 2].toInt() and 0xFF
            val high = buffer[i * 2 + 1].toInt()
            val sample = (high shl 8) or low
            if (sample != 0) nonZero++
            sumSquare += sample * sample
            val absVal = abs(sample)
            if (absVal > peak) peak = absVal
        }

        val rms = if (samples > 0) sqrt(sumSquare / samples) else 0.0
        VaniAudioDiagnosticsHud.recordAudioFrame(samples, nonZero, rms, peak)
    }
}

/**
 * Fallback when no audio capture is possible.
 */
object VaniUnavailableAudioSource : VaniCallAudioSource {
    override val sourceType: AudioSourceType = AudioSourceType.UNAVAILABLE
    override val audioSourceId: Int = -1
    override fun isAvailable(): Boolean = false
    override fun start(listener: (pcm: ByteArray, readBytes: Int) -> Unit) {}
    override fun stop() {}
    override fun release() {}
}
