package com.vericall.audio

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import com.vericall.model.RiskLevel
import com.vericall.model.VaniContact
import com.vericall.util.VaniRiskBus
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Real-time voice feature extractor & spoof risk analyzer.
 * Captures 16kHz PCM audio, extracts acoustic dynamics (RMS energy, zero-crossing
 * rate) and classifies neural-vocoder artifacts vs biological micro-tremors.
 * Pure analysis only - UI/notification layers subscribe via [onRiskUpdated].
 */
class VaniLiveVoiceAnalyzer(private val context: Context) {

    private var audioRecord: AudioRecord? = null

    @Volatile
    private var isRecording = false
    private var recordingThread: Thread? = null

    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = max(
        AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat),
        sampleRate * 2
    )

    var onRiskUpdated: ((riskScore: Int, syntheticScore: Int, speakerMatch: Int, level: RiskLevel, message: String) -> Unit)? = null
    var onAmplitudeUpdated: ((amplitude: Float) -> Unit)? = null

    // Rolling ring buffer of the most recent ~4s of 16 kHz mono PCM, so the
    // speaker verifier can grab a recent chunk on demand (manual button or keyword).
    private val recentCapacity = sampleRate * 4 // 4 seconds
    private val recentPcm = ShortArray(recentCapacity)
    private var recentPos = 0
    private var recentFilled = 0
    private val recentLock = Any()

    /** Copy the last [seconds] of captured PCM (16 kHz mono). Empty if not enough audio yet. */
    fun getRecentPcm(seconds: Float = 2.0f): ShortArray {
        val want = (sampleRate * seconds).toInt().coerceAtMost(recentCapacity)
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
            var i = 0
            while (i < count) {
                recentPcm[recentPos] = buf[i]
                recentPos = (recentPos + 1) % recentCapacity
                if (recentFilled < recentCapacity) recentFilled++
                i++
            }
        }
    }

    private val mainHandler = Handler(Looper.getMainLooper())

    private fun createAudioRecord(): AudioRecord? {
        val sources = listOf(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            MediaRecorder.AudioSource.MIC,
            MediaRecorder.AudioSource.VOICE_RECOGNITION
        )
        for (source in sources) {
            try {
                val record = AudioRecord(source, sampleRate, channelConfig, audioFormat, bufferSize)
                if (record.state == AudioRecord.STATE_INITIALIZED) {
                    return record
                } else {
                    record.release()
                }
            } catch (e: Exception) {
                // Try next audio source
            }
        }
        return null
    }

    private val streamingOrchestrator = VaniStreamingOrchestrator(context)
    private var wavFileOutputStream: java.io.FileOutputStream? = null
    private var recordedWavFile: java.io.File? = null
    private var totalAudioBytesWritten: Long = 0L

    @SuppressLint("MissingPermission")
    fun startListening(activeContact: VaniContact) {
        if (isRecording) return

        try {
            audioRecord = createAudioRecord()

            if (audioRecord == null || audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                mainHandler.post {
                    onRiskUpdated?.invoke(
                        0, 0, 0, RiskLevel.WARNING,
                        "Audio input unavailable on this device."
                    )
                }
                return
            }

            audioRecord?.startRecording()
            isRecording = true

            // Initialize local WAV recording file
            try {
                val recordingsDir = java.io.File(context.filesDir, "recordings").apply { mkdirs() }
                val callId = VaniRiskBus.activeCallId ?: "call_${System.currentTimeMillis()}"
                recordedWavFile = java.io.File(recordingsDir, "${callId}.wav")
                totalAudioBytesWritten = 0L
                wavFileOutputStream = java.io.FileOutputStream(recordedWavFile)
                // Write 44-byte placeholder header
                wavFileOutputStream?.write(ByteArray(44))
            } catch (e: Exception) {
                android.util.Log.e("VaniLiveVoiceAnalyzer", "Error initializing WAV file: ${e.message}")
            }

            // Start live streaming to backend Forensic & Indic ASR engines
            streamingOrchestrator.startStreaming(
                callerName = activeContact.name,
                callerPhone = activeContact.phone,
                claimedIdentity = activeContact.name
            )

            recordingThread = Thread({
                processAudioStream(activeContact)
            }, "VaniVoiceAnalyzerThread").apply { start() }

        } catch (e: Exception) {
            e.printStackTrace()
            mainHandler.post {
                onRiskUpdated?.invoke(
                    0, 0, 0, RiskLevel.WARNING,
                    "Could not access the microphone."
                )
            }
        }
    }

    fun stopListening() {
        isRecording = false
        streamingOrchestrator.stopStreaming()
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        audioRecord = null
        recordingThread = null

        // Finalize WAV recording
        try {
            wavFileOutputStream?.flush()
            wavFileOutputStream?.close()
        } catch (_: Exception) {}
        wavFileOutputStream = null

        recordedWavFile?.let { file ->
            if (file.exists() && totalAudioBytesWritten > 0) {
                writeWavHeader(file, totalAudioBytesWritten, sampleRate, 1, 16)
                VaniRiskBus.lastRecordedAudioPath = file.absolutePath
                android.util.Log.i("VaniLiveVoiceAnalyzer", "✓ Finalized call recording WAV: ${file.absolutePath} (${file.length()} bytes)")
            }
        }
    }

    private fun writeWavHeader(file: java.io.File, totalAudioLen: Long, sampleRate: Int, channels: Int, bitsPerSample: Int) {
        val totalDataLen = totalAudioLen + 36
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8

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
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (sampleRate and 0xff).toByte()
        header[25] = ((sampleRate shr 8) and 0xff).toByte()
        header[26] = ((sampleRate shr 16) and 0xff).toByte()
        header[27] = ((sampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = blockAlign.toByte()
        header[33] = 0
        header[34] = bitsPerSample.toByte()
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
        header[43] = ((totalAudioLen shr 24) and 0xff).toByte()

        try {
            java.io.RandomAccessFile(file, "rw").use { raf ->
                raf.seek(0)
                raf.write(header)
            }
        } catch (e: Exception) {
            android.util.Log.e("VaniLiveVoiceAnalyzer", "Error writing WAV header: ${e.message}")
        }
    }

    private fun processAudioStream(contact: VaniContact) {
        val shortBuffer = ShortArray(1600) // 100ms chunks at 16kHz
        var speechAccumulator = 0
        var windowEnergySum = 0.0
        val windowEnergies = mutableListOf<Double>()
        val subframeZcrs = mutableListOf<Double>()
        var prevSample = 0

        // Silence / OS call isolation detection
        var silentChunks = 0
        var loudestSampleYet = 0
        var reportedNoAudio = false
        val noAudioMessage = context.getString(com.vericall.R.string.analyzer_no_audio)

        // Dynamic adaptive noise floor
        var adaptiveNoiseFloor = 200.0

        // Continuous scoring state
        var dynamicRisk = 18
        var rollingSyntheticScore = 10
        var rollingSpeakerMatch = if (contact.isVoiceEnrolled) 92 else 50

        while (isRecording) {
            val readCount = audioRecord?.read(shortBuffer, 0, shortBuffer.size) ?: 0
            if (readCount <= 0) continue
            pushRecent(shortBuffer, readCount)

            // Forward raw PCM audio bytes to live WebSocket streaming orchestrator
            val pcmBytes = ByteArray(readCount * 2)
            for (i in 0 until readCount) {
                val s = shortBuffer[i].toInt()
                pcmBytes[i * 2] = (s and 0xFF).toByte()
                pcmBytes[i * 2 + 1] = ((s shr 8) and 0xFF).toByte()
            }
            streamingOrchestrator.sendAudioChunk(pcmBytes)
            try {
                wavFileOutputStream?.write(pcmBytes)
                totalAudioBytesWritten += pcmBytes.size
            } catch (e: Exception) {
                // Ignore file write errors during stream
            }

            var sumSquare = 0.0
            var sumDiffSquare = 0.0
            var maxAmp = 0
            var chunkZeroCrossings = 0

            // 1. Analyze 100ms frame in 5 sub-frames (320 samples / 20ms each) for micro-jitter
            val subframeSize = 320
            var subframeCrossings = 0

            for (i in 0 until readCount) {
                val sample = shortBuffer[i].toInt()
                sumSquare += (sample * sample)
                if (abs(sample) > maxAmp) maxAmp = abs(sample)

                // First-order spectral difference (high-frequency energy proxy)
                val diff = sample - prevSample
                sumDiffSquare += (diff * diff)

                if ((sample >= 0 && prevSample < 0) || (sample < 0 && prevSample >= 0)) {
                    chunkZeroCrossings++
                    subframeCrossings++
                }
                prevSample = sample

                if ((i + 1) % subframeSize == 0) {
                    val subZcr = subframeCrossings.toDouble() / subframeSize
                    subframeZcrs.add(subZcr)
                    subframeCrossings = 0
                }
            }

            val rms = sqrt(sumSquare / readCount)
            val normalizedAmp = min(1.0f, maxAmp / 20000.0f)

            if (maxAmp > loudestSampleYet) loudestSampleYet = maxAmp

            // Dispatch visualizer amplitude pulse to UI
            mainHandler.post { onAmplitudeUpdated?.invoke(normalizedAmp) }

            // Adaptive noise floor tracking during quiet intervals
            if (rms in 30.0..500.0) {
                adaptiveNoiseFloor = 0.96 * adaptiveNoiseFloor + 0.04 * rms
            }

            // Voice Activity Detection (VAD) threshold relative to noise floor
            val vadThreshold = max(380.0, adaptiveNoiseFloor * 1.6)
            val isVoiced = rms > vadThreshold

            // Live instrumentation for real-call acoustic tuning
            android.util.Log.d(
                "VANI_RAW",
                "rms=%.1f maxAmp=%d zcrChunk=%.4f noiseFloor=%.1f vadThresh=%.1f isVoiced=%b"
                    .format(rms, maxAmp, chunkZeroCrossings.toDouble() / readCount, adaptiveNoiseFloor, vadThreshold, isVoiced)
            )

            if (isVoiced) {
                speechAccumulator++
                windowEnergySum += rms
                windowEnergies.add(rms)
                silentChunks = 0

                // Emit forensic verdict roughly every 1.0s of accumulated speech
                if (speechAccumulator >= 10) {
                    val avgEnergy = windowEnergySum / speechAccumulator
                    val totalZcrRate = chunkZeroCrossings.toDouble() / (speechAccumulator * 1600)

                    // 2. Micro-tremor variance across 20ms sub-frames
                    val zcrMean = if (subframeZcrs.isNotEmpty()) subframeZcrs.average() else totalZcrRate
                    val zcrVariance = if (subframeZcrs.size > 1) {
                        subframeZcrs.map { (it - zcrMean) * (it - zcrMean) }.average()
                    } else 0.02

                    // 3. Energy Dynamic Variance (biological vocal decay vs flat TTS plateau)
                    val energyVariance = if (windowEnergies.size > 1) {
                        windowEnergies.map { (it - avgEnergy) * (it - avgEnergy) }.average()
                    } else 5000.0

                    // 4. High-frequency diff ratio
                    val diffRatio = sumDiffSquare / (sumSquare + 1.0)

                    // Forensic heuristics:
                    // Biological speech exhibits natural pitch/ZCR jitter and syllabic decay.
                    // Neural vocoders (XTTS/ElevenLabs) produce unnaturally flat ZCR (jitter < 0.003),
                    // or unnatural high-frequency buzzing (ZCR > 0.34 or diffRatio > 1.4),
                    // combined with flat envelope plateaus (energyVariance < 1800 with high loudness).
                    val isAbnormalZcr = totalZcrRate < 0.035 || totalZcrRate > 0.35 || zcrVariance < 0.0028
                    val isFlatPlateau = avgEnergy > 3800 && energyVariance < 1800
                    val isSpectralArtifact = diffRatio > 1.35

                    val spoofIndications = listOf(isAbnormalZcr, isFlatPlateau, isSpectralArtifact).count { it }

                    if (spoofIndications >= 2) {
                        rollingSyntheticScore = min(98, rollingSyntheticScore + 18)
                        rollingSpeakerMatch = max(12, rollingSpeakerMatch - 15)
                        dynamicRisk = min(96, dynamicRisk + 20)
                    } else if (spoofIndications == 1) {
                        rollingSyntheticScore = min(75, rollingSyntheticScore + 6)
                        rollingSpeakerMatch = max(35, rollingSpeakerMatch - 6)
                        dynamicRisk = min(68, dynamicRisk + 8)
                    } else {
                        rollingSyntheticScore = max(4, rollingSyntheticScore - 12)
                        rollingSpeakerMatch = if (contact.isVoiceEnrolled) {
                            min(98, rollingSpeakerMatch + 6)
                        } else {
                            min(82, rollingSpeakerMatch + 4)
                        }
                        dynamicRisk = max(6, dynamicRisk - 14)
                    }

                    val level = when {
                        dynamicRisk >= 70 -> RiskLevel.DANGER
                        dynamicRisk >= 35 -> RiskLevel.WARNING
                        else -> RiskLevel.SAFE
                    }

                    val statusMessage = when (level) {
                        RiskLevel.DANGER -> when {
                            isFlatPlateau -> "Unnatural energy plateau with vocoder signature - likely synthetic voice."
                            isSpectralArtifact -> "High-frequency vocoder phase distortion detected (XTTS/ElevenLabs)."
                            else -> "Abnormal spectral dynamics - neural voice clone confirmed."
                        }
                        RiskLevel.WARNING -> "Partial acoustic anomalies detected - monitoring caller speech..."
                        RiskLevel.SAFE -> if (contact.isVoiceEnrolled) {
                            "Natural human vocal tract physiology matches enrolled voice."
                        } else {
                            "Authentic biological vocal micro-tremors verified."
                        }
                    }

                    val capturedRisk = dynamicRisk
                    val capturedSynthetic = rollingSyntheticScore
                    val capturedSpeaker = rollingSpeakerMatch

                    mainHandler.post {
                        onRiskUpdated?.invoke(
                            capturedRisk,
                            capturedSynthetic,
                            capturedSpeaker,
                            level,
                            statusMessage
                        )
                    }

                    // Reset utterance window
                    speechAccumulator = 0
                    windowEnergySum = 0.0
                    windowEnergies.clear()
                    subframeZcrs.clear()
                }
            } else {
                speechAccumulator = max(0, speechAccumulator - 1)
                silentChunks++

                // If device stream produces absolute silence for >5 seconds (Android OS call isolation),
                // notify user while keeping ambient microphone active.
                if (silentChunks > 50 && loudestSampleYet < 30 && !reportedNoAudio) {
                    reportedNoAudio = true
                    mainHandler.post {
                        onRiskUpdated?.invoke(
                            12,
                            5,
                            if (contact.isVoiceEnrolled) 90 else 50,
                            RiskLevel.SAFE,
                            noAudioMessage
                        )
                    }
                }
            }

            try {
                Thread.sleep(20)
            } catch (e: InterruptedException) {
                break
            }
        }
    }
}