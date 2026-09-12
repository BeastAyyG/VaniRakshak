package com.vericall.verification

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.vericall.R
import com.vericall.databinding.ActivityVoiceVerificationBinding
import java.io.ByteArrayOutputStream
import kotlin.math.sin

class VoiceVerificationActivity : AppCompatActivity() {

    private lateinit var binding: ActivityVoiceVerificationBinding
    private var isRecording = false
    private var audioRecord: AudioRecord? = null
    private var recordingSeconds = 0
    private val mainHandler = Handler(Looper.getMainLooper())
    private var recordingThread: Thread? = null
    private val pcmBuffer = ByteArrayOutputStream()
    private var latestRmsAmplitude: Float = 0.1f

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startAudioCapture()
        } else {
            Toast.makeText(this, "Microphone permission needed to listen to spoken claims", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private val timerRunnable = object : Runnable {
        override fun run() {
            if (!isRecording) return
            recordingSeconds++
            val mm = String.format("%02d", recordingSeconds / 60)
            val ss = String.format("%02d", recordingSeconds % 60)
            binding.tvDuration.text = "$mm:$ss"
            mainHandler.postDelayed(this, 1000)
        }
    }

    private val waveformRunnable = object : Runnable {
        private var tick = 0.0
        override fun run() {
            if (!isRecording) return
            tick += 0.35
            val amp = latestRmsAmplitude.coerceIn(0.1f, 1.0f)
            val baseH = (amp * 28).toInt()

            binding.bar1.layoutParams.height = ((sin(tick) * 6 + baseH + 10).coerceIn(8.0, 48.0)).toInt() * 2
            binding.bar2.layoutParams.height = ((sin(tick + 1.0) * 8 + baseH + 14).coerceIn(10.0, 56.0)).toInt() * 2
            binding.bar3.layoutParams.height = ((sin(tick + 2.0) * 10 + baseH + 18).coerceIn(12.0, 64.0)).toInt() * 2
            binding.bar4.layoutParams.height = ((sin(tick + 3.0) * 7 + baseH + 12).coerceIn(8.0, 52.0)).toInt() * 2
            binding.bar5.layoutParams.height = ((sin(tick + 4.0) * 9 + baseH + 16).coerceIn(10.0, 58.0)).toInt() * 2
            binding.bar6.layoutParams.height = ((sin(tick + 5.0) * 5 + baseH + 8).coerceIn(8.0, 42.0)).toInt() * 2
            binding.layoutWaveform.requestLayout()

            mainHandler.postDelayed(this, 75)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVoiceVerificationBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        binding.btnBackVoice.setOnClickListener { finish() }

        binding.btnStopRecord.setOnClickListener {
            stopAndAnalyze()
        }

        binding.btnEditTranscript.setOnClickListener {
            binding.inputTranscribedText.isEnabled = true
            binding.inputTranscribedText.requestFocus()
            Toast.makeText(this, "You can edit the claim text directly", Toast.LENGTH_SHORT).show()
        }

        // Check & request RECORD_AUDIO permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            startAudioCapture()
        } else {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun startAudioCapture() {
        if (isRecording) return
        isRecording = true
        recordingSeconds = 0
        synchronized(pcmBuffer) {
            pcmBuffer.reset()
        }
        mainHandler.post(timerRunnable)
        mainHandler.post(waveformRunnable)

        recordingThread = Thread {
            val sampleRate = 16000
            val channelConfig = AudioFormat.CHANNEL_IN_MONO
            val audioFormat = AudioFormat.ENCODING_PCM_16BIT
            val minBuf = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
            val bufSize = minBuf.coerceAtLeast(4096)
            val readBuf = ByteArray(bufSize)

            try {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    audioRecord = AudioRecord(
                        MediaRecorder.AudioSource.MIC,
                        sampleRate,
                        channelConfig,
                        audioFormat,
                        bufSize
                    )
                    audioRecord?.startRecording()

                    while (isRecording) {
                        val read = audioRecord?.read(readBuf, 0, readBuf.size) ?: -1
                        if (read > 0) {
                            synchronized(pcmBuffer) {
                                pcmBuffer.write(readBuf, 0, read)
                            }

                            // Dynamic RMS calculation
                            var sum = 0.0
                            for (i in 0 until read step 2) {
                                val sample = (readBuf[i].toInt() and 0xFF) or (readBuf[i + 1].toInt() shl 8)
                                val signedSample = if (sample >= 32768) sample - 65536 else sample
                                sum += signedSample * signedSample
                            }
                            val rms = kotlin.math.sqrt(sum / (read / 2))
                            latestRmsAmplitude = (rms / 7000.0).toFloat().coerceIn(0.08f, 1.0f)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("VoiceVerification", "Audio capture exception: ${e.message}", e)
            } finally {
                try {
                    audioRecord?.stop()
                    audioRecord?.release()
                } catch (_: Exception) {}
                audioRecord = null
            }
        }
        recordingThread?.start()
    }

    private fun stopAndAnalyze() {
        isRecording = false
        mainHandler.removeCallbacks(timerRunnable)
        mainHandler.removeCallbacks(waveformRunnable)

        try {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
        } catch (_: Exception) {}

        binding.layoutListeningState.visibility = View.GONE
        binding.layoutAnalyzingState.visibility = View.VISIBLE

        // Step-by-step progress animation
        mainHandler.postDelayed({
            binding.iconStep3.text = "✓ "
            binding.iconStep3.setTextColor(ContextCompat.getColor(this, R.color.verdict_supported))
            binding.iconStep4.text = "◉ "
            binding.iconStep4.setTextColor(ContextCompat.getColor(this, R.color.brand_primary))
        }, 600)

        mainHandler.postDelayed({
            binding.iconStep4.text = "✓ "
            binding.iconStep4.setTextColor(ContextCompat.getColor(this, R.color.verdict_supported))
            binding.iconStep5.text = "◉ "
            binding.iconStep5.setTextColor(ContextCompat.getColor(this, R.color.brand_primary))
        }, 1200)

        val pcmData = synchronized(pcmBuffer) { pcmBuffer.toByteArray() }
        val wavData = createWavFile(pcmData, 16000, 1, 16)

        VaniVerificationClient.verifyAudioWav(this, wavData) { outcome ->
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread

                val spokenText = if (outcome.hinglishText.isNotBlank()) outcome.hinglishText else outcome.transcript
                if (spokenText.isNotBlank()) {
                    binding.inputTranscribedText.setText(spokenText)
                }

                binding.iconStep5.text = "✓ "
                binding.iconStep5.setTextColor(ContextCompat.getColor(this, R.color.verdict_supported))

                mainHandler.postDelayed({
                    if (outcome.scamResult != null) {
                        val intent = Intent(this, ScamResultActivity::class.java).apply {
                            putExtra(ScamResultActivity.EXTRA_STATEMENT, outcome.scamResult.suspiciousStatement)
                            putExtra(ScamResultActivity.EXTRA_THREAT_SCORE, outcome.scamResult.threatScore)
                            putExtra(ScamResultActivity.EXTRA_SCAM_TYPE, outcome.scamResult.scamType)
                            putExtra(ScamResultActivity.EXTRA_RED_FLAGS_JSON, outcome.redFlagsJson)
                            putExtra(ScamResultActivity.EXTRA_ACTION_ADVICE, outcome.actionAdvice)
                        }
                        startActivity(intent)
                    } else if (outcome.claimResult != null) {
                        val intent = Intent(this, ClaimResultActivity::class.java).apply {
                            putExtra(ClaimResultActivity.EXTRA_CLAIM, outcome.claimResult.claim)
                            putExtra(ClaimResultActivity.EXTRA_VERDICT, outcome.claimResult.verdict.name)
                            putExtra(ClaimResultActivity.EXTRA_CONFIDENCE, outcome.claimResult.confidence)
                            putExtra(ClaimResultActivity.EXTRA_EXPLANATION, outcome.claimResult.explanation)
                            putExtra(ClaimResultActivity.EXTRA_ACTION, outcome.claimResult.recommendedAction)
                            putExtra(ClaimResultActivity.EXTRA_SOURCES_JSON, outcome.sourcesJson)
                        }
                        startActivity(intent)
                    } else {
                        val intent = Intent(this, ClaimResultActivity::class.java).apply {
                            putExtra(ClaimResultActivity.EXTRA_CLAIM, spokenText.ifBlank { "Spoken claim" })
                            putExtra(ClaimResultActivity.EXTRA_VERDICT, "UNVERIFIABLE")
                            putExtra(ClaimResultActivity.EXTRA_CONFIDENCE, 70)
                            putExtra(ClaimResultActivity.EXTRA_EXPLANATION, "No definitive evidence found for this statement.")
                            putExtra(ClaimResultActivity.EXTRA_ACTION, "Verify with official channels before acting.")
                            putExtra(ClaimResultActivity.EXTRA_SOURCES_JSON, outcome.sourcesJson)
                        }
                        startActivity(intent)
                    }
                    finish()
                }, 700)
            }
        }
    }

    private fun createWavFile(pcmData: ByteArray, sampleRate: Int, channels: Int, bitsPerSample: Int): ByteArray {
        val totalAudioLen = pcmData.size
        val totalDataLen = totalAudioLen + 36
        val byteRate = sampleRate * channels * bitsPerSample / 8

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
        header[20] = 1 // PCM
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
        header[32] = (channels * bitsPerSample / 8).toByte()
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

        val wavOut = ByteArrayOutputStream(44 + pcmData.size)
        wavOut.write(header)
        wavOut.write(pcmData)
        return wavOut.toByteArray()
    }

    override fun onDestroy() {
        isRecording = false
        mainHandler.removeCallbacks(timerRunnable)
        mainHandler.removeCallbacks(waveformRunnable)
        try {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
        } catch (_: Exception) {}
        super.onDestroy()
    }
}
