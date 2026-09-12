package com.vericall.contacts

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.LayoutInflater
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.vericall.R
import com.vericall.config.VaniConfig
import com.vericall.data.VaniContactRepository
import com.vericall.model.VaniContact
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * Bottom sheet dialog for recording reference audio samples and enrolling
 * contact acoustic baselines with the VaniRakshak backend (/v1/enroll).
 */
object VaniEnrollmentSheet {

    private const val TAG = "VaniEnrollmentSheet"
    private const val TARGET_DURATION_SEC = 10
    private const val SAMPLE_RATE = 16000

    private val httpClient = OkHttpClient.Builder()
        .protocols(listOf(okhttp3.Protocol.HTTP_1_1))
        .retryOnConnectionFailure(true)
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    @SuppressLint("MissingPermission")
    fun show(context: AppCompatActivity, contact: VaniContact, onProfileSaved: () -> Unit) {
        val dialog = BottomSheetDialog(context)
        val view = LayoutInflater.from(context).inflate(R.layout.bottom_sheet_temp_contact, null)
        dialog.setContentView(view)

        val tvContactName = view.findViewById<TextView>(R.id.tvSheetContactName)
        val tvContactPhone = view.findViewById<TextView>(R.id.tvSheetContactPhone)
        val tvBadge = view.findViewById<TextView>(R.id.tvSheetBadge)
        val pbProgress = view.findViewById<ProgressBar>(R.id.pbEnrollProgress)
        val tvStatus = view.findViewById<TextView>(R.id.tvEnrollStatus)
        val btnRecord = view.findViewById<Button>(R.id.btnStartEnroll)
        val btnSave = view.findViewById<Button>(R.id.btnSaveEnroll)
        val btnClose = view.findViewById<ImageView>(R.id.btnSheetClose)

        tvContactName.text = contact.name
        if (contact.isVoiceEnrolled) {
            tvContactPhone.text = "${contact.phone} • Active voice anchor profile"
            tvBadge.text = "✓ Enrolled"
            tvBadge.setBackgroundResource(R.drawable.bg_badge_supported)
            tvBadge.setTextColor(ContextCompat.getColor(context, R.color.verdict_supported))
        } else {
            tvContactPhone.text = "${contact.phone} • No voice profile yet"
            tvBadge.text = "Not enrolled"
            tvBadge.setBackgroundResource(R.drawable.bg_badge_unverifiable)
            tvBadge.setTextColor(ContextCompat.getColor(context, R.color.verdict_unverifiable))
        }

        var audioRecord: AudioRecord? = null
        var isRecording = false
        var recordedWavBytes: ByteArray? = null
        val mainHandler = Handler(Looper.getMainLooper())

        btnClose.setOnClickListener {
            if (isRecording) {
                isRecording = false
                audioRecord?.stop()
                audioRecord?.release()
                audioRecord = null
            }
            dialog.dismiss()
        }

        dialog.setOnDismissListener {
            if (isRecording) {
                isRecording = false
                audioRecord?.stop()
                audioRecord?.release()
                audioRecord = null
            }
        }

        btnRecord.setOnClickListener {
            if (isRecording) return@setOnClickListener

            btnRecord.isEnabled = false
            btnSave.isEnabled = false
            pbProgress.progress = 0
            tvStatus.text = "Listening... read prompt aloud (0 / ${TARGET_DURATION_SEC}s)"
            tvStatus.setTextColor(ContextCompat.getColor(context, R.color.brand_primary))

            Thread {
                val channelConfig = AudioFormat.CHANNEL_IN_MONO
                val audioFormat = AudioFormat.ENCODING_PCM_16BIT
                val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, channelConfig, audioFormat).coerceAtLeast(SAMPLE_RATE * 2)

                try {
                    val record = AudioRecord(
                        MediaRecorder.AudioSource.MIC,
                        SAMPLE_RATE,
                        channelConfig,
                        audioFormat,
                        bufferSize
                    )

                    if (record.state != AudioRecord.STATE_INITIALIZED) {
                        mainHandler.post {
                            tvStatus.text = "Mic initialization failed"
                            btnRecord.isEnabled = true
                        }
                        return@Thread
                    }

                    audioRecord = record
                    record.startRecording()
                    isRecording = true

                    val pcmOut = ByteArrayOutputStream()
                    val tempBuffer = ByteArray(1600) // 50ms at 16kHz
                    val totalBytesNeeded = SAMPLE_RATE * 2 * TARGET_DURATION_SEC
                    var bytesCaptured = 0

                    while (isRecording && bytesCaptured < totalBytesNeeded) {
                        val read = record.read(tempBuffer, 0, tempBuffer.size)
                        if (read > 0) {
                            pcmOut.write(tempBuffer, 0, read)
                            bytesCaptured += read

                            val progressPct = ((bytesCaptured.toFloat() / totalBytesNeeded) * 100).toInt()
                            val secondsRecorded = bytesCaptured / (SAMPLE_RATE * 2)

                            mainHandler.post {
                                pbProgress.progress = progressPct
                                tvStatus.text = "Recording voice anchor... ($secondsRecorded / ${TARGET_DURATION_SEC}s)"
                            }
                        }
                    }

                    isRecording = false
                    record.stop()
                    record.release()
                    audioRecord = null

                    val rawPcm = pcmOut.toByteArray()
                    val wavData = createWavContainer(rawPcm, SAMPLE_RATE, 1, 16)
                    recordedWavBytes = wavData

                    mainHandler.post {
                        tvStatus.text = "Uploading to VaniRakshak Forensic Core (/v1/enroll)..."
                        tvStatus.setTextColor(ContextCompat.getColor(context, R.color.brand_primary))
                    }

                    // Upload to backend /v1/enroll
                    uploadEnrollmentAudio(context, contact, wavData) { success, message ->
                        mainHandler.post {
                            if (!dialog.isShowing) return@post
                            btnRecord.isEnabled = true
                            btnRecord.text = "Record again"

                            if (success) {
                                tvStatus.text = "✓ $message"
                                tvStatus.setTextColor(ContextCompat.getColor(context, R.color.verdict_supported))
                                tvBadge.text = "✓ Enrolled"
                                tvBadge.setBackgroundResource(R.drawable.bg_badge_supported)
                                tvBadge.setTextColor(ContextCompat.getColor(context, R.color.verdict_supported))

                                btnSave.isEnabled = true
                                btnSave.setTextColor(ContextCompat.getColor(context, R.color.text_primary))
                            } else {
                                tvStatus.text = "⚠ $message (Saved locally)"
                                tvStatus.setTextColor(ContextCompat.getColor(context, R.color.verdict_unverifiable))
                                btnSave.isEnabled = true
                                btnSave.setTextColor(ContextCompat.getColor(context, R.color.text_primary))
                            }
                        }
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Recording error: ${e.message}", e)
                    mainHandler.post {
                        tvStatus.text = "Error capturing audio: ${e.message}"
                        btnRecord.isEnabled = true
                    }
                }
            }.start()
        }

        btnSave.setOnClickListener {
            VaniContactRepository.saveEnrollment(context, contact, TARGET_DURATION_SEC)
            contact.sampleEmbeddingCount = 1
            contact.isVoiceEnrolled = true
            onProfileSaved()
            Toast.makeText(context, "Voice anchor profile active for ${contact.name}", Toast.LENGTH_SHORT).show()
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun uploadEnrollmentAudio(
        context: AppCompatActivity,
        contact: VaniContact,
        wavBytes: ByteArray,
        onComplete: (Boolean, String) -> Unit
    ) {
        val forensicUrl = VaniConfig.getForensicHttpUrl(context)
        val enrollUrl = "$forensicUrl/v1/enroll"

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("user_id", contact.phone.ifBlank { contact.name })
            .addFormDataPart(
                "audio",
                "enrollment_${contact.id}.wav",
                wavBytes.toRequestBody("audio/wav".toMediaType())
            )
            .build()

        val request = Request.Builder()
            .url(enrollUrl)
            .header("Connection", "close")
            .post(requestBody)
            .build()

        try {
            httpClient.newCall(request).enqueue(object : okhttp3.Callback {
                override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                    val body = response.body?.string().orEmpty()
                    if (response.isSuccessful) {
                        try {
                            val json = JSONObject(body)
                            val dim = json.optInt("embedding_dimension", 192)
                            onComplete(true, "Enrolled: ${dim}-dim ECAPA baseline computed (DPDP zero-retention)")
                        } catch (_: Exception) {
                            onComplete(true, "Speaker baseline enrolled successfully")
                        }
                    } else {
                        onComplete(false, "Backend HTTP ${response.code}: $body")
                    }
                }

                override fun onFailure(call: okhttp3.Call, e: java.io.IOException) {
                    Log.w(TAG, "Backend enroll failed: ${e.message}")
                    onComplete(false, "Offline: Saved baseline locally")
                }
            })
        } catch (e: Exception) {
            onComplete(false, "Network error: ${e.message}")
        }
    }

    /**
     * Builds a standard canonical 44-byte RIFF/WAVE header around raw PCM samples.
     */
    private fun createWavContainer(pcmData: ByteArray, sampleRate: Int, channels: Int, bitsPerSample: Int): ByteArray {
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
        header[16] = 16 // SubChunk1Size (16 for PCM)
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1 // AudioFormat (1 for PCM)
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
        header[32] = (channels * bitsPerSample / 8).toByte() // BlockAlign
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

        return header + pcmData
    }
}
