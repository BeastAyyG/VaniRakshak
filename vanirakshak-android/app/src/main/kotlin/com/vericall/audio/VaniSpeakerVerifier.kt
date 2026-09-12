package com.vericall.audio

import android.util.Log
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors

/**
 * Bridges the on-device call audio to the VaniRakshak backend's trained model.
 *
 * HONEST SCOPE: the model answers ONE question — "is this the enrolled target
 * voice (Modi ji)?", NOT "is this a synthetic/clone voice?". It was trained on a
 * single target speaker vs a single non-target speaker and has never seen cloned
 * audio. Present the result as an identity check only.
 *
 * The phone reaches the backend over `adb reverse tcp:8000 tcp:8000`, so the
 * app simply calls http://127.0.0.1:8000 — no external network, no HTTPS needed.
 */
object VaniSpeakerVerifier {

    // Default or fallback URL; can be dynamically overridden or read from VaniConfig
    var backendBaseUrl: String = com.vericall.config.VaniConfig.DEFAULT_BACKEND_URL

    private val executor = Executors.newSingleThreadExecutor()
    private const val TAG = "VaniSpeakerVerifier"

    data class Verdict(
        val available: Boolean,
        val isTarget: Boolean,      // true => "Modi ji", false => "not Modi ji"
        val label: String,          // "modi_ji" | "not_modi_ji" | "unavailable"
        val llr: Double?,
        val probTarget: Double?,
        val reason: String?
    )

    /**
     * Encode 16-bit PCM as a WAV file and POST it to /v1/verify-speaker.
     * @param pcm         mono 16-bit PCM samples
     * @param sampleRate  Hz (typically 16000)
     * @param context     optional Context to resolve configured server URL from VaniConfig
     * @param onResult    called on the calling thread with the model verdict
     */
    fun verify(
        pcm: ShortArray,
        sampleRate: Int,
        context: android.content.Context? = null,
        onResult: (Verdict) -> Unit
    ) {
        val activeBaseUrl = context?.let { com.vericall.config.VaniConfig.getBackendUrl(it) } ?: backendBaseUrl
        executor.execute {
            try {
                val wav = encodeWav(pcm, sampleRate)
                val verdict = postWav(wav, activeBaseUrl)
                onResult(verdict)
            } catch (e: Exception) {
                Log.e(TAG, "verify failed: ${e.message}")
                onResult(Verdict(false, false, "unavailable", null, null, e.message))
            }
        }
    }

    private fun encodeWav(pcm: ShortArray, sampleRate: Int): ByteArray {
        val baos = ByteArrayOutputStream()
        val dataOut = DataOutputStream(baos)
        val dataSize = pcm.size * 2
        // RIFF header
        dataOut.writeBytes("RIFF")
        dataOut.writeInt(Integer.reverseBytes(36 + dataSize))
        dataOut.writeBytes("WAVE")
        // fmt chunk
        dataOut.writeBytes("fmt ")
        dataOut.writeInt(Integer.reverseBytes(16))
        dataOut.writeShort(java.lang.Short.reverseBytes(1).toInt()) // PCM
        dataOut.writeShort(java.lang.Short.reverseBytes(1).toInt()) // mono
        dataOut.writeInt(Integer.reverseBytes(sampleRate))
        dataOut.writeInt(Integer.reverseBytes(sampleRate * 2))      // byte rate
        dataOut.writeShort(java.lang.Short.reverseBytes(2).toInt()) // block align
        dataOut.writeShort(java.lang.Short.reverseBytes(16).toInt()) // bits/sample
        // data chunk
        dataOut.writeBytes("data")
        dataOut.writeInt(Integer.reverseBytes(dataSize))
        for (s in pcm) {
            dataOut.writeShort(java.lang.Short.reverseBytes(s).toInt())
        }
        dataOut.flush()
        return baos.toByteArray()
    }

    private fun postWav(wav: ByteArray, baseUrl: String = backendBaseUrl): Verdict {
        val url = URL("$baseUrl/v1/verify-speaker")
        val conn = url.openConnection() as HttpURLConnection
        conn.requestMethod = "POST"
        conn.doOutput = true
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=VANI")
        conn.connectTimeout = 5000
        conn.readTimeout = 15000

        val boundary = "VANI"
        val body = ByteArrayOutputStream()
        val b = DataOutputStream(body)
        b.writeBytes("--$boundary\r\n")
        b.writeBytes("Content-Disposition: form-data; name=\"audio\"; filename=\"chunk.wav\"\r\n")
        b.writeBytes("Content-Type: audio/wav\r\n\r\n")
        b.write(wav)
        b.writeBytes("\r\n--$boundary--\r\n")
        b.flush()

        conn.outputStream.use { it.write(body.toByteArray()) }

        val code = conn.responseCode
        val text = if (code in 200..299) {
            conn.inputStream.bufferedReader().readText()
        } else {
            conn.errorStream?.bufferedReader()?.readText() ?: ""
        }
        if (code !in 200..299) {
            return Verdict(false, false, "unavailable", null, null, "HTTP $code: $text")
        }
        val json = JSONObject(text)
        return Verdict(
            available = json.optBoolean("available", false),
            isTarget = json.optBoolean("is_target", false),
            label = json.optString("label", "unavailable"),
            llr = if (json.isNull("llr")) null else json.optDouble("llr"),
            probTarget = if (json.isNull("prob_target")) null else json.optDouble("prob_target"),
            reason = if (json.isNull("reason")) null else json.optString("reason")
        )
    }
}
