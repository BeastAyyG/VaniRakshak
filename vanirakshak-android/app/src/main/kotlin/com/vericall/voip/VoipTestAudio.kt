package com.vericall.voip

import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.UUID

/**
 * Headless-test audio: synthesizes a spoken sentence with the on-device TTS
 * engine into a WAV file, then serves its PCM in a loop as a deterministic
 * stand-in for the live microphone. Used only when the voip_test_audio=tts
 * intent extra is set, so emulator verification scores real speech instead
 * of a silent virtual mic.
 */
object VoipTestAudio {

    private const val TAG = "VoipTestAudio"
    const val TEST_SENTENCE =
        "Hello, this is a voice verification test. Please confirm that you can hear me clearly."

    /**
     * Synthesize [TEST_SENTENCE] to a WAV file in the app cache dir.
     * Calls back on the main thread with the file, or null on failure/timeout.
     */
    fun synthesizeTtsWav(context: Context, onDone: (File?) -> Unit) {
        val mainHandler = Handler(Looper.getMainLooper())
        var tts: TextToSpeech? = null
        var finished = false
        lateinit var timeoutTask: Runnable
        fun done(file: File?) {
            if (finished) return
            finished = true
            // Remove only our own timeout, never the whole main-looper queue.
            runCatching { mainHandler.removeCallbacks(timeoutTask) }
            runCatching { tts?.shutdown() }
            mainHandler.post { onDone(file) }
        }
        // Hard timeout: never hang the test on TTS.
        timeoutTask = Runnable { done(null) }
        mainHandler.postDelayed(timeoutTask, 45_000)

        tts = TextToSpeech(context.applicationContext) { status ->
            if (status != TextToSpeech.SUCCESS) {
                Log.w(TAG, "tts init failed: $status")
                done(null)
                return@TextToSpeech
            }
            val engine = tts!!
            engine.language = Locale.US
            val outFile = File(context.cacheDir, "voip-test-${UUID.randomUUID()}.wav")
            val utteranceId = "voip-test-tts"
            engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String) {}
                override fun onDone(utteranceId: String) {
                    Log.i(TAG, "tts synthesized: ${outFile.length()} bytes")
                    done(if (outFile.exists() && outFile.length() > 1000) outFile else null)
                }

                @Deprecated("legacy callback")
                override fun onError(utteranceId: String) {
                    Log.w(TAG, "tts synthesis error")
                    done(null)
                }

                override fun onError(utteranceId: String, errorCode: Int) {
                    Log.w(TAG, "tts synthesis error: $errorCode")
                    done(null)
                }
            })
            val params = Bundle()
            val rc = engine.synthesizeToFile(TEST_SENTENCE, params, outFile, utteranceId)
            if (rc != TextToSpeech.SUCCESS) {
                Log.w(TAG, "synthesizeToFile returned $rc")
                done(null)
            }
        }
    }

    /**
     * Read a 16-bit PCM WAV file into 16 kHz mono int16 samples.
     * Returns null when the file cannot be parsed.
     */
    fun wavToPcm16kMono(file: File): ShortArray? {
        return try {
            val bytes = FileInputStream(file).use { it.readBytes() }
            if (bytes.size < 44) return null
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            // Minimal RIFF parse: find "fmt " and "data" chunks.
            var audioFormat = 1
            var channels = 1
            var sampleRate = 16000
            var dataStart = -1
            var dataLen = 0
            var pos = 12
            while (pos + 8 <= bytes.size) {
                val chunkId = String(bytes, pos, 4, Charsets.US_ASCII)
                val chunkSize = buf.getInt(pos + 4)
                if (chunkId == "fmt ") {
                    audioFormat = buf.getShort(pos + 8).toInt()
                    channels = buf.getShort(pos + 10).toInt()
                    sampleRate = buf.getInt(pos + 12)
                } else if (chunkId == "data") {
                    dataStart = pos + 8
                    dataLen = chunkSize
                    break
                }
                pos += 8 + chunkSize + (chunkSize % 2)
            }
            if (dataStart < 0 || audioFormat != 1) return null
            val frames = dataLen / 2 / channels
            val interleaved = ShortArray(frames * channels)
            ByteBuffer.wrap(bytes, dataStart, frames * channels * 2)
                .order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(interleaved)
            val mono = if (channels == 1) interleaved else ShortArray(frames) { i ->
                var acc = 0
                for (c in 0 until channels) acc += interleaved[i * channels + c].toInt()
                (acc / channels).toShort()
            }
            resampleTo16k(mono, sampleRate)
        } catch (e: Exception) {
            Log.w(TAG, "wav parse failed: ${e.message}")
            null
        }
    }

    private fun resampleTo16k(mono: ShortArray, rate: Int): ShortArray {
        if (rate == 16000) return mono
        if (rate == 48000) {
            val out = ShortArray(mono.size / 3)
            var j = 0
            var i = 0
            while (i + 2 < mono.size && j < out.size) {
                out[j++] = ((mono[i].toInt() + mono[i + 1].toInt() + mono[i + 2].toInt()) / 3).toShort()
                i += 3
            }
            return if (j < out.size) out.copyOf(j) else out
        }
        if (rate <= 0) return ShortArray(0)
        val ratio = rate / 16000.0
        val outLen = (mono.size / ratio).toInt()
        return ShortArray(outLen) { j ->
            val p = j * ratio
            val i0 = p.toInt().coerceIn(0, mono.size - 1)
            val i1 = (i0 + 1).coerceIn(0, mono.size - 1)
            val frac = p - i0
            (mono[i0] * (1 - frac) + mono[i1] * frac).toInt().toShort()
        }
    }

    /**
     * Build the manager's test PCM feeder: serves [pcm] in [chunkSamples]
     * slices, looping forever, mimicking 10 ms mic callbacks.
     */
    fun loopingFeeder(pcm: ShortArray, chunkSamples: Int = 160): () -> ShortArray? {
        var pos = 0
        val lock = Any()
        return {
            synchronized(lock) {
                if (pcm.isEmpty()) {
                    null
                } else {
                    val out = ShortArray(chunkSamples)
                    var filled = 0
                    while (filled < chunkSamples) {
                        val take = minOf(chunkSamples - filled, pcm.size - pos)
                        System.arraycopy(pcm, pos, out, filled, take)
                        filled += take
                        pos += take
                        if (pos >= pcm.size) pos = 0
                    }
                    out
                }
            }
        }
    }
}
