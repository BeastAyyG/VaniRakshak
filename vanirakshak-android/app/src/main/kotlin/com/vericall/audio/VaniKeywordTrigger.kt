package com.vericall.audio

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log

/**
 * Best-effort on-device keyword trigger for the speaker-verify flow.
 *
 * Listens for banking / transaction keywords (bank, rupee, payment, account, OTP,
 * and their Hindi equivalents) and fires [onKeyword] when one is heard, so the app
 * can auto-ask the backend "is this the enrolled voice (Modi ji)?".
 *
 * IMPORTANT: SpeechRecognizer and [VaniLiveVoiceAnalyzer] both open the microphone.
 * Running them at the same time makes one of them receive silence. Therefore this
 * trigger is OPT-IN: call [start] only when the analyzer is paused (e.g. a dedicated
 * "listen for keyword" mode), and call [stop] before resuming the analyzer.
 * If SpeechRecognizer is unavailable on the device, [start] simply does nothing.
 */
class VaniKeywordTrigger(
    private val context: Context,
    private val keywords: Set<String> = VaniCallGuardKeywords,
    private val onKeyword: (String) -> Unit
) {
    private var recognizer: SpeechRecognizer? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    var isRunning: Boolean = false
        private set

    fun start() {
        if (isRunning) return
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            Log.w(TAG, "SpeechRecognizer unavailable on this device; using manual verify only.")
            return
        }
        try {
            recognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
                setRecognitionListener(listener)
            }
            isRunning = true
            beginListening()
        } catch (e: Exception) {
            Log.e(TAG, "Keyword trigger init failed: ${e.message}")
        }
    }

    fun stop() {
        isRunning = false
        runCatching { recognizer?.stopListening() }
        runCatching { recognizer?.destroy() }
        recognizer = null
    }

    private fun beginListening() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        runCatching { recognizer?.startListening(intent) }
    }

    private val listener = object : RecognitionListener {
        override fun onResults(results: Bundle?) = handle(results)
        override fun onPartialResults(results: Bundle?) = handle(results)
        override fun onError(error: Int) {
            // Restart on transient errors (e.g. no speech). Stop only on fatal ones.
            if (isRunning && error != SpeechRecognizer.ERROR_RECOGNIZER_BUSY) {
                mainHandler.postDelayed({ beginListening() }, 500)
            }
        }
        override fun onReadyForSpeech(p: Bundle?) {}
        override fun onBeginningOfSpeech() {}
        override fun onRmsChanged(r: Float) {}
        override fun onBufferReceived(b: ByteArray?) {}
        override fun onEndOfSpeech() {}
        override fun onEvent(e: Int, p: Bundle?) {}
    }

    private fun handle(results: Bundle?) {
        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION) ?: return
        val text = matches.joinToString(" ").lowercase()
        val hit = keywords.firstOrNull { kw -> text.contains(kw) }
        if (hit != null) {
            isRunning = false
            onKeyword(hit)
            stop()
        } else if (isRunning) {
            // Keep listening for the keyword.
            mainHandler.postDelayed({ beginListening() }, 300)
        }
    }

    private companion object {
        const val TAG = "VaniKeywordTrigger"
        // Mirrors VaniCallGuardService.TRIGGER_KEYWORDS without a hard import cycle.
        val VaniCallGuardKeywords = setOf(
            "bank", "banking", "rupee", "rupees", "rupay", "rs ", "rs.", "payment",
            "transaction", "transfer", "account", "otp", "पैसा", "बैंक", "रुपया", "भुगतान"
        )
    }
}
