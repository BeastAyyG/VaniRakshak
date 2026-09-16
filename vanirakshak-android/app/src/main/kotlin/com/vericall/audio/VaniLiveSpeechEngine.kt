package com.vericall.audio

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import com.vericall.model.RiskLevel
import com.vericall.model.TranscriptUtterance
import com.vericall.util.VaniRiskBus

/**
 * VaniLiveSpeechEngine — Android-native real-time speech transcription.
 *
 * ARCHITECTURE:
 *   Uses Android's [SpeechRecognizer] for speech-to-text.
 *   The recognizer is created ONCE at session start and REUSED for the
 *   entire call. On results/errors, startListening() is called on the
 *   SAME instance — no destroy/recreate cycle, no mid-call mic churn.
 *
 *   All events, errors, and latencies are reported in real-time to
 *   [VaniAudioDiagnosticsHud] for complete observability.
 *
 * AUDIO CAPABILITY DISCLOSURE:
 *   If direct telephony capture is unavailable on this device, this engine
 *   transcribes the LOCAL USER'S OWN VOICE only.
 *   Transcripts are honestly labeled "You (Microphone)".
 */
class VaniLiveSpeechEngine(private val context: Context) {

    companion object {
        private const val TAG = "VaniLiveSpeechEngine"

        // Preferred recognition languages in order
        private val LANGUAGES = listOf("hi-IN", "en-IN")

        // Debounce delays to prevent mic churn
        private const val RESULT_RESTART_DELAY_MS = 600L
        private const val NATURAL_PAUSE_RESTART_MS = 1000L
        private const val MAX_CONSECUTIVE_ERRORS = 4
        private const val CIRCUIT_BREAKER_COOLDOWN_MS = 10000L
        private const val AUDIO_ERROR_BASE_BACKOFF_MS = 3000L
    }

    private var recognizer: SpeechRecognizer? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile
    private var isRunning = false
    val isEngineRunning: Boolean get() = isRunning

    private var isCurrentlyListening = false
    private var currentLanguageIndex = 0
    private var consecutiveErrorCount = 0
    private val committedUtterances = mutableListOf<TranscriptUtterance>()
    private var lastPartialText = ""
    private var lastPartialTimestamp = 0L
    private var sessionStartTime = 0L

    // Callbacks
    var onTranscriptUpdated: ((List<TranscriptUtterance>) -> Unit)? = null
    var onScamAlert: ((VaniScamDetector.DetectionResult) -> Unit)? = null
    var onClaimExtracted: ((String) -> Unit)? = null

    // Honest speaker labeling based on capability
    var speakerLabel: String = "You (Microphone)"

    /**
     * Creates the SpeechRecognizer ONCE and starts listening.
     * Must be called from the main thread.
     */
    fun start(sessionCallId: String = "", label: String = "You (Microphone)") {
        if (isRunning) {
            Log.d(TAG, "start() called while already running for session: $sessionCallId")
            return
        }

        speakerLabel = label
        val isAvailable = SpeechRecognizer.isRecognitionAvailable(context)
        Log.i(TAG, "start() initiated for session: '$sessionCallId'. isRecognitionAvailable=$isAvailable")
        VaniAudioDiagnosticsHud.updateSttState(isActive = true, provider = "Android SpeechRecognizer ($speakerLabel)")
        VaniAudioDiagnosticsHud.recordSttEvent("Speech engine initialized for session $sessionCallId")

        isRunning = true
        isCurrentlyListening = false
        sessionStartTime = System.currentTimeMillis()
        currentLanguageIndex = 0
        consecutiveErrorCount = 0
        committedUtterances.clear()
        lastPartialText = ""

        createRecognizer()
        beginListening()
    }

    /**
     * Stops recognition and flushes any uncommitted partial text as a final utterance.
     * Destroys the recognizer only at call termination.
     */
    fun stop() {
        if (!isRunning) return
        isRunning = false
        isCurrentlyListening = false

        VaniAudioDiagnosticsHud.updateSttState(isActive = false)
        VaniAudioDiagnosticsHud.recordSttEvent("Speech engine stopped")

        mainHandler.removeCallbacksAndMessages(null)
        runCatching { recognizer?.cancel() }
        runCatching { recognizer?.destroy() }
        recognizer = null

        if (lastPartialText.isNotBlank()) {
            commitUtterance(lastPartialText, isFinal = true)
        }
        Log.i(TAG, "VaniLiveSpeechEngine stopped cleanly. Total utterances: ${committedUtterances.size}")
    }

    /**
     * Returns a snapshot of all transcript utterances collected so far.
     */
    fun getTranscript(): List<TranscriptUtterance> = committedUtterances.toList()

    private fun createRecognizer() {
        recognizer = runCatching {
            SpeechRecognizer.createSpeechRecognizer(context)
        }.getOrElse { e ->
            Log.w(TAG, "Standard SpeechRecognizer creation failed: ${e.message}. Trying on-device.")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && SpeechRecognizer.isOnDeviceRecognitionAvailable(context)) {
                runCatching { SpeechRecognizer.createOnDeviceSpeechRecognizer(context) }.getOrNull()
            } else null
        }?.apply {
            setRecognitionListener(recognitionListener)
        }

        if (recognizer == null) {
            val err = "SpeechRecognizer could not be instantiated on this device."
            Log.e(TAG, err)
            VaniAudioDiagnosticsHud.recordError(err)
        } else {
            Log.i(TAG, "✓ SpeechRecognizer created (will be reused for entire call)")
        }
    }

    private fun buildRecognitionIntent(): Intent {
        val lang = LANGUAGES[currentLanguageIndex % LANGUAGES.size]
        return Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, lang)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, lang)
            putExtra(RecognizerIntent.EXTRA_ONLY_RETURN_LANGUAGE_PREFERENCE, false)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 3000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 2000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, 500L)
        }
    }

    private fun beginListening() {
        if (!isRunning || isCurrentlyListening) return

        if (recognizer == null) {
            Log.e(TAG, "Recognizer is null — cannot listen. Creating fresh instance.")
            createRecognizer()
            if (recognizer == null) return
        }

        val intent = buildRecognitionIntent()
        val lang = intent.getStringExtra(RecognizerIntent.EXTRA_LANGUAGE)

        runCatching {
            isCurrentlyListening = true
            recognizer?.startListening(intent)
            Log.d(TAG, "Listening with language: $lang (reusing existing recognizer)")
            VaniAudioDiagnosticsHud.recordSttEvent("startListening($lang)")
        }.onFailure { e ->
            isCurrentlyListening = false
            Log.e(TAG, "startListening failed: ${e.message}")
            VaniAudioDiagnosticsHud.recordError("startListening failed: ${e.message}")
            handleRecognitionFailure(isAudioError = true)
        }
    }

    private fun handleRecognitionFailure(isAudioError: Boolean) {
        if (!isRunning) return
        isCurrentlyListening = false
        consecutiveErrorCount++

        if (consecutiveErrorCount >= MAX_CONSECUTIVE_ERRORS) {
            Log.w(TAG, "Circuit breaker tripped: $consecutiveErrorCount consecutive errors. Cooldown $CIRCUIT_BREAKER_COOLDOWN_MS ms.")
            VaniAudioDiagnosticsHud.recordError("Circuit breaker tripped ($consecutiveErrorCount errors)")
            val prev = VaniRiskBus.last
            val baseSnapshot = prev ?: VaniRiskBus.Snapshot(
                risk = 0, synthetic = 0, speakerMatch = 50,
                level = RiskLevel.SAFE, message = "🛡 Protection Active"
            )
            VaniRiskBus.publish(
                baseSnapshot.copy(
                    message = "🛡 Protection Active • Speech monitor in standby"
                )
            )
            runCatching { recognizer?.cancel() }
            mainHandler.postDelayed({
                if (isRunning) {
                    Log.i(TAG, "Circuit breaker reset: attempting to resume speech recognition.")
                    consecutiveErrorCount = 0
                    beginListening()
                }
            }, CIRCUIT_BREAKER_COOLDOWN_MS)
            return
        }

        val backoffDelay = if (isAudioError) {
            AUDIO_ERROR_BASE_BACKOFF_MS * consecutiveErrorCount.coerceAtMost(3)
        } else {
            NATURAL_PAUSE_RESTART_MS
        }

        scheduleRestart(backoffDelay)
    }

    private fun scheduleRestart(delayMs: Long) {
        if (!isRunning) return
        isCurrentlyListening = false
        mainHandler.removeCallbacksAndMessages(null)
        mainHandler.postDelayed({
            if (isRunning) {
                beginListening()
            }
        }, delayMs)
    }

    private fun commitUtterance(text: String, isFinal: Boolean) {
        val cleaned = text.trim()
        if (cleaned.isBlank()) return

        val detection = VaniScamDetector.analyze(cleaned)

        val utterance = TranscriptUtterance(
            speaker = speakerLabel,
            text = VaniScamDetector.redact(cleaned),
            timestamp = System.currentTimeMillis(),
            isFinal = isFinal
        )

        if (isFinal) {
            if (committedUtterances.isNotEmpty() && !committedUtterances.last().isFinal) {
                committedUtterances[committedUtterances.size - 1] = utterance
            } else {
                committedUtterances.add(utterance)
            }
        } else {
            if (committedUtterances.isNotEmpty() && !committedUtterances.last().isFinal) {
                committedUtterances[committedUtterances.size - 1] = utterance
            } else {
                committedUtterances.add(utterance)
            }
        }

        lastPartialText = if (isFinal) "" else cleaned
        lastPartialTimestamp = System.currentTimeMillis()

        VaniAudioDiagnosticsHud.recordSttEvent("Utterance (${if (isFinal) "FINAL" else "PARTIAL"}): $cleaned")

        onTranscriptUpdated?.invoke(committedUtterances.toList())

        val fullTranscript = committedUtterances.joinToString(" ") { it.text }
        val prev = VaniRiskBus.last
        val effectiveRisk = maxOf(prev?.risk ?: 0, detection.threatScore)
        val level = when {
            effectiveRisk >= 70 -> RiskLevel.DANGER
            effectiveRisk >= 35 -> RiskLevel.WARNING
            else -> RiskLevel.SAFE
        }

        val updatedRedFlags = ((prev?.redFlags ?: emptyList()) + detection.redFlags).distinct()
        val baseSnapshot = prev ?: VaniRiskBus.Snapshot(
            risk = 0, synthetic = 0, speakerMatch = 50,
            level = RiskLevel.SAFE, message = "🛡 Protection Active"
        )

        VaniRiskBus.publish(
            baseSnapshot.copy(
                risk = effectiveRisk,
                level = level,
                transcript = fullTranscript,
                utterances = committedUtterances.toList(),
                threatScore = detection.threatScore,
                redFlags = updatedRedFlags,
                message = when {
                    detection.isHighRisk -> "⚠️ Suspicious speech pattern detected"
                    level == RiskLevel.DANGER -> "🚨 High-risk conversation detected"
                    else -> prev?.message ?: "🛡 Protection Active"
                }
            )
        )

        if (detection.isHighRisk) {
            onScamAlert?.invoke(detection)
            Log.w(TAG, "HIGH RISK utterance: score=${detection.threatScore} flags=${detection.redFlags}")
        }

        detection.extractedClaim?.let { claim ->
            Log.i(TAG, "Claim extracted for verification: $claim")
            onClaimExtracted?.invoke(claim)
        }
    }

    private val recognitionListener = object : RecognitionListener {

        override fun onReadyForSpeech(params: Bundle?) {
            isCurrentlyListening = true
            VaniAudioDiagnosticsHud.recordSttEvent("onReadyForSpeech")
        }

        override fun onBeginningOfSpeech() {
            VaniAudioDiagnosticsHud.recordSttEvent("onBeginningOfSpeech")
        }

        override fun onRmsChanged(rmsdB: Float) {
            val normalized = ((rmsdB + 2f) / 12f).coerceIn(0f, 1f)
            VaniRiskBus.publishAmplitude(normalized)
        }

        override fun onBufferReceived(buffer: ByteArray?) {}

        override fun onEndOfSpeech() {
            isCurrentlyListening = false
            VaniAudioDiagnosticsHud.recordSttEvent("onEndOfSpeech")
        }

        override fun onPartialResults(partialResults: Bundle?) {
            val results = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            val partial = results?.firstOrNull()?.trim() ?: return
            if (partial.isBlank() || partial == lastPartialText) return

            if (VaniScamDetector.hasImmediateThreat(partial)) {
                val prev = VaniRiskBus.last
                val baseSnapshot = prev ?: VaniRiskBus.Snapshot(
                    risk = 0, synthetic = 0, speakerMatch = 50,
                    level = RiskLevel.SAFE, message = "🛡 Protection Active"
                )
                VaniRiskBus.publish(
                    baseSnapshot.copy(
                        level = RiskLevel.DANGER,
                        risk = maxOf(prev?.risk ?: 0, 70),
                        message = "🚨 Critical: Possible credential request detected"
                    )
                )
            }

            commitUtterance(partial, isFinal = false)
        }

        override fun onResults(results: Bundle?) {
            isCurrentlyListening = false
            consecutiveErrorCount = 0
            val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            val finalResult = matches?.firstOrNull()?.trim()

            if (!finalResult.isNullOrBlank()) {
                Log.i(TAG, "Final result: $finalResult")
                commitUtterance(finalResult, isFinal = true)
            }

            if (isRunning) {
                currentLanguageIndex++
                scheduleRestart(RESULT_RESTART_DELAY_MS)
            }
        }

        override fun onError(error: Int) {
            isCurrentlyListening = false
            val errorName = when (error) {
                SpeechRecognizer.ERROR_AUDIO -> "ERROR_AUDIO"
                SpeechRecognizer.ERROR_CLIENT -> "ERROR_CLIENT"
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "ERROR_INSUFFICIENT_PERMISSIONS"
                SpeechRecognizer.ERROR_NETWORK -> "ERROR_NETWORK"
                SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "ERROR_NETWORK_TIMEOUT"
                SpeechRecognizer.ERROR_NO_MATCH -> "ERROR_NO_MATCH"
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "ERROR_RECOGNIZER_BUSY"
                SpeechRecognizer.ERROR_SERVER -> "ERROR_SERVER"
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "ERROR_SPEECH_TIMEOUT"
                else -> "ERROR_$error"
            }
            Log.w(TAG, "SpeechRecognizer error: $errorName ($error)")
            VaniAudioDiagnosticsHud.recordError("SpeechRecognizer: $errorName")

            when (error) {
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> {
                    isRunning = false
                    Log.e(TAG, "RECORD_AUDIO permission not granted — cannot transcribe")
                }
                SpeechRecognizer.ERROR_AUDIO,
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY,
                SpeechRecognizer.ERROR_CLIENT -> {
                    runCatching { recognizer?.cancel() }
                    handleRecognitionFailure(isAudioError = true)
                }
                SpeechRecognizer.ERROR_NETWORK,
                SpeechRecognizer.ERROR_NETWORK_TIMEOUT,
                SpeechRecognizer.ERROR_SERVER -> {
                    val prev = VaniRiskBus.last
                    val baseSnapshot = prev ?: VaniRiskBus.Snapshot(
                        risk = 0, synthetic = 0, speakerMatch = 50,
                        level = RiskLevel.SAFE, message = "🛡 Protection Active"
                    )
                    VaniRiskBus.publish(
                        baseSnapshot.copy(
                            message = "⚠️ Live transcription in offline mode"
                        )
                    )
                    handleRecognitionFailure(isAudioError = false)
                }
                else -> {
                    consecutiveErrorCount = 0
                    if (isRunning) {
                        scheduleRestart(NATURAL_PAUSE_RESTART_MS)
                    }
                }
            }
        }

        override fun onEvent(eventType: Int, params: Bundle?) {}
    }
}
