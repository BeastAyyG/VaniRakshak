package com.vericall.audio

import android.os.Handler
import android.os.Looper
import android.util.Log
import com.vericall.model.RiskLevel
import com.vericall.util.VaniRiskBus
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt
import kotlin.random.Random

/**
 * Server-driven forensic stream per docs/PROTOCOL.md.
 *
 * Flow: JSON handshake (session_id first) → binary 16-bit PCM @16kHz →
 * verdict frames. The server owns tier/interlock; this client only renders.
 * Reconnect: 5 attempts, 1s→2s→4s→8s→15s + jitter, fresh handshake each time.
 */
class VaniStreamingClient(
    private val wsUrl: String,
    private val callerId: String = "",
    private val claimedIdentity: String = ""
) {

    companion object {
        private const val TAG = "VaniStreamingClient"
        private const val SAMPLE_RATE = 16000
        private val BACKOFF_MS = longArrayOf(1000, 2000, 4000, 8000, 15000)
        private const val MAX_ATTEMPTS = 5
    }

    var onSnapshot: ((VaniRiskBus.Snapshot) -> Unit)? = null
    /**
     * Raw verdict JSON frames (for paths that need fields beyond the
     * Snapshot, e.g. metrics.tier_override). Fired for every frame that
     * carries risk_score; session_ack is skipped.
     */
    var onVerdictJson: ((JSONObject) -> Unit)? = null
    var onStatus: ((String) -> Unit)? = null
    /** Fires when the stream dies permanently so the caller can fall back to on-device analysis. */
    var onExhausted: (() -> Unit)? = null
    /** Current session id (client-generated per PROTOCOL §1, ack may echo it). */
    var sessionId: String = "android-${UUID.randomUUID()}"
        private set

    private val mainHandler: Handler by lazy { Handler(Looper.getMainLooper()) }
    private val client = OkHttpClient.Builder()
        .pingInterval(20, TimeUnit.SECONDS)
        .build()

    @Volatile private var socket: WebSocket? = null
    @Volatile private var manualClose = false
    /**
     * True only after the JSON handshake has been sent. Gates sendPcm so a
     * binary frame can never overtake the handshake (the server closes with
     * 1003 when the first frame is not JSON).
     */
    @Volatile private var handshakeSent = false
    @Volatile private var attempt = 0
    @Volatile private var everConnected = false

    fun start() {
        manualClose = false
        attempt = 0
        connect()
    }

    private fun connect() {
        if (manualClose) return
        // Fresh session per reconnect — server discards old call state (PROTOCOL §6).
        sessionId = "android-${UUID.randomUUID()}"
        val request = Request.Builder().url(wsUrl).build()
        socket = client.newWebSocket(request, Listener())
    }

    /** Push one PCM chunk (mono 16-bit @16kHz) as a binary frame. No-op until the handshake is sent. */
    fun sendPcm(pcm: ShortArray, count: Int = pcm.size) {
        if (!handshakeSent) return
        val ws = socket ?: return
        val n = count.coerceAtMost(pcm.size)
        if (n <= 0) return
        val bytes = ByteArray(n * 2)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(pcm, 0, n)
        ws.send(ByteString.of(*bytes))
    }

    fun stop() {
        manualClose = true
        handshakeSent = false
        runCatching { socket?.close(1000, "call ended") }
        socket = null
    }

    /**
     * Release the per-call OkHttp resources (dispatcher threads, connection
     * pool). Without this every call leaks threads until their idle timeout.
     * Do not call on an instance you intend to start() again.
     */
    fun release() {
        stop()
        runCatching { client.dispatcher.executorService.shutdown() }
        runCatching { client.connectionPool.evictAll() }
    }

    private fun scheduleReconnect() {
        if (manualClose) return
        if (attempt >= MAX_ATTEMPTS) {
            Log.w(TAG, "reconnect exhausted after $MAX_ATTEMPTS attempts")
            mainHandler.post { onExhausted?.invoke() }
            return
        }
        val delay = BACKOFF_MS[attempt.coerceAtMost(BACKOFF_MS.lastIndex)] +
            Random.nextLong(0, 500)
        attempt++
        postStatus("Connection lost — retrying ($attempt/$MAX_ATTEMPTS)…")
        mainHandler.postDelayed({ connect() }, delay)
    }

    private fun postStatus(s: String) = mainHandler.post { onStatus?.invoke(s) }

    private inner class Listener : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            everConnected = true
            attempt = 0
            val handshake = JSONObject().apply {
                put("session_id", sessionId)
                put("sample_rate", SAMPLE_RATE)
                put("caller_id", callerId)
                put("claimed_identity", claimedIdentity)
                put("transaction_value_inr", 0)
                put("origin_country", "IN")
                put("prior_fraud_score", 0)
            }
            webSocket.send(handshake.toString())
            handshakeSent = true
            Log.i(TAG, "ws connected, session=$sessionId")
            postStatus("Connected — streaming to forensic backend…")
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            try {
                val raw = JSONObject(text)
                if (raw.optString("type") != "session_ack" && raw.has("risk_score")) {
                    mainHandler.post { onVerdictJson?.invoke(raw) }
                }
            } catch (e: Exception) {
                Log.w(TAG, "raw verdict forward failed: " + e.message)
            }
            parseVerdict(text)?.let { snapshot ->
                mainHandler.post { onSnapshot?.invoke(snapshot) }
            }
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            Log.w(TAG, "ws failure: ${t.message}")
            handshakeSent = false
            socket = null
            if (!everConnected && attempt == 0) {
                // Fast-fail the very first attempt so offline fallback kicks in
                // without waiting through the whole backoff chain.
                postStatus("Backend unreachable — using on-device analysis.")
                mainHandler.post { onExhausted?.invoke() }
                return
            }
            scheduleReconnect()
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            handshakeSent = false
            socket = null
            if (!manualClose) scheduleReconnect()
        }
    }

    /** Verdict parser shared with unit tests (pure function of the payload). */
    fun parseVerdict(text: String): VaniRiskBus.Snapshot? {
        return try {
        val json = JSONObject(text)
        // Handshake ack carries no tier/risk_score — skip by shape (PROTOCOL §1).
        if (json.optString("type") == "session_ack") {
            json.optString("session_id").takeIf { it.isNotBlank() }?.let { sessionId = it }
            return null
        }
        if (!json.has("risk_score")) return null

        // Adopt the server-echoed session id when present (same as the ack path):
        // challenge grading must address the session the server knows.
        json.optString("session_id").takeIf { it.isNotBlank() }?.let { sessionId = it }

        val risk = json.optDouble("risk_score", 0.0).roundToInt().coerceIn(0, 100)
        val tier = json.optString("tier", "").ifBlank { null }
        val pSpoof = (json.optDouble("p_spoof", 0.0) * 100).roundToInt().coerceIn(0, 100)
        val metrics = json.optJSONObject("metrics")
        val asvEvaluated = metrics?.optBoolean("asv_evaluated", false) == true
        val speakerMatch = if (asvEvaluated) {
            ((metrics?.optDouble("asv_consistency", 0.5) ?: 0.5) * 100).roundToInt().coerceIn(0, 100)
        } else 50
        val interlock = json.optBoolean("interlock_active", false)
        val receipt = json.optString("receipt_hash").takeIf { it.isNotBlank() }

        val challengeObj = json.optJSONObject("challenge")
        val phrase = challengeObj?.optString("phrase")?.takeIf { it.isNotBlank() }
        val instructions = challengeObj?.optString("instructions")?.takeIf { it.isNotBlank() }

        // Trust the server tier authoritatively (PROTOCOL §5); derive display
        // level from tier first, risk bands only when tier is absent.
        val level = when (tier) {
            "BLOCK" -> RiskLevel.DANGER
            "CHALLENGE" -> RiskLevel.WARNING
            "ALLOW" -> RiskLevel.SAFE
            else -> when {
                risk >= 70 || interlock -> RiskLevel.DANGER
                risk >= 35 -> RiskLevel.WARNING
                else -> RiskLevel.SAFE
            }
        }
        val message = when {
            interlock || tier == "BLOCK" ->
                "Server verdict: BLOCK — do not proceed with any transaction."
            tier == "CHALLENGE" && phrase != null ->
                "Server requests liveness check — ask the caller to repeat the digits."
            tier == "CHALLENGE" ->
                "Server verdict: CHALLENGE — verify the caller before any payment step."
            else -> "Server verdict: ${tier ?: level.name} — risk $risk%."
        }
        VaniRiskBus.Snapshot(
            risk = risk, synthetic = pSpoof, speakerMatch = speakerMatch,
            level = level, message = message, tier = tier, serverDriven = true,
            challengePhrase = phrase, challengeInstructions = instructions,
            receiptHash = receipt, interlockActive = interlock, sessionId = sessionId
        )
    } catch (e: Exception) {
        Log.w(TAG, "verdict parse failed: ${e.message}")
        null
    }
    }
}
