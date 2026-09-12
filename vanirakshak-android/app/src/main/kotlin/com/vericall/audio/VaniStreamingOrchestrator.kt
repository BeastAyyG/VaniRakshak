package com.vericall.audio

import android.content.Context
import android.util.Log
import com.vericall.config.VaniConfig
import com.vericall.model.RiskLevel
import com.vericall.util.VaniRiskBus
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.TimeUnit

/**
 * Dual WebSocket audio streaming orchestrator.
 * Multiplexes 16 kHz 16-bit mono PCM audio from the mic to:
 *   1. VaniRakshak Forensic Core (Port 8000 /ws/stream)
 *   2. IndicConformer Multi-Lingual ASR & Scam Matrix (Port 8084 /ws/live-call-stream)
 *
 * Publishes fused verdicts directly to [VaniRiskBus].
 */
class VaniStreamingOrchestrator(private val context: Context) {

    companion object {
        private const val TAG = "VaniStreamOrchestrator"
    }

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .connectTimeout(5000, TimeUnit.MILLISECONDS)
        .build()

    private var forensicWs: WebSocket? = null
    private var indicWs: WebSocket? = null

    @Volatile
    private var isStreaming = false

    private var currentSessionId: String = ""
    private var currentCallId: String = ""

    // Current accumulated state
    private var lastRiskScore: Int = 0
    private var lastTier: String = "ALLOW"
    private var lastTranscript: String = ""
    private var lastScamType: String = ""
    private var lastThreatScore: Int = 0
    private var lastRedFlags: List<String> = emptyList()
    private var lastActionAdvice: String = ""
    private var lastReceiptHash: String = ""
    private var lastMetrics = mutableMapOf<String, Float>()

    /**
     * Connects both streaming WebSockets and begins session streaming.
     */
    fun startStreaming(callerName: String, callerPhone: String, claimedIdentity: String = "") {
        if (isStreaming) return
        isStreaming = true

        currentSessionId = "session_${UUID.randomUUID().toString().take(8)}"
        currentCallId = "call_${System.currentTimeMillis()}"

        // Reset state
        lastRiskScore = 0
        lastTier = "ALLOW"
        lastTranscript = ""
        lastScamType = ""
        lastThreatScore = 0
        lastRedFlags = emptyList()
        lastActionAdvice = ""
        lastReceiptHash = ""
        lastMetrics.clear()

        connectForensicWebSocket(callerPhone, claimedIdentity.ifBlank { callerName })
        connectIndicWebSocket(callerPhone)
    }

    private fun connectForensicWebSocket(callerPhone: String, identity: String) {
        val url = VaniConfig.getForensicWsUrl(context)
        Log.d(TAG, "Connecting to Forensic Core WebSocket: $url")

        val request = Request.Builder().url(url).build()
        forensicWs = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "✓ Forensic WebSocket connected. Sending handshake frame...")
                val handshake = JSONObject().apply {
                    put("session_id", currentSessionId)
                    put("sample_rate", 16000)
                    put("caller_id", callerPhone)
                    put("claimed_identity", identity)
                    put("transaction_value_inr", 0)
                    put("origin_country", "IN")
                    put("prior_fraud_score", 0)
                }
                webSocket.send(handshake.toString())
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    handleForensicVerdict(json)
                } catch (e: Exception) {
                    Log.w(TAG, "Error parsing forensic frame: ${e.message}")
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.w(TAG, "Forensic WebSocket failed/disconnected: ${t.message}")
            }
        })
    }

    private fun connectIndicWebSocket(callerPhone: String) {
        val url = VaniConfig.getIndicWsUrl(context, currentCallId, callerPhone)
        Log.d(TAG, "Connecting to IndicConformer WebSocket: $url")

        val request = Request.Builder().url(url).build()
        indicWs = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "✓ IndicConformer Live Stream connected.")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    handleIndicStreamUpdate(json)
                } catch (e: Exception) {
                    Log.w(TAG, "Error parsing Indic stream frame: ${e.message}")
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.w(TAG, "IndicConformer WebSocket failed/disconnected: ${t.message}")
            }
        })
    }

    /**
     * Push raw 16-bit mono 16 kHz PCM chunk to both sockets.
     */
    fun sendAudioChunk(pcmChunk: ByteArray) {
        if (!isStreaming || pcmChunk.isEmpty()) return
        val byteString = pcmChunk.toByteString()
        forensicWs?.send(byteString)
        indicWs?.send(byteString)
    }

    private fun handleForensicVerdict(json: JSONObject) {
        val riskDouble = json.optDouble("risk_score", 0.0)
        lastRiskScore = riskDouble.toInt().coerceIn(0, 100)
        lastTier = json.optString("tier", if (lastRiskScore >= 70) "BLOCK" else if (lastRiskScore >= 35) "CHALLENGE" else "ALLOW")
        lastReceiptHash = json.optString("receipt_hash", "")

        val metricsObj = json.optJSONObject("metrics")
        if (metricsObj != null) {
            lastMetrics["cm_score"] = metricsObj.optDouble("cm_score", 0.0).toFloat()
            lastMetrics["cm_genuine_prob"] = metricsObj.optDouble("cm_genuine_prob", 0.0).toFloat()
            lastMetrics["asv_consistency"] = metricsObj.optDouble("asv_consistency", 0.0).toFloat()
            lastMetrics["channel_anomaly"] = metricsObj.optDouble("channel_anomaly", 0.0).toFloat()
            lastMetrics["snr_db"] = metricsObj.optDouble("snr_db", 18.0).toFloat()
            lastMetrics["prosody_unnatural"] = metricsObj.optDouble("prosody_unnatural", 0.0).toFloat()
        }

        publishFusedSnapshot()
    }

    private fun handleIndicStreamUpdate(json: JSONObject) {
        val event = json.optString("event", "")
        if (event == "stream_update" || event == "call_completed") {
            val transcript = json.optString("full_transcript", json.optString("incremental_text", ""))
            if (transcript.isNotBlank()) {
                lastTranscript = transcript
            }
            lastThreatScore = json.optInt("threat_score", lastThreatScore)
            val scamType = json.optString("scam_type", "")
            if (scamType.isNotBlank()) lastScamType = scamType

            val advice = json.optString("action_advice", "")
            if (advice.isNotBlank()) lastActionAdvice = advice

            val redFlagsArr = json.optJSONArray("red_flags")
            if (redFlagsArr != null) {
                val list = mutableListOf<String>()
                for (i in 0 until redFlagsArr.length()) {
                    list.add(redFlagsArr.optString(i))
                }
                lastRedFlags = list
            }

            publishFusedSnapshot()
        }
    }

    private fun publishFusedSnapshot() {
        // Compute effective risk: highest of acoustic forensic risk or scam coercion score
        val effectiveRisk = maxOf(lastRiskScore, lastThreatScore).coerceIn(0, 100)
        val level = when {
            effectiveRisk >= 70 || lastTier == "BLOCK" -> RiskLevel.DANGER
            effectiveRisk >= 35 || lastTier == "CHALLENGE" -> RiskLevel.WARNING
            else -> RiskLevel.SAFE
        }

        val syntheticScore = ((lastMetrics["cm_score"] ?: (effectiveRisk / 100f)) * 100).toInt().coerceIn(0, 100)
        val speakerMatch = ((lastMetrics["asv_consistency"] ?: ((100 - effectiveRisk) / 100f)) * 100).toInt().coerceIn(0, 100)

        val message = when (level) {
            RiskLevel.DANGER -> if (lastScamType.isNotBlank()) "CRITICAL: $lastScamType detected" else "Synthetic Voice Clone Detected — High Impersonation Risk"
            RiskLevel.WARNING -> "Acoustic Anomaly / Suspicious Coercion Cues"
            RiskLevel.SAFE -> "Authentic Target Voice • No Scam Flags"
        }

        val snapshot = VaniRiskBus.Snapshot(
            risk = effectiveRisk,
            synthetic = syntheticScore,
            speakerMatch = speakerMatch,
            level = level,
            message = message,
            tier = lastTier,
            pSpoof = syntheticScore / 100f,
            transcript = lastTranscript,
            scamType = lastScamType,
            threatScore = lastThreatScore,
            redFlags = lastRedFlags,
            actionAdvice = lastActionAdvice,
            receiptHash = lastReceiptHash,
            cmScore = lastMetrics["cm_score"] ?: 0f,
            cmGenuineProb = lastMetrics["cm_genuine_prob"] ?: 0f,
            asvConsistency = lastMetrics["asv_consistency"] ?: 0f,
            channelAnomaly = lastMetrics["channel_anomaly"] ?: 0f,
            snrDb = lastMetrics["snr_db"] ?: 18f,
            vocoderFlatness = 0.15f + (syntheticScore / 100f) * 0.70f,
            microTremor = if (level == RiskLevel.DANGER) 0.0085f else 0.0412f,
            phaseJumps = if (level == RiskLevel.DANGER) (1 + syntheticScore / 20) else 0
        )

        VaniRiskBus.publish(snapshot)
    }

    /**
     * Gracefully terminates both streams.
     */
    fun stopStreaming() {
        if (!isStreaming) return
        isStreaming = false

        try {
            indicWs?.send("{\"event\": \"end_call\"}")
            indicWs?.close(1000, "Call Ended")
        } catch (_: Exception) {}

        try {
            forensicWs?.close(1000, "Call Ended")
        } catch (_: Exception) {}

        forensicWs = null
        indicWs = null
    }
}
