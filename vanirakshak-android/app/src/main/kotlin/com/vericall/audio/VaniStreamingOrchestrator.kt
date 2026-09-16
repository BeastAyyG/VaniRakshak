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
 * Optional dual WebSocket audio streaming orchestrator for REMOTE backend integration.
 *
 * In the default (no-PC) configuration, this class does nothing — transcription is
 * handled on-device by [VaniLiveSpeechEngine] and fraud detection by [VaniScamDetector].
 *
 * This class only activates when the user has configured a remote backend server in
 * [com.vericall.config.VaniConfig]. It then multiplexes 16kHz 16-bit mono PCM to:
 *   1. VaniRakshak Forensic Core (/ws/stream)
 *   2. IndicConformer Multi-Lingual ASR (/ws/live-call-stream) — optional
 *
 * Publishes fused verdicts to [VaniRiskBus] alongside the on-device pipeline.
 */
class VaniStreamingOrchestrator(private val context: Context) {

    companion object {
        private const val TAG = "VaniStreamOrchestrator"
    }

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .connectTimeout(5000, TimeUnit.MILLISECONDS)
        .build()

    private val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())

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
    private var lastCrawlerStatus: String = ""
    private var lastSources: List<com.vericall.model.WebSourceItem> = emptyList()
    private var lastFactChecks: List<com.vericall.model.FactCheckItem> = emptyList()
    private val accumulatedUtterances = mutableListOf<com.vericall.model.TranscriptUtterance>()

    var onTranscriptUpdated: ((List<com.vericall.model.TranscriptUtterance>) -> Unit)? = null
    var onConnectionStateChanged: ((Boolean) -> Unit)? = null

    @Volatile
    var isGatewayConnected: Boolean = false
        private set

    private var candidateHosts: List<String> = emptyList()
    private var hostIndex = 0

    fun getTranscript(): List<com.vericall.model.TranscriptUtterance> = accumulatedUtterances.toList()

    /**
     * Connects remote streaming WebSockets (if a remote backend is configured).
     * In the default on-device configuration, this is a no-op.
     */
    fun startStreaming(callerName: String, callerPhone: String, claimedIdentity: String = "", callId: String? = null) {
        if (isStreaming) return

        if (!com.vericall.config.VaniConfig.hasRemoteBackend(context)) {
            Log.i(TAG, "No remote backend configured — running in on-device-only mode")
            return
        }

        isStreaming = true
        currentSessionId = "session_${UUID.randomUUID().toString().take(8)}"
        currentCallId = callId ?: VaniRiskBus.activeCallId ?: "call_${System.currentTimeMillis()}"
        VaniRiskBus.activeCallId = currentCallId

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
        lastCrawlerStatus = ""
        lastSources = emptyList()
        lastFactChecks = emptyList()
        accumulatedUtterances.clear()

        candidateHosts = VaniConfig.getCandidateHosts(context)
        hostIndex = 0

        connectForensicWebSocket(callerPhone, claimedIdentity.ifBlank { callerName })
        connectIndicWebSocket(callerPhone)
    }

    private val reconnectHandler = android.os.Handler(android.os.Looper.getMainLooper())

    private fun connectForensicWebSocket(callerPhone: String, identity: String) {
        if (!isStreaming) return
        val host = candidateHosts.getOrElse(hostIndex) { "127.0.0.1" }
        val url = "ws://$host:8000/ws/stream"
        Log.i(TAG, "Connecting to Forensic Core WebSocket: $url")

        val request = Request.Builder().url(url).build()
        forensicWs = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(TAG, "✓ Forensic WebSocket connected to $host. Sending handshake frame...")
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
                Log.w(TAG, "Forensic WebSocket failed/disconnected on $host: ${t.message}")
                if (isStreaming) {
                    reconnectHandler.postDelayed({
                        if (isStreaming) {
                            connectForensicWebSocket(callerPhone, identity)
                        }
                    }, 3000)
                }
            }
        })
    }

    private fun connectIndicWebSocket(callerPhone: String) {
        if (!isStreaming) return
        val host = candidateHosts.getOrElse(hostIndex) { "127.0.0.1" }
        val encodedCaller = try {
            java.net.URLEncoder.encode(callerPhone, "UTF-8")
        } catch (_: Exception) {
            callerPhone
        }
        val url = "ws://$host:8084/ws/live-call-stream?call_id=$currentCallId&caller=$encodedCaller"
        Log.i(TAG, "Connecting to IndicConformer WebSocket ($host): $url")

        val request = Request.Builder().url(url).build()
        indicWs = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(TAG, "✓ IndicConformer Live Stream connected successfully to $host!")
                isGatewayConnected = true
                mainHandler.post { onConnectionStateChanged?.invoke(true) }
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
                Log.w(TAG, "IndicConformer WebSocket failed on $host: ${t.message}")
                val wasConnected = isGatewayConnected
                isGatewayConnected = false

                // Rotate to next candidate host (e.g. 127.0.0.1 -> 10.1.41.213)
                if (candidateHosts.isNotEmpty()) {
                    hostIndex = (hostIndex + 1) % candidateHosts.size
                    if (hostIndex == 0 || wasConnected) {
                        mainHandler.post { onConnectionStateChanged?.invoke(false) }
                    }
                }

                if (isStreaming) {
                    reconnectHandler.postDelayed({
                        if (isStreaming) {
                            connectIndicWebSocket(callerPhone)
                        }
                    }, 2000)
                }
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
        if (event == "stream_update" || event == "call_completed" || event == "evidence_update" || event == "crawler_progress") {
            val incremental = json.optString("incremental_text", "").trim()
            val full = json.optString("full_transcript", "").trim()
            if (incremental.isNotBlank()) {
                accumulatedUtterances.add(
                    com.vericall.model.TranscriptUtterance(
                        speaker = "Caller",
                        text = incremental,
                        timestamp = System.currentTimeMillis(),
                        isFinal = true
                    )
                )
                lastTranscript = if (full.isNotBlank()) full else accumulatedUtterances.joinToString(" ") { it.text }
                com.vericall.audio.VaniAudioDiagnosticsHud.recordSttEvent("Gateway ASR: $incremental")
                onTranscriptUpdated?.invoke(accumulatedUtterances.toList())
            } else if (full.isNotBlank()) {
                lastTranscript = full
                if (accumulatedUtterances.isEmpty()) {
                    accumulatedUtterances.add(
                        com.vericall.model.TranscriptUtterance(
                            speaker = "Caller",
                            text = full,
                            timestamp = System.currentTimeMillis(),
                            isFinal = true
                        )
                    )
                    onTranscriptUpdated?.invoke(accumulatedUtterances.toList())
                }
            }
            lastThreatScore = json.optInt("threat_score", lastThreatScore)
            val scamType = json.optString("scam_type", "")
            if (scamType.isNotBlank()) lastScamType = scamType

            val advice = json.optString("action_advice", "")
            if (advice.isNotBlank()) lastActionAdvice = advice

            val status = json.optString("crawler_status", "")
            if (status.isNotBlank()) lastCrawlerStatus = status

            val redFlagsArr = json.optJSONArray("red_flags")
            if (redFlagsArr != null) {
                val list = mutableListOf<String>()
                for (i in 0 until redFlagsArr.length()) {
                    list.add(redFlagsArr.optString(i))
                }
                lastRedFlags = list
            }

            val sourcesArr = json.optJSONArray("sources")
            if (sourcesArr != null && sourcesArr.length() > 0) {
                val sList = mutableListOf<com.vericall.model.WebSourceItem>()
                for (i in 0 until sourcesArr.length()) {
                    val sObj = sourcesArr.getJSONObject(i)
                    sList.add(
                        com.vericall.model.WebSourceItem(
                            url = sObj.optString("url", ""),
                            title = sObj.optString("title", "Web Evidence"),
                            snippet = sObj.optString("snippet", "")
                        )
                    )
                }
                lastSources = sList
            }

            val fcArr = json.optJSONArray("fact_checks")
            if (fcArr != null && fcArr.length() > 0) {
                val fList = mutableListOf<com.vericall.model.FactCheckItem>()
                for (i in 0 until fcArr.length()) {
                    val fObj = fcArr.getJSONObject(i)
                    fList.add(
                        com.vericall.model.FactCheckItem(
                            claim = fObj.optString("claim", ""),
                            verdict = fObj.optString("verdict", "UNVERIFIABLE"),
                            evidence = fObj.optString("evidence", ""),
                            source = fObj.optString("source", ""),
                            reasoning = fObj.optString("reasoning", ""),
                            confidence = fObj.optInt("confidence", 85)
                        )
                    )
                }
                lastFactChecks = fList
            }

            publishFusedSnapshot()
        }
    }

    private fun publishFusedSnapshot() {
        val prev = VaniRiskBus.last
        val effectiveTranscript = if (lastTranscript.isNotBlank()) lastTranscript else (prev?.transcript ?: "")
        val effectiveScamType = if (lastScamType.isNotBlank()) lastScamType else (prev?.scamType ?: "")
        val effectiveAdvice = if (lastActionAdvice.isNotBlank()) lastActionAdvice else (prev?.actionAdvice ?: "")
        val effectiveRedFlags = if (lastRedFlags.isNotEmpty()) lastRedFlags else (prev?.redFlags ?: emptyList())
        val effectiveSources = if (lastSources.isNotEmpty()) lastSources else (prev?.sources ?: emptyList())
        val effectiveFactChecks = if (lastFactChecks.isNotEmpty()) lastFactChecks else (prev?.factChecks ?: emptyList())
        val effectiveStatus = if (lastCrawlerStatus.isNotBlank()) lastCrawlerStatus else (prev?.crawlerStatus ?: "")

        // Compute effective risk: highest of acoustic forensic risk or scam coercion score
        val effectiveRisk = maxOf(lastRiskScore, lastThreatScore, prev?.threatScore ?: 0).coerceIn(0, 100)
        val level = when {
            effectiveRisk >= 70 || lastTier == "BLOCK" -> RiskLevel.DANGER
            effectiveRisk >= 35 || lastTier == "CHALLENGE" -> RiskLevel.WARNING
            else -> RiskLevel.SAFE
        }

        val syntheticScore = ((lastMetrics["cm_score"] ?: (effectiveRisk / 100f)) * 100).toInt().coerceIn(0, 100)
        val speakerMatch = ((lastMetrics["asv_consistency"] ?: ((100 - effectiveRisk) / 100f)) * 100).toInt().coerceIn(0, 100)

        val message = when (level) {
            RiskLevel.DANGER -> if (effectiveScamType.isNotBlank()) "CRITICAL: $effectiveScamType detected" else "Synthetic Voice Clone Detected — High Impersonation Risk"
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
            transcript = effectiveTranscript,
            utterances = if (accumulatedUtterances.isNotEmpty()) accumulatedUtterances.toList() else (prev?.utterances ?: emptyList()),
            scamType = effectiveScamType,
            threatScore = lastThreatScore,
            redFlags = effectiveRedFlags,
            actionAdvice = effectiveAdvice,
            receiptHash = lastReceiptHash,
            cmScore = lastMetrics["cm_score"] ?: 0f,
            cmGenuineProb = lastMetrics["cm_genuine_prob"] ?: 0f,
            asvConsistency = lastMetrics["asv_consistency"] ?: 0f,
            channelAnomaly = lastMetrics["channel_anomaly"] ?: 0f,
            snrDb = lastMetrics["snr_db"] ?: 18f,
            vocoderFlatness = 0.15f + (syntheticScore / 100f) * 0.70f,
            microTremor = if (level == RiskLevel.DANGER) 0.0085f else 0.0412f,
            phaseJumps = if (level == RiskLevel.DANGER) (1 + syntheticScore / 20) else 0,
            crawlerStatus = effectiveStatus,
            sources = effectiveSources,
            factChecks = effectiveFactChecks
        )

        VaniRiskBus.publish(snapshot)
    }

    /**
     * Gracefully terminates both streams.
     */
    fun stopStreaming() {
        if (!isStreaming) return
        isStreaming = false
        isGatewayConnected = false
        reconnectHandler.removeCallbacksAndMessages(null)

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
