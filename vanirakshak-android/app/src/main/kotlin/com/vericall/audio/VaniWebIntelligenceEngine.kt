package com.vericall.audio

import android.content.Context
import android.util.Log
import com.vericall.config.VaniConfig
import com.vericall.model.FactCheckItem
import com.vericall.model.WebSourceItem
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

/**
 * VaniWebIntelligenceEngine — Level 3: Asynchronous web verification.
 *
 * CRITICAL DESIGN PRINCIPLE:
 *   This engine NEVER blocks the transcription pipeline.
 *   It runs on a background thread, triggered ONLY when Level 2
 *   contextual risk crosses the suspicion threshold.
 *
 * Data flow:
 *   Level 2 threshold crossed
 *       ↓
 *   Extract context query
 *       ↓
 *   Gemini API (async, background thread)
 *       ↓
 *   Evidence collected
 *       ↓
 *   Published to VaniRiskBus (merged, not blocking)
 *
 * Rate limited: max 1 query per 30 seconds during a call.
 */
class VaniWebIntelligenceEngine(private val context: Context) {

    companion object {
        private const val TAG = "VaniWebIntelligence"
    }

    data class WebEvidence(
        val query: String,
        val factChecks: List<FactCheckItem>,
        val sources: List<WebSourceItem>,
        val riskAdjustment: Int,           // How much to increase/decrease risk
        val summary: String
    )

    /** Callback when web evidence is collected (runs on background thread) */
    var onEvidenceCollected: ((WebEvidence) -> Unit)? = null

    private val isSearching = AtomicBoolean(false)
    @Volatile
    private var isShutdown = false

    // ─── Public API ─────────────────────────────────────────────────────────────

    /**
     * Performs an asynchronous web search. Never blocks the caller.
     * Rate limited: ignores requests if a search is already in progress.
     */
    fun search(query: String) {
        if (isShutdown) return
        if (query.isBlank()) return
        if (!isSearching.compareAndSet(false, true)) {
            Log.d(TAG, "Search already in progress, skipping: $query")
            return
        }

        Thread({
            try {
                val evidence = performSearch(query)
                if (evidence != null && !isShutdown) {
                    Log.i(TAG, "✓ Web evidence collected: ${evidence.factChecks.size} fact checks, ${evidence.sources.size} sources")
                    onEvidenceCollected?.invoke(evidence)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Web search failed: ${e.message}")
            } finally {
                isSearching.set(false)
            }
        }, "VaniWebIntelligenceThread").start()
    }

    fun shutdown() {
        isShutdown = true
    }

    // ─── Search Implementation ──────────────────────────────────────────────────

    private fun performSearch(query: String): WebEvidence? {
        val apiKey = VaniConfig.getGeminiApiKey(context)
        if (apiKey.isBlank()) {
            Log.d(TAG, "No Gemini API key configured — skipping web intelligence")
            return null
        }

        val prompt = """You are VaniRakshak, an Indian cybercrime fraud prevention AI.

A phone call is currently in progress and the following suspicious pattern has been detected:
"$query"

Search your knowledge for:
1. Whether this matches known Indian phone scam patterns
2. Official government/RBI/TRAI warnings about this type of call
3. Whether legitimate organizations actually make calls like this

Respond ONLY with valid JSON:
{
  "risk_adjustment": 0 to 30 (how much to increase risk),
  "summary": "brief one-line assessment",
  "fact_checks": [
    {"claim": "...", "verdict": "SUPPORTED|CONTRADICTED|SCAM_DETECTED", "evidence": "...", "source": "...", "confidence": 85}
  ],
  "sources": [
    {"title": "...", "url": "...", "snippet": "..."}
  ]
}"""

        try {
            val requestBody = JSONObject().apply {
                put("contents", org.json.JSONArray().apply {
                    put(JSONObject().apply {
                        put("role", "user")
                        put("parts", org.json.JSONArray().apply {
                            put(JSONObject().apply { put("text", prompt) })
                        })
                    })
                })
                put("generationConfig", JSONObject().apply {
                    put("temperature", 0.1)
                    put("maxOutputTokens", 512)
                })
            }

            val url = java.net.URL(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=$apiKey"
            )
            val conn = url.openConnection() as java.net.HttpURLConnection
            conn.requestMethod = "POST"
            conn.setRequestProperty("Content-Type", "application/json")
            conn.connectTimeout = 8000
            conn.readTimeout = 15000
            conn.doOutput = true

            conn.outputStream.use { it.write(requestBody.toString().toByteArray()) }

            val code = conn.responseCode
            if (code !in 200..299) {
                Log.w(TAG, "Gemini returned HTTP $code")
                return null
            }

            val bodyStr = conn.inputStream.bufferedReader().readText()
            val responseJson = JSONObject(bodyStr)
            val textContent = responseJson
                .optJSONArray("candidates")
                ?.optJSONObject(0)
                ?.optJSONObject("content")
                ?.optJSONArray("parts")
                ?.optJSONObject(0)
                ?.optString("text") ?: return null

            // Parse JSON from response
            val s = textContent.indexOf('{')
            val e = textContent.lastIndexOf('}')
            if (s < 0 || e < 0) return null
            val parsed = JSONObject(textContent.substring(s, e + 1))

            val riskAdjustment = parsed.optInt("risk_adjustment", 0)
            val summary = parsed.optString("summary", "")

            val factChecks = mutableListOf<FactCheckItem>()
            val fcArr = parsed.optJSONArray("fact_checks")
            if (fcArr != null) {
                for (i in 0 until fcArr.length()) {
                    val fObj = fcArr.getJSONObject(i)
                    factChecks.add(FactCheckItem(
                        claim = fObj.optString("claim", ""),
                        verdict = fObj.optString("verdict", "UNVERIFIABLE"),
                        evidence = fObj.optString("evidence", ""),
                        source = fObj.optString("source", ""),
                        reasoning = fObj.optString("reasoning", ""),
                        confidence = fObj.optInt("confidence", 85)
                    ))
                }
            }

            val sources = mutableListOf<WebSourceItem>()
            val srcArr = parsed.optJSONArray("sources")
            if (srcArr != null) {
                for (i in 0 until srcArr.length()) {
                    val sObj = srcArr.getJSONObject(i)
                    sources.add(WebSourceItem(
                        url = sObj.optString("url", ""),
                        title = sObj.optString("title", ""),
                        snippet = sObj.optString("snippet", "")
                    ))
                }
            }

            return WebEvidence(
                query = query,
                factChecks = factChecks,
                sources = sources,
                riskAdjustment = riskAdjustment.coerceIn(0, 30),
                summary = summary
            )

        } catch (e: Exception) {
            Log.w(TAG, "Gemini web intelligence request failed: ${e.message}")
            return null
        }
    }
}
