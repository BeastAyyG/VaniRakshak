package com.vericall.data

import android.content.Context
import android.util.Log
import com.vericall.config.VaniConfig
import com.vericall.model.FactCheckItem
import com.vericall.model.TranscriptUtterance
import com.vericall.model.VaniCallRecord
import com.vericall.model.WebSourceItem
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Executes the complete post-call forensic pipeline against VaniRakshak IndicConformer server.
 * Uploads the recorded WAV audio (or transcribed text), extracts factual claims,
 * runs Crawl4AI web search & verification, and saves the final forensic report.
 */
object VaniCallFinalizer {

    private const val TAG = "VaniCallFinalizer"

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    const val ACTION_CALL_FINALIZED = "com.vericall.action.CALL_FINALIZED"
    const val EXTRA_FINALIZED_CALL_ID = "extra_finalized_call_id"

    fun finalizeCallAsync(
        context: Context,
        record: VaniCallRecord,
        onComplete: ((VaniCallRecord) -> Unit)? = null
    ) {
        Thread({
            val updated = finalizeCallSync(context, record)
            try {
                val intent = android.content.Intent(ACTION_CALL_FINALIZED).apply {
                    putExtra(EXTRA_FINALIZED_CALL_ID, updated.id)
                    setPackage(context.packageName)
                }
                context.sendBroadcast(intent)
            } catch (_: Exception) {}
            onComplete?.invoke(updated)
        }, "VaniCallFinalizerThread").start()
    }

    private fun finalizeCallSync(context: Context, record: VaniCallRecord): VaniCallRecord {
        val fullText = record.transcript.joinToString(" ") { it.text }.trim()
        Log.d(TAG, "Finalizing call ${record.id} via Gemini (transcript: ${fullText.length} chars)")

        // Try Gemini-powered post-call analysis
        val apiKey = com.vericall.config.VaniConfig.getGeminiApiKey(context)
        if (apiKey.isNotBlank() && fullText.isNotBlank()) {
            try {
                return finalizeWithGemini(context, record, fullText, apiKey)
            } catch (e: Exception) {
                Log.w(TAG, "Gemini finalization unavailable: ${e.message} — using local analysis")
            }
        }

        // Fallback: try remote backend if configured
        if (com.vericall.config.VaniConfig.hasRemoteBackend(context)) {
            try {
                return finalizeWithRemoteBackend(context, record, fullText)
            } catch (e: Exception) {
                Log.w(TAG, "Remote backend finalization unavailable: ${e.message} — using local analysis")
            }
        }

        // Final fallback: local analysis from what we already know
        return deriveLocalFinalRecord(record, fullText)
    }

    /**
     * Post-call analysis using Gemini API on the transcript text.
     * Runs entirely over HTTPS — no PC, no USB required.
     */
    private fun finalizeWithGemini(
        context: Context,
        record: VaniCallRecord,
        transcript: String,
        apiKey: String
    ): VaniCallRecord {
        // Redact credentials before sending
        val redactedTranscript = com.vericall.audio.VaniScamDetector.redact(transcript)
        val callerId = record.callerName.take(50)

        val prompt = """You are VaniRakshak, an Indian cybercrime fraud prevention AI.

Analyze this phone call transcript and provide a post-call forensic report.
Caller name/number: $callerId
Transcript (user's own voice): "$redactedTranscript"

Respond ONLY with valid JSON:
{
  "risk_score": 0-100,
  "risk_level": "SAFE|LOW_RISK|MODERATE_RISK|HIGH_RISK|CRITICAL",
  "scam_type": "string or empty",
  "action_advice": "brief protective advice",
  "red_flags": ["flag1", "flag2"],
  "fact_checks": [
    {"claim": "...", "verdict": "SUPPORTED|CONTRADICTED|UNVERIFIABLE|SCAM_DETECTED", "evidence": "...", "confidence": 85}
  ],
  "sources": [{"title": "...", "url": "...", "snippet": "..."}]
}"""

        val requestBody = org.json.JSONObject().apply {
            put("contents", org.json.JSONArray().apply {
                put(org.json.JSONObject().apply {
                    put("role", "user")
                    put("parts", org.json.JSONArray().apply {
                        put(org.json.JSONObject().apply { put("text", prompt) })
                    })
                })
            })
            put("generationConfig", org.json.JSONObject().apply {
                put("temperature", 0.1)
                put("maxOutputTokens", 1024)
            })
        }

        val url = java.net.URL("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=$apiKey")
        val conn = url.openConnection() as java.net.HttpURLConnection
        conn.requestMethod = "POST"
        conn.setRequestProperty("Content-Type", "application/json")
        conn.connectTimeout = 10000
        conn.readTimeout = 30000
        conn.doOutput = true

        conn.outputStream.use { it.write(requestBody.toString().toByteArray()) }

        val code = conn.responseCode
        if (code !in 200..299) {
            Log.w(TAG, "Gemini finalization returned HTTP $code")
            return deriveLocalFinalRecord(record, transcript)
        }

        val bodyStr = conn.inputStream.bufferedReader().readText()
        val responseJson = org.json.JSONObject(bodyStr)
        val textContent = responseJson
            .optJSONArray("candidates")
            ?.optJSONObject(0)
            ?.optJSONObject("content")
            ?.optJSONArray("parts")
            ?.optJSONObject(0)
            ?.optString("text") ?: return deriveLocalFinalRecord(record, transcript)

        val parsed = try {
            val s = textContent.indexOf('{')
            val e = textContent.lastIndexOf('}')
            if (s < 0 || e < 0) return deriveLocalFinalRecord(record, transcript)
            org.json.JSONObject(textContent.substring(s, e + 1))
        } catch (ex: Exception) {
            Log.w(TAG, "Could not parse Gemini finalization response: ${ex.message}")
            return deriveLocalFinalRecord(record, transcript)
        }

        val riskScore = parsed.optInt("risk_score", record.riskScore)
        val riskLevel = parsed.optString("risk_level", record.riskLevel)
        val scamType = parsed.optString("scam_type", record.scamType)
        val advice = parsed.optString("action_advice", "")

        val redFlags = mutableListOf<String>()
        val rfArr = parsed.optJSONArray("red_flags")
        if (rfArr != null) for (i in 0 until rfArr.length()) redFlags.add(rfArr.optString(i))

        val factChecks = mutableListOf<com.vericall.model.FactCheckItem>()
        val fcArr = parsed.optJSONArray("fact_checks")
        if (fcArr != null) for (i in 0 until fcArr.length()) {
            val fObj = fcArr.getJSONObject(i)
            factChecks.add(
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

        val sources = mutableListOf<com.vericall.model.WebSourceItem>()
        val srcArr = parsed.optJSONArray("sources")
        if (srcArr != null) for (i in 0 until srcArr.length()) {
            val sObj = srcArr.getJSONObject(i)
            sources.add(
                com.vericall.model.WebSourceItem(
                    url = sObj.optString("url", ""),
                    title = sObj.optString("title", ""),
                    snippet = sObj.optString("snippet", "")
                )
            )
        }

        val isProtected = riskScore >= 50 || record.isProtected
        val updatedRecord = record.copy(
            riskScore = riskScore,
            riskLevel = riskLevel,
            scamType = scamType,
            hasTranscript = record.transcript.isNotEmpty(),
            aiSummary = if (advice.isNotBlank()) advice else record.aiSummary,
            redFlags = if (redFlags.isNotEmpty()) redFlags else record.redFlags,
            factChecks = if (factChecks.isNotEmpty()) factChecks else record.factChecks,
            sources = if (sources.isNotEmpty()) sources else record.sources,
            isProtected = isProtected
        )

        VaniCallHistoryRepository.saveCallRecord(context, updatedRecord)
        Log.d(TAG, "✓ Call ${record.id} finalized via Gemini: risk=$riskScore%, factChecks=${factChecks.size}")
        return updatedRecord
    }

    /**
     * Remote backend finalization (only used if user has configured a server).
     */
    private fun finalizeWithRemoteBackend(context: Context, record: VaniCallRecord, fullText: String): VaniCallRecord {
        val baseUrl = VaniConfig.getIndicHttpUrl(context)
        val endpoint = "$baseUrl/api/call/finalize"

        val builder = okhttp3.MultipartBody.Builder().setType(okhttp3.MultipartBody.FORM)
        builder.addFormDataPart("call_id", record.id)
        builder.addFormDataPart("caller", record.callerName)
        builder.addFormDataPart("verify_fact", "true")
        if (fullText.isNotBlank()) builder.addFormDataPart("transcript", fullText)

        record.audioFilePath?.let { path ->
            val file = java.io.File(path)
            if (file.exists() && file.length() > 500) {
                builder.addFormDataPart("file", file.name, file.asRequestBody("audio/wav".toMediaTypeOrNull()))
            }
        }

        val request = Request.Builder().url(endpoint).post(builder.build()).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) return deriveLocalFinalRecord(record, fullText)

        val bodyString = response.body?.string() ?: return deriveLocalFinalRecord(record, fullText)
        val json = JSONObject(bodyString)

        val riskScore = json.optInt("risk_score", record.riskScore)
        val riskLevel = json.optString("risk_level", record.riskLevel)
        val scamType = json.optString("scam_type", record.scamType)
        val advice = json.optString("action_advice", "")

        val redFlags = mutableListOf<String>()
        val rfArr = json.optJSONArray("red_flags")
        if (rfArr != null) for (i in 0 until rfArr.length()) redFlags.add(rfArr.optString(i))

        val factChecks = mutableListOf<com.vericall.model.FactCheckItem>()
        val fcArr = json.optJSONArray("fact_checks")
        if (fcArr != null) for (i in 0 until fcArr.length()) {
            val fObj = fcArr.getJSONObject(i)
            factChecks.add(com.vericall.model.FactCheckItem(
                claim = fObj.optString("claim", ""),
                verdict = fObj.optString("verdict", "UNVERIFIABLE"),
                evidence = fObj.optString("evidence", ""),
                source = fObj.optString("source", ""),
                reasoning = fObj.optString("reasoning", ""),
                confidence = fObj.optInt("confidence", 85)
            ))
        }

        val sources = mutableListOf<com.vericall.model.WebSourceItem>()
        val srcArr = json.optJSONArray("sources")
        if (srcArr != null) for (i in 0 until srcArr.length()) {
            val sObj = srcArr.getJSONObject(i)
            sources.add(com.vericall.model.WebSourceItem(
                url = sObj.optString("url", ""),
                title = sObj.optString("title", ""),
                snippet = sObj.optString("snippet", "")
            ))
        }

        val isProtected = riskScore >= 50 || record.isProtected
        val updatedRecord = record.copy(
            riskScore = riskScore, riskLevel = riskLevel, scamType = scamType,
            hasTranscript = record.transcript.isNotEmpty(),
            aiSummary = if (advice.isNotBlank()) advice else record.aiSummary,
            redFlags = if (redFlags.isNotEmpty()) redFlags else record.redFlags,
            factChecks = if (factChecks.isNotEmpty()) factChecks else record.factChecks,
            sources = if (sources.isNotEmpty()) sources else record.sources,
            isProtected = isProtected
        )
        VaniCallHistoryRepository.saveCallRecord(context, updatedRecord)
        Log.d(TAG, "✓ Call ${record.id} finalized via remote backend: risk=$riskScore%")
        return updatedRecord
    }

    /**
     * Local-only finalization using already-computed transcript + VaniScamDetector.
     * Used when both Gemini and remote backend are unavailable.
     */
    private fun deriveLocalFinalRecord(record: VaniCallRecord, fullText: String): VaniCallRecord {
        if (fullText.isBlank()) return record

        val detection = com.vericall.audio.VaniScamDetector.analyze(fullText)
        val riskScore = maxOf(record.riskScore, detection.threatScore)
        val isProtected = riskScore >= 50 || record.isProtected
        val riskLevel = when {
            riskScore >= 70 -> "HIGH_RISK"
            riskScore >= 35 -> "MODERATE_RISK"
            else -> "LOW_RISK"
        }
        val scamType = if (detection.matchedCategories.isNotEmpty())
            detection.matchedCategories.first().label
        else record.scamType

        val updatedRecord = record.copy(
            riskScore = riskScore,
            riskLevel = riskLevel,
            scamType = scamType,
            hasTranscript = record.transcript.isNotEmpty(),
            isProtected = isProtected,
            redFlags = if (detection.redFlags.isNotEmpty()) detection.redFlags else record.redFlags
        )
        Log.d(TAG, "✓ Call ${record.id} finalized locally: risk=$riskScore%")
        return updatedRecord
    }
}
