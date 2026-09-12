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
        val baseUrl = VaniConfig.getIndicHttpUrl(context)
        val endpoint = "$baseUrl/api/call/finalize"
        Log.d(TAG, "Finalizing call ${record.id} via $endpoint")

        val builder = MultipartBody.Builder().setType(MultipartBody.FORM)
        builder.addFormDataPart("call_id", record.id)
        builder.addFormDataPart("caller", record.callerName)
        builder.addFormDataPart("verify_fact", "true")

        val fullText = record.transcript.joinToString(" ") { it.text }.trim()
        if (fullText.isNotBlank()) {
            builder.addFormDataPart("transcript", fullText)
        }

        record.audioFilePath?.let { path ->
            val file = File(path)
            if (file.exists() && file.length() > 500) {
                val body = file.asRequestBody("audio/wav".toMediaTypeOrNull())
                builder.addFormDataPart("file", file.name, body)
            }
        }

        val request = Request.Builder()
            .url(endpoint)
            .post(builder.build())
            .build()

        try {
            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                Log.w(TAG, "Server returned error: ${response.code}")
                return record
            }

            val bodyString = response.body?.string() ?: return record
            val json = JSONObject(bodyString)

            val riskScore = json.optInt("risk_score", record.riskScore)
            val riskLevel = json.optString("risk_level", record.riskLevel)
            val scamType = json.optString("scam_type", record.scamType)
            val serverTranscript = json.optString("transcript", "").trim()
            val advice = json.optString("action_advice", "")

            // Parse Red Flags
            val redFlags = mutableListOf<String>()
            val rfArr = json.optJSONArray("red_flags")
            if (rfArr != null) {
                for (i in 0 until rfArr.length()) {
                    redFlags.add(rfArr.optString(i))
                }
            }

            // Parse Fact Checks
            val factChecks = mutableListOf<FactCheckItem>()
            val fcArr = json.optJSONArray("fact_checks")
            if (fcArr != null) {
                for (i in 0 until fcArr.length()) {
                    val fObj = fcArr.getJSONObject(i)
                    factChecks.add(
                        FactCheckItem(
                            claim = fObj.optString("claim", ""),
                            verdict = fObj.optString("verdict", "UNVERIFIABLE"),
                            evidence = fObj.optString("evidence", ""),
                            source = fObj.optString("source", ""),
                            reasoning = fObj.optString("reasoning", ""),
                            confidence = fObj.optInt("confidence", 85)
                        )
                    )
                }
            }

            // Parse Sources
            val sources = mutableListOf<WebSourceItem>()
            val srcArr = json.optJSONArray("sources")
            if (srcArr != null) {
                for (i in 0 until srcArr.length()) {
                    val sObj = srcArr.getJSONObject(i)
                    sources.add(
                        WebSourceItem(
                            url = sObj.optString("url", ""),
                            title = sObj.optString("title", ""),
                            snippet = sObj.optString("snippet", "")
                        )
                    )
                }
            }

            // If server returned a fuller transcript and client transcript was empty
            val updatedTranscript = if (record.transcript.isEmpty() && serverTranscript.isNotBlank()) {
                listOf(TranscriptUtterance(speaker = record.callerName, text = serverTranscript))
            } else {
                record.transcript
            }

            val isProtected = riskScore >= 50 || record.isProtected

            val updatedRecord = record.copy(
                riskScore = riskScore,
                riskLevel = riskLevel,
                scamType = scamType,
                transcript = updatedTranscript,
                hasTranscript = updatedTranscript.isNotEmpty(),
                aiSummary = if (advice.isNotBlank()) advice else record.aiSummary,
                redFlags = if (redFlags.isNotEmpty()) redFlags else record.redFlags,
                factChecks = if (factChecks.isNotEmpty()) factChecks else record.factChecks,
                sources = if (sources.isNotEmpty()) sources else record.sources,
                isProtected = isProtected
            )

            // Save updated record (triggers retention and high-risk preservation)
            VaniCallHistoryRepository.saveCallRecord(context, updatedRecord)
            Log.d(TAG, "✓ Call ${record.id} finalized successfully with risk=$riskScore%, factChecks=${factChecks.size}")
            return updatedRecord

        } catch (e: Exception) {
            Log.w(TAG, "Failed to reach finalizer endpoint: ${e.message}")
            return record
        }
    }
}
