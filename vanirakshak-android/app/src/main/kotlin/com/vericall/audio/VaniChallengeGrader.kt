package com.vericall.audio

import android.util.Log
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors

/**
 * Grades a spoken challenge response via POST /v1/sessions/{sid}/challenge.
 * Grading checks digit order + 400–4000ms latency only (PROTOCOL §3) — a pass
 * never un-blocks the interlock; enforcement stays one-way (see GuardService).
 */
object VaniChallengeGrader {

    private val executor = Executors.newSingleThreadExecutor()
    private const val TAG = "VaniChallengeGrader"

    data class Result(
        val passed: Boolean,
        val confidence: Double?,
        val latencyOk: Boolean,
        val raw: String
    )

    fun grade(
        httpBase: String,
        sessionId: String,
        responseText: String,
        latencyMs: Long,
        onResult: (Result?) -> Unit
    ) {
        executor.execute {
            try {
                val url = URL("$httpBase/v1/sessions/$sessionId/challenge")
                val conn = url.openConnection() as HttpURLConnection
                conn.requestMethod = "POST"
                conn.doOutput = true
                conn.setRequestProperty("Content-Type", "application/json")
                conn.connectTimeout = 8000
                conn.readTimeout = 15000
                val body = JSONObject().apply {
                    put("response_text", responseText)
                    put("response_latency_ms", latencyMs)
                }.toString()
                conn.outputStream.use { it.write(body.toByteArray()) }
                val code = conn.responseCode
                val text = if (code in 200..299) {
                    conn.inputStream.bufferedReader().readText()
                } else {
                    conn.errorStream?.bufferedReader()?.readText() ?: ""
                }
                if (code !in 200..299) {
                    Log.w(TAG, "grade HTTP $code: $text")
                    onResult(null)
                    return@execute
                }
                val json = JSONObject(text)
                onResult(
                    Result(
                        passed = json.optBoolean("passed", json.optBoolean("challenge_passed", false)),
                        confidence = json.optDouble("confidence", json.optDouble("challenge_confidence", Double.NaN))
                            .takeIf { !it.isNaN() },
                        latencyOk = json.optBoolean("latency_ok", true),
                        raw = text
                    )
                )
            } catch (e: Exception) {
                Log.w(TAG, "grade failed: ${e.message}")
                onResult(null)
            }
        }
    }
}
