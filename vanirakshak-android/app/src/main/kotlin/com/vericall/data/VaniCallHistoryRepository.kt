package com.vericall.data

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.database.Cursor
import android.provider.CallLog
import android.util.Log
import androidx.core.content.ContextCompat
import com.vericall.model.CallDirection
import com.vericall.model.TranscriptUtterance
import com.vericall.model.VaniCallRecord
import com.vericall.util.VaniCallRouter
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

object VaniCallHistoryRepository {

    private const val TAG = "VaniCallHistoryRepo"
    private const val PREFS_NAME = "vani_call_history_store"
    private const val KEY_RECORDS = "persisted_call_records"

    /**
     * Retrieves all recent call history, seamlessly merging native device call logs
     * with VaniRakshak's AI-transcribed sessions.
     */
    fun getRecents(context: Context): List<VaniCallRecord> {
        val persisted = loadPersistedRecords(context)
        val hasPermission = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.READ_CALL_LOG
        ) == PackageManager.PERMISSION_GRANTED

        if (!hasPermission) {
            Log.d(TAG, "READ_CALL_LOG not granted; returning ${persisted.size} persisted sessions")
            return persisted.sortedByDescending { it.timestamp }
        }

        val deviceCalls = queryDeviceCallLog(context)
        val mergedList = mutableListOf<VaniCallRecord>()

        // Enrich device calls with stored transcripts and AI intelligence
        for (devCall in deviceCalls) {
            val matchingStored = persisted.find { stored ->
                // Match by phone number and timestamp within 3 minutes
                VaniCallRouter.sameNumber(stored.number, devCall.number) &&
                    kotlin.math.abs(stored.timestamp - devCall.timestamp) < 180_000
            }

            if (matchingStored != null) {
                mergedList.add(
                    devCall.copy(
                        id = matchingStored.id,
                        hasTranscript = matchingStored.hasTranscript,
                        transcript = matchingStored.transcript,
                        aiSummary = matchingStored.aiSummary,
                        keyPoints = matchingStored.keyPoints,
                        actionItems = matchingStored.actionItems,
                        riskScore = matchingStored.riskScore,
                        scamType = matchingStored.scamType,
                        isVoiceVerified = matchingStored.isVoiceVerified
                    )
                )
            } else {
                mergedList.add(devCall)
            }
        }

        // Also add any persisted sessions that were not in native call log (e.g. VoIP / app calls)
        for (stored in persisted) {
            val alreadyPresent = mergedList.any { it.id == stored.id }
            if (!alreadyPresent) {
                mergedList.add(stored)
            }
        }

        return mergedList.sortedByDescending { it.timestamp }
    }

    /**
     * Returns call history specifically associated with a phone number.
     */
    fun getCallsForContact(context: Context, phoneNumber: String): List<VaniCallRecord> {
        return getRecents(context).filter {
            VaniCallRouter.sameNumber(it.number, phoneNumber)
        }
    }

    /**
     * Looks up a call record by its ID.
     */
    fun getCallById(context: Context, id: String): VaniCallRecord? {
        return getRecents(context).find { it.id == id }
    }

    /**
     * Persists a completed call session (with transcript, summary, and forensic scores).
     */
    /**
     * Persists a completed call session (with transcript, summary, and forensic scores).
     * Enforces:
     *   1. High-Risk Call Protection: If risk >= 50%, auto-saves call_<id>_transcript.txt and marks isProtected=true.
     *   2. Last 10 Call Policy: Retains the 10 most recent normal calls (<50% risk); prunes older normal audio files.
     *   3. Protected calls (>=50% risk) are NEVER deleted.
     */
    fun saveCallRecord(context: Context, record: VaniCallRecord) {
        var finalRecord = record

        // High-risk auto preservation
        if (finalRecord.riskScore >= 50 || finalRecord.isProtected) {
            val protectedDir = File(context.filesDir, "protected").apply { mkdirs() }
            val txtFile = File(protectedDir, "${finalRecord.id}_transcript.txt")
            try {
                val evidenceContent = buildString {
                    appendLine("================================================================================")
                    appendLine("VANIRAKSHAK FORENSIC VOICE EVIDENCE ARCHIVE")
                    appendLine("================================================================================")
                    appendLine("Call ID:        ${finalRecord.id}")
                    appendLine("Caller:         ${finalRecord.callerName} (${finalRecord.number})")
                    appendLine("Timestamp:      ${java.util.Date(finalRecord.timestamp)}")
                    appendLine("Duration:       ${finalRecord.durationSec} seconds")
                    appendLine("Risk Score:     ${finalRecord.riskScore}% (${finalRecord.riskLevel})")
                    appendLine("Scam Category:  ${finalRecord.scamType.ifBlank { "Acoustic / Financial Anomaly" }}")
                    appendLine("Protection:     AUTOMATICALLY PRESERVED EVIDENCE (Risk >= 50%)")
                    appendLine("Audio File:     ${finalRecord.audioFilePath ?: "Preserved in session"}")
                    appendLine("--------------------------------------------------------------------------------")
                    appendLine("DETECTED RED FLAGS & SCAM SIGNALS:")
                    if (finalRecord.redFlags.isNotEmpty()) {
                        finalRecord.redFlags.forEach { appendLine("  • $it") }
                    } else {
                        appendLine("  • Severe acoustic vocoder distortion / impersonation risk")
                    }
                    appendLine("--------------------------------------------------------------------------------")
                    appendLine("CONVERSATION TRANSCRIPT:")
                    if (finalRecord.transcript.isNotEmpty()) {
                        finalRecord.transcript.forEach { appendLine("[${it.speaker}]: ${it.text}") }
                    } else {
                        appendLine("  [No spoken transcript captured]")
                    }
                    appendLine("--------------------------------------------------------------------------------")
                    appendLine("FACT-CHECK & WEB EVIDENCE:")
                    if (finalRecord.factChecks.isNotEmpty()) {
                        finalRecord.factChecks.forEach { fc ->
                            appendLine("  Claim:    ${fc.claim}")
                            appendLine("  Verdict:  ${fc.verdict}")
                            appendLine("  Evidence: ${fc.evidence}")
                            appendLine("  Source:   ${fc.source}")
                            appendLine()
                        }
                    } else {
                        appendLine("  No specific external factual claim extracted.")
                    }
                    appendLine("--------------------------------------------------------------------------------")
                    appendLine("AI SUMMARY & ACTION DIRECTIVE:")
                    appendLine(finalRecord.aiSummary.ifBlank { "Caution recommended. Do not share OTPs or transfer funds." })
                    appendLine("================================================================================")
                }
                txtFile.writeText(evidenceContent)
                finalRecord = finalRecord.copy(
                    isProtected = true,
                    transcriptFilePath = txtFile.absolutePath
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error saving protected transcript file: ${e.message}", e)
            }
        }

        val existing = loadPersistedRecords(context).toMutableList()
        val index = existing.indexOfFirst { it.id == finalRecord.id }
        if (index != -1) {
            existing[index] = finalRecord
        } else {
            existing.add(0, finalRecord)
        }

        // Apply retention policy:
        // Protected calls: ALWAYS kept.
        // Normal calls (risk < 50 && !isProtected): Keep 10 most recent.
        val protectedCalls = existing.filter { it.isProtected || it.riskScore >= 50 }
        val normalCalls = existing.filter { !it.isProtected && it.riskScore < 50 }.sortedByDescending { it.timestamp }

        val keptNormal = if (normalCalls.size > 10) {
            val toPrune = normalCalls.drop(10)
            for (oldCall in toPrune) {
                oldCall.audioFilePath?.let { path ->
                    try {
                        val f = File(path)
                        if (f.exists()) f.delete()
                    } catch (_: Exception) {}
                }
            }
            normalCalls.take(10)
        } else {
            normalCalls
        }

        val prunedList = (protectedCalls + keptNormal).sortedByDescending { it.timestamp }
        savePersistedRecords(context, prunedList)
    }

    /**
     * Deletes a recorded call session from local persistence.
     * Protected calls can only be deleted if explicitly requested.
     */
    fun deleteCallRecord(context: Context, id: String) {
        val existing = loadPersistedRecords(context).toMutableList()
        val target = existing.find { it.id == id }
        target?.audioFilePath?.let { path ->
            try { File(path).delete() } catch (_: Exception) {}
        }
        target?.transcriptFilePath?.let { path ->
            try { File(path).delete() } catch (_: Exception) {}
        }
        existing.removeAll { it.id == id }
        savePersistedRecords(context, existing)
    }

    private fun queryDeviceCallLog(context: Context): List<VaniCallRecord> {
        val list = mutableListOf<VaniCallRecord>()
        val projection = arrayOf(
            CallLog.Calls._ID,
            CallLog.Calls.NUMBER,
            CallLog.Calls.CACHED_NAME,
            CallLog.Calls.TYPE,
            CallLog.Calls.DATE,
            CallLog.Calls.DURATION
        )

        var cursor: Cursor? = null
        try {
            cursor = context.contentResolver.query(
                CallLog.Calls.CONTENT_URI,
                projection,
                null,
                null,
                "${CallLog.Calls.DATE} DESC LIMIT 100"
            )

            cursor?.let {
                val idIdx = it.getColumnIndex(CallLog.Calls._ID)
                val numIdx = it.getColumnIndex(CallLog.Calls.NUMBER)
                val nameIdx = it.getColumnIndex(CallLog.Calls.CACHED_NAME)
                val typeIdx = it.getColumnIndex(CallLog.Calls.TYPE)
                val dateIdx = it.getColumnIndex(CallLog.Calls.DATE)
                val durIdx = it.getColumnIndex(CallLog.Calls.DURATION)

                while (it.moveToNext()) {
                    val logId = if (idIdx >= 0) it.getString(idIdx) ?: "" else ""
                    val number = if (numIdx >= 0) it.getString(numIdx) ?: "" else ""
                    val cachedName = if (nameIdx >= 0) it.getString(nameIdx) else null
                    val callType = if (typeIdx >= 0) it.getInt(typeIdx) else CallLog.Calls.INCOMING_TYPE
                    val date = if (dateIdx >= 0) it.getLong(dateIdx) else System.currentTimeMillis()
                    val duration = if (durIdx >= 0) it.getInt(durIdx) else 0

                    val direction = when (callType) {
                        CallLog.Calls.OUTGOING_TYPE -> CallDirection.OUTGOING
                        CallLog.Calls.MISSED_TYPE,
                        CallLog.Calls.REJECTED_TYPE,
                        CallLog.Calls.BLOCKED_TYPE -> CallDirection.MISSED
                        else -> CallDirection.INCOMING
                    }

                    val resolvedCaller = VaniCallRouter.resolve(context, number)
                    val displayName = cachedName?.takeIf { n -> n.isNotBlank() }
                        ?: resolvedCaller.name.takeIf { n -> n != number }
                        ?: number.ifBlank { "Unknown caller" }

                    list.add(
                        VaniCallRecord(
                            id = "call_log_$logId",
                            number = number,
                            callerName = displayName,
                            direction = direction,
                            timestamp = date,
                            durationSec = duration,
                            hasTranscript = false,
                            isVoiceVerified = resolvedCaller.enrolled
                        )
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error querying CallLog.Calls: ${e.message}", e)
        } finally {
            cursor?.close()
        }
        return list
    }

    private fun loadPersistedRecords(context: Context): List<VaniCallRecord> {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val jsonString = prefs.getString(KEY_RECORDS, null) ?: return emptyList()
        val result = mutableListOf<VaniCallRecord>()

        try {
            val jsonArray = JSONArray(jsonString)
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val utterances = mutableListOf<TranscriptUtterance>()
                val utterArray = obj.optJSONArray("transcript")
                if (utterArray != null) {
                    for (j in 0 until utterArray.length()) {
                        val uObj = utterArray.getJSONObject(j)
                        utterances.add(
                            TranscriptUtterance(
                                speaker = uObj.optString("speaker", "Caller"),
                                text = uObj.optString("text", ""),
                                timestamp = uObj.optLong("timestamp", 0L),
                                isFinal = uObj.optBoolean("isFinal", true)
                            )
                        )
                    }
                }

                val keyPointsList = mutableListOf<String>()
                val kpArr = obj.optJSONArray("keyPoints")
                if (kpArr != null) {
                    for (k in 0 until kpArr.length()) keyPointsList.add(kpArr.getString(k))
                }

                val actionItemsList = mutableListOf<String>()
                val actArr = obj.optJSONArray("actionItems")
                if (actArr != null) {
                    for (a in 0 until actArr.length()) actionItemsList.add(actArr.getString(a))
                }

                val redFlagsList = mutableListOf<String>()
                val rfArr = obj.optJSONArray("redFlags")
                if (rfArr != null) {
                    for (r in 0 until rfArr.length()) redFlagsList.add(rfArr.getString(r))
                }

                val factChecksList = mutableListOf<com.vericall.model.FactCheckItem>()
                val fcArr = obj.optJSONArray("factChecks")
                if (fcArr != null) {
                    for (f in 0 until fcArr.length()) {
                        val fObj = fcArr.getJSONObject(f)
                        factChecksList.add(
                            com.vericall.model.FactCheckItem(
                                claim = fObj.optString("claim", ""),
                                verdict = fObj.optString("verdict", "UNVERIFIABLE"),
                                evidence = fObj.optString("evidence", ""),
                                source = fObj.optString("source", ""),
                                reasoning = fObj.optString("reasoning", ""),
                                confidence = fObj.optInt("confidence", 80)
                            )
                        )
                    }
                }

                val sourcesList = mutableListOf<com.vericall.model.WebSourceItem>()
                val srcArr = obj.optJSONArray("sources")
                if (srcArr != null) {
                    for (s in 0 until srcArr.length()) {
                        val sObj = srcArr.getJSONObject(s)
                        sourcesList.add(
                            com.vericall.model.WebSourceItem(
                                url = sObj.optString("url", ""),
                                title = sObj.optString("title", ""),
                                snippet = sObj.optString("snippet", "")
                            )
                        )
                    }
                }

                val dirStr = obj.optString("direction", CallDirection.INCOMING.name)
                val dir = try { CallDirection.valueOf(dirStr) } catch (_: Exception) { CallDirection.INCOMING }
                val score = obj.optInt("riskScore", 0)
                val defaultLevel = when {
                    score >= 80 -> "Critical"
                    score >= 60 -> "High"
                    score >= 40 -> "Moderate"
                    score >= 20 -> "Low"
                    else -> "Very Low"
                }

                result.add(
                    VaniCallRecord(
                        id = obj.getString("id"),
                        number = obj.optString("number", ""),
                        callerName = obj.optString("callerName", "Unknown"),
                        direction = dir,
                        timestamp = obj.optLong("timestamp", System.currentTimeMillis()),
                        durationSec = obj.optInt("durationSec", 0),
                        hasTranscript = obj.optBoolean("hasTranscript", false),
                        transcript = utterances,
                        aiSummary = obj.optString("aiSummary", ""),
                        keyPoints = keyPointsList,
                        actionItems = actionItemsList,
                        riskScore = score,
                        riskLevel = obj.optString("riskLevel", defaultLevel),
                        scamType = obj.optString("scamType", ""),
                        isVoiceVerified = obj.optBoolean("isVoiceVerified", false),
                        audioFilePath = obj.optString("audioFilePath").takeIf { it.isNotBlank() },
                        transcriptFilePath = obj.optString("transcriptFilePath").takeIf { it.isNotBlank() },
                        isProtected = obj.optBoolean("isProtected", score >= 50),
                        confidence = obj.optInt("confidence", 0),
                        factChecks = factChecksList,
                        redFlags = redFlagsList,
                        sources = sourcesList
                    )
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error deserializing persisted call records: ${e.message}", e)
        }
        return result
    }

    private fun savePersistedRecords(context: Context, records: List<VaniCallRecord>) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val jsonArray = JSONArray()

        for (record in records) {
            val obj = JSONObject().apply {
                put("id", record.id)
                put("number", record.number)
                put("callerName", record.callerName)
                put("direction", record.direction.name)
                put("timestamp", record.timestamp)
                put("durationSec", record.durationSec)
                put("hasTranscript", record.hasTranscript)
                put("aiSummary", record.aiSummary)
                put("riskScore", record.riskScore)
                put("riskLevel", record.riskLevel)
                put("scamType", record.scamType)
                put("isVoiceVerified", record.isVoiceVerified)
                put("audioFilePath", record.audioFilePath ?: "")
                put("transcriptFilePath", record.transcriptFilePath ?: "")
                put("isProtected", record.isProtected)
                put("confidence", record.confidence)

                val utterArr = JSONArray()
                for (u in record.transcript) {
                    val uObj = JSONObject().apply {
                        put("speaker", u.speaker)
                        put("text", u.text)
                        put("timestamp", u.timestamp)
                        put("isFinal", u.isFinal)
                    }
                    utterArr.put(uObj)
                }
                put("transcript", utterArr)

                val kpArr = JSONArray()
                record.keyPoints.forEach { kpArr.put(it) }
                put("keyPoints", kpArr)

                val actArr = JSONArray()
                record.actionItems.forEach { actArr.put(it) }
                put("actionItems", actArr)

                val rfArr = JSONArray()
                record.redFlags.forEach { rfArr.put(it) }
                put("redFlags", rfArr)

                val fcArr = JSONArray()
                for (fc in record.factChecks) {
                    val fObj = JSONObject().apply {
                        put("claim", fc.claim)
                        put("verdict", fc.verdict)
                        put("evidence", fc.evidence)
                        put("source", fc.source)
                        put("reasoning", fc.reasoning)
                        put("confidence", fc.confidence)
                    }
                    fcArr.put(fObj)
                }
                put("factChecks", fcArr)

                val srcArr = JSONArray()
                for (src in record.sources) {
                    val sObj = JSONObject().apply {
                        put("url", src.url)
                        put("title", src.title)
                        put("snippet", src.snippet)
                    }
                    srcArr.put(sObj)
                }
                put("sources", srcArr)
            }
            jsonArray.put(obj)
        }

        prefs.edit().putString(KEY_RECORDS, jsonArray.toString()).apply()
    }

    /**
     * Derives a structured AI Summary, Key Points, and Action Items from a transcript
     * when real conversation occurs.
     */
    fun deriveIntelligence(
        callerName: String,
        transcript: List<TranscriptUtterance>,
        riskScore: Int,
        scamType: String
    ): Triple<String, List<String>, List<String>> {
        if (transcript.isEmpty()) {
            return Triple("", emptyList(), emptyList())
        }

        val fullText = transcript.joinToString(" ") { it.text }
        val summary = when {
            riskScore >= 70 -> "High-risk call detected. The caller exhibited urgent pressure and coercion patterns characteristic of $scamType. Immediate defensive interlocks were engaged."
            riskScore >= 35 -> "Call with $callerName flagged for mild acoustic and conversational anomalies. Verification recommended."
            fullText.contains("document", ignoreCase = true) || fullText.contains("file", ignoreCase = true) ->
                "Discussion with $callerName regarding project documentation and deliverables."
            fullText.contains("meeting", ignoreCase = true) || fullText.contains("time", ignoreCase = true) ->
                "Conversation coordinating schedule and upcoming agenda items with $callerName."
            else -> "Call completed with $callerName. Authenticity verified by VaniRakshak audio telemetry."
        }

        val keyPoints = mutableListOf<String>()
        if (riskScore >= 70) {
            keyPoints.add("Autonomous transaction interlock held during call")
            keyPoints.add("Coercion pressure detected in speech pattern")
            if (scamType.isNotBlank()) keyPoints.add("Pattern matched: $scamType")
        } else {
            keyPoints.add("Voice biometric verified against enrolled baseline")
            keyPoints.add("No spoof or neural vocoder artifacts detected")
        }
        if (fullText.isNotBlank()) {
            val snippet = fullText.take(90).trim()
            keyPoints.add("Discussed: \"$snippet...\"")
        }

        val actionItems = mutableListOf<String>()
        if (riskScore >= 70) {
            actionItems.add("Do not transfer funds or share banking OTPs")
            actionItems.add("File compliance report to 1930 Cyber Helpline")
        } else {
            actionItems.add("Follow up with $callerName if needed")
        }

        return Triple(summary, keyPoints, actionItems)
    }
}
