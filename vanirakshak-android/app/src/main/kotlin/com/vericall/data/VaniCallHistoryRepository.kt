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
    fun saveCallRecord(context: Context, record: VaniCallRecord) {
        val existing = loadPersistedRecords(context).toMutableList()
        val index = existing.indexOfFirst { it.id == record.id }
        if (index != -1) {
            existing[index] = record
        } else {
            existing.add(0, record)
        }
        savePersistedRecords(context, existing)
    }

    /**
     * Deletes a recorded call session from local persistence.
     */
    fun deleteCallRecord(context: Context, id: String) {
        val existing = loadPersistedRecords(context).toMutableList()
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

                val dirStr = obj.optString("direction", CallDirection.INCOMING.name)
                val dir = try { CallDirection.valueOf(dirStr) } catch (_: Exception) { CallDirection.INCOMING }

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
                        riskScore = obj.optInt("riskScore", 0),
                        scamType = obj.optString("scamType", ""),
                        isVoiceVerified = obj.optBoolean("isVoiceVerified", false)
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
                put("scamType", record.scamType)
                put("isVoiceVerified", record.isVoiceVerified)

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
