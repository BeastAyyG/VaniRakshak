package com.vericall.model

/**
 * Direction / type of a phone call.
 */
enum class CallDirection {
    INCOMING,
    OUTGOING,
    MISSED
}

/**
 * Individual transcript utterance within a live or recorded conversation.
 */
data class TranscriptUtterance(
    val speaker: String,       // "You", "Caller", or resolved Contact Name
    val text: String,
    val timestamp: Long = System.currentTimeMillis(),
    val isFinal: Boolean = true
)

/**
 * Represents a complete call history entry with associated forensic & AI metadata.
 */
data class VaniCallRecord(
    val id: String,
    val number: String,
    val callerName: String,
    val direction: CallDirection,
    val timestamp: Long,
    val durationSec: Int = 0,
    val hasTranscript: Boolean = false,
    val transcript: List<TranscriptUtterance> = emptyList(),
    val aiSummary: String = "",
    val keyPoints: List<String> = emptyList(),
    val actionItems: List<String> = emptyList(),
    val riskScore: Int = 0,
    val scamType: String = "",
    val isVoiceVerified: Boolean = false
)
