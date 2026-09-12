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
 * Single fact-check claim and web evidence finding.
 */
data class FactCheckItem(
    val claim: String,
    val verdict: String,       // "SUPPORTED", "CONTRADICTED", "FRAUDULENT_CLAIM", "UNVERIFIABLE"
    val evidence: String,
    val source: String,
    val reasoning: String = "",
    val confidence: Int = 85
)

/**
 * Grounded web source information scraped by Crawl4AI.
 */
data class WebSourceItem(
    val url: String,
    val title: String,
    val snippet: String = ""
)

/**
 * Represents a complete call history entry with associated forensic, recording & AI metadata.
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
    val riskLevel: String = "Very Low",
    val scamType: String = "",
    val isVoiceVerified: Boolean = false,
    val audioFilePath: String? = null,
    val transcriptFilePath: String? = null,
    val isProtected: Boolean = false,
    val confidence: Int = 0,
    val factChecks: List<FactCheckItem> = emptyList(),
    val redFlags: List<String> = emptyList(),
    val sources: List<WebSourceItem> = emptyList()
)
