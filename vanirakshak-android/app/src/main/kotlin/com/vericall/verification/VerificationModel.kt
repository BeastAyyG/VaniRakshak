package com.vericall.verification

enum class VerificationVerdict {
    SUPPORTED,
    CONTRADICTED,
    UNVERIFIABLE,
    SCAM_DETECTED
}

data class EvidenceSource(
    val title: String,
    val url: String,
    val domain: String,
    val sourceType: String,
    val snippet: String,
    val date: String? = null
)

data class RedFlag(
    val number: String,
    val category: String,
    val quote: String
)

data class ClaimVerificationResult(
    val id: String,
    val query: String,
    val claim: String,
    val verdict: VerificationVerdict,
    val confidence: Int,
    val explanation: String,
    val sources: List<EvidenceSource>,
    val recommendedAction: String,
    val timestampMs: Long = System.currentTimeMillis()
)

data class ScamCheckResult(
    val id: String,
    val query: String,
    val isScam: Boolean,
    val threatScore: Int,
    val threatLevel: String,
    val scamType: String,
    val suspiciousStatement: String,
    val redFlags: List<RedFlag>,
    val safetyChecklist: List<String>,
    val timestampMs: Long = System.currentTimeMillis()
)
