package com.vericall.audio

import android.util.Log
import com.vericall.model.RiskLevel
import com.vericall.model.TranscriptUtterance
import com.vericall.util.VaniRiskBus

/**
 * VaniContextualRiskEngine — Level 2: Multi-sentence conversational pattern analysis.
 *
 * Unlike Level 1 (instant keyword detection on individual utterances),
 * this engine accumulates transcript segments and detects combinatorial
 * threat patterns across the conversation context.
 *
 * Example:
 *   Segment 1: "I am calling from RBI"             → IMPERSONATION
 *   Segment 2: "Your account will be blocked"       → ACCOUNT_THREAT
 *   Segment 3: "Tell me the OTP immediately"        → CREDENTIAL_REQUEST + URGENCY
 *   Combined:  IMPERSONATION + THREAT + CREDENTIAL  → CRITICAL RISK (95)
 *
 * This engine runs on every finalized utterance but does NOT block transcription.
 * When the suspicion threshold is crossed, it triggers Level 3 (web verification).
 */
class VaniContextualRiskEngine {

    companion object {
        private const val TAG = "VaniContextualRisk"

        /** Maximum number of utterances to keep in the sliding window */
        private const val CONTEXT_WINDOW_SIZE = 15

        /** Suspicion threshold to trigger Level 3 web search */
        private const val WEB_SEARCH_THRESHOLD = 55

        /** Minimum seconds between web search triggers */
        private const val WEB_SEARCH_COOLDOWN_MS = 30_000L
    }

    data class ContextualResult(
        val contextScore: Int,                          // 0–100
        val level: RiskLevel,
        val escalationReason: String,
        val activeCategories: Set<VaniScamDetector.ScamCategory>,
        val conversationThreatLevel: String,             // "NORMAL", "ELEVATED", "HIGH", "CRITICAL"
        val shouldTriggerWebSearch: Boolean,
        val suggestedQuery: String?
    )

    /** Called when contextual risk score changes */
    var onRiskUpdated: ((ContextualResult) -> Unit)? = null

    /** Called when risk crosses the web search threshold */
    var onWebSearchTriggered: ((String) -> Unit)? = null

    // ─── State ──────────────────────────────────────────────────────────────────

    private val contextWindow = mutableListOf<TranscriptUtterance>()
    private val allCategoriesHit = mutableSetOf<VaniScamDetector.ScamCategory>()
    private var lastWebSearchTime = 0L
    private var previousUtteranceCount = 0

    // ─── Public API ─────────────────────────────────────────────────────────────

    /**
     * Called whenever the transcript is updated. Analyzes the newest utterances
     * and produces an updated contextual risk assessment.
     */
    fun onUtterancesUpdated(utterances: List<TranscriptUtterance>) {
        if (utterances.size <= previousUtteranceCount) return
        previousUtteranceCount = utterances.size

        // Update sliding window
        synchronized(contextWindow) {
            contextWindow.clear()
            contextWindow.addAll(utterances.takeLast(CONTEXT_WINDOW_SIZE))
        }

        // Analyze the full context window
        val result = analyzeContext()

        // Publish to risk bus
        val prev = VaniRiskBus.last
        val baseSnapshot = prev ?: VaniRiskBus.Snapshot(
            risk = 0, synthetic = 0, speakerMatch = 50,
            level = RiskLevel.SAFE, message = "🛡 Protection Active"
        )

        val effectiveRisk = maxOf(prev?.risk ?: 0, result.contextScore)
        VaniRiskBus.publish(
            baseSnapshot.copy(
                risk = effectiveRisk,
                level = result.level,
                utterances = utterances,
                transcript = utterances.joinToString(" ") { it.text },
                message = when (result.conversationThreatLevel) {
                    "CRITICAL" -> "🚨 ${result.escalationReason}"
                    "HIGH" -> "⚠️ ${result.escalationReason}"
                    "ELEVATED" -> "⚠ ${result.escalationReason}"
                    else -> prev?.message ?: "🛡 Protection Active"
                }
            )
        )

        // Notify listener
        onRiskUpdated?.invoke(result)

        // Trigger web search if threshold crossed
        if (result.shouldTriggerWebSearch && !result.suggestedQuery.isNullOrBlank()) {
            val now = System.currentTimeMillis()
            if (now - lastWebSearchTime > WEB_SEARCH_COOLDOWN_MS) {
                lastWebSearchTime = now
                Log.i(TAG, "Level 3 web search triggered: ${result.suggestedQuery}")
                onWebSearchTriggered?.invoke(result.suggestedQuery)
            }
        }
    }

    // ─── Context Analysis ───────────────────────────────────────────────────────

    private fun analyzeContext(): ContextualResult {
        val windowText: String
        synchronized(contextWindow) {
            windowText = contextWindow.joinToString(" ") { it.text }
        }

        if (windowText.isBlank()) {
            return ContextualResult(
                contextScore = 0, level = RiskLevel.SAFE,
                escalationReason = "", activeCategories = emptySet(),
                conversationThreatLevel = "NORMAL",
                shouldTriggerWebSearch = false, suggestedQuery = null
            )
        }

        // Run Level 1 scam detector on the FULL window text (cross-utterance patterns)
        val windowDetection = VaniScamDetector.analyze(windowText)
        allCategoriesHit.addAll(windowDetection.matchedCategories)

        // Combinatorial escalation: multiple categories together are much more dangerous
        val comboScore = calculateCombinatorialScore(allCategoriesHit)
        val effectiveScore = maxOf(windowDetection.threatScore, comboScore)

        val threatLevel = when {
            effectiveScore >= 80 -> "CRITICAL"
            effectiveScore >= 55 -> "HIGH"
            effectiveScore >= 30 -> "ELEVATED"
            else -> "NORMAL"
        }

        val level = when {
            effectiveScore >= 70 -> RiskLevel.DANGER
            effectiveScore >= 35 -> RiskLevel.WARNING
            else -> RiskLevel.SAFE
        }

        val reason = buildEscalationReason(allCategoriesHit, effectiveScore)
        val shouldSearch = effectiveScore >= WEB_SEARCH_THRESHOLD
        val query = if (shouldSearch) buildSearchQuery(allCategoriesHit, windowText) else null

        Log.d(TAG, "Context analysis: score=$effectiveScore, categories=$allCategoriesHit, level=$threatLevel")

        return ContextualResult(
            contextScore = effectiveScore,
            level = level,
            escalationReason = reason,
            activeCategories = allCategoriesHit.toSet(),
            conversationThreatLevel = threatLevel,
            shouldTriggerWebSearch = shouldSearch,
            suggestedQuery = query
        )
    }

    /**
     * Calculates a combinatorial risk score based on which threat categories
     * have been detected across the ENTIRE conversation (not just one utterance).
     *
     * Key insight: A single "bank" mention is low risk. But "bank" + "OTP request"
     * + "urgency" + "account threat" in the same conversation is a classic scam pattern.
     */
    private fun calculateCombinatorialScore(categories: Set<VaniScamDetector.ScamCategory>): Int {
        if (categories.isEmpty()) return 0

        // Base: sum of individual category weights
        val baseScore = categories.sumOf { it.weight }

        // Multiplier for known dangerous combinations
        val comboMultiplier = when {
            // Classic bank scam: impersonation + credential request + urgency
            categories.containsAll(setOf(
                VaniScamDetector.ScamCategory.IMPERSONATION,
                VaniScamDetector.ScamCategory.CREDENTIAL_REQUEST,
                VaniScamDetector.ScamCategory.URGENCY
            )) -> 1.5

            // Digital arrest scam
            categories.contains(VaniScamDetector.ScamCategory.DIGITAL_ARREST) &&
            categories.contains(VaniScamDetector.ScamCategory.FINANCIAL_ACTION) -> 1.4

            // Remote access + impersonation
            categories.contains(VaniScamDetector.ScamCategory.REMOTE_ACCESS) &&
            categories.contains(VaniScamDetector.ScamCategory.IMPERSONATION) -> 1.4

            // Account threat + credential request
            categories.contains(VaniScamDetector.ScamCategory.ACCOUNT_THREAT) &&
            categories.contains(VaniScamDetector.ScamCategory.CREDENTIAL_REQUEST) -> 1.3

            // Multiple categories detected
            categories.size >= 3 -> 1.2

            else -> 1.0
        }

        return (baseScore * comboMultiplier).toInt().coerceAtMost(100)
    }

    private fun buildEscalationReason(
        categories: Set<VaniScamDetector.ScamCategory>,
        score: Int
    ): String {
        if (categories.isEmpty()) return ""

        return when {
            categories.contains(VaniScamDetector.ScamCategory.DIGITAL_ARREST) ->
                "Digital arrest scam pattern detected"
            categories.contains(VaniScamDetector.ScamCategory.CREDENTIAL_REQUEST) &&
            categories.contains(VaniScamDetector.ScamCategory.IMPERSONATION) ->
                "Authority impersonation with credential request"
            categories.contains(VaniScamDetector.ScamCategory.REMOTE_ACCESS) ->
                "Remote access/screen sharing scam pattern"
            categories.contains(VaniScamDetector.ScamCategory.CREDENTIAL_REQUEST) ->
                "Credential solicitation detected (OTP/PIN/CVV)"
            categories.contains(VaniScamDetector.ScamCategory.ACCOUNT_THREAT) ->
                "Account blocking threat detected"
            categories.size >= 3 ->
                "Multiple fraud indicators: ${categories.joinToString(", ") { it.label }}"
            else ->
                categories.first().label
        }
    }

    /**
     * Builds a focused web search query from detected threat categories.
     * Does NOT search for every word — only constructs a targeted query
     * based on the specific scam type detected.
     */
    private fun buildSearchQuery(
        categories: Set<VaniScamDetector.ScamCategory>,
        windowText: String
    ): String {
        val queryParts = mutableListOf<String>()

        if (categories.contains(VaniScamDetector.ScamCategory.IMPERSONATION)) {
            // Extract which authority is being impersonated
            val authorityMatch = Regex(
                "(RBI|SEBI|TRAI|CBI|income tax|cybercrime|police|bank)",
                RegexOption.IGNORE_CASE
            ).find(windowText)
            queryParts.add(authorityMatch?.value ?: "government authority")
            queryParts.add("phone call scam India")
        }

        if (categories.contains(VaniScamDetector.ScamCategory.DIGITAL_ARREST)) {
            queryParts.add("digital arrest scam India warning")
        }

        if (categories.contains(VaniScamDetector.ScamCategory.ACCOUNT_THREAT)) {
            queryParts.add("bank account block KYC scam")
        }

        if (categories.contains(VaniScamDetector.ScamCategory.REMOTE_ACCESS)) {
            queryParts.add("AnyDesk TeamViewer phone scam")
        }

        if (categories.contains(VaniScamDetector.ScamCategory.CREDENTIAL_REQUEST)) {
            queryParts.add("OTP sharing fraud warning")
        }

        if (queryParts.isEmpty()) {
            queryParts.add("phone call scam India 2025")
        }

        return queryParts.joinToString(" ").take(200)
    }
}
