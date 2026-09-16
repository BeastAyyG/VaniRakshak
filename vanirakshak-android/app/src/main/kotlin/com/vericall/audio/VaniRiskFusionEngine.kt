package com.vericall.audio

import android.util.Log
import com.vericall.model.FactCheckItem
import com.vericall.model.RiskLevel
import com.vericall.model.WebSourceItem
import com.vericall.util.VaniRiskBus

/**
 * VaniRiskFusionEngine — Combines signals from all three risk tiers + acoustic analysis.
 *
 * Fusion rules:
 *   Level 1 (instant keywords)    → immediate UI indicator, sub-second
 *   Level 2 (context patterns)    → risk score adjustment, possible Level 3 trigger
 *   Level 3 (web evidence)        → confirms/denies scam, adds sources
 *   Acoustic (spoof detection)    → modifies confidence in voice authenticity
 *
 * The fused result is published to VaniRiskBus for the UI to render.
 */
class VaniRiskFusionEngine {

    companion object {
        private const val TAG = "VaniRiskFusion"
    }

    // Accumulated signals from each tier
    private var level1Score: Int = 0
    private var level1Flags: List<String> = emptyList()
    private var level2Score: Int = 0
    private var level2Reason: String = ""
    private var level3Adjustment: Int = 0
    private var level3FactChecks: List<FactCheckItem> = emptyList()
    private var level3Sources: List<WebSourceItem> = emptyList()
    private var level3Summary: String = ""
    private var acousticScore: Int = 0

    // ─── Tier Inputs ────────────────────────────────────────────────────────────

    /**
     * Level 1: Instant keyword/pattern detection from VaniScamDetector.
     * Called on every finalized utterance with high-risk patterns.
     */
    fun onLevel1Alert(detection: VaniScamDetector.DetectionResult) {
        level1Score = maxOf(level1Score, detection.threatScore)
        level1Flags = (level1Flags + detection.redFlags).distinct()
        Log.d(TAG, "Level 1 alert: score=${detection.threatScore}, flags=${detection.redFlags}")
        publishFused()
    }

    /**
     * Level 2: Contextual multi-sentence analysis from VaniContextualRiskEngine.
     */
    fun onLevel2Update(result: VaniContextualRiskEngine.ContextualResult) {
        level2Score = result.contextScore
        level2Reason = result.escalationReason
        Log.d(TAG, "Level 2 update: score=${result.contextScore}, reason=${result.escalationReason}")
        publishFused()
    }

    /**
     * Level 3: Web evidence from VaniWebIntelligenceEngine.
     * This arrives asynchronously and may come seconds after Levels 1+2.
     */
    fun onLevel3Evidence(evidence: VaniWebIntelligenceEngine.WebEvidence) {
        level3Adjustment = evidence.riskAdjustment
        level3FactChecks = evidence.factChecks
        level3Sources = evidence.sources
        level3Summary = evidence.summary
        Log.i(TAG, "Level 3 evidence: adjustment=+${evidence.riskAdjustment}, " +
                "factChecks=${evidence.factChecks.size}, sources=${evidence.sources.size}")
        publishFused()
    }

    /**
     * Acoustic analysis update (from VaniLiveVoiceAnalyzer via VaniContinuousAudioEngine).
     */
    fun onAcousticUpdate(risk: Int) {
        acousticScore = risk
    }

    // ─── Fusion Logic ───────────────────────────────────────────────────────────

    private fun publishFused() {
        // Fuse all signals: highest of all tiers + web adjustment
        val fusedScore = (maxOf(level1Score, level2Score, acousticScore) + level3Adjustment)
            .coerceIn(0, 100)

        val level = when {
            fusedScore >= 70 -> RiskLevel.DANGER
            fusedScore >= 35 -> RiskLevel.WARNING
            else -> RiskLevel.SAFE
        }

        // Build the most informative message
        val message = when {
            level3Summary.isNotBlank() && level == RiskLevel.DANGER ->
                "🚨 $level3Summary"
            level3Summary.isNotBlank() && level == RiskLevel.WARNING ->
                "⚠️ $level3Summary"
            level2Reason.isNotBlank() && level == RiskLevel.DANGER ->
                "🚨 $level2Reason"
            level2Reason.isNotBlank() && level == RiskLevel.WARNING ->
                "⚠️ $level2Reason"
            level1Flags.isNotEmpty() && level == RiskLevel.DANGER ->
                "🚨 ${level1Flags.first()}"
            else -> null  // Don't override existing message
        }

        // Merge with existing snapshot (preserve transcript, utterances, etc.)
        val prev = VaniRiskBus.last ?: return
        val updated = prev.copy(
            risk = fusedScore,
            level = level,
            threatScore = maxOf(prev.threatScore, fusedScore),
            redFlags = (prev.redFlags + level1Flags).distinct(),
            factChecks = if (level3FactChecks.isNotEmpty()) level3FactChecks else prev.factChecks,
            sources = if (level3Sources.isNotEmpty()) level3Sources else prev.sources,
            message = message ?: prev.message,
            crawlerStatus = if (level3Sources.isNotEmpty()) "evidence_collected" else prev.crawlerStatus
        )

        VaniRiskBus.publish(updated)
    }
}
