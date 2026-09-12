package com.vericall.model

enum class RiskLevel {
    SAFE,
    WARNING,
    DANGER
}

enum class CallScenario(
    val riskScore: Int,
    val acousticSyntheticScore: Int,
    val speakerCosineScore: Int,
    val channelInfo: String,
    val latencyMs: Int,
    val verdictTitle: String,
    val verdictDescription: String,
    val level: RiskLevel
) {
    AI_CLONE_ATTACK(
        riskScore = 89,
        acousticSyntheticScore = 94,
        speakerCosineScore = 26,
        channelInfo = "PSTN / G.711 Transcoded (8kHz)",
        latencyMs = 124,
        verdictTitle = "CRITICAL: AI VOICE-CLONE IMPERSONATION",
        verdictDescription = "High synthetic vocoder artifacts detected (XTTS / ElevenLabs signature). Speaker characteristics fail ECAPA-TDNN baseline. High financial fraud probability.",
        level = RiskLevel.DANGER
    ),
    GENUINE_CALL(
        riskScore = 11,
        acousticSyntheticScore = 5,
        speakerCosineScore = 96,
        channelInfo = "VoIP / Opus Wideband (16kHz)",
        latencyMs = 98,
        verdictTitle = "AUTHENTIC HUMAN VOICE VERIFIED",
        verdictDescription = "Speech acoustics match biological vocal tract physiology. Speaker identity confirmed against enrolled 192-dim voice embedding.",
        level = RiskLevel.SAFE
    ),
    SUSPICIOUS_ANOMALY(
        riskScore = 62,
        acousticSyntheticScore = 58,
        speakerCosineScore = 54,
        channelInfo = "GSM-EFR Cellular (8kHz)",
        latencyMs = 142,
        verdictTitle = "WARNING: VOICE ANOMALY DETECTED",
        verdictDescription = "Lossy transcode with micro-tremor irregularities. Discrepancy in pitch contour. Trigger active challenge to verify caller liveness.",
        level = RiskLevel.WARNING
    )
}

data class VaniContact(
    val id: String,
    val name: String,
    val phone: String,
    val relation: String,
    var isVoiceEnrolled: Boolean,
    var enrollmentSec: Int = 15,
    var sampleEmbeddingCount: Int = 3,
    var defaultScenario: CallScenario = CallScenario.AI_CLONE_ATTACK,
    var lastAlertText: String? = null,
    var isFavorite: Boolean = false
)
