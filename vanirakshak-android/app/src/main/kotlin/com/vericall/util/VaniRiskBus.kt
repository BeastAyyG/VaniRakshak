package com.vericall.util

import com.vericall.model.RiskLevel

/**
 * Single in-process channel for live analysis results.
 *
 * The foreground guard service owns the microphone, so it is the only publisher.
 * The guard screen subscribes instead of opening a second AudioRecord - two
 * AudioRecord instances on the same source fight each other and one silently
 * returns zero frames.
 */
object VaniRiskBus {

    data class Snapshot(
        val risk: Int,
        val synthetic: Int,
        val speakerMatch: Int,
        val level: RiskLevel,
        val message: String,
        val tier: String = when (level) {
            RiskLevel.DANGER -> "BLOCK"
            RiskLevel.WARNING -> "CHALLENGE"
            RiskLevel.SAFE -> "ALLOW"
        },
        val pSpoof: Float = synthetic / 100f,
        val transcript: String = "",
        val scamType: String = "",
        val threatScore: Int = 0,
        val redFlags: List<String> = emptyList(),
        val actionAdvice: String = "",
        val receiptHash: String = "",
        val cmScore: Float = 0.0f,
        val cmGenuineProb: Float = 0.0f,
        val asvConsistency: Float = 0.0f,
        val channelAnomaly: Float = 0.0f,
        val snrDb: Float = 18.0f,
        val vocoderFlatness: Float = 0.0f,
        val microTremor: Float = 0.0f,
        val phaseJumps: Int = 0
    )

    private val listeners = LinkedHashSet<(Snapshot) -> Unit>()
    private val amplitudeListeners = LinkedHashSet<(Float) -> Unit>()

    @Volatile
    var last: Snapshot? = null
        private set

    @Volatile
    var activeCallId: String? = null

    @Volatile
    var lastRecordedAudioPath: String? = null

    fun publish(snapshot: Snapshot) {
        last = snapshot
        listeners.toList().forEach { it(snapshot) }
    }

    fun publishAmplitude(amplitude: Float) {
        amplitudeListeners.toList().forEach { it(amplitude) }
    }

    fun addListener(listener: (Snapshot) -> Unit) {
        listeners += listener
        last?.let(listener)
    }

    fun removeListener(listener: (Snapshot) -> Unit) {
        listeners -= listener
    }

    fun addAmplitudeListener(listener: (Float) -> Unit) {
        amplitudeListeners += listener
    }

    fun removeAmplitudeListener(listener: (Float) -> Unit) {
        amplitudeListeners -= listener
    }

    fun reset() {
        last = null
    }
}
