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
        /** Server-driven tier per docs/PROTOCOL.md §5 (ALLOW/CHALLENGE/BLOCK). Null = on-device fallback. */
        val tier: String? = null,
        /** True when the server computed this verdict (WS stream); false = local heuristic. */
        val serverDriven: Boolean = false,
        /** Active challenge phrase from verdict.challenge.phrase, if any. */
        val challengePhrase: String? = null,
        val challengeInstructions: String? = null,
        /** SHA-256 audit receipt from verdict.receipt_hash, if any. */
        val receiptHash: String? = null,
        /** Server interlock_active flag. One-way latch enforced by callers. */
        val interlockActive: Boolean = false,
        /** WS session id that produced this verdict (needed for challenge grading). Null when offline. */
        val sessionId: String? = null
    )

    private val listeners = LinkedHashSet<(Snapshot) -> Unit>()
    private val amplitudeListeners = LinkedHashSet<(Float) -> Unit>()

    @Volatile
    var last: Snapshot? = null
        private set

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
