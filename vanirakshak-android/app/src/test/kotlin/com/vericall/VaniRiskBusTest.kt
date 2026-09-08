package com.vericall

import com.vericall.model.RiskLevel
import com.vericall.util.VaniRiskBus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

class VaniRiskBusTest {

    @Before
    fun setUp() {
        VaniRiskBus.reset()
    }

    @Test
    fun publishedSnapshotIsRetainedAsLast() {
        assertNull(VaniRiskBus.last)

        val snapshot = VaniRiskBus.Snapshot(
            risk = 85,
            synthetic = 92,
            speakerMatch = 20,
            level = RiskLevel.DANGER,
            message = "AI Voice clone detected"
        )
        VaniRiskBus.publish(snapshot)

        assertNotNull(VaniRiskBus.last)
        assertEquals(85, VaniRiskBus.last?.risk)
        assertEquals(RiskLevel.DANGER, VaniRiskBus.last?.level)
    }

    @Test
    fun listenerReceivesImmediateReplayOfLastSnapshot() {
        val snapshot = VaniRiskBus.Snapshot(
            risk = 15,
            synthetic = 5,
            speakerMatch = 95,
            level = RiskLevel.SAFE,
            message = "Authentic voice verified"
        )
        VaniRiskBus.publish(snapshot)

        var deliveredSnapshot: VaniRiskBus.Snapshot? = null
        val listener: (VaniRiskBus.Snapshot) -> Unit = { deliveredSnapshot = it }

        VaniRiskBus.addListener(listener)
        assertNotNull(deliveredSnapshot)
        assertEquals(15, deliveredSnapshot?.risk)
        assertEquals(RiskLevel.SAFE, deliveredSnapshot?.level)

        VaniRiskBus.removeListener(listener)
    }

    @Test
    fun amplitudeListenerReceivesUpdates() {
        var receivedAmp = 0f
        val listener: (Float) -> Unit = { receivedAmp = it }

        VaniRiskBus.addAmplitudeListener(listener)
        VaniRiskBus.publishAmplitude(0.75f)

        assertEquals(0.75f, receivedAmp, 0.001f)
        VaniRiskBus.removeAmplitudeListener(listener)
    }

    @Test
    fun resetClearsLastSnapshot() {
        val snapshot = VaniRiskBus.Snapshot(50, 50, 50, RiskLevel.WARNING, "Warning")
        VaniRiskBus.publish(snapshot)
        assertNotNull(VaniRiskBus.last)

        VaniRiskBus.reset()
        assertNull(VaniRiskBus.last)
    }
}
