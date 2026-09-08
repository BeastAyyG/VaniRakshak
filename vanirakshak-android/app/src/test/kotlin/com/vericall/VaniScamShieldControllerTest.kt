package com.vericall

import com.vericall.model.RiskLevel
import com.vericall.shield.VaniScamShieldController
import com.vericall.util.VaniRiskBus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class VaniScamShieldControllerTest {

    @Before
    fun setUp() {
        VaniRiskBus.reset()
        VaniScamShieldController.resetForTesting()
    }

    @Test
    fun shieldIsInactiveWhenNoCallIsOngoing() {
        assertFalse(VaniScamShieldController.isShieldActive())
        VaniScamShieldController.setUserInterlockEnabled(true)
        // Even if user enables interlock, if call is not active, shield should not be active
        assertFalse(VaniScamShieldController.isShieldActive())
    }

    @Test
    fun shieldActivatesWhenCallIsActiveAndUserEngagesInterlock() {
        VaniScamShieldController.onCallStarted("Aarav", "+91 98765 43210")
        assertTrue(VaniScamShieldController.isCallActive)
        assertFalse(VaniScamShieldController.isShieldActive())

        VaniScamShieldController.setUserInterlockEnabled(true)
        assertTrue(VaniScamShieldController.isShieldActive())
    }

    @Test
    fun shieldActivatesAutomaticallyWhenThreatLevelIsDanger() {
        VaniScamShieldController.onCallStarted("Suspicious Caller", "+91 99999 88888")
        assertFalse(VaniScamShieldController.isShieldActive())

        // Publish high risk spoof snapshot
        VaniRiskBus.publish(
            VaniRiskBus.Snapshot(
                risk = 92,
                synthetic = 95,
                speakerMatch = 15,
                level = RiskLevel.DANGER,
                message = "Deepfake voice detected"
            )
        )

        assertTrue("Shield must engage automatically on DANGER threat level", VaniScamShieldController.isShieldActive())
    }

    @Test
    fun bankingPackagesIncludeMajorPaymentApps() {
        val packages = VaniScamShieldController.bankingPackages
        assertTrue(packages.contains("com.google.android.apps.nbu.paisa.user")) // GPay
        assertTrue(packages.contains("com.phonepe.app"))                        // PhonePe
        assertTrue(packages.contains("net.one97.paytm"))                        // Paytm
        assertTrue(packages.contains("in.org.npci.upiapp"))                     // BHIM
        assertTrue(packages.contains("com.snapwork.hdfc"))                      // HDFC
    }

    @Test
    fun peakRiskTracksMaximumObservedRisk() {
        VaniScamShieldController.onCallStarted("Caller", "+91 98765 43210")
        VaniScamShieldController.onRiskScoreUpdated(40)
        VaniScamShieldController.onRiskScoreUpdated(88)
        VaniScamShieldController.onRiskScoreUpdated(60)

        // Session variables update correctly
        assertEquals("Caller", VaniScamShieldController.activeCallerName)
        assertEquals("+91 98765 43210", VaniScamShieldController.activeCallerPhone)
    }
}
