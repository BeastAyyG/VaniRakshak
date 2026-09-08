package com.vericall

import com.vericall.shield.VaniIncidentLogger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class VaniIncidentLoggerTest {

    @Before
    fun setUp() {
        VaniIncidentLogger.clearActiveHeldNotifications()
        VaniIncidentLogger.clearActiveOverlayTriggers()
    }

    @Test
    fun activeHeldNotificationsCanBeStashedAndCleared() {
        assertEquals(0, VaniIncidentLogger.getActiveHeldNotifications().size)

        // Mock stashing without SQLite database (test in-memory session cache)
        VaniIncidentLogger.stashHeldNotification(
            context = null as? android.content.Context ?: object : android.content.ContextWrapper(null) {},
            packageName = "com.google.android.apps.nbu.paisa.user",
            text = "Your OTP is 123456",
            timestamp = 1000L
        )

        val held = VaniIncidentLogger.getActiveHeldNotifications()
        assertEquals(1, held.size)
        assertEquals("com.google.android.apps.nbu.paisa.user", held[0].packageName)
        assertEquals("Your OTP is 123456", held[0].text)

        VaniIncidentLogger.clearActiveHeldNotifications()
        assertTrue(VaniIncidentLogger.getActiveHeldNotifications().isEmpty())
    }

    @Test
    fun activeOverlayTriggersTrackedAndCleared() {
        assertEquals(0, VaniIncidentLogger.getActiveOverlayTriggers().size)

        VaniIncidentLogger.logOverlayTrigger("com.phonepe.app")
        VaniIncidentLogger.logOverlayTrigger("net.one97.paytm")

        val triggers = VaniIncidentLogger.getActiveOverlayTriggers()
        assertEquals(2, triggers.size)
        assertEquals("com.phonepe.app", triggers[0])
        assertEquals("net.one97.paytm", triggers[1])

        VaniIncidentLogger.clearActiveOverlayTriggers()
        assertTrue(VaniIncidentLogger.getActiveOverlayTriggers().isEmpty())
    }
}
