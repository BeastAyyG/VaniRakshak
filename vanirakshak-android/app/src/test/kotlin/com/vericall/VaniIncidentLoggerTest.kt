package com.vericall

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.vericall.shield.VaniIncidentLogger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
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

    @Test
    fun receiptAndTierRoundTripThroughSqlite() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val id = VaniIncidentLogger.recordIncident(
            context = context,
            callerNumber = "+919876543210",
            callerName = "Unknown",
            peakRisk = 91,
            verdictSummary = "Server verdict: BLOCK",
            receiptHash = "deadbeef",
            serverTier = "BLOCK"
        )
        assertTrue(id > 0)
        val read = VaniIncidentLogger.getIncident(context, id)
        assertEquals("+919876543210", read?.callerNumber)
        assertEquals(91, read?.peakRisk)
        assertEquals("deadbeef", read?.receiptHash)
        assertEquals("BLOCK", read?.serverTier)
    }

    @Test
    fun missingReceiptReadsBackNull() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val id = VaniIncidentLogger.recordIncident(
            context = context,
            callerNumber = "123",
            callerName = "Local",
            peakRisk = 20,
            verdictSummary = "offline estimate"
        )
        val read = VaniIncidentLogger.getIncident(context, id)
        assertNull(read?.receiptHash)
        assertNull(read?.serverTier)
        assertNull(VaniIncidentLogger.getIncident(context, -999L))
    }
}
