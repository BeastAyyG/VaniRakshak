package com.vericall

import android.Manifest
import android.app.Application
import android.app.Notification
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.telephony.TelephonyManager
import androidx.test.core.app.ApplicationProvider
import com.vericall.receiver.VaniCallReceiver
import com.vericall.util.VaniNotificationHelper
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config

/**
 * Exercises the real notification pipeline on the JVM via Robolectric, so the
 * incoming-call alert is actually verified rather than just compiled.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class VaniCallReceiverTest {

    private lateinit var context: Context
    private lateinit var notificationManager: NotificationManager

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        notificationManager =
            context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        shadowOf(context as Application)
            .grantPermissions(Manifest.permission.POST_NOTIFICATIONS)
    }

    private fun phoneStateIntent(state: String, number: String?): Intent =
        Intent(TelephonyManager.ACTION_PHONE_STATE_CHANGED).apply {
            putExtra(TelephonyManager.EXTRA_STATE, state)
            if (number != null) {
                putExtra(TelephonyManager.EXTRA_INCOMING_NUMBER, number)
            }
        }

    private fun postedNotifications() = shadowOf(notificationManager).allNotifications

    private fun Notification.titleText(): String =
        extras.getCharSequence(Notification.EXTRA_TITLE)?.toString().orEmpty()

    private fun Notification.bodyText(): String =
        extras.getCharSequence(Notification.EXTRA_TEXT)?.toString().orEmpty()

    @Test
    fun ringingFromKnownContactPostsAlertNamingTheCaller() {
        VaniCallReceiver().onReceive(
            context,
            phoneStateIntent(TelephonyManager.EXTRA_STATE_RINGING, "+91 98765 43210")
        )

        val notifications = postedNotifications()
        assertEquals(1, notifications.size)
        assertTrue(
            "Expected the resolved contact name in the title, got: ${notifications[0].titleText()}",
            notifications[0].titleText().contains("Aarav Sharma")
        )
    }

    @Test
    fun ringingFromUnknownNumberPostsAlertWithTheNumber() {
        VaniCallReceiver().onReceive(
            context,
            phoneStateIntent(TelephonyManager.EXTRA_STATE_RINGING, "+91 11111 22222")
        )

        val notifications = postedNotifications()
        assertEquals(1, notifications.size)
        assertTrue(
            "Expected the raw number in the title, got: ${notifications[0].titleText()}",
            notifications[0].titleText().contains("+91 11111 22222")
        )
    }

    @Test
    fun ringingAlertShowsNoRiskScoreBeforeAnyAudioIsAnalyzed() {
        VaniCallReceiver().onReceive(
            context,
            phoneStateIntent(TelephonyManager.EXTRA_STATE_RINGING, "+91 98765 43210")
        )

        val notification = postedNotifications().first()
        val combined = notification.titleText() + notification.bodyText()
        assertFalse(
            "A risk percentage must not be shown before analysis, got: $combined",
            combined.contains("%")
        )
    }

    @Test
    fun missedCallClearsTheRingingAlert() {
        VaniCallReceiver().onReceive(
            context,
            phoneStateIntent(TelephonyManager.EXTRA_STATE_RINGING, "+91 98765 43210")
        )
        assertEquals(1, postedNotifications().size)

        // Never answered - the call goes straight to IDLE.
        VaniCallReceiver().onReceive(context, phoneStateIntent(TelephonyManager.EXTRA_STATE_IDLE, null))

        assertTrue(
            "A missed call must not leave the alert in the shade",
            postedNotifications().isEmpty()
        )
    }

    @Test
    fun channelsAreCreatedWhenTheCallArrives() {
        VaniCallReceiver().onReceive(
            context,
            phoneStateIntent(TelephonyManager.EXTRA_STATE_RINGING, "+91 98765 43210")
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            assertNotNull(notificationManager.getNotificationChannel(VaniNotificationHelper.CHANNEL_ALERT))
            assertNotNull(notificationManager.getNotificationChannel(VaniNotificationHelper.CHANNEL_LIVE))
        }
    }
}
