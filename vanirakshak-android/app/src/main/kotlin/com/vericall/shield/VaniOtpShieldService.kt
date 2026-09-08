package com.vericall.shield

import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification

/**
 * Legitimate Android NotificationListenerService that shields banking/UPI
 * notifications and OTPs from the screen while a high-risk or interlocked
 * call is active, preventing real-time verbal extortion or coerced transfers.
 */
class VaniOtpShieldService : NotificationListenerService() {

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        if (sbn == null) return
        if (!VaniScamShieldController.isShieldActive()) return

        val pkg = sbn.packageName ?: return
        if (pkg !in VaniScamShieldController.bankingPackages) return

        val extras = sbn.notification?.extras
        val title = extras?.getCharSequence("android.title")?.toString().orEmpty()
        val text = extras?.getCharSequence("android.text")?.toString().orEmpty()
        val bigText = extras?.getCharSequence("android.bigText")?.toString().orEmpty()
        val fullContent = listOf(title, text.ifEmpty { bigText }).filter { it.isNotBlank() }.joinToString(" — ")

        // Suppress live display and stash for safe post-call review
        VaniIncidentLogger.stashHeldNotification(
            context = this,
            packageName = pkg,
            text = fullContent.ifEmpty { "Protected financial notification from $pkg" }
        )

        try {
            cancelNotification(sbn.key)
        } catch (e: Exception) {
            e.printStackTrace()
        }

        VaniScamShieldController.onNotificationHeld()
    }
}
