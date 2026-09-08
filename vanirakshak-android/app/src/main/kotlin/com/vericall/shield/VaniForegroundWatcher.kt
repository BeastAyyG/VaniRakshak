package com.vericall.shield

import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.os.Handler
import android.os.Looper

/**
 * Periodically detects if a sensitive banking/UPI app is resumed in the foreground
 * while a call is under high risk or interlock defense.
 */
class VaniForegroundWatcher(private val context: Context) {

    private val handler = Handler(Looper.getMainLooper())

    @Volatile
    private var isPolling = false

    private val pollRunnable = object : Runnable {
        override fun run() {
            if (!isPolling) return
            checkForegroundApp()
            handler.postDelayed(this, POLL_INTERVAL_MS)
        }
    }

    fun start() {
        if (isPolling) return
        isPolling = true
        handler.post(pollRunnable)
    }

    fun stop() {
        isPolling = false
        handler.removeCallbacks(pollRunnable)
    }

    private fun checkForegroundApp() {
        if (!VaniScamShieldController.isShieldActive()) return

        try {
            val usm = context.getSystemService(Context.USAGE_STATS_SERVICE) as? UsageStatsManager ?: return
            val end = System.currentTimeMillis()
            val begin = end - 3000
            val events = usm.queryEvents(begin, end)
            val event = UsageEvents.Event()
            var lastForegroundPkg: String? = null

            while (events.hasNextEvent()) {
                events.getNextEvent(event)
                if (event.eventType == UsageEvents.Event.ACTIVITY_RESUMED) {
                    lastForegroundPkg = event.packageName
                }
            }

            if (lastForegroundPkg != null && lastForegroundPkg in VaniScamShieldController.bankingPackages) {
                VaniCoercionOverlay.show(context, lastForegroundPkg)
            }
        } catch (e: Exception) {
            // Usage stats not granted or restricted
        }
    }

    companion object {
        private const val POLL_INTERVAL_MS = 2000L
    }
}
