package com.vericall.receiver

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.widget.Toast
import com.vericall.util.VaniCallRouter
import com.vericall.util.VaniNotificationHelper

/**
 * Handles the "Block & report" action on the live risk notification.
 *
 * Terminating the call itself needs the InCallService / MANAGE_OWN_CALLS
 * capability, so this stops the analysis, clears the alert and files the
 * incident - the same outcome as the block button on the guard screen.
 */
class VaniNotificationActionReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION_BLOCK) return

        val callerName = intent.getStringExtra(VaniNotificationHelper.EXTRA_CALLER_NAME)
            ?: context.getString(com.vericall.R.string.unknown_caller)

        VaniCallRouter.onEnded(context)
        VaniNotificationHelper.cancelNotification(context)

        Toast.makeText(
            context,
            context.getString(com.vericall.R.string.toast_call_blocked, callerName),
            Toast.LENGTH_LONG
        ).show()
    }

    companion object {
        const val ACTION_BLOCK = "com.vericall.action.BLOCK_CALL"

        fun blockIntent(context: Context, callerName: String): PendingIntent {
            val intent = Intent(context, VaniNotificationActionReceiver::class.java).apply {
                action = ACTION_BLOCK
                putExtra(VaniNotificationHelper.EXTRA_CALLER_NAME, callerName)
            }
            val flags = PendingIntent.FLAG_UPDATE_CURRENT or
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE else 0
            return PendingIntent.getBroadcast(context, 31, intent, flags)
        }
    }
}
