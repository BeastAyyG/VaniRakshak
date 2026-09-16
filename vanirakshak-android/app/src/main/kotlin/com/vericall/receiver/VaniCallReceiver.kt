package com.vericall.receiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.telephony.TelephonyManager
import com.vericall.state.CallStateStore
import com.vericall.util.VaniCallRouter

/**
 * Manifest-broadcast detection path.
 *
 * PHONE_STATE keeps working after the app process has been killed and needs no
 * special role. On Android 7+ the OS additionally binds
 * [com.vericall.service.VaniCallScreeningService], which is the more reliable
 * path. Both funnel into [VaniCallRouter] so behaviour never drifts.
 */
class VaniCallReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != TelephonyManager.ACTION_PHONE_STATE_CHANGED) return

        // EXTRA_INCOMING_NUMBER only arrives with the RINGING broadcast, so it
        // has to be cached for the OFFHOOK and IDLE broadcasts that follow.
        val incomingNumber = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER)
        if (!incomingNumber.isNullOrBlank()) {
            CallStateStore.remember(context, incomingNumber)
        }
        val number = incomingNumber ?: CallStateStore.recall(context)

        when (intent.getStringExtra(TelephonyManager.EXTRA_STATE)) {
            TelephonyManager.EXTRA_STATE_RINGING -> {
                if (!com.vericall.calling.VaniCallManager.hasActiveTelecomCall()) {
                    VaniCallRouter.onRinging(context, number)
                }
            }
            TelephonyManager.EXTRA_STATE_OFFHOOK -> {
                if (!com.vericall.calling.VaniCallManager.hasActiveTelecomCall()) {
                    VaniCallRouter.onAnswered(context, number)
                }
            }
            TelephonyManager.EXTRA_STATE_IDLE -> {
                if (!com.vericall.calling.VaniCallManager.hasActiveTelecomCall()) {
                    VaniCallRouter.onEnded(context)
                    CallStateStore.clear(context)
                }
            }
        }
    }
}
