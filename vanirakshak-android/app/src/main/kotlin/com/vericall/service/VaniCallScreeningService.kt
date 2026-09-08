package com.vericall.service

import android.os.Build
import android.telecom.Call
import android.telecom.CallScreeningService
import com.vericall.util.VaniCallRouter

/**
 * Android 7+ call screening hook.
 *
 * Telecom binds this service directly, so it receives incoming calls far more
 * reliably than a manifest broadcast - including when the app has never been
 * opened in this boot. The user has to grant the call-screening role once
 * (Settings > Caller ID & spam, or the button on the main screen).
 *
 * We never reject or silence the call: VaniRakshak only observes and reports.
 */
class VaniCallScreeningService : CallScreeningService() {

    override fun onScreenCall(callDetails: Call.Details) {
        // setSilenceCall landed in API 29. Omitting it is equivalent to false,
        // so on older devices we simply skip it rather than crashing.
        val response = CallResponse.Builder().apply {
            setDisallowCall(false)
            setRejectCall(false)
            setSkipCallLog(false)
            setSkipNotification(false)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                setSilenceCall(false)
            }
        }.build()

        // getCallDirection is also API 29. Below that, onScreenCall is only ever
        // invoked for incoming calls, so treating them all as incoming is correct.
        val isIncoming = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            callDetails.callDirection == Call.Details.DIRECTION_INCOMING
        } else {
            true
        }

        if (isIncoming) {
            VaniCallRouter.onRinging(this, callDetails.handle?.schemeSpecificPart)
        }

        respondToCall(callDetails, response)
    }
}
