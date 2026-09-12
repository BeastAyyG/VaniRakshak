package com.vericall.service

import android.content.Intent
import android.telecom.Call
import android.telecom.CallAudioState
import android.telecom.InCallService
import com.vericall.calling.VaniActiveCallActivity
import com.vericall.calling.VaniCallManager
import com.vericall.util.VaniCallRouter

/**
 * Android Telecom InCallService implementation.
 * When VaniRakshak is set as the default phone app (or in-call UI), Android
 * binds to this service for every incoming and outgoing cellular phone call,
 * routing call state and audio controls directly into VaniRakshak.
 */
class VaniInCallService : InCallService() {

    override fun onCallAdded(call: Call) {
        super.onCallAdded(call)
        VaniCallManager.setCall(call, this)

        val handle = call.details?.handle?.schemeSpecificPart ?: ""
        val caller = VaniCallRouter.resolve(this, handle)
        val isOutgoing = call.details?.callDirection == Call.Details.DIRECTION_OUTGOING ||
                call.state == Call.STATE_CONNECTING ||
                call.state == Call.STATE_DIALING
        val isRinging = call.state == Call.STATE_RINGING

        // Register callback on the specific Call object
        call.registerCallback(object : Call.Callback() {
            override fun onStateChanged(c: Call, state: Int) {
                super.onStateChanged(c, state)
                VaniCallManager.onStateChanged?.invoke(state)
                if (state == Call.STATE_DISCONNECTED) {
                    VaniCallManager.clearCall(c)
                }
            }
        })

        // Launch VaniRakshak's native In-Call Activity
        val intent = Intent(this, VaniActiveCallActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or
                    Intent.FLAG_ACTIVITY_SINGLE_TOP or
                    Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra(VaniActiveCallActivity.EXTRA_CALL_NUMBER, caller.phone.ifBlank { handle })
            putExtra(VaniActiveCallActivity.EXTRA_CALL_NAME, caller.name)
            putExtra(VaniActiveCallActivity.EXTRA_IS_OUTGOING, isOutgoing)
            putExtra(VaniActiveCallActivity.EXTRA_IS_INCOMING_RINGING, isRinging)
        }
        startActivity(intent)
    }

    override fun onCallRemoved(call: Call) {
        super.onCallRemoved(call)
        VaniCallManager.clearCall(call)
    }

    override fun onCallAudioStateChanged(audioState: CallAudioState) {
        super.onCallAudioStateChanged(audioState)
        VaniCallManager.onAudioStateChanged?.invoke(audioState)
    }
}
