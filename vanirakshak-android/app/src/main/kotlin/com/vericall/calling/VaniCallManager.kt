package com.vericall.calling

import android.os.Build
import android.telecom.Call
import android.telecom.CallAudioState
import android.telecom.VideoProfile
import com.vericall.service.VaniInCallService

/**
 * Singleton that manages the active Android Telecom cellular Call object
 * and bridges it to VaniRakshak's in-call UI (VaniActiveCallActivity).
 */
object VaniCallManager {

    var currentCall: Call? = null
    var inCallService: VaniInCallService? = null

    // UI Listeners
    var onStateChanged: ((Int) -> Unit)? = null
    var onAudioStateChanged: ((CallAudioState) -> Unit)? = null
    var onCallEnded: (() -> Unit)? = null

    fun setCall(call: Call, service: VaniInCallService) {
        currentCall = call
        inCallService = service
    }

    fun clearCall(call: Call) {
        if (currentCall == call) {
            currentCall = null
            onCallEnded?.invoke()
        }
    }

    fun hasActiveTelecomCall(): Boolean {
        val call = currentCall ?: return false
        return call.state != Call.STATE_DISCONNECTED
    }

    fun isCallActive(): Boolean {
        return currentCall?.state == Call.STATE_ACTIVE
    }

    fun isIncomingRinging(): Boolean {
        return currentCall?.state == Call.STATE_RINGING
    }

    fun disconnect() {
        currentCall?.disconnect()
    }

    fun answer() {
        currentCall?.answer(VideoProfile.STATE_AUDIO_ONLY)
    }

    fun reject() {
        currentCall?.reject(false, null)
    }

    fun setMuted(muted: Boolean) {
        inCallService?.setMuted(muted)
    }

    fun setSpeaker(speakerOn: Boolean) {
        val route = if (speakerOn) {
            CallAudioState.ROUTE_SPEAKER
        } else {
            CallAudioState.ROUTE_EARPIECE
        }
        inCallService?.setAudioRoute(route)
    }

    fun isSpeakerOn(): Boolean {
        val audioState = inCallService?.callAudioState ?: return false
        return audioState.route == CallAudioState.ROUTE_SPEAKER
    }

    fun isMuted(): Boolean {
        return inCallService?.callAudioState?.isMuted == true
    }

    fun sendDtmf(digit: Char) {
        currentCall?.playDtmfTone(digit)
        currentCall?.stopDtmfTone()
    }

    fun hold() {
        currentCall?.hold()
    }

    fun unhold() {
        currentCall?.unhold()
    }
}
