package com.vericall.voip

import android.os.Handler
import android.os.Looper
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * JSON-over-WebSocket client for the VaniRakshak Call signaling server
 * (ws://host:8001/ws/signal). Wire protocol is documented in
 * hidden_files/voip-build/SIGNALING_PROTOCOL.md. Text frames only.
 */
class VoipSignalingClient(private val wsUrl: String) {

    interface Listener {
        fun onRegistered(userId: String)
        fun onPresence(users: List<String>)
        fun onIncomingCall(from: String, callId: String)
        fun onAccepted(callId: String)
        fun onRejected(callId: String, reason: String)
        fun onRemoteSdp(callId: String, kind: String, sdp: JSONObject)
        fun onRemoteIce(callId: String, candidate: JSONObject)
        fun onHangup(callId: String)
        fun onSignalingError(code: String, message: String)
        fun onDisconnected()
    }

    companion object {
        private const val TAG = "VoipSignaling"
    }

    var listener: Listener? = null
    /** Fires on the main thread once the WebSocket handshake completes. */
    var onConnected: (() -> Unit)? = null

    private val mainHandler = Handler(Looper.getMainLooper())
    private val httpClient = OkHttpClient.Builder()
        .pingInterval(20, TimeUnit.SECONDS)
        .build()

    @Volatile private var socket: WebSocket? = null
    @Volatile private var closed = false

    fun connect() {
        closed = false
        val request = Request.Builder().url(wsUrl).build()
        socket = httpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(TAG, "signaling connected: $wsUrl")
                mainHandler.post { onConnected?.invoke() }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                handleMessage(text)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.w(TAG, "signaling failure: ${t.message}")
                socket = null
                if (!closed) mainHandler.post { listener?.onDisconnected() }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                socket = null
                if (!closed) mainHandler.post { listener?.onDisconnected() }
            }
        })
    }

    fun disconnect() {
        closed = true
        runCatching { socket?.close(1000, "done") }
        socket = null
    }

    private fun send(obj: JSONObject) {
        val ws = socket
        if (ws == null) {
            Log.w(TAG, "send while disconnected: ${obj.optString("type")}")
            return
        }
        ws.send(obj.toString())
    }

    fun register(userId: String) =
        send(JSONObject().put("type", "register").put("user_id", userId))

    fun presenceRequest() =
        send(JSONObject().put("type", "presence_request"))

    fun call(from: String, to: String, callId: String) =
        send(JSONObject().put("type", "call").put("from", from).put("to", to).put("call_id", callId))

    fun accept(callId: String) =
        send(JSONObject().put("type", "accept").put("call_id", callId))

    fun reject(callId: String, reason: String = "declined") =
        send(JSONObject().put("type", "reject").put("call_id", callId).put("reason", reason))

    fun sendSdp(callId: String, kind: String, sdp: JSONObject) =
        send(JSONObject().put("type", "sdp").put("call_id", callId).put("kind", kind).put("sdp", sdp))

    fun sendIce(callId: String, candidate: JSONObject) =
        send(JSONObject().put("type", "ice").put("call_id", callId).put("candidate", candidate))

    fun hangup(callId: String) =
        send(JSONObject().put("type", "hangup").put("call_id", callId))

    private fun handleMessage(text: String) {
        val json = try {
            JSONObject(text)
        } catch (e: Exception) {
            Log.w(TAG, "bad signaling JSON: ${e.message}")
            return
        }
        val type = json.optString("type")
        val callId = json.optString("call_id")
        mainHandler.post {
            val l = listener ?: return@post
            when (type) {
                "registered" -> l.onRegistered(json.optString("user_id"))
                "presence" -> {
                    val arr = json.optJSONArray("users")
                    val users = mutableListOf<String>()
                    if (arr != null) {
                        for (i in 0 until arr.length()) users += arr.optString(i)
                    }
                    l.onPresence(users)
                }
                "incoming_call" -> l.onIncomingCall(json.optString("from"), callId)
                "accepted" -> l.onAccepted(callId)
                "rejected" -> l.onRejected(callId, json.optString("reason", "declined"))
                "sdp" -> json.optJSONObject("sdp")?.let { l.onRemoteSdp(callId, json.optString("kind"), it) }
                "ice" -> json.optJSONObject("candidate")?.let { l.onRemoteIce(callId, it) }
                "hangup" -> l.onHangup(callId)
                "error" -> l.onSignalingError(json.optString("code", "?"), json.optString("message", ""))
                else -> Log.w(TAG, "unknown signaling type: $type")
            }
        }
    }
}
