package com.vericall.voip

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.vericall.audio.VaniStreamingClient
import org.json.JSONObject
import org.webrtc.DataChannel
import org.webrtc.IceCandidate
import org.webrtc.MediaConstraints
import org.webrtc.MediaStream
import org.webrtc.PeerConnection
import org.webrtc.PeerConnectionFactory
import org.webrtc.RtpReceiver
import org.webrtc.SdpObserver
import org.webrtc.SessionDescription
import org.webrtc.audio.JavaAudioDeviceModule
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Owns one WebRTC voice call: signaling, peer connection, audio transport,
 * and the scoring stream.
 *
 * AUDIO TAP (why this shape): the WebRTC Android SDK no longer exposes the
 * remote AudioTrack's PCM to Java (AudioSink was removed upstream; verified
 * against the SDK classes). Instead this manager taps WebRTC's own
 * microphone capture via JavaAudioDeviceModule.SamplesReadyCallback, converts
 * it to 16 kHz mono int16, and forwards it to the peer over a reliable
 * WebRTC DataChannel ("voip-pcm"). Each side therefore scores the REMOTE
 * party's voice on its own device: incoming DataChannel PCM goes straight
 * into VaniStreamingClient.sendPcm(), which speaks the documented scorer
 * wire protocol (JSON handshake, then binary int16-LE PCM). Verdict frames
 * come back through VaniStreamingClient.onVerdictJson and are mapped to the
 * in-call banner by VoipVerdictPolicy. No second AudioRecord is opened, so
 * there is no mic contention with WebRTC.
 */
class VoipCallManager(
    private val appContext: Context,
    private val userId: String,
    private val signalingHost: String,
    private val scorerHost: String,
    private val turnUri: String?,
    private val turnUser: String?,
    private val turnPass: String?,
    private val listener: CallListener
) {

    enum class CallState { IDLE, READY, CALLING, RINGING, ACTIVE, ENDED }

    interface CallListener {
        fun onStateChanged(state: CallState, peer: String?, callId: String?)
        fun onPresenceUsers(users: List<String>)
        fun onBanner(banner: VoipBanner)
        fun onTestLog(line: String)
        fun onAudioStats(sentSamples: Long, receivedSamples: Long, rms: Double)
    }

    companion object {
        private const val TAG = "VoipCall"
        const val TEST_TAG = "VOIPTEST"
        private val factoryInit = AtomicBoolean(false)
        private const val PCM_FRAME_SAMPLES = 1600 // 100 ms @ 16 kHz
        private const val PCM_CHANNEL_LABEL = "voip-pcm"
    }

    private val mainHandler = Handler(Looper.getMainLooper())

    private var signaling: VoipSignalingClient? = null
    private var factory: PeerConnectionFactory? = null
    private var peerConnection: PeerConnection? = null
    private var pcmChannel: DataChannel? = null
    @Volatile private var pcmChannelOpen = false
    private var streamingClient: VaniStreamingClient? = null

    @Volatile private var state = CallState.IDLE
    @Volatile private var activeCallId: String? = null
    @Volatile private var activePeer: String? = null
    @Volatile private var isCaller = false
    @Volatile private var remoteDescSet = false
    private val pendingIce = mutableListOf<IceCandidate>()

    // Outgoing mic PCM accumulation (16 kHz mono) before DataChannel send.
    private val micLock = Any()
    private val micBuf = ShortArray(PCM_FRAME_SAMPLES)
    private var micPos = 0
    private var micSentSamples = 0L
    private var micSumSq = 0.0
    private var remoteReceivedSamples = 0L
    private var lastStatsLog = 0L

    /** Test hook: when set, this PCM source replaces the live mic. */
    @Volatile var testPcmFeeder: (() -> ShortArray?)? = null

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    fun start() {
        ensureFactory()
        val client = VoipSignalingClient("ws://$signalingHost:8001/ws/signal")
        client.listener = signalingListener
        client.onConnected = { client.register(userId) }
        signaling = client
        client.connect()
        testLog("signaling_connect host=$signalingHost user=$userId")
    }

    fun shutdown() {
        runCatching { signaling?.disconnect() }
        stopCallResources()
        factory = null
        setState(CallState.IDLE, null, null)
    }

    fun refreshPresence() {
        signaling?.presenceRequest()
    }

    // ------------------------------------------------------------------
    // Call control
    // ------------------------------------------------------------------

    fun placeCall(peer: String) {
        if (state != CallState.READY) {
            testLog("placeCall ignored state=$state")
            return
        }
        val callId = UUID.randomUUID().toString()
        activeCallId = callId
        activePeer = peer
        isCaller = true
        remoteDescSet = false
        pendingIce.clear()
        peerConnection = createPeerConnection(callId)
        // Caller creates the PCM data channel up front.
        pcmChannel = peerConnection?.createDataChannel(
            PCM_CHANNEL_LABEL, DataChannel.Init()
        )?.also { it.registerObserver(dataObserver) }
        setState(CallState.CALLING, peer, callId)
        signaling?.call(userId, peer, callId)
        testLog("call_placed to=$peer call=$callId")
    }

    fun acceptCall() {
        val callId = activeCallId ?: return
        if (state != CallState.RINGING) return
        signaling?.accept(callId)
        setState(CallState.ACTIVE, activePeer, callId)
        startScoring()
        testLog("call_accepted call=$callId")
    }

    fun rejectCall(reason: String = "declined") {
        val callId = activeCallId ?: return
        signaling?.reject(callId, reason)
        testLog("call_rejected call=$callId reason=$reason")
        endCallLocal("rejected")
    }

    fun hangup() {
        val callId = activeCallId
        if (callId != null && (state == CallState.ACTIVE || state == CallState.CALLING || state == CallState.RINGING)) {
            signaling?.hangup(callId)
            testLog("hangup_sent call=$callId")
        }
        endCallLocal("local-hangup")
    }

    private fun endCallLocal(reason: String) {
        testLog("call_ended reason=$reason")
        stopCallResources()
        setState(CallState.ENDED, null, null)
        mainHandler.postDelayed({
            if (state == CallState.ENDED) {
                activeCallId = null
                activePeer = null
                setState(CallState.READY, null, null)
            }
        }, 500)
    }

    private fun stopCallResources() {
        runCatching { streamingClient?.stop() }
        streamingClient = null
        runCatching { pcmChannel?.close() }
        pcmChannel = null
        pcmChannelOpen = false
        synchronized(micLock) { micPos = 0 }
        micSentSamples = 0L
        micSumSq = 0.0
        remoteReceivedSamples = 0L
        runCatching { peerConnection?.close() }
        peerConnection = null
    }

    // ------------------------------------------------------------------
    // Signaling events
    // ------------------------------------------------------------------

    private val signalingListener = object : VoipSignalingClient.Listener {
        override fun onRegistered(userId: String) {
            testLog("registered user=$userId")
            setState(CallState.READY, null, null)
        }

        override fun onPresence(users: List<String>) {
            testLog("presence users=${users.joinToString(",")}")
            listener.onPresenceUsers(users)
        }

        override fun onIncomingCall(from: String, callId: String) {
            if (state != CallState.READY) {
                signaling?.reject(callId, "busy")
                testLog("incoming_rejected_busy from=$from call=$callId")
                return
            }
            activeCallId = callId
            activePeer = from
            isCaller = false
            remoteDescSet = false
            pendingIce.clear()
            peerConnection = createPeerConnection(callId)
            // Callee waits for onDataChannel for the PCM channel.
            setState(CallState.RINGING, from, callId)
            testLog("incoming_call from=$from call=$callId")
        }

        override fun onAccepted(callId: String) {
            if (callId != activeCallId || !isCaller) return
            testLog("call_accepted_by_peer call=$callId")
            setState(CallState.ACTIVE, activePeer, callId)
            startScoring()
            peerConnection?.createOffer(object : SimpleSdpObserver() {
                override fun onCreateSuccess(desc: SessionDescription) {
                    peerConnection?.setLocalDescription(SimpleSdpObserver(), desc)
                    signaling?.sendSdp(
                        callId, "offer",
                        JSONObject().put("type", "offer").put("sdp", desc.description)
                    )
                    testLog("sdp_offer_sent call=$callId")
                }

                override fun onCreateFailure(err: String) {
                    testLog("sdp_offer_failed err=$err")
                    endCallLocal("offer-failed")
                }
            }, MediaConstraints())
        }

        override fun onRejected(callId: String, reason: String) {
            if (callId != activeCallId) return
            testLog("call_rejected_by_peer call=$callId reason=$reason")
            endCallLocal("rejected-by-peer")
        }

        override fun onRemoteSdp(callId: String, kind: String, sdp: JSONObject) {
            if (callId != activeCallId) return
            val type = if (kind == "offer") SessionDescription.Type.OFFER else SessionDescription.Type.ANSWER
            val desc = SessionDescription(type, sdp.optString("sdp"))
            testLog("sdp_remote_${kind}_received call=$callId")
            peerConnection?.setRemoteDescription(object : SimpleSdpObserver() {
                override fun onSetSuccess() {
                    remoteDescSet = true
                    drainPendingIce()
                    if (type == SessionDescription.Type.OFFER && !isCaller) {
                        peerConnection?.createAnswer(object : SimpleSdpObserver() {
                            override fun onCreateSuccess(answer: SessionDescription) {
                                peerConnection?.setLocalDescription(SimpleSdpObserver(), answer)
                                signaling?.sendSdp(
                                    callId, "answer",
                                    JSONObject().put("type", "answer").put("sdp", answer.description)
                                )
                                testLog("sdp_answer_sent call=$callId")
                            }

                            override fun onCreateFailure(err: String) {
                                testLog("sdp_answer_failed err=$err")
                                endCallLocal("answer-failed")
                            }
                        }, MediaConstraints())
                    }
                }

                override fun onSetFailure(err: String) {
                    testLog("sdp_remote_set_failed err=$err")
                    endCallLocal("remote-sdp-failed")
                }
            }, desc)
        }

        override fun onRemoteIce(callId: String, candidate: JSONObject) {
            if (callId != activeCallId) return
            val ice = IceCandidate(
                candidate.optString("sdpMid"),
                candidate.optInt("sdpMLineIndex"),
                candidate.optString("candidate")
            )
            if (remoteDescSet) {
                peerConnection?.addIceCandidate(ice)
            } else {
                synchronized(pendingIce) { pendingIce.add(ice) }
            }
        }

        override fun onHangup(callId: String) {
            if (callId != activeCallId) return
            testLog("hangup_received call=$callId")
            endCallLocal("remote-hangup")
        }

        override fun onSignalingError(code: String, message: String) {
            testLog("signaling_error code=$code msg=$message")
        }

        override fun onDisconnected() {
            testLog("signaling_disconnected")
            if (state == CallState.ACTIVE || state == CallState.CALLING || state == CallState.RINGING) {
                endCallLocal("signaling-lost")
            }
            setState(CallState.IDLE, null, null)
        }
    }

    private fun drainPendingIce() {
        synchronized(pendingIce) {
            for (ice in pendingIce) peerConnection?.addIceCandidate(ice)
            if (pendingIce.isNotEmpty()) testLog("ice_drained count=${pendingIce.size}")
            pendingIce.clear()
        }
    }

    // ------------------------------------------------------------------
    // Peer connection
    // ------------------------------------------------------------------

    private fun ensureFactory() {
        if (factoryInit.compareAndSet(false, true)) {
            PeerConnectionFactory.initialize(
                PeerConnectionFactory.InitializationOptions.builder(appContext)
                    .createInitializationOptions()
            )
            Log.i(TAG, "PeerConnectionFactory initialized")
        }
        if (factory == null) {
            val adm = JavaAudioDeviceModule.builder(appContext)
                .setSamplesReadyCallback(micTap)
                .createAudioDeviceModule()
            factory = PeerConnectionFactory.builder()
                .setAudioDeviceModule(adm)
                .createPeerConnectionFactory()
            Log.i(TAG, "PeerConnectionFactory created (audio-only, mic tap armed)")
        }
    }

    private fun createPeerConnection(callId: String): PeerConnection {
        val iceServers = mutableListOf<PeerConnection.IceServer>()
        if (!turnUri.isNullOrBlank()) {
            iceServers += PeerConnection.IceServer.builder(turnUri)
                .setUsername(turnUser ?: "")
                .setPassword(turnPass ?: "")
                .createIceServer()
            testLog("turn_configured uri=$turnUri")
        }
        val config = PeerConnection.RTCConfiguration(iceServers).apply {
            sdpSemantics = PeerConnection.SdpSemantics.UNIFIED_PLAN
            tcpCandidatePolicy = PeerConnection.TcpCandidatePolicy.ENABLED
            bundlePolicy = PeerConnection.BundlePolicy.MAXBUNDLE
            rtcpMuxPolicy = PeerConnection.RtcpMuxPolicy.REQUIRE
            continualGatheringPolicy = PeerConnection.ContinualGatheringPolicy.GATHER_CONTINUALLY
        }
        val pc = factory!!.createPeerConnection(config, pcObserver)
            ?: throw IllegalStateException("createPeerConnection returned null")
        val audioSource = factory!!.createAudioSource(MediaConstraints())
        val localAudioTrack = factory!!.createAudioTrack("vr-audio-$userId", audioSource)
        pc.addTrack(localAudioTrack, listOf("vr-stream-$callId"))
        testLog("peerconnection_created call=$callId")
        return pc
    }

    private val pcObserver = object : PeerConnection.Observer {
        override fun onIceCandidate(candidate: IceCandidate) {
            val callId = activeCallId ?: return
            signaling?.sendIce(
                callId,
                JSONObject()
                    .put("candidate", candidate.sdp)
                    .put("sdpMid", candidate.sdpMid)
                    .put("sdpMLineIndex", candidate.sdpMLineIndex)
            )
        }

        override fun onAddTrack(receiver: RtpReceiver, mediaStreams: Array<MediaStream>) {
            testLog("remote_track added kind=${receiver.track()?.kind()}")
        }

        override fun onAddStream(stream: MediaStream) {
            testLog("remote_stream added (legacy)")
        }

        override fun onDataChannel(channel: DataChannel) {
            testLog("datachannel_incoming label=${channel.label()}")
            if (channel.label() == PCM_CHANNEL_LABEL) {
                pcmChannel = channel
                channel.registerObserver(dataObserver)
            }
        }

        override fun onConnectionChange(newState: PeerConnection.PeerConnectionState) {
            testLog("pc_state=$newState")
            if (newState == PeerConnection.PeerConnectionState.FAILED) {
                testLog("pc_failed")
                mainHandler.post { endCallLocal("pc-failed") }
            }
        }

        override fun onIceConnectionChange(newState: PeerConnection.IceConnectionState) {
            testLog("ice_state=$newState")
        }

        override fun onSignalingChange(newState: PeerConnection.SignalingState) {}
        override fun onIceConnectionReceivingChange(receiving: Boolean) {}
        override fun onIceGatheringChange(newState: PeerConnection.IceGatheringState) {}
        override fun onIceCandidatesRemoved(candidates: Array<IceCandidate>) {}
        override fun onRemoveStream(stream: MediaStream) {}
        override fun onRenegotiationNeeded() {}
    }

    private val dataObserver = object : DataChannel.Observer {
        override fun onStateChange() {
            val open = pcmChannel?.state() == DataChannel.State.OPEN
            pcmChannelOpen = open
            testLog("datachannel_state open=$open")
        }

        override fun onMessage(buffer: DataChannel.Buffer) {
            if (!buffer.binary) return
            val bytes = ByteArray(buffer.data.remaining())
            buffer.data.get(bytes)
            if (bytes.size % 2 != 0) return
            val pcm = ShortArray(bytes.size / 2)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(pcm)
            remoteReceivedSamples += pcm.size
            // This PCM is the remote party's voice at 16 kHz mono: score it.
            streamingClient?.sendPcm(pcm)
        }

        override fun onBufferedAmountChange(previousAmount: Long) {}
    }

    private open class SimpleSdpObserver : SdpObserver {
        override fun onCreateSuccess(desc: SessionDescription) {}
        override fun onSetSuccess() {}
        override fun onCreateFailure(err: String) { Log.w(TAG, "sdp create failed: $err") }
        override fun onSetFailure(err: String) { Log.w(TAG, "sdp set failed: $err") }
    }

    // ------------------------------------------------------------------
    // Mic tap: WebRTC's own capture -> 16 kHz mono -> DataChannel -> peer
    // ------------------------------------------------------------------

    private val micTap = JavaAudioDeviceModule.SamplesReadyCallback { samples ->
        try {
            if (state != CallState.ACTIVE && state != CallState.CALLING && state != CallState.RINGING) return@SamplesReadyCallback
            val data = samples.data ?: return@SamplesReadyCallback
            if (data.isEmpty()) return@SamplesReadyCallback
            // Test hook: file-based PCM replaces the live mic when set.
            val feeder = testPcmFeeder
            val mono16k: ShortArray = if (feeder != null) {
                feeder.invoke() ?: return@SamplesReadyCallback
            } else {
                val shorts = ShortArray(data.size / 2)
                ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts)
                resampleTo16kMono(shorts, samples.sampleRate, samples.channelCount)
            }
            if (mono16k.isEmpty()) return@SamplesReadyCallback

            var sum = 0.0
            for (s in mono16k) sum += (s * s).toDouble()
            micSentSamples += mono16k.size
            micSumSq += sum

            // Frame into 100 ms chunks and forward to the peer.
            synchronized(micLock) {
                var offset = 0
                while (offset < mono16k.size) {
                    val room = micBuf.size - micPos
                    val take = minOf(room, mono16k.size - offset)
                    System.arraycopy(mono16k, offset, micBuf, micPos, take)
                    micPos += take
                    offset += take
                    if (micPos >= micBuf.size) {
                        micPos = 0
                        sendPcmFrame(micBuf.copyOf())
                    }
                }
            }

            val now = SystemClock.elapsedRealtime()
            if (now - lastStatsLog > 5000) {
                lastStatsLog = now
                val rms = sqrt(micSumSq / max(1L, micSentSamples).toDouble())
                val sent = micSentSamples
                val recv = remoteReceivedSamples
                mainHandler.post {
                    listener.onAudioStats(sent, recv, rms)
                    testLog(
                        "audio_stats mic_sent=%ds remote_recv=%ds mic_rms=%.1f dc_open=%b".format(
                            sent / 16000, recv / 16000, rms, pcmChannelOpen
                        )
                    )
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "mic tap error: ${e.message}")
        }
    }

    private fun sendPcmFrame(frame: ShortArray) {
        val ch = pcmChannel
        if (!pcmChannelOpen || ch == null) return
        if (state != CallState.ACTIVE) return
        val bytes = ByteArray(frame.size * 2)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(frame)
        runCatching { ch.send(DataChannel.Buffer(ByteBuffer.wrap(bytes), true)) }
    }

    /**
     * Downmix to mono and resample to 16 kHz int16.
     * 48 kHz -> boxcar-of-3 decimation (its nulls sit at 16/32 kHz, killing
     * the fold-back); 16 kHz passes through; anything else uses linear
     * interpolation.
     */
    private fun resampleTo16kMono(input: ShortArray, rate: Int, channels: Int): ShortArray {
        val mono: ShortArray = if (channels <= 1) {
            input
        } else {
            val frames = input.size / channels
            ShortArray(frames) { i ->
                var acc = 0
                for (c in 0 until channels) acc += input[i * channels + c].toInt()
                (acc / channels).toShort()
            }
        }
        return when (rate) {
            16000 -> mono
            48000 -> {
                val outLen = mono.size / 3
                val out = ShortArray(outLen)
                var j = 0
                var i = 0
                while (i + 2 < mono.size && j < outLen) {
                    out[j++] = ((mono[i].toInt() + mono[i + 1].toInt() + mono[i + 2].toInt()) / 3).toShort()
                    i += 3
                }
                if (j < outLen) out.copyOf(j) else out
            }
            else -> {
                if (rate <= 0) return ShortArray(0)
                val ratio = rate / 16000.0
                val outLen = (mono.size / ratio).toInt()
                ShortArray(outLen) { j ->
                    val pos = j * ratio
                    val i0 = pos.toInt().coerceIn(0, mono.size - 1)
                    val i1 = (i0 + 1).coerceIn(0, mono.size - 1)
                    val frac = pos - i0
                    (mono[i0] * (1 - frac) + mono[i1] * frac).toInt().toShort()
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Scoring (remote party's voice, computed on this device)
    // ------------------------------------------------------------------

    private fun startScoring() {
        val peer = activePeer ?: "unknown"
        val client = VaniStreamingClient(
            wsUrl = "ws://$scorerHost:8000/ws/stream",
            callerId = "voip:$peer"
        )
        client.onVerdictJson = { json -> handleVerdictJson(json) }
        client.onExhausted = {
            mainHandler.post {
                listener.onBanner(
                    VoipBanner(VoipBannerLevel.NEUTRAL, "Scoring unavailable", "backend unreachable")
                )
                testLog("scoring_exhausted")
            }
        }
        streamingClient = client
        client.start()
        testLog("scoring_started peer=$peer")
    }

    private fun handleVerdictJson(json: JSONObject) {
        val pSpoof = json.optDouble("p_spoof", Double.NaN)
        if (pSpoof.isNaN()) return
        val tier = json.optString("tier", null)
        val tierOverride = json.optJSONObject("metrics")?.optString("tier_override", null)
        val banner = VoipVerdictPolicy.decide(pSpoof, tier, tierOverride)
        mainHandler.post {
            listener.onBanner(banner)
            testLog(
                "verdict p_spoof=%.4f tier=%s override=%s banner=%s".format(pSpoof, tier, tierOverride, banner.level)
            )
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private fun setState(state: CallState, peer: String?, callId: String?) {
        this.state = state
        mainHandler.post {
            listener.onStateChanged(state, peer, callId)
            testLog("state=$state peer=${peer ?: "-"} call=${callId ?: "-"}")
        }
    }

    private fun testLog(line: String) {
        Log.i(TEST_TAG, line)
        mainHandler.post { listener.onTestLog(line) }
    }
}
