package com.vericall.voip

import android.Manifest
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

/**
 * In-app internet calling screen for VaniRakshak Call (Layer 1).
 *
 * Register with a user id, see who is online, place a WebRTC voice call,
 * accept or reject incoming calls, and watch the live deepfake verdict
 * banner while the remote audio is scored.
 *
 * Headless test hooks (intent extras, for emulator verification):
 *   voip_user_id        auto-register as this user id
 *   voip_backend_host   override backend host (default 127.0.0.1; use
 *                       10.0.2.2 on an emulator to reach the host)
 *   voip_auto_call      peer user id to call automatically once ready
 *   voip_auto_accept    "1" to auto-accept incoming calls after 1 s
 *   voip_auto_hangup_ms hang up automatically N ms after the call goes active
 *   voip_turn_uri / voip_turn_user / voip_turn_pass  TURN override
 */
class VoipCallActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_USER_ID = "voip_user_id"
        const val EXTRA_BACKEND_HOST = "voip_backend_host"
        const val EXTRA_AUTO_CALL = "voip_auto_call"
        const val EXTRA_AUTO_ACCEPT = "voip_auto_accept"
        const val EXTRA_AUTO_HANGUP_MS = "voip_auto_hangup_ms"
        const val EXTRA_TURN_URI = "voip_turn_uri"
        const val EXTRA_TURN_USER = "voip_turn_user"
        const val EXTRA_TURN_PASS = "voip_turn_pass"
        const val EXTRA_TEST_AUDIO = "voip_test_audio"
        private const val TAG = "VoipCallActivity"
        private const val REQ_AUDIO = 1001
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private var manager: VoipCallManager? = null

    private lateinit var statusView: TextView
    private lateinit var userIdInput: EditText
    private lateinit var registerButton: Button
    private lateinit var presenceList: ListView
    private lateinit var refreshButton: Button
    private lateinit var callButton: Button
    private lateinit var bannerView: TextView
    private lateinit var bannerDetail: TextView
    private lateinit var hangupButton: Button
    private lateinit var logView: TextView
    private lateinit var statsView: TextView

    private val presenceAdapter by lazy {
        ArrayAdapter<String>(this, android.R.layout.simple_list_item_single_choice, mutableListOf())
    }
    private var selectedPeer: String? = null
    private var selfId: String = ""
    private var autoAccept = false
    private var incomingDialog: AlertDialog? = null
    private val logLines = ArrayDeque<String>(60)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()

        val extras = intent.extras
        val backendHost = extras?.getString(EXTRA_BACKEND_HOST) ?: "127.0.0.1"
        autoAccept = extras?.getString(EXTRA_AUTO_ACCEPT) == "1"
        val autoCall = extras?.getString(EXTRA_AUTO_CALL)
        val autoHangupMs = extras?.getString(EXTRA_AUTO_HANGUP_MS)?.toLongOrNull() ?: 0L
        val presetUser = extras?.getString(EXTRA_USER_ID)
        if (!presetUser.isNullOrBlank()) userIdInput.setText(presetUser)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQ_AUDIO)
        }

        manager = VoipCallManager(
            appContext = applicationContext,
            userId = "",
            signalingHost = backendHost,
            scorerHost = backendHost,
            turnUri = extras?.getString(EXTRA_TURN_URI),
            turnUser = extras?.getString(EXTRA_TURN_USER),
            turnPass = extras?.getString(EXTRA_TURN_PASS),
            listener = callListener
        )

        registerButton.setOnClickListener { registerFromInput(backendHost) }
        refreshButton.setOnClickListener { manager?.refreshPresence() }
        callButton.setOnClickListener {
            selectedPeer?.let { manager?.placeCall(it) }
        }
        hangupButton.setOnClickListener { manager?.hangup() }
        presenceList.setOnItemClickListener { _, _, position, _ ->
            selectedPeer = presenceAdapter.getItem(position)
            appendLog("selected peer=$selectedPeer")
        }

        // Auto-register when a user id was supplied via intent.
        if (!presetUser.isNullOrBlank()) {
            mainHandler.postDelayed({ registerFromInput(backendHost) }, 800)
        }
        if (!autoCall.isNullOrBlank()) {
            pendingAutoCall = autoCall
        }
        if (autoHangupMs > 0) {
            pendingAutoHangupMs = autoHangupMs
        }
        pendingTestAudio = extras?.getString(EXTRA_TEST_AUDIO)
    }

    private var pendingAutoCall: String? = null
    private var pendingAutoHangupMs: Long = 0L
    private var pendingTestAudio: String? = null

    private fun registerFromInput(backendHost: String) {
        val id = userIdInput.text.toString().trim()
        if (id.isEmpty()) {
            statusView.text = "Enter a user id first"
            return
        }
        selfId = id
        // Recreate the manager with the real user id, then start.
        manager?.shutdown()
        val extras = intent.extras
        manager = VoipCallManager(
            appContext = applicationContext,
            userId = id,
            signalingHost = backendHost,
            scorerHost = backendHost,
            turnUri = extras?.getString(EXTRA_TURN_URI),
            turnUser = extras?.getString(EXTRA_TURN_USER),
            turnPass = extras?.getString(EXTRA_TURN_PASS),
            listener = callListener
        )
        statusView.text = "Registering as $id ..."
        armTestAudio()
        manager?.start()
    }

    private val callListener = object : VoipCallManager.CallListener {
        override fun onStateChanged(state: VoipCallManager.CallState, peer: String?, callId: String?) {
            statusView.text = when (state) {
                VoipCallManager.CallState.IDLE -> "Idle"
                VoipCallManager.CallState.READY -> "Online as $selfId"
                VoipCallManager.CallState.CALLING -> "Calling $peer ..."
                VoipCallManager.CallState.RINGING -> "Incoming call from $peer"
                VoipCallManager.CallState.ACTIVE -> "In call with $peer"
                VoipCallManager.CallState.ENDED -> "Call ended"
            }
            hangupButton.visibility =
                if (state == VoipCallManager.CallState.ACTIVE ||
                    state == VoipCallManager.CallState.CALLING
                ) View.VISIBLE else View.GONE

            if (state == VoipCallManager.CallState.READY) {
                manager?.refreshPresence()
                pendingAutoCall?.let { peerId ->
                    pendingAutoCall = null
                    mainHandler.postDelayed({
                        appendLog("auto_call -> $peerId")
                        manager?.placeCall(peerId)
                    }, 1500)
                }
            }
            if (state == VoipCallManager.CallState.ACTIVE && pendingAutoHangupMs > 0) {
                val ms = pendingAutoHangupMs
                pendingAutoHangupMs = 0L
                mainHandler.postDelayed({
                    appendLog("auto_hangup after ${ms}ms")
                    manager?.hangup()
                }, ms)
            }
        }

        override fun onPresenceUsers(users: List<String>) {
            val others = users.filter { it != selfId }
            presenceAdapter.clear()
            presenceAdapter.addAll(others)
            presenceAdapter.notifyDataSetChanged()
        }

        override fun onBanner(banner: VoipBanner) {
            bannerView.text = banner.title
            bannerDetail.text = banner.detail
            val (bg, fg) = when (banner.level) {
                VoipBannerLevel.RED -> Color.parseColor("#B3261E") to Color.WHITE
                VoipBannerLevel.AMBER -> Color.parseColor("#E8A33D") to Color.BLACK
                VoipBannerLevel.NEUTRAL -> Color.parseColor("#616161") to Color.WHITE
            }
            bannerView.setBackgroundColor(bg)
            bannerView.setTextColor(fg)
            bannerView.visibility = View.VISIBLE
        }

        override fun onTestLog(line: String) {
            appendLog(line)
            if (line.startsWith("incoming_call") || line.startsWith("state=RINGING")) {
                showIncomingUi()
            }
        }

        override fun onAudioStats(sentSamples: Long, receivedSamples: Long, rms: Double) {
            statsView.text = "mic sent: ${sentSamples / 16000}s, remote scored: ${receivedSamples / 16000}s, rms=%.1f".format(rms)
        }
    }

    private fun showIncomingUi() {
        val peer = selectedPeer
        if (autoAccept) {
            mainHandler.postDelayed({
                appendLog("auto_accept")
                manager?.acceptCall()
            }, 1000)
            return
        }
        if (incomingDialog?.isShowing == true) return
        runOnUiThread {
            incomingDialog = AlertDialog.Builder(this)
                .setTitle("Incoming internet call")
                .setMessage("Accept the call?")
                .setPositiveButton("Accept") { _, _ -> manager?.acceptCall() }
                .setNegativeButton("Reject") { _, _ -> manager?.rejectCall() }
                .setCancelable(false)
                .show()
        }
        Log.i(VoipCallManager.TEST_TAG, "incoming_ui peer=$peer")
    }

    private fun appendLog(line: String) {
        mainHandler.post {
            logLines.addLast(line)
            while (logLines.size > 60) logLines.removeFirst()
            logView.text = logLines.joinToString("\n")
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_AUDIO) {
            val granted = grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED
            appendLog("mic_permission granted=$granted")
        }
    }

    override fun onDestroy() {
        incomingDialog?.dismiss()
        manager?.shutdown()
        manager = null
        super.onDestroy()
    }

    /**
     * Headless-test hook: when voip_test_audio=tts, synthesize a spoken
     * sentence and loop it as this side's microphone PCM, so verification
     * scores real speech instead of a silent virtual mic. When
     * voip_test_audio=file:<path>, loop that WAV file instead (used to feed
     * genuine recorded speech for the neutral-banner path).
     */
    private fun armTestAudio() {
        val mode = pendingTestAudio ?: return
        pendingTestAudio = null
        if (mode == "tts") {
            armTtsTestAudio()
            return
        }
        if (mode.startsWith("file:")) {
            val wavPath = mode.removePrefix("file:")
            appendLog("test_audio: loading wav $wavPath ...")
            val pcm = VoipTestAudio.wavToPcm16kMono(java.io.File(wavPath))
            if (pcm == null || pcm.isEmpty()) {
                appendLog("test_audio: wav parse failed, using live mic")
                return
            }
            manager?.testPcmFeeder = VoipTestAudio.loopingFeeder(pcm)
            appendLog("test_audio: looping ${pcm.size / 16000}s of wav speech as mic")
            Log.i(VoipCallManager.TEST_TAG, "test_audio file_ready seconds=${pcm.size / 16000}")
        }
    }

    private fun armTtsTestAudio() {
        appendLog("test_audio: synthesizing TTS ...")
        VoipTestAudio.synthesizeTtsWav(this) { file ->
            if (file == null) {
                appendLog("test_audio: TTS failed, using live mic")
                Log.i(VoipCallManager.TEST_TAG, "test_audio tts_failed")
                return@synthesizeTtsWav
            }
            val pcm = VoipTestAudio.wavToPcm16kMono(file)
            if (pcm == null || pcm.isEmpty()) {
                appendLog("test_audio: wav parse failed, using live mic")
                return@synthesizeTtsWav
            }
            manager?.testPcmFeeder = VoipTestAudio.loopingFeeder(pcm)
            appendLog("test_audio: looping ${pcm.size / 16000}s of TTS speech as mic")
            Log.i(VoipCallManager.TEST_TAG, "test_audio tts_ready seconds=${pcm.size / 16000}")
        }
    }

    // ------------------------------------------------------------------
    // Programmatic UI (keeps the feature self-contained in one file)
    // ------------------------------------------------------------------

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }

        val title = TextView(this).apply {
            text = "VaniRakshak Call (internet)"
            textSize = 20f
        }
        root.addView(title)

        statusView = TextView(this).apply { text = "Idle"; textSize = 16f }
        root.addView(statusView)

        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        userIdInput = EditText(this).apply {
            hint = "your user id"
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        registerButton = Button(this).apply { text = "Register" }
        row.addView(userIdInput)
        row.addView(registerButton)
        root.addView(row)

        val presRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val presLabel = TextView(this).apply {
            text = "Online users"
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        refreshButton = Button(this).apply { text = "Refresh" }
        presRow.addView(presLabel)
        presRow.addView(refreshButton)
        root.addView(presRow)

        presenceList = ListView(this).apply {
            adapter = presenceAdapter
            choiceMode = ListView.CHOICE_MODE_SINGLE
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 300
            )
        }
        root.addView(presenceList)

        val callRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        callButton = Button(this).apply {
            text = "Call selected"
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        hangupButton = Button(this).apply {
            text = "Hang up"
            visibility = View.GONE
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        callRow.addView(callButton)
        callRow.addView(hangupButton)
        root.addView(callRow)

        bannerView = TextView(this).apply {
            text = ""
            textSize = 18f
            setPadding(24, 24, 24, 24)
            visibility = View.GONE
        }
        root.addView(bannerView)

        bannerDetail = TextView(this).apply { textSize = 12f }
        root.addView(bannerDetail)

        statsView = TextView(this).apply { textSize = 12f }
        root.addView(statsView)

        logView = TextView(this).apply { textSize = 11f }
        val scroll = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
            addView(logView)
        }
        root.addView(scroll)

        setContentView(root)
    }
}
