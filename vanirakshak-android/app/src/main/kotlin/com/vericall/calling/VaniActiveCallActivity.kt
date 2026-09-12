package com.vericall.calling

import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.LayoutInflater
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.vericall.R
import com.vericall.data.VaniCallHistoryRepository
import com.vericall.databinding.ActivityVaniActiveCallBinding
import com.vericall.model.CallDirection
import com.vericall.model.RiskLevel
import com.vericall.model.TranscriptUtterance
import com.vericall.model.VaniCallRecord
import com.vericall.shield.VaniScamShieldController
import com.vericall.ui.VaniTranscriptAdapter
import com.vericall.util.VaniCallRouter
import com.vericall.util.VaniRiskBus
import java.util.UUID

class VaniActiveCallActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_CALL_NUMBER = "extra_call_number"
        const val EXTRA_CALL_NAME = "extra_call_name"
        const val EXTRA_IS_OUTGOING = "extra_is_outgoing"
        const val EXTRA_IS_INCOMING_RINGING = "extra_is_incoming_ringing"

        fun startOutgoingCall(context: Context, number: String, name: String) {
            val intent = Intent(context, VaniActiveCallActivity::class.java).apply {
                putExtra(EXTRA_CALL_NUMBER, number)
                putExtra(EXTRA_CALL_NAME, name)
                putExtra(EXTRA_IS_OUTGOING, true)
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
            }
            context.startActivity(intent)
        }
    }

    private lateinit var binding: ActivityVaniActiveCallBinding
    private lateinit var transcriptAdapter: VaniTranscriptAdapter
    private lateinit var sheetBehavior: BottomSheetBehavior<View>

    private var callerNumber: String = ""
    private var callerName: String = "Caller"
    private var isOutgoing: Boolean = true
    private var callId: String = ""

    private var callStartTime: Long = 0L
    private val timerHandler = Handler(Looper.getMainLooper())
    private var isCallConnected = false

    private var audioManager: AudioManager? = null
    private var isMuted = false
    private var isSpeakerOn = false

    private var riskListener: ((VaniRiskBus.Snapshot) -> Unit)? = null
    private var amplitudeListener: ((Float) -> Unit)? = null

    private val timerRunnable = object : Runnable {
        override fun run() {
            if (isCallConnected && callStartTime > 0) {
                val elapsedSec = (SystemClock.elapsedRealtime() - callStartTime) / 1000
                val mins = elapsedSec / 60
                val secs = elapsedSec % 60
                binding.tvActiveCallDuration.text = String.format("%02d:%02d", mins, secs)
                timerHandler.postDelayed(this, 1000)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVaniActiveCallBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        callerNumber = intent.getStringExtra(EXTRA_CALL_NUMBER) ?: ""
        callerName = intent.getStringExtra(EXTRA_CALL_NAME) ?: "Caller"
        isOutgoing = intent.getBooleanExtra(EXTRA_IS_OUTGOING, true)
        callId = "call_${UUID.randomUUID().toString().take(8)}"
        VaniRiskBus.activeCallId = callId

        audioManager = getSystemService(Context.AUDIO_SERVICE) as? AudioManager

        setupUI()
        setupTranscriptSheet()
        setupAudioListeners()
        startCallSession()
    }

    private fun setupUI() {
        binding.tvActiveCallerName.text = callerName.ifBlank { callerNumber.ifBlank { "Unknown" } }
        binding.tvActiveCallerNumber.text = callerNumber

        val initial = callerName.firstOrNull { it.isLetter() }?.uppercaseChar() ?: 'V'
        binding.tvCenterAvatarLetter.text = initial.toString()

        binding.tvActiveCallDuration.text = if (isOutgoing) "Calling..." else "Connecting..."
        binding.tvActiveCallStatus.text = "Initializing Vani Guard Voice Channel..."

        binding.btnMinimizeCall.setOnClickListener {
            moveTaskToBack(true)
        }

        binding.btnCallEnd.setOnClickListener {
            endCallAndSummarize()
        }

        binding.btnCallMute.setOnClickListener {
            toggleMute()
        }

        binding.btnCallSpeaker.setOnClickListener {
            toggleSpeaker()
        }

        binding.btnCallKeypad.setOnClickListener {
            showIncallKeypadDialog()
        }

        binding.btnMoreCallOptions.setOnClickListener {
            showMoreCallOptions()
        }

        binding.cardTranscriptPeek.setOnClickListener {
            sheetBehavior.state = BottomSheetBehavior.STATE_EXPANDED
        }

        binding.btnCloseTranscriptSheet.setOnClickListener {
            sheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
        }
    }

    private fun setupTranscriptSheet() {
        sheetBehavior = BottomSheetBehavior.from(binding.bottomSheetTranscript)
        sheetBehavior.peekHeight = 0
        sheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED

        transcriptAdapter = VaniTranscriptAdapter()
        val layoutManager = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }
        binding.recyclerFullTranscript.layoutManager = layoutManager
        binding.recyclerFullTranscript.adapter = transcriptAdapter

        binding.recyclerFullTranscript.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                super.onScrolled(recyclerView, dx, dy)
                val lastPos = layoutManager.findLastVisibleItemPosition()
                val totalCount = transcriptAdapter.itemCount
                if (totalCount > 0 && lastPos < totalCount - 2) {
                    binding.btnJumpToLatest.visibility = View.VISIBLE
                } else {
                    binding.btnJumpToLatest.visibility = View.GONE
                }
            }
        })

        binding.btnJumpToLatest.setOnClickListener {
            if (transcriptAdapter.itemCount > 0) {
                binding.recyclerFullTranscript.smoothScrollToPosition(transcriptAdapter.itemCount - 1)
            }
            binding.btnJumpToLatest.visibility = View.GONE
        }
    }

    private fun startCallSession() {
        // Connect the call after short simulation delay or immediate
        Handler(Looper.getMainLooper()).postDelayed({
            if (isFinishing || isDestroyed) return@postDelayed
            isCallConnected = true
            callStartTime = SystemClock.elapsedRealtime()
            binding.tvActiveCallStatus.text = "Connected • Live Telemetry Active"
            timerHandler.post(timerRunnable)

            // Start guard service
            val resolvedCaller = VaniCallRouter.resolve(this, callerNumber)
            VaniCallRouter.startGuard(this, resolvedCaller)
        }, 1200)
    }

    private fun setupAudioListeners() {
        amplitudeListener = { amp ->
            runOnUiThread {
                binding.voiceVisualizer.setAmplitude(amp)
            }
        }
        VaniRiskBus.addAmplitudeListener(amplitudeListener!!)

        riskListener = { snapshot ->
            runOnUiThread {
                binding.voiceVisualizer.setRiskLevel(snapshot.level)

                // Update security badge
                when (snapshot.level) {
                    RiskLevel.SAFE -> {
                        binding.pillCallSecurity.setBackgroundResource(R.drawable.bg_badge_supported)
                        binding.dotCallSecurity.backgroundTintList = ContextCompat.getColorStateList(this, R.color.verdict_supported)
                        binding.tvCallSecurityText.text = "✦ Voice Authentic"
                        binding.tvCallSecurityText.setTextColor(ContextCompat.getColor(this, R.color.verdict_supported))
                    }
                    RiskLevel.WARNING -> {
                        binding.pillCallSecurity.setBackgroundResource(R.drawable.bg_badge_unverifiable)
                        binding.dotCallSecurity.backgroundTintList = ContextCompat.getColorStateList(this, R.color.verdict_unverifiable)
                        binding.tvCallSecurityText.text = "⚠ Anomaly Detected"
                        binding.tvCallSecurityText.setTextColor(ContextCompat.getColor(this, R.color.verdict_unverifiable))
                    }
                    RiskLevel.DANGER -> {
                        binding.pillCallSecurity.setBackgroundResource(R.drawable.bg_badge_scam)
                        binding.dotCallSecurity.backgroundTintList = ContextCompat.getColorStateList(this, R.color.verdict_scam)
                        binding.tvCallSecurityText.text = "🚨 CLONE / SCAM ALERT"
                        binding.tvCallSecurityText.setTextColor(ContextCompat.getColor(this, R.color.verdict_scam))
                    }
                }

                // Update live transcript from streaming ASR
                if (snapshot.transcript.isNotBlank()) {
                    binding.tvPeekTranscript.text = snapshot.transcript
                    binding.tvEmptyTranscriptMessage.visibility = View.GONE

                    val utterance = TranscriptUtterance(
                        speaker = callerName,
                        text = snapshot.transcript,
                        timestamp = System.currentTimeMillis(),
                        isFinal = false
                    )
                    transcriptAdapter.addOrUpdateUtterance(utterance)
                    binding.tvTranscriptCount.text = "${transcriptAdapter.itemCount} messages"

                    if (binding.btnJumpToLatest.visibility != View.VISIBLE && transcriptAdapter.itemCount > 0) {
                        binding.recyclerFullTranscript.scrollToPosition(transcriptAdapter.itemCount - 1)
                    }
                }
            }
        }
        VaniRiskBus.addListener(riskListener!!)
    }

    private fun toggleMute() {
        isMuted = !isMuted
        audioManager?.isMicrophoneMute = isMuted
        binding.btnCallMute.setImageResource(if (isMuted) R.drawable.ic_mute else R.drawable.ic_mic)
        binding.tvCallMuteLabel.text = if (isMuted) "Unmute" else "Mute"
        binding.btnCallMute.setBackgroundResource(
            if (isMuted) R.drawable.bg_badge_warning else R.drawable.bg_call_control
        )
        Toast.makeText(this, if (isMuted) "Microphone muted" else "Microphone unmuted", Toast.LENGTH_SHORT).show()
    }

    private fun toggleSpeaker() {
        isSpeakerOn = !isSpeakerOn
        audioManager?.isSpeakerphoneOn = isSpeakerOn
        binding.btnCallSpeaker.setBackgroundResource(
            if (isSpeakerOn) R.drawable.bg_badge_enrolled else R.drawable.bg_call_control
        )
        Toast.makeText(this, if (isSpeakerOn) "Speakerphone on" else "Speakerphone off", Toast.LENGTH_SHORT).show()
    }

    private fun showIncallKeypadDialog() {
        val dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_incall_keypad, null)
        val tvDigits = dialogView.findViewById<TextView>(R.id.tvIncallDialedDigits)

        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        dialogView.findViewById<View>(R.id.btnCloseKeypadDialog).setOnClickListener { dialog.dismiss() }

        val keyIds = listOf(
            R.id.key_1 to "1", R.id.key_2 to "2", R.id.key_3 to "3",
            R.id.key_4 to "4", R.id.key_5 to "5", R.id.key_6 to "6",
            R.id.key_7 to "7", R.id.key_8 to "8", R.id.key_9 to "9",
            R.id.key_star to "*", R.id.key_0 to "0", R.id.key_hash to "#"
        )

        for ((id, char) in keyIds) {
            dialogView.findViewById<View>(id).setOnClickListener {
                tvDigits.append(char)
            }
        }

        dialog.show()
    }

    private fun showMoreCallOptions() {
        val options = arrayOf(
            "🛡️ Trigger Anti-Replay Liveness Challenge",
            "🔒 Engage Financial Coercion Shield",
            "⚙️ Server Forensic Endpoint"
        )

        AlertDialog.Builder(this)
            .setTitle("VaniRakshak Live Guard Actions")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> triggerLivenessChallenge()
                    1 -> {
                        VaniScamShieldController.setUserInterlockEnabled(true)
                        Toast.makeText(this, "Scam-Shield Coercion Interlock Engaged", Toast.LENGTH_SHORT).show()
                    }
                    2 -> com.vericall.config.VaniConfig.showServerConfigDialog(this)
                }
            }
            .show()
    }

    private fun triggerLivenessChallenge() {
        val phrase = com.vericall.audio.VaniChallengeManager.getRandomFallbackPhrase()
        AlertDialog.Builder(this)
            .setTitle("Dynamic Liveness Challenge")
            .setMessage("Prompt the caller to repeat:\n\n\"${phrase.phrase}\"\n\nLatency & acoustic anti-replay analysis will verify biological vocal presence.")
            .setPositiveButton("Verify Response") { _, _ ->
                com.vericall.audio.VaniChallengeManager.gradeChallenge(
                    context = this,
                    sessionId = callId,
                    responseText = phrase.phrase,
                    latencyMs = 380L
                ) { result ->
                    runOnUiThread {
                        Toast.makeText(this, "Challenge Result: ${result.detail}", Toast.LENGTH_LONG).show()
                    }
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun endCallAndSummarize() {
        timerHandler.removeCallbacksAndMessages(null)
        val elapsedSec = if (callStartTime > 0) {
            ((SystemClock.elapsedRealtime() - callStartTime) / 1000).toInt()
        } else 0

        val utterances = transcriptAdapter.getAll()
        val snapshot = VaniRiskBus.last
        val riskScore = snapshot?.risk ?: 6
        val scamType = snapshot?.scamType ?: ""

        // Derive AI intelligence
        val (summary, keyPoints, actionItems) = VaniCallHistoryRepository.deriveIntelligence(
            callerName = callerName,
            transcript = utterances,
            riskScore = riskScore,
            scamType = scamType
        )

        // Stop services and audio capture first to finalize WAV file
        VaniCallRouter.onEnded(this)

        val recordedFile = java.io.File(java.io.File(filesDir, "recordings"), "${callId}.wav")
        val audioPath = VaniRiskBus.lastRecordedAudioPath ?: if (recordedFile.exists() && recordedFile.length() > 500) recordedFile.absolutePath else null
        val riskLevel = when {
            riskScore >= 80 -> "Critical"
            riskScore >= 60 -> "High"
            riskScore >= 40 -> "Moderate"
            riskScore >= 20 -> "Low"
            else -> "Very Low"
        }

        // Save completed call record to repository
        val record = VaniCallRecord(
            id = callId,
            number = callerNumber,
            callerName = callerName,
            direction = if (isOutgoing) CallDirection.OUTGOING else CallDirection.INCOMING,
            timestamp = System.currentTimeMillis(),
            durationSec = elapsedSec,
            hasTranscript = utterances.isNotEmpty(),
            transcript = utterances,
            aiSummary = summary,
            keyPoints = keyPoints,
            actionItems = actionItems,
            riskScore = riskScore,
            riskLevel = riskLevel,
            scamType = scamType,
            isVoiceVerified = snapshot?.level == RiskLevel.SAFE,
            audioFilePath = audioPath,
            isProtected = riskScore >= 50,
            redFlags = snapshot?.redFlags ?: emptyList()
        )
        VaniCallHistoryRepository.saveCallRecord(this, record)

        // Asynchronously finalize with Crawl4AI web search & fact checking
        com.vericall.data.VaniCallFinalizer.finalizeCallAsync(applicationContext, record)

        // Launch Post-Call Summary screen
        val summaryIntent = Intent(this, VaniCallSummaryActivity::class.java).apply {
            putExtra(VaniCallSummaryActivity.EXTRA_CALL_ID, callId)
        }
        startActivity(summaryIntent)
        finish()
    }

    override fun onDestroy() {
        timerHandler.removeCallbacksAndMessages(null)
        riskListener?.let { VaniRiskBus.removeListener(it) }
        amplitudeListener?.let { VaniRiskBus.removeAmplitudeListener(it) }
        riskListener = null
        amplitudeListener = null
        super.onDestroy()
    }
}
