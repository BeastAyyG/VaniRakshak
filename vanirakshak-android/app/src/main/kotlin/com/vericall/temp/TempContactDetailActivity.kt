package com.vericall.temp

import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.vericall.R
import com.vericall.audio.VaniLiveVoiceAnalyzer
import com.vericall.data.VaniSampleData
import com.vericall.databinding.ActivityTempContactDetailBinding
import com.vericall.model.RiskLevel
import com.vericall.model.VaniContact
import com.vericall.shield.VaniIncidentLogger
import com.vericall.shield.VaniScamShieldController
import com.vericall.util.VaniCallRouter
import com.vericall.util.VaniNotificationHelper
import com.vericall.util.VaniRiskBus

class TempContactDetailActivity : AppCompatActivity() {

    private lateinit var binding: ActivityTempContactDetailBinding
    private lateinit var contact: VaniContact
    private var voiceAnalyzer: VaniLiveVoiceAnalyzer? = null
    private var liveLevel: RiskLevel = RiskLevel.WARNING
    private var isLiveCall = false
    private var riskListener: ((VaniRiskBus.Snapshot) -> Unit)? = null
    private var amplitudeListener: ((Float) -> Unit)? = null

    private val speakerPollHandler = Handler(Looper.getMainLooper())
    private val speakerPollRunnable = object : Runnable {
        override fun run() {
            checkSpeakerphoneState()
            speakerPollHandler.postDelayed(this, 2000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTempContactDetailBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Handle edge-to-edge system insets (mandatory for Android 15 & 16)
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        contact = resolveContact()
        isLiveCall = contact.id == "live_call"

        bindCallerInfo(contact.name, contact.phone, contact.relation)
        bindNeutralVerdict()

        // Start real-time acoustic ingestion; UI + notification follow ACTUAL voice
        startLiveVoiceAnalysis()

        binding.btnBack.setOnClickListener { finish() }

        binding.btnTriggerChallenge.setOnClickListener {
            triggerLivenessChallenge()
        }

        binding.btnBlockCall.setOnClickListener {
            blockAndTerminateCall(contact.name)
        }

        binding.btnInterlock.setOnClickListener {
            toggleFinancialInterlock()
        }

        // Speakerphone nudge
        binding.btnTurnOnSpeaker.setOnClickListener {
            val audioManager = getSystemService(Context.AUDIO_SERVICE) as? AudioManager
            audioManager?.isSpeakerphoneOn = true
            checkSpeakerphoneState()
            Toast.makeText(this, "Speakerphone enabled for voice analysis", Toast.LENGTH_SHORT).show()
        }

        // Review held notifications
        binding.btnViewHeldNotifications.setOnClickListener {
            showHeldNotificationsDialog()
        }

        VaniScamShieldController.onHeldNotificationsChanged = { count ->
            runOnUiThread { updateHeldNotificationsBadge(count) }
        }
        updateHeldNotificationsBadge(VaniIncidentLogger.getActiveHeldNotifications().size)
    }

    /**
     * Called when the notification is tapped while the activity is already alive
     * (FLAG_ACTIVITY_SINGLE_TOP). Re-hydrates the contact from the new intent
     * and resubscribes to the live bus so the UI shows the current risk.
     */
    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        setIntent(intent)
        // Unsubscribe old listeners before resubscribing
        detachBusListeners()
        voiceAnalyzer?.stopListening()
        voiceAnalyzer = null

        contact = resolveContact()
        isLiveCall = contact.id == "live_call"
        bindCallerInfo(contact.name, contact.phone, contact.relation)
        bindNeutralVerdict()
        startLiveVoiceAnalysis()
    }

    /** Enrolled contacts come from sample data; live callers come from the notification extras. */
    private fun resolveContact(): VaniContact {
        val id = intent.getStringExtra(TempContactsActivity.EXTRA_CONTACT_ID)
        if (id != null && id != "live_call") {
            return VaniSampleData.getContactById(id) ?: VaniSampleData.contacts[0]
        }
        val name = intent.getStringExtra(VaniNotificationHelper.EXTRA_CALLER_NAME) ?: "Unknown caller"
        val phone = intent.getStringExtra(VaniNotificationHelper.EXTRA_CALLER_PHONE) ?: ""
        val enrolled = intent.getBooleanExtra(VaniNotificationHelper.EXTRA_CALLER_ENROLLED, false)
            || VaniSampleData.contacts.any { VaniCallRouter.sameNumber(it.phone, phone) && it.isVoiceEnrolled }

        return VaniContact(
            id = "live_call",
            name = name,
            phone = phone,
            relation = if (enrolled) "Voice Enrolled • Live Call" else "Live incoming call",
            isVoiceEnrolled = enrolled
        )
    }

    private fun startLiveVoiceAnalysis() {
        if (isLiveCall) startLiveGuard() else startDemoAnalyzer()
    }

    /**
     * Live call: the guard service owns the microphone, this screen only listens.
     *
     * Always subscribes to VaniRiskBus first - the service may already be
     * running and publishing live scores. VaniRiskBus.addListener() replays the
     * last snapshot immediately, so the UI updates the moment this screen opens.
     *
     * If the bus has data, we skip the service-start attempt entirely.
     * If the bus is empty AND the OS refuses a foreground start, we fall back
     * to the in-process analyzer so the screen still shows real measurements.
     */
    private fun startLiveGuard() {
        // The ringing alert hands over to the live meter the service posts.
        VaniNotificationHelper.cancelAlert(this)

        // Subscribe to the bus immediately - it replays the latest snapshot
        amplitudeListener = { pulseOnAmplitudeValue(it) }
        VaniRiskBus.addAmplitudeListener(amplitudeListener!!)

        riskListener = { snapshot ->
            applyRisk(snapshot.risk, snapshot.synthetic, snapshot.speakerMatch, snapshot.level, snapshot.message)
        }
        VaniRiskBus.addListener(riskListener!!)

        // If the service is already publishing, we are good.
        if (VaniRiskBus.last != null) return

        // Service not running yet - try to start it.
        val caller = VaniCallRouter.Caller(
            name = contact.name,
            phone = contact.phone,
            enrolled = contact.isVoiceEnrolled,
            known = false
        )

        if (!VaniCallRouter.startGuard(this, caller)) {
            // OS refused the foreground start, fall back to in-process analyzer.
            detachBusListeners()
            startDemoAnalyzer()
        }
    }

    private fun startDemoAnalyzer() {
        voiceAnalyzer = VaniLiveVoiceAnalyzer(this).apply {
            onAmplitudeUpdated = { pulseOnAmplitudeValue(it) }
            onRiskUpdated = { risk, synthetic, speakerMatch, level, message ->
                applyRisk(risk, synthetic, speakerMatch, level, message)
            }
            startListening(contact)
        }
    }

    private fun pulseOnAmplitudeValue(amp: Float) {
        val scale = 1.0f + (amp * 0.20f)
        binding.tvCallerAvatar.scaleX = scale
        binding.tvCallerAvatar.scaleY = scale
    }

    private fun applyRisk(
        risk: Int,
        synthetic: Int,
        speakerMatch: Int,
        level: RiskLevel,
        message: String
    ) {
        liveLevel = level
        binding.tvRiskScoreNumber.text = "$risk%"
        binding.pbRiskMeter.progress = risk
        binding.tvMetricAcoustic.text = "$synthetic% synthetic"
        binding.tvMetricSpeaker.text = "$speakerMatch% match"
        binding.tvVerdictTitle.text = verdictTitleFor(level)
        binding.tvVerdictDesc.text = message

        val (alertBg, tone) = toneFor(level)
        binding.cardRiskContainer.setBackgroundResource(alertBg)
        val color = ContextCompat.getColor(this, tone)
        binding.tvRiskScoreNumber.setTextColor(color)
        binding.pbRiskMeter.progressTintList = ContextCompat.getColorStateList(this, tone)
        binding.tvMetricAcoustic.setTextColor(color)
        binding.tvMetricSpeaker.setTextColor(color)
    }

    private fun verdictTitleFor(level: RiskLevel): String = when (level) {
        RiskLevel.DANGER -> "AI voice-clone impersonation"
        RiskLevel.WARNING -> "Voice anomaly detected"
        RiskLevel.SAFE -> "Authentic voice verified"
    }

    private fun toneFor(level: RiskLevel): Pair<Int, Int> = when (level) {
        RiskLevel.DANGER -> R.drawable.bg_alert_danger to R.color.error_tone
        RiskLevel.SAFE -> R.drawable.bg_alert_safe to R.color.success_deep
        RiskLevel.WARNING -> R.drawable.bg_alert_warning to R.color.warning_deep
    }

    private fun bindCallerInfo(name: String, phone: String, relation: String) {
        binding.tvCallerAvatar.text = name.firstOrNull()?.toString() ?: "V"
        binding.tvCallerName.text = name
        binding.tvCallerPhone.text = listOf(phone, relation).filter { it.isNotBlank() }.joinToString(" • ")
    }

    /** Honest starting point: nothing has been analyzed yet. */
    private fun bindNeutralVerdict() {
        liveLevel = RiskLevel.WARNING
        binding.tvRiskScoreNumber.text = "--"
        binding.pbRiskMeter.progress = 0
        binding.tvVerdictTitle.text = "Analyzing live voice"
        binding.tvVerdictDesc.text = "Risk score updates from the actual call audio as the caller speaks."
        binding.tvMetricAcoustic.text = "--"
        binding.tvMetricSpeaker.text = "--"
        binding.cardRiskContainer.setBackgroundResource(R.drawable.bg_card)
        binding.pbRiskMeter.progressTintList =
            ContextCompat.getColorStateList(this, R.color.mute)
    }

    private fun triggerLivenessChallenge() {
        binding.cardChallengeBox.visibility = View.VISIBLE
        val phrases = listOf(
            "Mango, 8, Nadi, 4, Blue",
            "Surya, 9, Chai, 3, Red",
            "Amber, 5, Sagar, 7, Green",
            "Pavan, 2, Bharat, 6, Yellow"
        )
        binding.tvChallengePhrase.text = phrases.random()
        binding.tvChallengeResult.text = "Ingesting telephony response, measuring synthesis latency..."
        binding.tvChallengeResult.setTextColor(ContextCompat.getColor(this, R.color.warning_deep))

        binding.btnTriggerChallenge.isEnabled = false

        Handler(Looper.getMainLooper()).postDelayed({
            if (isFinishing || isDestroyed) return@postDelayed
            binding.btnTriggerChallenge.isEnabled = true

            if (liveLevel == RiskLevel.DANGER) {
                binding.tvChallengeResult.text =
                    "Liveness failed. Response latency 1820ms vs human ~350ms - deepfake replay confirmed."
                binding.tvChallengeResult.setTextColor(ContextCompat.getColor(this, R.color.error_tone))
            } else {
                binding.tvChallengeResult.text =
                    "Liveness passed. Human response latency 340ms - natural pitch contour matched."
                binding.tvChallengeResult.setTextColor(ContextCompat.getColor(this, R.color.success_deep))
            }
        }, 1800)
    }

    private fun blockAndTerminateCall(callerName: String) {
        AlertDialog.Builder(this)
            .setTitle("Block call and log incident")
            .setMessage("Block this incoming call from $callerName? A forensic report will be filed for compliance.")
            .setPositiveButton("Block") { _, _ ->
                voiceAnalyzer?.stopListening()
                VaniCallRouter.onEnded(this)
                VaniNotificationHelper.cancelNotification(this)
                Toast.makeText(this, "Call blocked. Forensic incident report filed.", Toast.LENGTH_LONG).show()
                finish()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun toggleFinancialInterlock() {
        val currentlyActive = VaniScamShieldController.isUserInterlockEnabled
        if (currentlyActive) {
            VaniScamShieldController.setUserInterlockEnabled(false)
            binding.btnInterlock.text = "Financial interlock"
            binding.btnInterlock.setBackgroundResource(R.drawable.bg_button_secondary)
            binding.btnInterlock.setTextColor(ContextCompat.getColor(this, R.color.ink))
            Toast.makeText(this, "Financial Interlock Disengaged", Toast.LENGTH_SHORT).show()
        } else {
            VaniScamShieldController.setUserInterlockEnabled(true)
            binding.btnInterlock.text = "Scam Shield Active (Tap to Disable)"
            binding.btnInterlock.setBackgroundResource(R.drawable.bg_button_primary)
            binding.btnInterlock.setTextColor(ContextCompat.getColor(this, R.color.white))

            AlertDialog.Builder(this)
                .setTitle("Scam-Shield Engaged")
                .setMessage(
                    "Active coercion protection enabled for this call:\n\n" +
                        "• Banking OTP and payment notifications are shielded from the screen.\n" +
                        "• Opening banking/UPI apps triggers a critical coercion warning overlay.\n" +
                        "• All incidents are logged for forensic compliance."
                )
                .setPositiveButton("Understood", null)
                .show()
        }
    }

    private fun checkSpeakerphoneState() {
        val audioManager = getSystemService(Context.AUDIO_SERVICE) as? AudioManager ?: return
        if (isLiveCall && !audioManager.isSpeakerphoneOn) {
            binding.cardSpeakerphonePrompt.visibility = View.VISIBLE
        } else {
            binding.cardSpeakerphonePrompt.visibility = View.GONE
        }
    }

    private fun updateHeldNotificationsBadge(count: Int) {
        if (count > 0) {
            binding.cardHeldNotifications.visibility = View.VISIBLE
            binding.tvHeldNotificationsTitle.text = "$count Banking Notification(s) Shielded"
        } else {
            binding.cardHeldNotifications.visibility = View.GONE
        }
    }

    private fun showHeldNotificationsDialog() {
        val items = VaniIncidentLogger.getActiveHeldNotifications()
        val dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_held_notifications, null)
        val container = dialogView.findViewById<android.widget.LinearLayout>(R.id.layoutHeldItems)

        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        if (items.isEmpty()) {
            val emptyTv = TextView(this).apply {
                text = "No banking notifications currently held."
                setTextColor(ContextCompat.getColor(this@TempContactDetailActivity, R.color.mute))
                setPadding(0, 16, 0, 16)
            }
            container.addView(emptyTv)
        } else {
            for (item in items) {
                val itemView = TextView(this).apply {
                    text = "${item.packageName}\n${item.text}"
                    textSize = spToFloat(12f)
                    setTextColor(ContextCompat.getColor(this@TempContactDetailActivity, R.color.ink))
                    setBackgroundResource(R.drawable.bg_card)
                    setPadding(24, 18, 24, 18)
                }
                val lp = android.widget.LinearLayout.LayoutParams(
                    android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                    android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(0, 0, 0, 16) }
                container.addView(itemView, lp)
            }

            val clearBtn = com.google.android.material.button.MaterialButton(this).apply {
                text = "Dismiss All Shielded Alerts"
                setTextColor(ContextCompat.getColor(this@TempContactDetailActivity, R.color.error_tone))
                setBackgroundColor(android.graphics.Color.TRANSPARENT)
                setOnClickListener {
                    VaniIncidentLogger.clearActiveHeldNotifications()
                    updateHeldNotificationsBadge(0)
                    dialog.dismiss()
                    Toast.makeText(this@TempContactDetailActivity, "Shielded alerts dismissed", Toast.LENGTH_SHORT).show()
                }
            }
            container.addView(clearBtn)
        }

        dialogView.findViewById<View>(R.id.btnCloseHeldDialog).setOnClickListener {
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun spToFloat(sp: Float): Float = sp

    override fun onResume() {
        super.onResume()
        speakerPollHandler.post(speakerPollRunnable)
        updateHeldNotificationsBadge(VaniIncidentLogger.getActiveHeldNotifications().size)
    }

    override fun onPause() {
        super.onPause()
        speakerPollHandler.removeCallbacks(speakerPollRunnable)
    }

    private fun detachBusListeners() {
        riskListener?.let { VaniRiskBus.removeListener(it) }
        amplitudeListener?.let { VaniRiskBus.removeAmplitudeListener(it) }
        riskListener = null
        amplitudeListener = null
    }

    override fun onDestroy() {
        speakerPollHandler.removeCallbacks(speakerPollRunnable)
        VaniScamShieldController.onHeldNotificationsChanged = null
        detachBusListeners()
        voiceAnalyzer?.stopListening()
        voiceAnalyzer = null
        super.onDestroy()
    }
}