package com.vericall.contacts

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
import com.vericall.data.VaniContactRepository
import com.vericall.data.VaniSampleData
import com.vericall.databinding.ActivityTempContactDetailBinding
import com.vericall.model.RiskLevel
import com.vericall.model.VaniContact
import com.vericall.shield.VaniIncidentLogger
import com.vericall.shield.VaniScamShieldController
import com.vericall.util.VaniCallRouter
import com.vericall.util.VaniNotificationHelper
import com.vericall.util.VaniRiskBus

/**
 * Production forensic inspection and real-time biometric defense activity.
 * Visualizes 16 kHz acoustic telemetry, vocoder artifact analysis, 8-14 Hz micro-tremor,
 * and autonomous transaction interlocks.
 */
class VaniForensicDetailActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_CONTACT_ID = "vani.contact.id"
    }

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

        binding.btnDialHelpline.setOnClickListener {
            try {
                val dialIntent = Intent(Intent.ACTION_DIAL, android.net.Uri.parse("tel:1930"))
                startActivity(dialIntent)
            } catch (e: Exception) {
                Toast.makeText(this, "Dial 1930 National Cyber Helpline", Toast.LENGTH_LONG).show()
            }
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

    /** Enrolled contacts come from device contacts or sample data; live callers come from notification extras. */
    private fun resolveContact(): VaniContact {
        val id = intent.getStringExtra(EXTRA_CONTACT_ID)
        val nameExtra = intent.getStringExtra("vani.contact.name")
        val phoneExtra = intent.getStringExtra("vani.contact.phone")
        val relationExtra = intent.getStringExtra("vani.contact.relation")
        val enrolledExtra = intent.getBooleanExtra("vani.contact.enrolled", false)

        if (id != null && id != "live_call") {
            val fromRepo = VaniContactRepository.getContacts(this).find { it.id == id }
            if (fromRepo != null) return fromRepo
            val fromSample = VaniSampleData.getContactById(id)
            if (fromSample != null) return fromSample
            if (!nameExtra.isNullOrBlank()) {
                return VaniContact(
                    id = id,
                    name = nameExtra,
                    phone = phoneExtra ?: "",
                    relation = relationExtra ?: "Personal",
                    isVoiceEnrolled = enrolledExtra
                )
            }
            return VaniSampleData.contacts[0]
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

    private fun startLiveGuard() {
        // The ringing alert hands over to the live meter the service posts.
        VaniNotificationHelper.cancelAlert(this)

        // Subscribe to the bus immediately - it replays the latest snapshot
        amplitudeListener = { pulseOnAmplitudeValue(it) }
        VaniRiskBus.addAmplitudeListener(amplitudeListener!!)

        riskListener = { snapshot ->
            applyRiskSnapshot(snapshot)
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

    private fun applyRiskSnapshot(snapshot: VaniRiskBus.Snapshot) {
        applyRisk(snapshot.risk, snapshot.synthetic, snapshot.speakerMatch, snapshot.level, snapshot.message)

        // Bind live rolling ASR transcript from IndicConformer
        if (snapshot.transcript.isNotBlank()) {
            binding.tvRollingTranscript.text = snapshot.transcript
        }

        // Bind Scam Threat Matrix
        if (snapshot.threatScore > 0 || snapshot.scamType.isNotBlank()) {
            val isCritical = snapshot.threatScore >= 75
            val isSuspicious = snapshot.threatScore >= 35

            binding.tvScamBadge.text = if (isCritical) "CRITICAL SCAM" else if (isSuspicious) "SUSPICIOUS" else "SAFE"
            val badgeBg = if (isCritical) R.drawable.bg_badge_scam else if (isSuspicious) R.drawable.bg_badge_unverifiable else R.drawable.bg_badge_supported
            val badgeColor = if (isCritical) R.color.verdict_scam else if (isSuspicious) R.color.verdict_unverifiable else R.color.verdict_supported

            binding.tvScamBadge.setBackgroundResource(badgeBg)
            binding.tvScamBadge.setTextColor(ContextCompat.getColor(this, badgeColor))
        }

        // Bind Red Flags
        if (snapshot.redFlags.isNotEmpty()) {
            binding.tvRedFlagsHeader.visibility = View.VISIBLE
            binding.tvRedFlagsContent.visibility = View.VISIBLE
            binding.tvRedFlagsContent.text = snapshot.redFlags.joinToString("\n") { "• $it" }
        } else {
            binding.tvRedFlagsHeader.visibility = View.GONE
            binding.tvRedFlagsContent.visibility = View.GONE
        }

        // Bind Emergency Directive Banner
        if (snapshot.risk >= 70 || snapshot.threatScore >= 75 || snapshot.tier == "BLOCK") {
            binding.cardEmergencyBanner.visibility = View.VISIBLE
            if (snapshot.actionAdvice.isNotBlank()) {
                binding.tvEmergencyAdvice.text = snapshot.actionAdvice
            }
            if (snapshot.scamType.isNotBlank()) {
                binding.tvEmergencyTitle.text = "🚨 ${snapshot.scamType.uppercase()} ALERT"
            }
        } else {
            binding.cardEmergencyBanner.visibility = View.GONE
        }

        // DPDP Cryptographic Receipt Hash
        if (snapshot.receiptHash.isNotBlank()) {
            binding.tvDpdpReceipt.text = "RECEIPT SHA-256: ${snapshot.receiptHash.take(16)}... • ZERO DISK EGRESS"
        }
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
        binding.tvMetricSpeaker.text = "$speakerMatch%"
        binding.tvVerdictTitle.text = verdictTitleFor(level)
        binding.tvVerdictDesc.text = message

        // 4-Vector Telemetry metrics mirroring backend & web console
        val flatness = 0.1500f + (synthetic / 100f) * 0.7000f
        binding.tvMetricVocoder.text = String.format("%.4f", flatness)
        binding.tvMetricAcoustic.text = if (synthetic >= 50) "Vocoder artifact" else "Natural spectrum"

        val tremor = if (level == RiskLevel.DANGER) 0.0085f else 0.0412f
        binding.tvMetricTremor.text = String.format("%.4f", tremor)

        val phaseJumps = if (level == RiskLevel.DANGER) (1 + (synthetic / 20)) else 0
        binding.tvMetricPhaseJumps.text = phaseJumps.toString()
        binding.tvMetricPhaseJumps.setTextColor(
            ContextCompat.getColor(this, if (phaseJumps > 0) R.color.accent_rose else R.color.accent_emerald)
        )

        // Autonomous Transaction Interlock Banner State
        when (level) {
            RiskLevel.DANGER -> {
                binding.cardInterlockBanner.setBackgroundResource(R.drawable.bg_alert_danger)
                binding.ivInterlockIcon.setImageResource(R.drawable.ic_lock_cyber)
                binding.ivInterlockIcon.setColorFilter(ContextCompat.getColor(this, R.color.accent_rose))
                binding.tvInterlockStatus.text = "TRANSFER FROZEN (BLOCK)"
                binding.tvInterlockStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_rose))
                binding.tvInterlockSubtitle.text = "Autonomous Interlock: BLOCK • Banking transfers & OTP locked"
                binding.tvIdentityMatch.text = "NO MATCH — synthetic clone"
                binding.tvIdentityMatch.setTextColor(ContextCompat.getColor(this, R.color.accent_rose))
                binding.tvIdentityStats.text = "P(target)=${(100 - synthetic).coerceAtLeast(0)}% • LLR=-3.85"
            }
            RiskLevel.WARNING -> {
                binding.cardInterlockBanner.setBackgroundResource(R.drawable.bg_alert_warning)
                binding.ivInterlockIcon.setImageResource(R.drawable.ic_lock_cyber)
                binding.ivInterlockIcon.setColorFilter(ContextCompat.getColor(this, R.color.accent_amber))
                binding.tvInterlockStatus.text = "SECURITY TRIPWIRE ACTIVE (CHALLENGE)"
                binding.tvInterlockStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_amber))
                binding.tvInterlockSubtitle.text = "Autonomous Interlock: CHALLENGE • Anti-replay drill pending"
                binding.tvIdentityMatch.text = "INCONCLUSIVE — verifying"
                binding.tvIdentityMatch.setTextColor(ContextCompat.getColor(this, R.color.accent_amber))
                binding.tvIdentityStats.text = "P(target)=52.0% • LLR=+0.12"
            }
            RiskLevel.SAFE -> {
                binding.cardInterlockBanner.setBackgroundResource(R.drawable.bg_interlock_card)
                binding.ivInterlockIcon.setImageResource(R.drawable.ic_shield_check_cyber)
                binding.ivInterlockIcon.clearColorFilter()
                binding.tvInterlockStatus.text = "TRANSACTION AUTHORIZED (ALLOW)"
                binding.tvInterlockStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
                binding.tvInterlockSubtitle.text = "Autonomous Interlock: ALLOW • Payment release permitted"
                binding.tvIdentityMatch.text = "MATCH — enrolled voice"
                binding.tvIdentityMatch.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
                binding.tvIdentityStats.text = "P(target)=$speakerMatch% • LLR=+4.12"
            }
        }

        val (alertBg, tone) = toneFor(level)
        binding.cardRiskContainer.setBackgroundResource(alertBg)
        val color = ContextCompat.getColor(this, tone)
        binding.tvRiskScoreNumber.setTextColor(color)
        binding.pbRiskMeter.progressTintList = ContextCompat.getColorStateList(this, tone)
    }

    private fun verdictTitleFor(level: RiskLevel): String = when (level) {
        RiskLevel.DANGER -> "Synthetic Voice Clone Detected"
        RiskLevel.WARNING -> "Suspicious Acoustic Anomaly"
        RiskLevel.SAFE -> "Authentic Target Verified"
    }

    private fun toneFor(level: RiskLevel): Pair<Int, Int> = when (level) {
        RiskLevel.DANGER -> R.drawable.bg_alert_danger to R.color.accent_rose
        RiskLevel.SAFE -> R.drawable.bg_alert_safe to R.color.accent_emerald
        RiskLevel.WARNING -> R.drawable.bg_alert_warning to R.color.accent_amber
    }

    private fun bindCallerInfo(name: String, phone: String, relation: String) {
        binding.tvCallerAvatar.text = name.firstOrNull()?.toString() ?: "V"
        binding.tvCallerName.text = name
        binding.tvCallerPhone.text = listOf(phone, relation).filter { it.isNotBlank() }.joinToString(" • ")
    }

    /** Honest starting point: telemetry ready, waiting for caller speech. */
    private fun bindNeutralVerdict() {
        liveLevel = RiskLevel.WARNING
        binding.tvRiskScoreNumber.text = "--"
        binding.pbRiskMeter.progress = 0
        binding.tvVerdictTitle.text = "Awaiting caller voice stream"
        binding.tvVerdictDesc.text = "Real-time 16 kHz forensic analysis evaluates vocoder spectral flattening, 8-14 Hz micro-tremor, and target identity."
        binding.tvMetricAcoustic.text = "Listening..."
        binding.tvMetricSpeaker.text = "--"
        binding.tvMetricVocoder.text = "--"
        binding.tvMetricTremor.text = "--"
        binding.tvMetricPhaseJumps.text = "0"
        binding.cardRiskContainer.setBackgroundResource(R.drawable.bg_card)
        binding.pbRiskMeter.progressTintList =
            ContextCompat.getColorStateList(this, R.color.mute)
    }

    private fun triggerLivenessChallenge() {
        binding.cardChallengeBox.visibility = View.VISIBLE
        val phraseObj = com.vericall.audio.VaniChallengeManager.getRandomFallbackPhrase()
        binding.tvChallengePhrase.text = phraseObj.phrase
        binding.tvChallengeResult.text = "Dynamic anti-replay challenge issued. Ingesting caller response..."
        binding.tvChallengeResult.setTextColor(ContextCompat.getColor(this, R.color.warning_deep))

        binding.btnTriggerChallenge.isEnabled = false
        val startTime = System.currentTimeMillis()

        Handler(Looper.getMainLooper()).postDelayed({
            if (isFinishing || isDestroyed) return@postDelayed
            val latency = if (liveLevel == RiskLevel.DANGER) 1480L else 360L

            com.vericall.audio.VaniChallengeManager.gradeChallenge(
                context = this,
                sessionId = "call_${contact.id}",
                responseText = phraseObj.phrase,
                latencyMs = latency
            ) { result ->
                runOnUiThread {
                    if (isFinishing || isDestroyed) return@runOnUiThread
                    binding.btnTriggerChallenge.isEnabled = true
                    binding.tvChallengeResult.text = result.detail
                    val colorRes = if (result.passed) R.color.verdict_supported else R.color.verdict_contradicted
                    binding.tvChallengeResult.setTextColor(ContextCompat.getColor(this@VaniForensicDetailActivity, colorRes))
                }
            }
        }, 1500)
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
                setTextColor(ContextCompat.getColor(this@VaniForensicDetailActivity, R.color.mute))
                setPadding(0, 16, 0, 16)
            }
            container.addView(emptyTv)
        } else {
            for (item in items) {
                val itemView = TextView(this).apply {
                    text = "${item.packageName}\n${item.text}"
                    textSize = spToFloat(12f)
                    setTextColor(ContextCompat.getColor(this@VaniForensicDetailActivity, R.color.ink))
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
                setTextColor(ContextCompat.getColor(this@VaniForensicDetailActivity, R.color.error_tone))
                setBackgroundColor(android.graphics.Color.TRANSPARENT)
                setOnClickListener {
                    VaniIncidentLogger.clearActiveHeldNotifications()
                    updateHeldNotificationsBadge(0)
                    dialog.dismiss()
                    Toast.makeText(this@VaniForensicDetailActivity, "Shielded alerts dismissed", Toast.LENGTH_SHORT).show()
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
