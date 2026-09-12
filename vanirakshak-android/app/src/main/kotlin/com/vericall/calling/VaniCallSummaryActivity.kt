package com.vericall.calling

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.recyclerview.widget.LinearLayoutManager
import com.vericall.R
import com.vericall.data.VaniCallHistoryRepository
import com.vericall.databinding.ActivityVaniCallSummaryBinding
import com.vericall.model.VaniCallRecord
import com.vericall.ui.VaniTranscriptAdapter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class VaniCallSummaryActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_CALL_ID = "extra_call_id"
    }

    private lateinit var binding: ActivityVaniCallSummaryBinding
    private val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())
    private val dateFormat = SimpleDateFormat("EEEE, MMM d", Locale.getDefault())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVaniCallSummaryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        val callId = intent.getStringExtra(EXTRA_CALL_ID) ?: ""
        val record = VaniCallHistoryRepository.getCallById(this, callId)

        if (record != null) {
            bindCallRecord(record)
        } else {
            finish()
        }

        binding.btnCloseSummary.setOnClickListener { finish() }
        binding.btnSummaryDone.setOnClickListener { finish() }
    }

    private fun bindCallRecord(record: VaniCallRecord) {
        val name = record.callerName.ifBlank { record.number.ifBlank { "Unknown caller" } }
        binding.tvSummaryCallerName.text = name
        binding.tvSummaryCallerNumber.text = record.number

        val initial = name.firstOrNull { it.isLetter() }?.uppercaseChar() ?: 'V'
        binding.tvSummaryAvatar.text = initial.toString()

        val mins = record.durationSec / 60
        val secs = record.durationSec % 60
        val durationStr = String.format("%02d:%02d", mins, secs)
        val dateStr = dateFormat.format(Date(record.timestamp))
        val timeStr = timeFormat.format(Date(record.timestamp))
        binding.tvSummaryCallMeta.text = "$durationStr • $dateStr at $timeStr"

        // Verdict Badge
        val isDanger = record.riskScore >= 70
        val isWarning = record.riskScore in 35..69
        val verdictColor = when {
            isDanger -> R.color.verdict_scam
            isWarning -> R.color.verdict_unverifiable
            else -> R.color.verdict_supported
        }
        val verdictBg = when {
            isDanger -> R.drawable.bg_badge_scam
            isWarning -> R.drawable.bg_badge_unverifiable
            else -> R.drawable.bg_badge_supported
        }
        val verdictTitle = when {
            isDanger -> if (record.scamType.isNotBlank()) "🚨 ${record.scamType.uppercase()}" else "🚨 SYNTHETIC CLONE DETECTED"
            isWarning -> "⚠ ACOUSTIC ANOMALY"
            else -> "✓ AUTHENTIC TARGET VERIFIED"
        }

        binding.cardSummaryVerdict.setBackgroundResource(verdictBg)
        binding.dotSummaryVerdict.backgroundTintList = ContextCompat.getColorStateList(this, verdictColor)
        binding.tvSummaryVerdictTitle.text = verdictTitle
        binding.tvSummaryVerdictTitle.setTextColor(ContextCompat.getColor(this, verdictColor))
        binding.tvSummaryRiskScore.text = "Risk: ${record.riskScore}%"
        binding.tvSummaryRiskScore.setTextColor(ContextCompat.getColor(this, verdictColor))

        // AI Summary
        if (record.aiSummary.isNotBlank()) {
            binding.tvSummaryAiText.text = record.aiSummary
            binding.sectionAiSummary.visibility = View.VISIBLE
        } else {
            binding.sectionAiSummary.visibility = View.GONE
        }

        // Key Points
        binding.containerKeyPoints.removeAllViews()
        if (record.keyPoints.isNotEmpty()) {
            binding.sectionKeyPoints.visibility = View.VISIBLE
            for (point in record.keyPoints) {
                val tv = TextView(this).apply {
                    text = "•  $point"
                    textSize = 14f
                    setTextColor(ContextCompat.getColor(this@VaniCallSummaryActivity, R.color.text_primary))
                    setLineSpacing(3f, 1f)
                    setPadding(0, 4, 0, 4)
                }
                binding.containerKeyPoints.addView(tv)
            }
        } else {
            binding.sectionKeyPoints.visibility = View.GONE
        }

        // Action Items
        binding.containerActionItems.removeAllViews()
        if (record.actionItems.isNotEmpty()) {
            binding.sectionActionItems.visibility = View.VISIBLE
            for (action in record.actionItems) {
                val cb = CheckBox(this).apply {
                    text = action
                    textSize = 14f
                    setTextColor(ContextCompat.getColor(this@VaniCallSummaryActivity, R.color.text_primary))
                    setPadding(8, 6, 0, 6)
                }
                binding.containerActionItems.addView(cb)
            }
        } else {
            binding.sectionActionItems.visibility = View.GONE
        }

        // Full Transcript
        if (record.transcript.isNotEmpty()) {
            binding.sectionFullTranscript.visibility = View.VISIBLE
            binding.tvSummaryNoTranscript.visibility = View.GONE
            binding.tvSummaryMessageCount.text = "${record.transcript.size} messages"

            val adapter = VaniTranscriptAdapter(record.transcript.toMutableList())
            binding.recyclerSummaryTranscript.layoutManager = LinearLayoutManager(this)
            binding.recyclerSummaryTranscript.adapter = adapter
        } else {
            binding.tvSummaryNoTranscript.visibility = View.VISIBLE
            binding.tvSummaryMessageCount.text = "0 messages"
        }

        // Quick Call Again
        binding.btnSummaryQuickCall.setOnClickListener {
            VaniActiveCallActivity.startOutgoingCall(this, record.number, record.callerName)
            finish()
        }

        // Share Summary
        binding.btnSummaryShare.setOnClickListener {
            val shareText = buildString {
                append("📞 Call Summary with ${record.callerName} (${record.number})\n")
                append("Date: $dateStr, $timeStr | Duration: $durationStr\n")
                append("VaniRakshak Verdict: $verdictTitle (Risk: ${record.riskScore}%)\n\n")
                if (record.aiSummary.isNotBlank()) {
                    append("✦ AI Summary:\n${record.aiSummary}\n\n")
                }
                if (record.keyPoints.isNotEmpty()) {
                    append("Key Takeaways:\n")
                    record.keyPoints.forEach { append("• $it\n") }
                    append("\n")
                }
                if (record.actionItems.isNotEmpty()) {
                    append("Action Items:\n")
                    record.actionItems.forEach { append("□ $it\n") }
                }
            }
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_TEXT, shareText)
            }
            startActivity(Intent.createChooser(intent, "Share Call Summary"))
        }
    }
}
