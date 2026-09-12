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
import com.vericall.util.VaniCallRouter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.media.MediaPlayer
import android.os.Handler
import android.os.Looper
import android.widget.SeekBar
import android.widget.Toast

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

        // Preserved Evidence Banner
        if (record.isProtected || record.riskScore >= 50) {
            binding.cardPreservedEvidence.visibility = View.VISIBLE
        } else {
            binding.cardPreservedEvidence.visibility = View.GONE
        }

        // Audio Player Setup
        val audioFile = record.audioFilePath?.let { File(it) }
        if (audioFile != null && audioFile.exists() && audioFile.length() > 500) {
            binding.layoutSummaryAudioPlayer.visibility = View.VISIBLE
            setupAudioPlayer(audioFile)
        } else {
            binding.layoutSummaryAudioPlayer.visibility = View.GONE
        }

        // Red Flags & Scam Signals
        binding.containerRedFlags.removeAllViews()
        if (record.redFlags.isNotEmpty()) {
            binding.sectionRedFlags.visibility = View.VISIBLE
            for (rf in record.redFlags) {
                val tv = TextView(this).apply {
                    text = "•  $rf"
                    textSize = 13f
                    setTextColor(ContextCompat.getColor(this@VaniCallSummaryActivity, R.color.verdict_scam))
                    setPadding(0, 4, 0, 4)
                    setTypeface(typeface, android.graphics.Typeface.BOLD)
                }
                binding.containerRedFlags.addView(tv)
            }
        } else if (record.riskScore >= 60) {
            binding.sectionRedFlags.visibility = View.VISIBLE
            val tv = TextView(this).apply {
                text = "•  Anomalous vocoder synthesis and high-frequency phase discontinuities detected."
                textSize = 13f
                setTextColor(ContextCompat.getColor(this@VaniCallSummaryActivity, R.color.verdict_scam))
                setPadding(0, 4, 0, 4)
            }
            binding.containerRedFlags.addView(tv)
        } else {
            binding.sectionRedFlags.visibility = View.GONE
        }

        // Fact Checking & Web Evidence
        binding.containerFactChecks.removeAllViews()
        if (record.factChecks.isNotEmpty()) {
            binding.sectionFactChecks.visibility = View.VISIBLE
            for (fc in record.factChecks) {
                val fcCard = createFactCheckCard(fc)
                binding.containerFactChecks.addView(fcCard)
            }
        } else {
            binding.sectionFactChecks.visibility = View.GONE
        }

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
            VaniCallRouter.makeProperPhoneCall(this, record.number)
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

    private var mediaPlayer: MediaPlayer? = null
    private val handler = Handler(Looper.getMainLooper())
    private var progressRunnable: Runnable? = null

    private fun setupAudioPlayer(file: File) {
        binding.btnSummaryPlayPause.setOnClickListener {
            if (mediaPlayer == null) {
                try {
                    mediaPlayer = MediaPlayer().apply {
                        setDataSource(file.absolutePath)
                        prepare()
                        start()
                        setOnCompletionListener {
                            binding.btnSummaryPlayPause.setImageResource(R.drawable.ic_play_arrow)
                            binding.seekSummaryAudio.progress = 0
                            binding.tvSummaryAudioDuration.text = "00:00 / 00:00"
                        }
                    }
                    binding.btnSummaryPlayPause.setImageResource(R.drawable.ic_pause)
                    startPlayerTracker()
                } catch (e: Exception) {
                    Toast.makeText(this, "Could not play audio", Toast.LENGTH_SHORT).show()
                }
            } else {
                if (mediaPlayer!!.isPlaying) {
                    mediaPlayer!!.pause()
                    binding.btnSummaryPlayPause.setImageResource(R.drawable.ic_play_arrow)
                } else {
                    mediaPlayer!!.start()
                    binding.btnSummaryPlayPause.setImageResource(R.drawable.ic_pause)
                    startPlayerTracker()
                }
            }
        }

        binding.seekSummaryAudio.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser && mediaPlayer != null) {
                    val pos = (mediaPlayer!!.duration * (progress / 100f)).toInt()
                    mediaPlayer!!.seekTo(pos)
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun startPlayerTracker() {
        handler.removeCallbacksAndMessages(null)
        progressRunnable = object : Runnable {
            override fun run() {
                val mp = mediaPlayer
                if (mp != null && mp.isPlaying) {
                    val cur = mp.currentPosition
                    val tot = mp.duration
                    val pct = if (tot > 0) ((cur.toFloat() / tot) * 100).toInt() else 0
                    binding.seekSummaryAudio.progress = pct

                    val cMin = (cur / 1000) / 60
                    val cSec = (cur / 1000) % 60
                    val tMin = (tot / 1000) / 60
                    val tSec = (tot / 1000) % 60
                    binding.tvSummaryAudioDuration.text = String.format("%02d:%02d / %02d:%02d", cMin, cSec, tMin, tSec)

                    handler.postDelayed(this, 300)
                }
            }
        }
        handler.post(progressRunnable!!)
    }

    private fun createFactCheckCard(fc: com.vericall.model.FactCheckItem): View {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.bg_card_inner)
            setPadding(12, 12, 12, 12)
            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = 8 }
            layoutParams = lp
        }

        // Header Row: Claim + Verdict
        val headerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
        }

        val verdictPill = TextView(this).apply {
            text = fc.verdict
            textSize = 10f
            textStyleBold()
            setPadding(8, 4, 8, 4)
            when (fc.verdict) {
                "SUPPORTED" -> {
                    setBackgroundResource(R.drawable.bg_badge_supported)
                    setTextColor(ContextCompat.getColor(this@VaniCallSummaryActivity, R.color.verdict_supported))
                }
                "FRAUDULENT_CLAIM", "CONTRADICTED" -> {
                    setBackgroundResource(R.drawable.bg_badge_scam)
                    setTextColor(ContextCompat.getColor(this@VaniCallSummaryActivity, R.color.verdict_scam))
                }
                else -> {
                    setBackgroundResource(R.drawable.bg_pill_neobrutal_paper)
                    setTextColor(ContextCompat.getColor(this@VaniCallSummaryActivity, R.color.neo_ink))
                }
            }
        }
        headerRow.addView(verdictPill)

        val claimTv = TextView(this).apply {
            text = "\"${fc.claim}\""
            textSize = 13f
            textStyleBold()
            setTextColor(ContextCompat.getColor(this@VaniCallSummaryActivity, R.color.neo_ink))
            setPadding(10, 0, 0, 0)
        }
        headerRow.addView(claimTv)
        card.addView(headerRow)

        // Evidence text
        if (fc.evidence.isNotBlank()) {
            val evidenceTv = TextView(this).apply {
                text = "Evidence: ${fc.evidence}"
                textSize = 12f
                setTextColor(ContextCompat.getColor(this@VaniCallSummaryActivity, R.color.text_secondary))
                setPadding(0, 6, 0, 0)
            }
            card.addView(evidenceTv)
        }

        // Source Citation
        if (fc.source.isNotBlank()) {
            val sourceTv = TextView(this).apply {
                text = "Source: ${fc.source}"
                textSize = 11f
                setTextColor(ContextCompat.getColor(this@VaniCallSummaryActivity, R.color.brand_primary))
                setPadding(0, 4, 0, 0)
            }
            card.addView(sourceTv)
        }

        return card
    }

    private fun TextView.textStyleBold() {
        setTypeface(typeface, android.graphics.Typeface.BOLD)
    }

    override fun onDestroy() {
        handler.removeCallbacksAndMessages(null)
        try {
            mediaPlayer?.stop()
            mediaPlayer?.release()
        } catch (_: Exception) {}
        mediaPlayer = null
        super.onDestroy()
    }
}
