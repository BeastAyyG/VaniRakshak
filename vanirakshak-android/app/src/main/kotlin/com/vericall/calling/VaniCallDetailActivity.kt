package com.vericall.calling

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.CheckBox
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.recyclerview.widget.LinearLayoutManager
import com.vericall.R
import com.vericall.data.VaniCallHistoryRepository
import com.vericall.databinding.ActivityVaniCallDetailBinding
import com.vericall.model.CallDirection
import com.vericall.model.VaniCallRecord
import com.vericall.ui.VaniTranscriptAdapter
import com.vericall.util.VaniCallRouter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class VaniCallDetailActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_RECORD_ID = "extra_record_id"
    }

    private lateinit var binding: ActivityVaniCallDetailBinding
    private val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())
    private val dateFormat = SimpleDateFormat("EEEE, MMM d", Locale.getDefault())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVaniCallDetailBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        val recordId = intent.getStringExtra(EXTRA_RECORD_ID) ?: ""
        val record = VaniCallHistoryRepository.getCallById(this, recordId)

        if (record != null) {
            bindRecord(record)
        } else {
            val fallback = VaniCallHistoryRepository.getRecents(this).firstOrNull()
            if (fallback != null) {
                bindRecord(fallback)
            } else {
                Toast.makeText(this, "Call record details not found", Toast.LENGTH_SHORT).show()
                finish()
            }
        }

        binding.btnBackCallDetail.setOnClickListener { finish() }
    }

    private fun bindRecord(record: VaniCallRecord) {
        val name = record.callerName.ifBlank { record.number.ifBlank { "Unknown caller" } }
        binding.tvDetailCallerName.text = name
        binding.tvDetailCallerNumber.text = record.number

        val initial = name.firstOrNull { it.isLetter() }?.uppercaseChar() ?: 'V'
        binding.tvDetailAvatar.text = initial.toString()

        val mins = record.durationSec / 60
        val secs = record.durationSec % 60
        val durationStr = String.format("%02d:%02d", mins, secs)
        val dateStr = dateFormat.format(Date(record.timestamp))
        val timeStr = timeFormat.format(Date(record.timestamp))
        binding.tvDetailDurationTime.text = "$durationStr • $dateStr at $timeStr"

        when (record.direction) {
            CallDirection.INCOMING -> {
                binding.ivDetailCallDirection.setImageResource(R.drawable.ic_call_incoming)
                binding.ivDetailCallDirection.setColorFilter(ContextCompat.getColor(this, R.color.verdict_supported))
                binding.tvDetailCallDirectionText.text = "Incoming Call"
            }
            CallDirection.OUTGOING -> {
                binding.ivDetailCallDirection.setImageResource(R.drawable.ic_call_outgoing)
                binding.ivDetailCallDirection.setColorFilter(ContextCompat.getColor(this, R.color.brand_primary))
                binding.tvDetailCallDirectionText.text = "Outgoing Call"
            }
            CallDirection.MISSED -> {
                binding.ivDetailCallDirection.setImageResource(R.drawable.ic_call_missed)
                binding.ivDetailCallDirection.setColorFilter(ContextCompat.getColor(this, R.color.verdict_contradicted))
                binding.tvDetailCallDirectionText.text = "Missed Call"
            }
        }

        // Primary Actions
        binding.btnDetailActionCall.setOnClickListener {
            VaniCallRouter.makeProperPhoneCall(this, record.number)
        }

        binding.btnDetailActionMessage.setOnClickListener {
            val smsIntent = Intent(Intent.ACTION_VIEW).apply {
                data = Uri.parse("sms:${record.number}")
            }
            try {
                startActivity(smsIntent)
            } catch (_: Exception) {
                Toast.makeText(this, "No SMS app found", Toast.LENGTH_SHORT).show()
            }
        }

        binding.btnDetailActionShare.setOnClickListener {
            val shareText = "Call with ${record.callerName} (${record.number}) on $dateStr at $timeStr.\n${record.aiSummary}"
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_TEXT, shareText)
            }
            startActivity(Intent.createChooser(shareIntent, "Share Call Detail"))
        }

        binding.btnDeleteCallRecord.setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("Delete Call Log Entry")
                .setMessage("Remove this call log entry and its transcript from VaniRakshak?")
                .setPositiveButton("Delete") { _, _ ->
                    VaniCallHistoryRepository.deleteCallRecord(this, record.id)
                    Toast.makeText(this, "Call entry deleted", Toast.LENGTH_SHORT).show()
                    finish()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }

        // Transcript & Intelligence Section
        if (record.hasTranscript && record.transcript.isNotEmpty()) {
            binding.tvDetailStatusTag.text = "Available"
            binding.tvDetailStatusTag.setTextColor(ContextCompat.getColor(this, R.color.verdict_supported))
            binding.tvDetailEmptyTranscript.visibility = View.GONE

            if (record.aiSummary.isNotBlank()) {
                binding.tvDetailSummaryText.text = record.aiSummary
                binding.tvDetailSummaryText.visibility = View.VISIBLE
            } else {
                binding.tvDetailSummaryText.visibility = View.GONE
            }

            // Key Points
            binding.containerDetailKeyPoints.removeAllViews()
            if (record.keyPoints.isNotEmpty()) {
                val header = TextView(this).apply {
                    text = "Key Takeaways:"
                    textSize = 12f
                    textStyle(android.graphics.Typeface.BOLD)
                    setTextColor(ContextCompat.getColor(this@VaniCallDetailActivity, R.color.text_secondary))
                    setPadding(0, 8, 0, 4)
                }
                binding.containerDetailKeyPoints.addView(header)
                for (kp in record.keyPoints) {
                    val tv = TextView(this).apply {
                        text = "• $kp"
                        textSize = 13f
                        setTextColor(ContextCompat.getColor(this@VaniCallDetailActivity, R.color.text_primary))
                        setPadding(0, 2, 0, 2)
                    }
                    binding.containerDetailKeyPoints.addView(tv)
                }
            }

            // Action Items
            binding.containerDetailActionItems.removeAllViews()
            if (record.actionItems.isNotEmpty()) {
                val actHeader = TextView(this).apply {
                    text = "Action Items:"
                    textSize = 12f
                    textStyle(android.graphics.Typeface.BOLD)
                    setTextColor(ContextCompat.getColor(this@VaniCallDetailActivity, R.color.text_secondary))
                    setPadding(0, 8, 0, 4)
                }
                binding.containerDetailActionItems.addView(actHeader)
                for (act in record.actionItems) {
                    val cb = CheckBox(this).apply {
                        text = act
                        textSize = 13f
                        setTextColor(ContextCompat.getColor(this@VaniCallDetailActivity, R.color.text_primary))
                    }
                    binding.containerDetailActionItems.addView(cb)
                }
            }

            val adapter = VaniTranscriptAdapter(record.transcript.toMutableList())
            binding.recyclerDetailTranscript.layoutManager = LinearLayoutManager(this)
            binding.recyclerDetailTranscript.adapter = adapter
            binding.recyclerDetailTranscript.visibility = View.VISIBLE
        } else {
            binding.tvDetailStatusTag.text = "Not Monitored"
            binding.tvDetailStatusTag.setTextColor(ContextCompat.getColor(this, R.color.text_tertiary))
            binding.tvDetailSummaryText.visibility = View.GONE
            binding.recyclerDetailTranscript.visibility = View.GONE
            binding.tvDetailEmptyTranscript.visibility = View.VISIBLE
        }
    }

    private fun TextView.textStyle(style: Int) {
        setTypeface(typeface, style)
    }
}
