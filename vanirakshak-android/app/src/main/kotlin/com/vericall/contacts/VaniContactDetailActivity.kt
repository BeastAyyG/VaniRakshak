package com.vericall.contacts

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.vericall.R
import com.vericall.calling.VaniActiveCallActivity
import com.vericall.calling.VaniCallDetailActivity
import com.vericall.data.VaniCallHistoryRepository
import com.vericall.data.VaniContactRepository
import com.vericall.data.VaniSampleData
import com.vericall.databinding.ActivityVaniContactDetailBinding
import com.vericall.model.CallDirection
import com.vericall.model.VaniCallRecord
import com.vericall.model.VaniContact
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class VaniContactDetailActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_CONTACT_ID = "vani.contact.id"
    }

    private lateinit var binding: ActivityVaniContactDetailBinding
    private lateinit var contact: VaniContact
    private val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())
    private val dateFormat = SimpleDateFormat("MMM d", Locale.getDefault())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVaniContactDetailBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        val contactId = intent.getStringExtra(EXTRA_CONTACT_ID) ?: ""
        contact = VaniContactRepository.getContacts(this).find { it.id == contactId }
            ?: VaniSampleData.getContactById(contactId)
            ?: VaniSampleData.contacts[0]

        bindContact(contact)

        binding.btnBackContactDetail.setOnClickListener { finish() }
    }

    private fun bindContact(contact: VaniContact) {
        binding.tvContactDetailName.text = contact.name
        binding.tvContactDetailPhone.text = contact.phone
        binding.tvContactDetailRelation.text = contact.relation.ifBlank { "Personal Contact" }

        val initial = contact.name.firstOrNull { it.isLetter() }?.uppercaseChar() ?: '?'
        binding.tvContactDetailAvatar.text = initial.toString()

        updateFavoriteIcon()

        binding.btnToggleFavorite.setOnClickListener {
            val newFav = !contact.isFavorite
            VaniContactRepository.setFavorite(this, contact, newFav)
            updateFavoriteIcon()
            Toast.makeText(
                this,
                if (newFav) "Added to Favorites" else "Removed from Favorites",
                Toast.LENGTH_SHORT
            ).show()
        }

        binding.btnContactCall.setOnClickListener {
            VaniActiveCallActivity.startOutgoingCall(this, contact.phone, contact.name)
        }

        binding.btnContactMessage.setOnClickListener {
            val smsIntent = Intent(Intent.ACTION_VIEW).apply {
                data = Uri.parse("sms:${contact.phone}")
            }
            try {
                startActivity(smsIntent)
            } catch (_: Exception) {
                Toast.makeText(this, "No SMS app found", Toast.LENGTH_SHORT).show()
            }
        }

        // Voice Enrollment Status
        updateEnrollmentUI()

        binding.btnContactEnroll.setOnClickListener {
            VaniEnrollmentSheet.show(this, contact) {
                updateEnrollmentUI()
            }
        }

        loadContactCalls()
    }

    private fun updateFavoriteIcon() {
        if (contact.isFavorite) {
            binding.btnToggleFavorite.setColorFilter(ContextCompat.getColor(this, R.color.brand_primary))
        } else {
            binding.btnToggleFavorite.setColorFilter(ContextCompat.getColor(this, R.color.text_tertiary))
        }
    }

    private fun updateEnrollmentUI() {
        if (contact.isVoiceEnrolled) {
            binding.cardEnrollmentStatus.setBackgroundResource(R.drawable.bg_badge_enrolled)
            binding.ivBadgeEnrollCheck.setImageResource(R.drawable.ic_shield_check_cyber)
            binding.ivBadgeEnrollCheck.setColorFilter(ContextCompat.getColor(this, R.color.success_deep))
            binding.tvEnrollmentStatusTitle.text = "Voice Biometric Enrolled"
            binding.tvEnrollmentStatusTitle.setTextColor(ContextCompat.getColor(this, R.color.success_deep))
            binding.tvEnrollmentStatusDesc.text = "${contact.enrollmentSec}s acoustic baseline stored. Protected against synthetic clones."
            binding.tvEnrollActionLabel.text = "Update Voice"
        } else {
            binding.cardEnrollmentStatus.setBackgroundResource(R.drawable.bg_badge_warning)
            binding.ivBadgeEnrollCheck.setImageResource(R.drawable.ic_mic)
            binding.ivBadgeEnrollCheck.setColorFilter(ContextCompat.getColor(this, R.color.warning_deep))
            binding.tvEnrollmentStatusTitle.text = "Voice Not Enrolled"
            binding.tvEnrollmentStatusTitle.setTextColor(ContextCompat.getColor(this, R.color.warning_deep))
            binding.tvEnrollmentStatusDesc.text = "Enroll caller voice to enable zero-latency synthetic clone defense."
            binding.tvEnrollActionLabel.text = "Enroll Voice"
        }
    }

    private fun loadContactCalls() {
        binding.containerContactRecentCalls.removeAllViews()
        val calls = VaniCallHistoryRepository.getCallsForContact(this, contact.phone)

        if (calls.isEmpty()) {
            binding.tvNoContactCalls.visibility = View.VISIBLE
        } else {
            binding.tvNoContactCalls.visibility = View.GONE
            for (call in calls.take(8)) {
                val row = createCallRow(call)
                binding.containerContactRecentCalls.addView(row)
            }
        }
    }

    private fun createCallRow(record: VaniCallRecord): View {
        val view = LayoutInflater.from(this).inflate(R.layout.item_call_history, null, false)
        val tvName = view.findViewById<TextView>(R.id.tvCallName)
        val ivType = view.findViewById<ImageView>(R.id.ivCallType)
        val tvTypeAndTime = view.findViewById<TextView>(R.id.tvCallTypeAndTime)
        val tvDuration = view.findViewById<TextView>(R.id.tvCallDuration)
        val tvTranscriptBadge = view.findViewById<TextView>(R.id.tvTranscriptAvailableBadge)
        val btnCall = view.findViewById<ImageView>(R.id.btnQuickCall)
        val tvAvatar = view.findViewById<TextView>(R.id.tvCallAvatar)

        tvAvatar.visibility = View.GONE

        val dateStr = dateFormat.format(Date(record.timestamp))
        val timeStr = timeFormat.format(Date(record.timestamp))

        when (record.direction) {
            CallDirection.INCOMING -> {
                ivType.setImageResource(R.drawable.ic_call_incoming)
                ivType.setColorFilter(ContextCompat.getColor(this, R.color.verdict_supported))
                tvTypeAndTime.text = "Incoming • $dateStr, $timeStr"
            }
            CallDirection.OUTGOING -> {
                ivType.setImageResource(R.drawable.ic_call_outgoing)
                ivType.setColorFilter(ContextCompat.getColor(this, R.color.brand_primary))
                tvTypeAndTime.text = "Outgoing • $dateStr, $timeStr"
            }
            CallDirection.MISSED -> {
                ivType.setImageResource(R.drawable.ic_call_missed)
                ivType.setColorFilter(ContextCompat.getColor(this, R.color.verdict_contradicted))
                tvTypeAndTime.text = "Missed • $dateStr, $timeStr"
            }
        }

        tvName.text = if (record.aiSummary.isNotBlank()) record.aiSummary else record.number
        tvName.textSize = 14f

        if (record.durationSec > 0) {
            val mins = record.durationSec / 60
            val secs = record.durationSec % 60
            tvDuration.text = String.format(" • %02d:%02d", mins, secs)
            tvDuration.visibility = View.VISIBLE
        } else {
            tvDuration.visibility = View.GONE
        }

        tvTranscriptBadge.visibility = if (record.hasTranscript) View.VISIBLE else View.GONE

        view.setOnClickListener {
            val intent = Intent(this, VaniCallDetailActivity::class.java).apply {
                putExtra(VaniCallDetailActivity.EXTRA_RECORD_ID, record.id)
            }
            startActivity(intent)
        }

        btnCall.setOnClickListener {
            VaniActiveCallActivity.startOutgoingCall(this, record.number, contact.name)
        }

        return view
    }
}
