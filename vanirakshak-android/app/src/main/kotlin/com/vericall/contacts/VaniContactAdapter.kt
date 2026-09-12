package com.vericall.contacts

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.vericall.R
import com.vericall.model.VaniContact
import kotlin.math.abs

/**
 * Production contact roster adapter presenting enrolled voice profiles,
 * biometric baseline indicators, and incoming threat alerts.
 */
class VaniContactAdapter(
    context: Context,
    private val contacts: List<VaniContact>,
    private val onContactClicked: (VaniContact) -> Unit,
    private val onEnrollClicked: (VaniContact) -> Unit
) : ArrayAdapter<VaniContact>(context, R.layout.list_item_temp_contact, contacts) {

    private var allContacts: List<VaniContact> = contacts
    private var visible: List<VaniContact> = contacts
    var onFilterChanged: ((Int) -> Unit)? = null

    // 8 curated avatar background/text color pairs
    private val avatarBgColors = intArrayOf(
        R.color.avatar_1, R.color.avatar_2, R.color.avatar_3, R.color.avatar_4,
        R.color.avatar_5, R.color.avatar_6, R.color.avatar_7, R.color.avatar_8
    )
    private val avatarTextColors = intArrayOf(
        R.color.avatar_text_1, R.color.avatar_text_2, R.color.avatar_text_3, R.color.avatar_text_4,
        R.color.avatar_text_5, R.color.avatar_text_6, R.color.avatar_text_7, R.color.avatar_text_8
    )

    fun updateData(newContacts: List<VaniContact>) {
        allContacts = newContacts
        visible = newContacts
        notifyDataSetChanged()
        onFilterChanged?.invoke(visible.size)
    }

    fun filter(query: String) {
        visible = if (query.isBlank()) allContacts
        else allContacts.filter {
            it.name.contains(query, ignoreCase = true) ||
                it.phone.contains(query) ||
                it.relation.contains(query, ignoreCase = true)
        }
        notifyDataSetChanged()
        onFilterChanged?.invoke(visible.size)
    }

    fun getEnrolledCount(): Int = allContacts.count { it.isVoiceEnrolled }

    fun getThreatCount(): Int = allContacts.count { it.lastAlertText != null }

    override fun getCount(): Int = visible.size

    override fun getItem(position: Int): VaniContact = visible[position]

    override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
        val view = convertView ?: LayoutInflater.from(context)
            .inflate(R.layout.list_item_temp_contact, parent, false)
        val contact = visible[position]

        val rowContact = view.findViewById<View>(R.id.rowContact)
        val tvAvatar = view.findViewById<TextView>(R.id.tvAvatar)
        val tvName = view.findViewById<TextView>(R.id.tvName)
        val tvRelation = view.findViewById<TextView>(R.id.tvRelation)
        val dotEnroll = view.findViewById<View>(R.id.dotEnrollStatus)
        val tvBadge = view.findViewById<TextView>(R.id.tvBadge)
        val btnEnroll = view.findViewById<TextView>(R.id.btnEnroll)
        val btnVerify = view.findViewById<ImageView>(R.id.btnVerify)
        val rowAlert = view.findViewById<View>(R.id.rowAlert)
        val tvAlertText = view.findViewById<TextView>(R.id.tvAlertText)

        // Avatar: first letter + name-hashed color
        val initial = contact.name.firstOrNull()?.toString() ?: "V"
        tvAvatar?.text = initial
        val colorIndex = abs(contact.name.hashCode()) % 8
        tvAvatar?.background?.setTint(ContextCompat.getColor(context, avatarBgColors[colorIndex]))
        tvAvatar?.setTextColor(ContextCompat.getColor(context, avatarTextColors[colorIndex]))

        // Name & relation
        tvName?.text = contact.name
        tvRelation?.text = contact.relation

        // Enrollment status
        if (contact.isVoiceEnrolled) {
            tvBadge?.text = "Voice enrolled · ${contact.enrollmentSec}s"
            tvBadge?.setTextColor(ContextCompat.getColor(context, R.color.verdict_supported))
            dotEnroll?.backgroundTintList = ContextCompat.getColorStateList(context, R.color.verdict_supported)
            btnEnroll?.text = "Voice"
            btnEnroll?.setBackgroundResource(R.drawable.bg_pill_secondary)
            btnEnroll?.setTextColor(ContextCompat.getColor(context, R.color.text_primary))
        } else {
            tvBadge?.text = "Not enrolled"
            tvBadge?.setTextColor(ContextCompat.getColor(context, R.color.text_tertiary))
            dotEnroll?.backgroundTintList = ContextCompat.getColorStateList(context, R.color.text_tertiary)
            btnEnroll?.text = "Enroll"
            btnEnroll?.setBackgroundResource(R.drawable.bg_pill_primary)
            btnEnroll?.setTextColor(ContextCompat.getColor(context, R.color.white))
        }

        // Alert row: show if contact has an active alert
        if (!contact.lastAlertText.isNullOrBlank()) {
            rowAlert?.visibility = View.VISIBLE
            tvAlertText?.text = contact.lastAlertText
        } else {
            rowAlert?.visibility = View.GONE
        }

        // Click listeners
        val clickListener = View.OnClickListener {
            onContactClicked(contact)
        }
        rowContact?.setOnClickListener(clickListener)
        view.setOnClickListener(clickListener)
        btnVerify?.setOnClickListener(clickListener)

        // Enroll/Voice button
        btnEnroll?.setOnClickListener {
            onEnrollClicked(contact)
        }

        return view
    }
}
