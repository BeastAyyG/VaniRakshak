package com.vericall.temp

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.vericall.R
import com.vericall.model.VaniContact

class TempContactAdapter(
    context: Context,
    private val contacts: List<VaniContact>,
    private val onContactClicked: (VaniContact) -> Unit,
    private val onEnrollClicked: (VaniContact) -> Unit
) : ArrayAdapter<VaniContact>(context, R.layout.list_item_temp_contact, contacts) {

    private var visible: List<VaniContact> = contacts

    fun filter(query: String) {
        visible = if (query.isBlank()) contacts
        else contacts.filter {
            it.name.contains(query, ignoreCase = true) ||
                it.phone.contains(query) ||
                it.relation.contains(query, ignoreCase = true)
        }
        notifyDataSetChanged()
    }

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
        val tvBadge = view.findViewById<TextView>(R.id.tvBadge)
        val btnVerify = view.findViewById<TextView>(R.id.btnVerify)
        val btnEnroll = view.findViewById<TextView>(R.id.btnEnroll)

        tvAvatar?.text = contact.name.firstOrNull()?.toString() ?: "V"
        tvName?.text = contact.name
        tvRelation?.text = contact.relation

        if (contact.isVoiceEnrolled) {
            tvBadge?.text = "Voice enrolled · ${contact.enrollmentSec}s"
            tvBadge?.setBackgroundResource(R.drawable.bg_badge_enrolled)
            tvBadge?.setTextColor(ContextCompat.getColor(context, R.color.success_deep))
            btnEnroll?.text = "Voice"
        } else {
            tvBadge?.text = "Not enrolled"
            tvBadge?.setBackgroundResource(R.drawable.bg_badge_warning)
            tvBadge?.setTextColor(ContextCompat.getColor(context, R.color.warning_deep))
            btnEnroll?.text = "Enroll"
        }

        // Tapping the card or the Call button triggers the live call screen & notification
        val clickListener = View.OnClickListener {
            onContactClicked(contact)
        }
        rowContact?.setOnClickListener(clickListener)
        view.setOnClickListener(clickListener)
        btnVerify?.setOnClickListener(clickListener)

        // Tapping Voice triggers the enrollment sheet
        btnEnroll?.setOnClickListener {
            onEnrollClicked(contact)
        }

        return view
    }
}