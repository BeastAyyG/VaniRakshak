package com.vericall.ui

import android.content.Context
import android.text.format.DateUtils
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.vericall.R
import com.vericall.model.CallDirection
import com.vericall.model.VaniCallRecord
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class VaniCallHistoryAdapter(
    private val context: Context,
    private var calls: List<VaniCallRecord> = emptyList(),
    private val onCallClicked: (VaniCallRecord) -> Unit,
    private val onQuickCallClicked: (VaniCallRecord) -> Unit
) : RecyclerView.Adapter<VaniCallHistoryAdapter.ViewHolder>() {

    private val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val tvAvatar: TextView = view.findViewById(R.id.tvCallAvatar)
        val tvName: TextView = view.findViewById(R.id.tvCallName)
        val ivType: ImageView = view.findViewById(R.id.ivCallType)
        val tvTypeAndTime: TextView = view.findViewById(R.id.tvCallTypeAndTime)
        val tvDuration: TextView = view.findViewById(R.id.tvCallDuration)
        val tvTranscriptBadge: TextView = view.findViewById(R.id.tvTranscriptAvailableBadge)
        val btnQuickCall: ImageView = view.findViewById(R.id.btnQuickCall)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val v = LayoutInflater.from(parent.context).inflate(R.layout.item_call_history, parent, false)
        return ViewHolder(v)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = calls[position]

        val displayName = item.callerName.ifBlank { item.number.ifBlank { "Unknown" } }
        holder.tvName.text = displayName

        val initial = displayName.firstOrNull { it.isLetter() }?.uppercaseChar() ?: '?'
        holder.tvAvatar.text = initial.toString()

        val timeStr = formatCallTime(item.timestamp)

        when (item.direction) {
            CallDirection.INCOMING -> {
                holder.ivType.setImageResource(R.drawable.ic_call_incoming)
                holder.ivType.setColorFilter(ContextCompat.getColor(context, R.color.verdict_supported))
                holder.tvTypeAndTime.text = "Incoming • $timeStr"
                holder.tvName.setTextColor(ContextCompat.getColor(context, R.color.text_primary))
            }
            CallDirection.OUTGOING -> {
                holder.ivType.setImageResource(R.drawable.ic_call_outgoing)
                holder.ivType.setColorFilter(ContextCompat.getColor(context, R.color.brand_primary))
                holder.tvTypeAndTime.text = "Outgoing • $timeStr"
                holder.tvName.setTextColor(ContextCompat.getColor(context, R.color.text_primary))
            }
            CallDirection.MISSED -> {
                holder.ivType.setImageResource(R.drawable.ic_call_missed)
                holder.ivType.setColorFilter(ContextCompat.getColor(context, R.color.verdict_contradicted))
                holder.tvTypeAndTime.text = "Missed • $timeStr"
                holder.tvName.setTextColor(ContextCompat.getColor(context, R.color.verdict_contradicted))
            }
        }

        if (item.durationSec > 0) {
            val mins = item.durationSec / 60
            val secs = item.durationSec % 60
            holder.tvDuration.text = String.format(" • %02d:%02d", mins, secs)
            holder.tvDuration.visibility = View.VISIBLE
        } else {
            holder.tvDuration.visibility = View.GONE
        }

        holder.tvTranscriptBadge.visibility = if (item.hasTranscript) View.VISIBLE else View.GONE

        holder.itemView.setOnClickListener { onCallClicked(item) }
        holder.btnQuickCall.setOnClickListener { onQuickCallClicked(item) }
    }

    override fun getItemCount(): Int = calls.size

    fun updateData(newCalls: List<VaniCallRecord>) {
        calls = newCalls
        notifyDataSetChanged()
    }

    private fun formatCallTime(epoch: Long): String {
        if (epoch <= 0) return ""
        val now = System.currentTimeMillis()
        return if (DateUtils.isToday(epoch)) {
            timeFormat.format(Date(epoch))
        } else {
            DateUtils.getRelativeTimeSpanString(epoch, now, DateUtils.DAY_IN_MILLIS).toString()
        }
    }
}
