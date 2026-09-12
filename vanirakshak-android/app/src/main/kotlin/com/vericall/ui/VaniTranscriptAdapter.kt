package com.vericall.ui

import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.vericall.R
import com.vericall.model.TranscriptUtterance
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class VaniTranscriptAdapter(
    private val utterances: MutableList<TranscriptUtterance> = mutableListOf()
) : RecyclerView.Adapter<VaniTranscriptAdapter.ViewHolder>() {

    private val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val rootLayout: LinearLayout = view as LinearLayout
        val headerLayout: LinearLayout = view.findViewById(R.id.layoutUtteranceHeader)
        val tvSpeaker: TextView = view.findViewById(R.id.tvSpeakerLabel)
        val tvTime: TextView = view.findViewById(R.id.tvUtteranceTime)
        val dotLive: View = view.findViewById(R.id.dotLiveTranscribing)
        val bubble: LinearLayout = view.findViewById(R.id.bubbleContainer)
        val tvText: TextView = view.findViewById(R.id.tvUtteranceText)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val v = LayoutInflater.from(parent.context).inflate(R.layout.item_transcript_utterance, parent, false)
        return ViewHolder(v)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = utterances[position]
        holder.tvText.text = item.text
        holder.tvSpeaker.text = item.speaker.uppercase(Locale.getDefault())
        holder.tvTime.text = timeFormat.format(Date(if (item.timestamp > 0) item.timestamp else System.currentTimeMillis()))

        val isYou = item.speaker.equals("You", ignoreCase = true) || item.speaker.equals("Me", ignoreCase = true)

        if (isYou) {
            holder.rootLayout.gravity = Gravity.END
            holder.headerLayout.gravity = Gravity.END
            holder.bubble.setBackgroundResource(R.drawable.bg_transcript_bubble_you)
            holder.tvSpeaker.setTextColor(ContextCompat.getColor(holder.itemView.context, R.color.brand_primary))
        } else {
            holder.rootLayout.gravity = Gravity.START
            holder.headerLayout.gravity = Gravity.START
            holder.bubble.setBackgroundResource(R.drawable.bg_transcript_bubble_caller)
            holder.tvSpeaker.setTextColor(ContextCompat.getColor(holder.itemView.context, R.color.text_secondary))
        }

        holder.dotLive.visibility = if (!item.isFinal) View.VISIBLE else View.GONE
        holder.tvText.alpha = if (item.isFinal) 1.0f else 0.75f
    }

    override fun getItemCount(): Int = utterances.size

    fun addOrUpdateUtterance(utterance: TranscriptUtterance) {
        if (utterances.isNotEmpty() && !utterances.last().isFinal && utterance.speaker == utterances.last().speaker) {
            utterances[utterances.lastIndex] = utterance
            notifyItemChanged(utterances.lastIndex)
        } else {
            utterances.add(utterance)
            notifyItemInserted(utterances.lastIndex)
        }
    }

    fun setAll(newUtterances: List<TranscriptUtterance>) {
        utterances.clear()
        utterances.addAll(newUtterances)
        notifyDataSetChanged()
    }

    fun getAll(): List<TranscriptUtterance> = utterances.toList()
}
