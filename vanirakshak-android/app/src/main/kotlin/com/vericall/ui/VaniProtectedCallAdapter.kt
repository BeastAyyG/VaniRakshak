package com.vericall.ui

import android.content.Context
import android.content.Intent
import android.media.MediaPlayer
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.vericall.R
import com.vericall.calling.VaniActiveCallActivity
import com.vericall.calling.VaniCallSummaryActivity
import com.vericall.databinding.ItemProtectedCallBinding
import com.vericall.model.VaniCallRecord
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class VaniProtectedCallAdapter(
    private var records: List<VaniCallRecord> = emptyList(),
    private val onRecordAction: ((VaniCallRecord) -> Unit)? = null
) : RecyclerView.Adapter<VaniProtectedCallAdapter.ViewHolder>() {

    private var currentlyPlayingId: String? = null
    private var mediaPlayer: MediaPlayer? = null
    private var currentPlayingHolder: ViewHolder? = null
    private val handler = Handler(Looper.getMainLooper())
    private var updateProgressRunnable: Runnable? = null

    private val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())
    private val dateFormat = SimpleDateFormat("MMM d", Locale.getDefault())

    fun updateRecords(newRecords: List<VaniCallRecord>) {
        stopPlayback()
        records = newRecords
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemProtectedCallBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val record = records[position]
        holder.bind(record)
    }

    override fun getItemCount(): Int = records.size

    fun stopPlayback() {
        handler.removeCallbacksAndMessages(null)
        updateProgressRunnable = null
        try {
            mediaPlayer?.stop()
            mediaPlayer?.release()
        } catch (_: Exception) {}
        mediaPlayer = null
        currentlyPlayingId = null
        currentPlayingHolder?.resetPlayerUI()
        currentPlayingHolder = null
    }

    inner class ViewHolder(val binding: ItemProtectedCallBinding) :
        RecyclerView.ViewHolder(binding.root) {

        fun bind(record: VaniCallRecord) {
            val context = binding.root.context
            val name = record.callerName.ifBlank { record.number.ifBlank { "Unknown Caller" } }
            binding.tvArchivedCallerName.text = name
            binding.tvArchivedNumber.text = record.number.ifBlank { "Unknown number" }

            val initial = name.firstOrNull { it.isLetter() }?.uppercaseChar() ?: 'V'
            binding.tvArchivedAvatar.text = initial.toString()

            val mins = record.durationSec / 60
            val secs = record.durationSec % 60
            val durStr = String.format("%02d:%02d", mins, secs)
            val dateStr = dateFormat.format(Date(record.timestamp))
            val timeStr = timeFormat.format(Date(record.timestamp))
            binding.tvArchivedMeta.text = "$dateStr • $timeStr • $durStr"

            // Risk badge styling
            val score = record.riskScore
            when {
                score >= 80 -> {
                    binding.pillArchivedRisk.setBackgroundResource(R.drawable.bg_badge_scam)
                    binding.dotArchivedRisk.backgroundTintList = ContextCompat.getColorStateList(context, R.color.verdict_scam)
                    binding.tvArchivedRiskText.text = "Critical • $score%"
                    binding.tvArchivedRiskText.setTextColor(ContextCompat.getColor(context, R.color.verdict_scam))
                }
                score >= 60 -> {
                    binding.pillArchivedRisk.setBackgroundResource(R.drawable.bg_badge_unverifiable)
                    binding.dotArchivedRisk.backgroundTintList = ContextCompat.getColorStateList(context, R.color.verdict_unverifiable)
                    binding.tvArchivedRiskText.text = "High • $score%"
                    binding.tvArchivedRiskText.setTextColor(ContextCompat.getColor(context, R.color.verdict_unverifiable))
                }
                score >= 40 -> {
                    binding.pillArchivedRisk.setBackgroundResource(R.drawable.bg_pill_neobrutal_butter)
                    binding.dotArchivedRisk.backgroundTintList = ContextCompat.getColorStateList(context, R.color.neo_ink)
                    binding.tvArchivedRiskText.text = "Moderate • $score%"
                    binding.tvArchivedRiskText.setTextColor(ContextCompat.getColor(context, R.color.neo_ink))
                }
                score >= 20 -> {
                    binding.pillArchivedRisk.setBackgroundResource(R.drawable.bg_pill_neobrutal_paper)
                    binding.dotArchivedRisk.backgroundTintList = ContextCompat.getColorStateList(context, R.color.text_secondary)
                    binding.tvArchivedRiskText.text = "Low • $score%"
                    binding.tvArchivedRiskText.setTextColor(ContextCompat.getColor(context, R.color.neo_ink))
                }
                else -> {
                    binding.pillArchivedRisk.setBackgroundResource(R.drawable.bg_badge_supported)
                    binding.dotArchivedRisk.backgroundTintList = ContextCompat.getColorStateList(context, R.color.verdict_supported)
                    binding.tvArchivedRiskText.text = "Safe • $score%"
                    binding.tvArchivedRiskText.setTextColor(ContextCompat.getColor(context, R.color.verdict_supported))
                }
            }

            // Threat alert row
            if (record.scamType.isNotBlank() && score >= 35) {
                binding.layoutThreatAlert.visibility = View.VISIBLE
                binding.tvThreatAlertText.text = "🚨 Scam Alert: ${record.scamType}"
            } else {
                binding.layoutThreatAlert.visibility = View.GONE
            }

            // High-risk evidence preservation marker (>=50%)
            if (record.isProtected || score >= 50) {
                binding.layoutProtectedEvidenceMarker.visibility = View.VISIBLE
                binding.tvProtectedMarkerText.text = "🔒 Preserved Evidence: Audio WAV & Transcript TXT archived on disk"
            } else {
                binding.layoutProtectedEvidenceMarker.visibility = View.GONE
            }

            // Audio player setup
            val audioFile = record.audioFilePath?.let { File(it) }
            val hasAudio = audioFile != null && audioFile.exists() && audioFile.length() > 500

            if (hasAudio) {
                binding.layoutAudioPlayer.visibility = View.VISIBLE
                binding.tvNoAudioMessage.visibility = View.GONE

                if (currentlyPlayingId == record.id && mediaPlayer != null) {
                    currentPlayingHolder = this
                    binding.btnPlayPauseAudio.setImageResource(R.drawable.ic_pause)
                } else {
                    resetPlayerUI()
                }

                binding.btnPlayPauseAudio.setOnClickListener {
                    togglePlay(record, audioFile!!)
                }

                binding.seekAudioProgress.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                        if (fromUser && currentlyPlayingId == record.id && mediaPlayer != null) {
                            val duration = mediaPlayer!!.duration
                            val newPos = (duration * (progress / 100f)).toInt()
                            mediaPlayer!!.seekTo(newPos)
                        }
                    }
                    override fun onStartTrackingTouch(seekBar: SeekBar?) {}
                    override fun onStopTrackingTouch(seekBar: SeekBar?) {}
                })

            } else {
                binding.layoutAudioPlayer.visibility = View.GONE
                binding.tvNoAudioMessage.visibility = View.VISIBLE
            }

            // Button: View Transcript
            binding.btnViewTranscript.setOnClickListener {
                showTranscriptDialog(context, record)
            }

            // Button: Safety Report
            binding.btnViewReport.setOnClickListener {
                val intent = Intent(context, VaniCallSummaryActivity::class.java).apply {
                    putExtra(VaniCallSummaryActivity.EXTRA_CALL_ID, record.id)
                }
                context.startActivity(intent)
            }

            // Button: Call
            binding.btnCallBack.setOnClickListener {
                VaniActiveCallActivity.startOutgoingCall(context, record.number, record.callerName)
            }
        }

        fun resetPlayerUI() {
            binding.btnPlayPauseAudio.setImageResource(R.drawable.ic_play_arrow)
            binding.seekAudioProgress.progress = 0
            binding.tvAudioDuration.text = "00:00 / 00:00"
        }

        private fun togglePlay(record: VaniCallRecord, audioFile: File) {
            if (currentlyPlayingId == record.id && mediaPlayer != null) {
                if (mediaPlayer!!.isPlaying) {
                    mediaPlayer!!.pause()
                    binding.btnPlayPauseAudio.setImageResource(R.drawable.ic_play_arrow)
                } else {
                    mediaPlayer!!.start()
                    binding.btnPlayPauseAudio.setImageResource(R.drawable.ic_pause)
                    startProgressTracker()
                }
                return
            }

            // Stop previous playing audio
            stopPlayback()

            try {
                mediaPlayer = MediaPlayer().apply {
                    setDataSource(audioFile.absolutePath)
                    prepare()
                    start()
                }
                currentlyPlayingId = record.id
                currentPlayingHolder = this
                binding.btnPlayPauseAudio.setImageResource(R.drawable.ic_pause)

                mediaPlayer?.setOnCompletionListener {
                    stopPlayback()
                }

                startProgressTracker()

            } catch (e: Exception) {
                Toast.makeText(binding.root.context, "Could not play audio file", Toast.LENGTH_SHORT).show()
                resetPlayerUI()
            }
        }

        private fun startProgressTracker() {
            handler.removeCallbacksAndMessages(null)
            updateProgressRunnable = object : Runnable {
                override fun run() {
                    val mp = mediaPlayer
                    if (mp != null && mp.isPlaying) {
                        val current = mp.currentPosition
                        val total = mp.duration
                        val progress = if (total > 0) ((current.toFloat() / total) * 100).toInt() else 0
                        binding.seekAudioProgress.progress = progress

                        val curMin = (current / 1000) / 60
                        val curSec = (current / 1000) % 60
                        val totMin = (total / 1000) / 60
                        val totSec = (total / 1000) % 60
                        binding.tvAudioDuration.text = String.format("%02d:%02d / %02d:%02d", curMin, curSec, totMin, totSec)

                        handler.postDelayed(this, 300)
                    }
                }
            }
            handler.post(updateProgressRunnable!!)
        }

        private fun showTranscriptDialog(context: Context, record: VaniCallRecord) {
            val transcriptText = if (record.transcript.isNotEmpty()) {
                record.transcript.joinToString("\n\n") { "[${it.speaker}]\n${it.text}" }
            } else {
                "No transcript available for this call."
            }

            AlertDialog.Builder(context)
                .setTitle("📄 Transcript • ${record.callerName}")
                .setMessage(transcriptText)
                .setPositiveButton("Copy") { _, _ ->
                    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                    val clip = android.content.ClipData.newPlainText("VaniRakshak Transcript", transcriptText)
                    clipboard.setPrimaryClip(clip)
                    Toast.makeText(context, "Transcript copied to clipboard", Toast.LENGTH_SHORT).show()
                }
                .setNeutralButton("Open Report") { _, _ ->
                    val intent = Intent(context, VaniCallSummaryActivity::class.java).apply {
                        putExtra(VaniCallSummaryActivity.EXTRA_CALL_ID, record.id)
                    }
                    context.startActivity(intent)
                }
                .setNegativeButton("Close", null)
                .show()
        }
    }
}
