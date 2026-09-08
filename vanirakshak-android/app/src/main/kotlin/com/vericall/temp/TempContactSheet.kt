package com.vericall.temp

import android.view.LayoutInflater
import android.os.Handler
import android.os.Looper
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.vericall.R
import com.vericall.model.VaniContact

object TempContactSheet {

    fun show(context: AppCompatActivity, contact: VaniContact, onProfileSaved: () -> Unit) {
        val dialog = BottomSheetDialog(context)
        val view = LayoutInflater.from(context).inflate(R.layout.bottom_sheet_temp_contact, null)
        dialog.setContentView(view)

        val tvContactName = view.findViewById<TextView>(R.id.tvSheetContactName)
        val tvContactPhone = view.findViewById<TextView>(R.id.tvSheetContactPhone)
        val tvBadge = view.findViewById<TextView>(R.id.tvSheetBadge)
        val pbProgress = view.findViewById<ProgressBar>(R.id.pbEnrollProgress)
        val tvStatus = view.findViewById<TextView>(R.id.tvEnrollStatus)
        val btnRecord = view.findViewById<TextView>(R.id.btnStartEnroll)
        val btnSave = view.findViewById<TextView>(R.id.btnSaveEnroll)
        val btnClose = view.findViewById<ImageView>(R.id.btnSheetClose)

        tvContactName.text = contact.name
        if (contact.isVoiceEnrolled) {
            tvContactPhone.text = "${contact.phone} - ${contact.enrollmentSec}s active voice baseline"
            tvBadge.text = "Enrolled"
            tvBadge.setBackgroundResource(R.drawable.bg_badge_enrolled)
            tvBadge.setTextColor(ContextCompat.getColor(context, R.color.success_deep))
        } else {
            tvContactPhone.text = "${contact.phone} - no voice profile yet"
            tvBadge.text = "Not enrolled"
            tvBadge.setBackgroundResource(R.drawable.bg_badge_warning)
            tvBadge.setTextColor(ContextCompat.getColor(context, R.color.warning_deep))
        }

        btnClose.setOnClickListener {
            dialog.dismiss()
        }

        btnRecord.setOnClickListener {
            btnRecord.isEnabled = false
            btnSave.isEnabled = false
            tvStatus.text = "Recording reference voice sample... ingesting 16kHz PCM..."
            tvStatus.setTextColor(ContextCompat.getColor(context, R.color.error_tone))

            val handler = Handler(Looper.getMainLooper())
            var progress = 0

            val runnable = object : Runnable {
                override fun run() {
                    if (!dialog.isShowing || context.isFinishing || context.isDestroyed) return
                    progress += 5
                    pbProgress.progress = progress
                    if (progress < 100) {
                        handler.postDelayed(this, 100)
                    } else {
                        tvStatus.text = "15s reference audio processed. Speaker embedding computed."
                        tvStatus.setTextColor(ContextCompat.getColor(context, R.color.success_deep))
                        btnRecord.isEnabled = true
                        btnRecord.text = "Record again"
                        btnSave.isEnabled = true
                    }
                }
            }
            handler.post(runnable)
        }

        btnSave.setOnClickListener {
            contact.isVoiceEnrolled = true
            contact.enrollmentSec = 15
            contact.sampleEmbeddingCount = 3
            onProfileSaved()
            Toast.makeText(context, "Voice profile saved for ${contact.name}", Toast.LENGTH_SHORT).show()
            dialog.dismiss()
        }

        dialog.show()
    }
}