package com.vericall.verification

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.vericall.R
import com.vericall.databinding.ActivityScamResultBinding

class ScamResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityScamResultBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityScamResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        binding.btnBack.setOnClickListener { finish() }

        val statement = intent.getStringExtra(EXTRA_STATEMENT) ?: "Suspicious statement under review"
        val threatScore = intent.getIntExtra(EXTRA_THREAT_SCORE, 82)
        val scamType = intent.getStringExtra(EXTRA_SCAM_TYPE) ?: "Financial Impersonation Fraud"

        binding.tvSuspiciousText.text = statement
        binding.tvThreatScore.text = "$threatScore% CRITICAL"
        binding.pbThreatMeter.progress = threatScore
        binding.tvScamTypeTag.text = scamType

        binding.btnHelpline.setOnClickListener {
            try {
                val dialIntent = Intent(Intent.ACTION_DIAL, Uri.parse("tel:1930"))
                startActivity(dialIntent)
            } catch (_: Exception) {}
        }

        val advice = intent.getStringExtra(EXTRA_ACTION_ADVICE)
        if (!advice.isNullOrBlank()) {
            binding.tvActionAdvice.text = advice
        }

        populateRedFlags()
    }

    private fun populateRedFlags() {
        binding.containerRedFlags.removeAllViews()
        val flags = mutableListOf<RedFlag>()
        val flagsJson = intent.getStringExtra(EXTRA_RED_FLAGS_JSON)

        if (!flagsJson.isNullOrBlank()) {
            try {
                val arr = org.json.JSONArray(flagsJson)
                for (i in 0 until arr.length()) {
                    val item = arr.get(i)
                    if (item is org.json.JSONObject) {
                        flags.add(
                            RedFlag(
                                number = item.optString("number", String.format("%02d", i + 1)),
                                category = item.optString("category", "Threat Trigger"),
                                quote = item.optString("quote", "")
                            )
                        )
                    } else if (item is String) {
                        flags.add(
                            RedFlag(
                                number = String.format("%02d", i + 1),
                                category = "High Risk Coercion",
                                quote = item
                            )
                        )
                    }
                }
            } catch (_: Exception) {}
        }

        if (flags.isEmpty()) {
            flags.add(RedFlag("01", "Artificial Urgency", "Demands immediate action before you can independently check with your branch."))
            flags.add(RedFlag("02", "Sensitive Credential Solicitation", "Attempts to capture confidential security codes or login OTPs."))
            flags.add(RedFlag("03", "Authority Impersonation", "Fraudulently claims official affiliation with regulatory or police agencies."))
        }

        for (flag in flags) {
            val itemView = LayoutInflater.from(this).inflate(R.layout.item_red_flag, binding.containerRedFlags, false)
            itemView.findViewById<TextView>(R.id.tvFlagNumber).text = flag.number
            itemView.findViewById<TextView>(R.id.tvFlagCategory).text = flag.category
            itemView.findViewById<TextView>(R.id.tvFlagQuote).text = flag.quote
            binding.containerRedFlags.addView(itemView)
        }
    }

    companion object {
        const val EXTRA_STATEMENT = "extra_statement"
        const val EXTRA_THREAT_SCORE = "extra_threat_score"
        const val EXTRA_SCAM_TYPE = "extra_scam_type"
        const val EXTRA_RED_FLAGS_JSON = "extra_red_flags_json"
        const val EXTRA_ACTION_ADVICE = "extra_action_advice"
    }
}
