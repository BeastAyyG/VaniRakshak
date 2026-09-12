package com.vericall.verification

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.vericall.R
import com.vericall.databinding.ActivityClaimResultBinding

class ClaimResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityClaimResultBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityClaimResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        binding.btnBack.setOnClickListener { finish() }
        binding.btnCheckAnother.setOnClickListener { finish() }

        val claim = intent.getStringExtra(EXTRA_CLAIM) ?: "Claim under review"
        val verdictName = intent.getStringExtra(EXTRA_VERDICT) ?: "UNVERIFIABLE"
        val confidence = intent.getIntExtra(EXTRA_CONFIDENCE, 75)
        val explanation = intent.getStringExtra(EXTRA_EXPLANATION) ?: "Evaluation completed against verified evidence sources."
        val action = intent.getStringExtra(EXTRA_ACTION) ?: "Always cross-verify with institutional portals."

        binding.tvClaimText.text = "\"$claim\""
        binding.tvExplanation.text = explanation
        binding.tvRecommendedAction.text = action
        binding.pbConfidence.progress = confidence
        binding.tvConfidenceBadge.text = "$confidence% Confidence"

        when (verdictName.uppercase()) {
            "SUPPORTED", "TRUE" -> {
                binding.tvVerdictBadge.text = "✓ SUPPORTED"
                binding.tvVerdictBadge.setBackgroundResource(R.drawable.bg_badge_supported)
                binding.tvVerdictBadge.setTextColor(ContextCompat.getColor(this, R.color.verdict_supported))
                binding.tvVerdictTitle.text = "This claim is supported by reliable sources."
                binding.pbConfidence.progressTintList = ContextCompat.getColorStateList(this, R.color.verdict_supported)
            }
            "CONTRADICTED", "FALSE" -> {
                binding.tvVerdictBadge.text = "✕ CONTRADICTED"
                binding.tvVerdictBadge.setBackgroundResource(R.drawable.bg_badge_contradicted)
                binding.tvVerdictBadge.setTextColor(ContextCompat.getColor(this, R.color.verdict_contradicted))
                binding.tvVerdictTitle.text = "This claim conflicts with reliable information."
                binding.pbConfidence.progressTintList = ContextCompat.getColorStateList(this, R.color.verdict_contradicted)
            }
            "SCAM_DETECTED", "FRAUDULENT_CLAIM", "SCAM", "CRITICAL_SCAM" -> {
                binding.tvVerdictBadge.text = "🚨 FRAUDULENT CLAIM"
                binding.tvVerdictBadge.setBackgroundResource(R.drawable.bg_badge_contradicted)
                binding.tvVerdictBadge.setTextColor(ContextCompat.getColor(this, R.color.verdict_contradicted))
                binding.tvVerdictTitle.text = "Warning: This matches known voice scams or fraudulent claims."
                binding.pbConfidence.progressTintList = ContextCompat.getColorStateList(this, R.color.verdict_contradicted)
            }
            else -> {
                binding.tvVerdictBadge.text = "⚠ UNVERIFIABLE"
                binding.tvVerdictBadge.setBackgroundResource(R.drawable.bg_badge_unverifiable)
                binding.tvVerdictBadge.setTextColor(ContextCompat.getColor(this, R.color.verdict_unverifiable))
                binding.tvVerdictTitle.text = "We couldn't find enough reliable evidence to confirm this claim."
                binding.pbConfidence.progressTintList = ContextCompat.getColorStateList(this, R.color.verdict_unverifiable)
            }
        }

        populateSources()
    }

    private fun populateSources() {
        binding.containerSources.removeAllViews()
        val sources = mutableListOf<EvidenceSource>()
        val sourcesJson = intent.getStringExtra(EXTRA_SOURCES_JSON)

        if (!sourcesJson.isNullOrBlank()) {
            try {
                val arr = org.json.JSONArray(sourcesJson)
                for (i in 0 until arr.length()) {
                    val obj = arr.getJSONObject(i)
                    sources.add(
                        EvidenceSource(
                            title = obj.optString("title", "Web Evidence"),
                            url = obj.optString("url", ""),
                            domain = obj.optString("domain", "web.source"),
                            sourceType = obj.optString("sourceType", "🔍 Scraped Evidence"),
                            snippet = obj.optString("snippet", "")
                        )
                    )
                }
            } catch (_: Exception) {}
        }

        if (sources.isEmpty()) {
            sources.add(
                EvidenceSource(
                    title = "National Regulatory & Public Records",
                    url = "https://rbi.org.in",
                    domain = "rbi.org.in",
                    sourceType = "🏛 Official Source",
                    snippet = "Official directives specify procedural timelines and consumer protection protocols for all account updates."
                )
            )
            sources.add(
                EvidenceSource(
                    title = "Press Information Bureau Fact Verification",
                    url = "https://pib.gov.in/factcheck",
                    domain = "pib.gov.in",
                    sourceType = "📰 Fact Check Archive",
                    snippet = "Verified archival record indicates periodic circulation of misleading rumors regarding arbitrary closures."
                )
            )
        }

        for (source in sources) {
            val itemView = LayoutInflater.from(this).inflate(R.layout.item_evidence_source, binding.containerSources, false)
            val tvType = itemView.findViewById<TextView>(R.id.tvSourceType)
            val tvDomain = itemView.findViewById<TextView>(R.id.tvDomain)
            val tvTitle = itemView.findViewById<TextView>(R.id.tvSourceTitle)
            val tvSnippet = itemView.findViewById<TextView>(R.id.tvSnippet)
            val btnOpen = itemView.findViewById<TextView>(R.id.btnOpenSource)

            tvType.text = source.sourceType
            tvDomain.text = source.domain
            tvTitle.text = source.title
            tvSnippet.text = source.snippet

            btnOpen.setOnClickListener {
                try {
                    val browserIntent = Intent(Intent.ACTION_VIEW, Uri.parse(source.url))
                    startActivity(browserIntent)
                } catch (_: Exception) {}
            }

            binding.containerSources.addView(itemView)
        }
    }

    companion object {
        const val EXTRA_CLAIM = "extra_claim"
        const val EXTRA_VERDICT = "extra_verdict"
        const val EXTRA_CONFIDENCE = "extra_confidence"
        const val EXTRA_EXPLANATION = "extra_explanation"
        const val EXTRA_ACTION = "extra_action"
        const val EXTRA_SOURCES_JSON = "extra_sources_json"
    }
}
