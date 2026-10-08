package com.vericall.voip

/**
 * Verdict policy for the VoIP (internet calling) screening path.
 *
 * Product decision (pending user review): the scorer backend returns tier
 * BLOCK with metrics.tier_override == "impostor" for ANY caller without an
 * enrolled voiceprint, regardless of the deepfake score. For the VoIP path,
 * where screened callers are unknown by definition, the app treats
 * "impostor" as "unenrolled / unknown caller" instead of a hard block, and
 * drives the in-call banner from the CM deepfake score (p_spoof).
 * The backend is untouched by this policy.
 */
enum class VoipBannerLevel { NEUTRAL, AMBER, RED }

data class VoipBanner(
    val level: VoipBannerLevel,
    val title: String,
    val detail: String
)

object VoipVerdictPolicy {

    fun decide(pSpoof: Double, tier: String?, tierOverride: String?): VoipBanner {
        val unknownCaller = tierOverride.equals("impostor", ignoreCase = true)
        val prefix = if (unknownCaller) "Unknown caller (not enrolled). " else ""
        val detail = buildString {
            append("server tier=")
            append(tier ?: "n/a")
            if (unknownCaller) append(", voiceprint=unenrolled")
            append(", p_spoof=").append("%.3f".format(pSpoof))
        }
        return when {
            pSpoof >= 0.7 -> VoipBanner(
                VoipBannerLevel.RED,
                prefix + "Possible AI-cloned voice",
                detail
            )
            pSpoof >= 0.4 -> VoipBanner(
                VoipBannerLevel.AMBER,
                prefix + "Uncertain",
                detail
            )
            else -> VoipBanner(
                VoipBannerLevel.NEUTRAL,
                prefix + "No deepfake detected",
                detail
            )
        }
    }
}
