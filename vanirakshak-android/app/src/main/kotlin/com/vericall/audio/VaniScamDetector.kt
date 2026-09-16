package com.vericall.audio

import android.util.Log

/**
 * VaniScamDetector — Android-native, zero-network fraud detection engine.
 *
 * Operates entirely on-device on finalized transcript text.
 * Covers all scam patterns in English, Hindi, and Hinglish.
 * Extracts verifiable claims for optional Gemini API verification.
 * Redacts credentials (OTP/PIN/CVV) before any external logging.
 *
 * SECURITY: Caller speech is untrusted data. This class NEVER uses
 * caller-supplied text to construct URLs or execute external commands.
 * Only sanitized, extracted claims are forwarded for verification.
 */
object VaniScamDetector {

    private const val TAG = "VaniScamDetector"

    data class DetectionResult(
        val threatScore: Int,               // 0–100
        val redFlags: List<String>,
        val matchedCategories: Set<ScamCategory>,
        val extractedClaim: String?,        // Sanitized claim for Gemini verification
        val redactedText: String,           // Text with credentials removed
        val isHighRisk: Boolean
    )

    enum class ScamCategory(val weight: Int, val label: String) {
        CREDENTIAL_REQUEST(40, "Credential Request"),
        ACCOUNT_THREAT(30, "Account Threat"),
        IMPERSONATION(25, "Government/Bank Impersonation"),
        URGENCY(20, "Artificial Urgency"),
        FINANCIAL_ACTION(20, "Financial Action Request"),
        REMOTE_ACCESS(35, "Remote Access Request"),
        DIGITAL_ARREST(45, "Digital Arrest Scam")
    }

    // ─── Pattern sets ──────────────────────────────────────────────────────────

    private val credentialPatterns = listOf(
        Regex("""\botp\b""", RegexOption.IGNORE_CASE),
        Regex("""\bpin\b""", RegexOption.IGNORE_CASE),
        Regex("""\bcvv\b""", RegexOption.IGNORE_CASE),
        Regex("""\bpassword\b""", RegexOption.IGNORE_CASE),
        Regex("""\bsecurity code\b""", RegexOption.IGNORE_CASE),
        Regex("""\bverification code\b""", RegexOption.IGNORE_CASE),
        Regex("""\batm pin\b""", RegexOption.IGNORE_CASE),
        Regex("""\bupi pin\b""", RegexOption.IGNORE_CASE),
        Regex("""ओटीपी"""),
        Regex("""पासवर्ड"""),
        Regex("""सुरक्षा कोड"""),
        Regex("""otp\s*(bata|share|send|do|dena|dijiye|batao|bhejo)""", RegexOption.IGNORE_CASE),
        Regex("""otp\s*kya\s*(hai|tha|hoga)""", RegexOption.IGNORE_CASE)
    )

    private val accountThreatPatterns = listOf(
        Regex("""\baccount\s*(will\s*be\s*)?(block|blocked|suspend|suspended|deactivat|close|closed)\b""", RegexOption.IGNORE_CASE),
        Regex("""\bkyc\s*(expir|pending|incomplet|verif)\b""", RegexOption.IGNORE_CASE),
        Regex("""\baccount\s*(fraud|hack|compromise)\b""", RegexOption.IGNORE_CASE),
        Regex("""(खाता|अकाउंट|अकाउन्ट|बैंक).*?(बंद|ब्लॉक|निलंबित|सस्पेंड)"""),
        Regex("""account\s*(band|block)\s*(hoga|ho\s*jayega|kar\s*diya|ho\s*gaya)""", RegexOption.IGNORE_CASE),
        Regex("""kyc\s*(pending|expired|incomplete|update\s*karna|complete\s*karo)""", RegexOption.IGNORE_CASE),
        Regex("""(केवाईसी|kyc).*?(पेंडिंग|एक्सपायर|इकसपाया|अपडेट)"""),
        Regex("""(बिजली|बिजली\s*बिल).*?(काट|कट|बकाया|बंद)""")
    )

    private val impersonationPatterns = listOf(
        Regex("""\b(rbi|reserve\s*bank)\b""", RegexOption.IGNORE_CASE),
        Regex("""\b(sebi|enforcement\s*directorate)\b""", RegexOption.IGNORE_CASE),
        Regex("""\bincome\s*tax\s*(department|officer|notice)\b""", RegexOption.IGNORE_CASE),
        Regex("""\bcybercrime\s*(department|officer|cell)\b""", RegexOption.IGNORE_CASE),
        Regex("""\b(trai|telecom\s*regulatory)\b""", RegexOption.IGNORE_CASE),
        Regex("""\b(government|govt)\s*(officer|official|department)\b""", RegexOption.IGNORE_CASE),
        Regex("""\bcalling\s*from\s*(your\s*)?(bank|rbi|government|telecom)\b""", RegexOption.IGNORE_CASE),
        Regex("""आरबीआई"""),
        Regex("""आयकर\s*(विभाग|अधिकारी)"""),
        Regex("""साइबर\s*क्राइम"""),
        Regex("""(rbi|trai|cbi)\s*(se|ki\s*taraf\s*se)\s*(bol\s*(raha|rahi)|call)\b""", RegexOption.IGNORE_CASE)
    )

    private val urgencyPatterns = listOf(
        Regex("""\b(immediately|right\s*now|within\s*\d+\s*(minutes?|hours?)|today\s*only)\b""", RegexOption.IGNORE_CASE),
        Regex("""\b(or\s*(else\s*)?your\s*account|otherwise\s*your\s*account)\b""", RegexOption.IGNORE_CASE),
        Regex("""\burgent(ly)?\b""", RegexOption.IGNORE_CASE),
        Regex("""(अभी|तुरंत)\s*(करें|karo|karna)"""),
        Regex("""आज\s*ही"""),
        Regex("""(turant|abhi)\s*(karo|bataiye|dijiye|share)""", RegexOption.IGNORE_CASE)
    )

    private val financialActionPatterns = listOf(
        Regex("""\b(transfer|send|deposit|pay)\s*(money|funds|amount|rs\.?|inr)\b""", RegexOption.IGNORE_CASE),
        Regex("""\brefund\s*(process|initiat|credit)\b""", RegexOption.IGNORE_CASE),
        Regex("""\b(upi|neft|rtgs|imps)\s*(transfer|send|pay)\b""", RegexOption.IGNORE_CASE),
        Regex("""पैसे\s*(भेजें|ट्रांसफर|जमा)"""),
        Regex("""(paisa|paise|rupees?|rs)\s*(bhejo|transfer|send)\s*(karo|karna|kijiye)""", RegexOption.IGNORE_CASE),
        Regex("""(refund|वापस)\s*(milega|aayega|bhejna\s*hai)""", RegexOption.IGNORE_CASE)
    )

    private val remoteAccessPatterns = listOf(
        Regex("""\b(anydesk|teamviewer|quicksupport|rustdesk)\b""", RegexOption.IGNORE_CASE),
        Regex("""\bscreen\s*shar(e|ing)\b""", RegexOption.IGNORE_CASE),
        Regex("""\bremote\s*(access|control|connect)\b""", RegexOption.IGNORE_CASE),
        Regex("""\b(download|install)\s*(this\s*)?(app|apk|software|tool)\b""", RegexOption.IGNORE_CASE),
        Regex("""\ballow\s*(me|us)\s*to\s*(access|control|view)\b""", RegexOption.IGNORE_CASE)
    )

    private val digitalArrestPatterns = listOf(
        Regex("""\bdigital\s*arrest\b""", RegexOption.IGNORE_CASE),
        Regex("""\b(arrest|arrested|warrant)\s*(will\s*be\s*)?(issu|sent|made|filed)\b""", RegexOption.IGNORE_CASE),
        Regex("""\bfir\s*(will\s*be\s*)?(filed|register)\b""", RegexOption.IGNORE_CASE),
        Regex("""\bpolice\s*(will\s*(come|arrive)|case|complaint)\b""", RegexOption.IGNORE_CASE),
        Regex("""\bjail\b""", RegexOption.IGNORE_CASE),
        Regex("""डिजिटल\s*अरेस्ट"""),
        Regex("""(police|fir|arrest)\s*(hoga|ho\s*jayega|karwa\s*denge)""", RegexOption.IGNORE_CASE),
        Regex("""(giraftari|गिरफ्तार)""", RegexOption.IGNORE_CASE)
    )

    // ─── Credential redaction ──────────────────────────────────────────────────

    private val otpNumberPattern = Regex(
        """(?i)(otp|code|pin|cvv|ओटीपी|पिन)\s*[:\-]?\s*(\d{4,8})"""
    )

    fun redact(text: String): String {
        return otpNumberPattern.replace(text) { mr ->
            "${mr.groupValues[1]} [REDACTED]"
        }
    }

    // ─── Main detection entry point ────────────────────────────────────────────

    /**
     * Analyzes a finalized utterance for scam indicators.
     * Input: raw transcript text from SpeechRecognizer.
     * Returns a [DetectionResult] with threat score, red flags, and optional verification claim.
     */
    fun analyze(text: String): DetectionResult {
        if (text.isBlank()) {
            return DetectionResult(0, emptyList(), emptySet(), null, text, false)
        }

        val redacted = redact(text)
        val matched = mutableSetOf<ScamCategory>()
        val flags = mutableListOf<String>()

        fun check(patterns: List<Regex>, category: ScamCategory, flagPrefix: String) {
            for (p in patterns) {
                if (p.containsMatchIn(text)) {
                    matched.add(category)
                    val snippet = p.find(text)?.value?.take(40) ?: ""
                    flags.add("$flagPrefix: \"$snippet\"")
                    break
                }
            }
        }

        check(credentialPatterns, ScamCategory.CREDENTIAL_REQUEST, "Credential request")
        check(accountThreatPatterns, ScamCategory.ACCOUNT_THREAT, "Account blocking threat")
        check(impersonationPatterns, ScamCategory.IMPERSONATION, "Authority impersonation")
        check(urgencyPatterns, ScamCategory.URGENCY, "Artificial urgency")
        check(financialActionPatterns, ScamCategory.FINANCIAL_ACTION, "Financial action")
        check(remoteAccessPatterns, ScamCategory.REMOTE_ACCESS, "Remote access request")
        check(digitalArrestPatterns, ScamCategory.DIGITAL_ARREST, "Digital arrest threat")

        val rawScore = matched.sumOf { it.weight }
        val threatScore = rawScore.coerceAtMost(100)

        val extractedClaim = if (matched.size >= 2) {
            buildClaimForVerification(matched, redacted)
        } else null

        val isHighRisk = threatScore >= 50
                || ScamCategory.DIGITAL_ARREST in matched
                || ScamCategory.CREDENTIAL_REQUEST in matched

        Log.d(TAG, "Analyzed: score=$threatScore categories=$matched")

        return DetectionResult(
            threatScore = threatScore,
            redFlags = flags,
            matchedCategories = matched,
            extractedClaim = extractedClaim,
            redactedText = redacted,
            isHighRisk = isHighRisk
        )
    }

    /**
     * Builds a sanitized, safe claim string for external Gemini verification.
     * Never includes raw credential values.
     */
    private fun buildClaimForVerification(
        categories: Set<ScamCategory>,
        redactedText: String
    ): String {
        val sb = StringBuilder()
        if (ScamCategory.IMPERSONATION in categories) {
            sb.append("Caller claims to represent a government or bank authority. ")
        }
        if (ScamCategory.ACCOUNT_THREAT in categories) {
            sb.append("Caller claims the recipient's bank account will be blocked. ")
        }
        if (ScamCategory.CREDENTIAL_REQUEST in categories) {
            sb.append("Caller is requesting sensitive credentials (OTP/PIN). ")
        }
        if (ScamCategory.DIGITAL_ARREST in categories) {
            sb.append("Caller is threatening arrest or legal action. ")
        }
        if (ScamCategory.REMOTE_ACCESS in categories) {
            sb.append("Caller is requesting installation of remote access software. ")
        }
        sb.append("(Verbatim, credentials redacted): \"$redactedText\"")
        return sb.toString().take(500)
    }

    /**
     * Quick immediate-threat check for sub-second UI alerting.
     * Checks only the highest-priority patterns.
     */
    fun hasImmediateThreat(text: String): Boolean {
        if (text.isBlank()) return false
        return credentialPatterns.any { it.containsMatchIn(text) }
                || digitalArrestPatterns.any { it.containsMatchIn(text) }
    }
}
