package com.vericall.shield

import android.content.Context
import com.vericall.model.RiskLevel
import com.vericall.util.VaniRiskBus

/**
 * Orchestrates the Scam-Shield subsystem during live telephony sessions.
 * Ties together OTP suppression, foreground app surveillance, and warning overlays.
 */
object VaniScamShieldController {

    // Target financial, UPI, and banking app identifiers
    val bankingPackages: Set<String> = setOf(
        "com.google.android.apps.nbu.paisa.user", // Google Pay
        "com.phonepe.app",                        // PhonePe
        "net.one97.paytm",                        // Paytm
        "in.org.npci.upiapp",                     // BHIM UPI
        "com.dreamplug.androidapp",               // CRED
        "com.snapwork.hdfc",                      // HDFC Bank MobileBanking
        "com.csam.icici.bank.imobile",            // ICICI iMobile Pay
        "com.sbi.lotusintouch",                   // SBI YONO
        "com.axis.mobile",                        // Axis Mobile
        "com.msf.kbank.mobile",                   // Kotak Mobile Banking
        "com.bankofbaroda.mconnect",              // bob World
        "com.fss.pnb"                             // PNB ONE
    )

    @Volatile
    var isCallActive: Boolean = false
        private set

    @Volatile
    var isUserInterlockEnabled: Boolean = false
        private set

    var activeCallerName: String = "Caller"
        private set

    var activeCallerPhone: String = ""
        private set

    private var peakRiskRecorded = 0

    var onHeldNotificationsChanged: ((Int) -> Unit)? = null

    fun setUserInterlockEnabled(enabled: Boolean) {
        isUserInterlockEnabled = enabled
    }

    /**
     * The shield is active if a call is ongoing AND either the user engaged the
     * interlock manually OR the voice analyzer flags the call as HIGH CLONE RISK (DANGER).
     */
    fun isShieldActive(): Boolean {
        if (!isCallActive) return false
        val isThreatLevel = VaniRiskBus.last?.level == RiskLevel.DANGER
        return isUserInterlockEnabled || isThreatLevel
    }

    fun onCallStarted(callerName: String, callerPhone: String) {
        isCallActive = true
        activeCallerName = callerName
        activeCallerPhone = callerPhone
        peakRiskRecorded = 0
        VaniIncidentLogger.clearActiveHeldNotifications()
        VaniIncidentLogger.clearActiveOverlayTriggers()
        onHeldNotificationsChanged?.invoke(0)
    }

    fun onRiskScoreUpdated(risk: Int) {
        if (risk > peakRiskRecorded) {
            peakRiskRecorded = risk
        }
    }

    fun onNotificationHeld() {
        val count = VaniIncidentLogger.getActiveHeldNotifications().size
        onHeldNotificationsChanged?.invoke(count)
    }

    fun onCallEnded(context: Context, lastSnapshot: VaniRiskBus.Snapshot?) {
        if (!isCallActive) return

        val summary = lastSnapshot?.let {
            "${it.level.name}: ${it.risk}% risk (${it.message})"
        } ?: "Call ended. Normal disconnect."

        val finalPeak = maxOf(peakRiskRecorded, lastSnapshot?.risk ?: 0)

        VaniIncidentLogger.recordIncident(
            context = context,
            callerNumber = activeCallerPhone,
            callerName = activeCallerName,
            peakRisk = finalPeak,
            verdictSummary = summary
        )

        isCallActive = false
        isUserInterlockEnabled = false
        activeCallerName = "Caller"
        activeCallerPhone = ""
    }

    fun resetForTesting() {
        isCallActive = false
        isUserInterlockEnabled = false
        activeCallerName = "Caller"
        activeCallerPhone = ""
        peakRiskRecorded = 0
    }
}
