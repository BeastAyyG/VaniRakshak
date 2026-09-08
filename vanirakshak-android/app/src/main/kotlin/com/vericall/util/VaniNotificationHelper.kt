package com.vericall.util

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.vericall.R
import com.vericall.receiver.VaniNotificationActionReceiver
import com.vericall.model.CallScenario
import com.vericall.model.RiskLevel
import com.vericall.model.VaniContact
import com.vericall.temp.TempContactDetailActivity
import com.vericall.temp.TempContactsActivity

/**
 * Every notification VaniRakshak shows during a call.
 *
 * Two channels, deliberately:
 *  - [CHANNEL_ALERT] is high importance and owns the moment a call arrives
 *    (heads-up + full-screen intent).
 *  - [CHANNEL_LIVE] is low importance so the ongoing risk meter can refresh
 *    every second without re-alerting the user.
 */
object VaniNotificationHelper {

    const val CHANNEL_ALERT = "vani_call_alert"
    const val CHANNEL_LIVE = "vani_live_risk"

    const val ID_ALERT = 1001
    const val ID_LIVE = 1002

    const val EXTRA_CALLER_NAME = "vani.caller.name"
    const val EXTRA_CALLER_PHONE = "vani.caller.phone"
    const val EXTRA_CALLER_ENROLLED = "vani.caller.enrolled"

    private const val TAG_ALERT = "vani_alert"
    private const val TAG_LIVE = "vani_live"
    private const val TAG_DEMO = "vani_demo"

    // ------------------------------------------------------------------ channels

    fun createChannels(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = context.getSystemService(NotificationManager::class.java) ?: return

        val alert = NotificationChannel(
            CHANNEL_ALERT,
            "Incoming call alerts",
            NotificationManager.IMPORTANCE_HIGH
        ).apply {
            description = "Alerts you the moment a call arrives, with the live clone-risk verdict."
            // The ringtone is already playing - do not stack a second sound on top.
            setSound(null, null)
            enableVibration(true)
            vibrationPattern = longArrayOf(0L, 250L, 150L, 250L)
        }

        val live = NotificationChannel(
            CHANNEL_LIVE,
            "Live voice analysis",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Ongoing clone-risk meter while a call is being analyzed."
            setSound(null, null)
            enableVibration(false)
        }

        manager.createNotificationChannel(alert)
        manager.createNotificationChannel(live)
    }

    // ------------------------------------------------------------------ intents

    private fun pendingIntentFlags(): Int =
        PendingIntent.FLAG_UPDATE_CURRENT or
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE else 0

    private fun guardIntent(
        context: Context,
        name: String,
        phone: String,
        enrolled: Boolean = false
    ): Intent = Intent(context, TempContactDetailActivity::class.java).apply {
        putExtra(TempContactsActivity.EXTRA_CONTACT_ID, "live_call")
        putExtra(EXTRA_CALLER_NAME, name)
        putExtra(EXTRA_CALLER_PHONE, phone)
        putExtra(EXTRA_CALLER_ENROLLED, enrolled)
        flags = Intent.FLAG_ACTIVITY_NEW_TASK or
            Intent.FLAG_ACTIVITY_CLEAR_TOP or
            Intent.FLAG_ACTIVITY_SINGLE_TOP
    }

    private fun guardPendingIntent(
        context: Context,
        name: String,
        phone: String,
        requestCode: Int,
        enrolled: Boolean = false
    ): PendingIntent = PendingIntent.getActivity(
        context,
        requestCode,
        guardIntent(context, name, phone, enrolled),
        pendingIntentFlags()
    )

    // ------------------------------------------------------------------ labels

    fun levelLabel(context: Context, level: RiskLevel): String = context.getString(
        when (level) {
            RiskLevel.DANGER -> R.string.risk_level_high
            RiskLevel.WARNING -> R.string.risk_level_medium
            RiskLevel.SAFE -> R.string.risk_level_low
        }
    )

    private fun levelColor(context: Context, level: RiskLevel?): Int = ContextCompat.getColor(
        context,
        when (level) {
            RiskLevel.DANGER -> R.color.risk_high
            RiskLevel.WARNING -> R.color.risk_medium
            RiskLevel.SAFE -> R.color.risk_low
            null -> R.color.risk_neutral
        }
    )

    // ------------------------------------------------------------------ incoming

    /**
     * RINGING state. Honest: no risk number is shown because nothing has been
     * analyzed yet. Returns true when the notification actually reached the shade.
     */
    fun showIncomingAlert(context: Context, caller: VaniCallRouter.Caller): Boolean {
        createChannels(context)

        val body = if (caller.known) {
            context.getString(R.string.notif_incoming_known, caller.name)
        } else {
            context.getString(R.string.notif_incoming_unknown)
        }

        val notification = NotificationCompat.Builder(context, CHANNEL_ALERT)
            .setSmallIcon(R.drawable.ic_stat_shield)
            .setContentTitle(context.getString(R.string.notif_incoming_title, caller.name))
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setContentIntent(guardPendingIntent(context, caller.name, caller.phone, 11, caller.enrolled))
            .setFullScreenIntent(guardPendingIntent(context, caller.name, caller.phone, 12, caller.enrolled), true)
            .addAction(
                0,
                context.getString(R.string.notif_action_open),
                guardPendingIntent(context, caller.name, caller.phone, 13, caller.enrolled)
            )
            .setColor(ContextCompat.getColor(context, R.color.risk_neutral))
            .setAutoCancel(true)
            .setOngoing(false)
            .build()

        return notifySafe(context, TAG_ALERT, ID_ALERT, notification)
    }

    // ------------------------------------------------------------------ live meter

    /** Neutral state used for startForeground before the first verdict exists. */
    fun buildAnalyzingNotification(
        context: Context,
        callerName: String,
        callerPhone: String = "",
        enrolled: Boolean = false
    ): Notification {
        createChannels(context)
        val body = context.getString(R.string.notif_analyzing_body, callerName)
        return NotificationCompat.Builder(context, CHANNEL_LIVE)
            .setSmallIcon(R.drawable.ic_stat_shield)
            .setContentTitle(context.getString(R.string.notif_analyzing_title))
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setProgress(100, 0, true)
            .setColor(ContextCompat.getColor(context, R.color.risk_neutral))
            .setContentIntent(guardPendingIntent(context, callerName, callerPhone, 21, enrolled))
            .build()
    }

    /**
     * The live verdict. Replaces the notification in place every time the
     * analyzer produces a new score, so the shade shows the current risk level.
     */
    fun showLiveRiskNotification(
        context: Context,
        callerName: String,
        callerPhone: String,
        riskScore: Int,
        syntheticScore: Int,
        speakerMatch: Int,
        level: RiskLevel,
        message: String,
        enrolled: Boolean = false
    ) {
        createChannels(context)

        val label = levelLabel(context, level)
        val detail = context.getString(
            R.string.notif_live_detail,
            syntheticScore,
            speakerMatch
        )
        val body = "$message • $detail"

        val notification = NotificationCompat.Builder(context, CHANNEL_LIVE)
            .setSmallIcon(R.drawable.ic_stat_shield)
            .setContentTitle(context.getString(R.string.notif_live_title, label, riskScore))
            .setContentText(body)
            .setStyle(
                NotificationCompat.BigTextStyle()
                    .bigText(body)
                    .setSummaryText(callerName)
            )
            .setSubText(callerName)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setProgress(100, riskScore.coerceIn(0, 100), false)
            .setColor(levelColor(context, level))
            .setColorized(true)
            .setContentIntent(guardPendingIntent(context, callerName, callerPhone, 22, enrolled))
            .addAction(
                0,
                context.getString(R.string.notif_action_open),
                guardPendingIntent(context, callerName, callerPhone, 23, enrolled)
            )
            .addAction(
                0,
                context.getString(R.string.notif_action_block),
                VaniNotificationActionReceiver.blockIntent(context, callerName)
            )
            .build()

        notifySafe(context, TAG_LIVE, ID_LIVE, notification)
    }

    // ------------------------------------------------------------------ demo + end

    /** Demo path: shows the preset scenario verdict for a sample contact. */
    fun showCallRiskNotification(
        context: Context,
        contact: VaniContact,
        scenario: CallScenario
    ) {
        createChannels(context)

        val label = levelLabel(context, scenario.level)
        val body = context.getString(
            R.string.notif_demo_detail,
            scenario.acousticSyntheticScore,
            scenario.speakerCosineScore
        )

        val intent = Intent(context, TempContactDetailActivity::class.java).apply {
            putExtra(TempContactsActivity.EXTRA_CONTACT_ID, contact.id)
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val pending = PendingIntent.getActivity(context, contact.id.hashCode(), intent, pendingIntentFlags())

        val notification = NotificationCompat.Builder(context, CHANNEL_ALERT)
            .setSmallIcon(R.drawable.ic_stat_shield)
            .setContentTitle(context.getString(R.string.notif_demo_title, label, scenario.riskScore))
            .setContentText(contact.name)
            .setStyle(
                NotificationCompat.BigTextStyle()
                    .bigText("${scenario.verdictTitle}\n$body")
                    .setSummaryText(contact.name)
            )
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setProgress(100, scenario.riskScore.coerceIn(0, 100), false)
            .setColor(levelColor(context, scenario.level))
            .setAutoCancel(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(pending)
            .build()

        notifySafe(context, "$TAG_DEMO:${contact.id}", ID_ALERT, notification)
    }

    /** Final verdict after the call disconnects. */
    fun showCallSummary(context: Context, callerName: String, snapshot: VaniRiskBus.Snapshot) {
        createChannels(context)
        val label = levelLabel(context, snapshot.level)
        val body = context.getString(
            R.string.notif_summary_detail,
            snapshot.risk,
            snapshot.synthetic,
            snapshot.speakerMatch
        )

        val notification = NotificationCompat.Builder(context, CHANNEL_ALERT)
            .setSmallIcon(R.drawable.ic_stat_shield)
            .setContentTitle(context.getString(R.string.notif_summary_title, callerName, label))
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText("${snapshot.message}\n$body"))
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setColor(levelColor(context, snapshot.level))
            .setAutoCancel(true)
            .setOnlyAlertOnce(true)
            .build()

        notifySafe(context, TAG_ALERT, ID_ALERT, notification)
    }

    // ------------------------------------------------------------------ plumbing

    private fun notifySafe(
        context: Context,
        tag: String,
        id: Int,
        notification: Notification
    ): Boolean = try {
        NotificationManagerCompat.from(context).notify(tag, id, notification)
        true
    } catch (e: SecurityException) {
        // POST_NOTIFICATIONS was never granted on Android 13+.
        false
    }

    fun cancelAlert(context: Context) {
        getManager(context)?.cancel(TAG_ALERT, ID_ALERT)
    }

    fun cancelLive(context: Context) {
        getManager(context)?.cancel(TAG_LIVE, ID_LIVE)
    }

    /** Back-compat entry point: clears every call-related notification. */
    fun cancelNotification(context: Context) {
        val manager = getManager(context) ?: return
        manager.cancel(TAG_ALERT, ID_ALERT)
        manager.cancel(TAG_LIVE, ID_LIVE)
    }

    fun areNotificationsEnabled(context: Context): Boolean =
        NotificationManagerCompat.from(context).areNotificationsEnabled()

    private fun getManager(context: Context): NotificationManager? =
        context.getSystemService(NotificationManager::class.java)
}
