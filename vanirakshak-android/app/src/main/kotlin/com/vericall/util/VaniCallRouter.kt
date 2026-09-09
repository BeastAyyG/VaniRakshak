package com.vericall.util

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.ContactsContract
import androidx.core.content.ContextCompat
import com.vericall.data.VaniSampleData
import com.vericall.service.VaniCallGuardService
import com.vericall.state.CallStateStore

/**
 * One place that decides what happens when the phone rings, is answered, or ends.
 *
 * Both the legacy PHONE_STATE broadcast and the modern CallScreeningService feed
 * into here, so the behaviour can never drift between the two detection paths.
 */
object VaniCallRouter {

    data class Caller(
        val name: String,
        val phone: String,
        val enrolled: Boolean,
        val known: Boolean
    )

    /** Keeps digits only, so "+91 98765 43210" and "09876543210" compare equal. */
    fun normalize(number: String): String = number.filter { it.isDigit() }

    fun sameNumber(a: String, b: String): Boolean {
        val da = normalize(a)
        val db = normalize(b)
        if (da.isEmpty() || db.isEmpty()) return false
        return da.takeLast(10) == db.takeLast(10)
    }

    /**
     * Resolves a caller's identity against enrolled sample profiles first,
     * then against the device's native address book (if READ_CONTACTS is granted),
     * and finally falls back to the raw phone number.
     */
    fun resolve(rawNumber: String?): Caller = resolve(null, rawNumber)

    fun resolve(context: Context?, rawNumber: String?): Caller {
        val phone = rawNumber?.takeIf { it.isNotBlank() } ?: ""
        if (normalize(phone).isEmpty()) {
            return Caller(name = "Unknown caller", phone = "", enrolled = false, known = false)
        }

        // 1. Check verified / enrolled sample data
        val sampleContact = VaniSampleData.contacts.find { sameNumber(it.phone, phone) }
        if (sampleContact != null) {
            return Caller(sampleContact.name, phone, sampleContact.isVoiceEnrolled, known = true)
        }

        // 2. Query device phonebook if context and permission are available
        if (context != null && ContextCompat.checkSelfPermission(
                context, Manifest.permission.READ_CONTACTS
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            val phonebookName = lookupDeviceContactName(context, phone)
            if (!phonebookName.isNullOrBlank()) {
                return Caller(phonebookName, phone, enrolled = false, known = true)
            }
        }

        return Caller(name = phone, phone = phone, enrolled = false, known = false)
    }

    private fun lookupDeviceContactName(context: Context, number: String): String? {
        return try {
            val uri = Uri.withAppendedPath(
                ContactsContract.PhoneLookup.CONTENT_FILTER_URI,
                Uri.encode(number)
            )
            context.contentResolver.query(
                uri,
                arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME),
                null,
                null,
                null
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val nameIdx = cursor.getColumnIndex(ContactsContract.PhoneLookup.DISPLAY_NAME)
                    if (nameIdx != -1) cursor.getString(nameIdx) else null
                } else null
            }
        } catch (e: Exception) {
            null
        }
    }

    /** RINGING - alert the user immediately, no risk number has been earned yet. */
    fun onRinging(context: Context, rawNumber: String?) {
        if (!rawNumber.isNullOrBlank()) {
            CallStateStore.remember(context, rawNumber)
        }
        val effectiveNumber = rawNumber?.takeIf { it.isNotBlank() } ?: CallStateStore.recall(context)
        val caller = resolve(context, effectiveNumber)
        VaniNotificationHelper.createChannels(context)
        VaniNotificationHelper.showIncomingAlert(context, caller)
    }

    /** OFFHOOK - hand over to the foreground guard so the live meter can run. */
    fun onAnswered(context: Context, rawNumber: String?) {
        val effectiveNumber = rawNumber?.takeIf { it.isNotBlank() } ?: CallStateStore.recall(context)
        val caller = resolve(context, effectiveNumber)
        VaniNotificationHelper.cancelAlert(context)
        startGuard(context, caller)
    }

    /**
     * IDLE - clear the alert and stop capture.
     *
     * Cancelling the alert has to happen before the service stops, because the
     * service posts its final summary on the same tag/id as it shuts down. For a
     * call that was never answered there is no service, so without this the
     * ringing alert would sit in the shade indefinitely.
     */
    fun onEnded(context: Context) {
        VaniNotificationHelper.cancelAlert(context)
        context.stopService(Intent(context, VaniCallGuardService::class.java))
        CallStateStore.clear(context)
    }

    /**
     * Starts the guard foreground service.
     *
     * Returns false when the OS refuses the start. Android 12+ blocks foreground
     * service launches from the background, which is exactly the situation a
     * broadcast receiver is in. Callers use the result to fall back to the
     * full-screen guard screen, which is in the foreground and therefore allowed
     * to start the service.
     */
    fun startGuard(context: Context, caller: Caller): Boolean = try {
        val intent = Intent(context, VaniCallGuardService::class.java).apply {
            putExtra(VaniNotificationHelper.EXTRA_CALLER_NAME, caller.name)
            putExtra(VaniNotificationHelper.EXTRA_CALLER_PHONE, caller.phone)
            putExtra(VaniNotificationHelper.EXTRA_CALLER_ENROLLED, caller.enrolled)
        }
        ContextCompat.startForegroundService(context, intent)
        true
    } catch (e: android.app.ForegroundServiceStartNotAllowedException) {
        // Android 12+: FGS launch from background is denied. Caller falls back
        // to the foreground guard screen, which may then start the service.
        android.util.Log.w("VaniCallRouter", "FGS start denied (background): ${e.message}")
        false
    } catch (e: SecurityException) {
        android.util.Log.w("VaniCallRouter", "FGS start denied (security): ${e.message}")
        false
    } catch (e: Exception) {
        android.util.Log.w("VaniCallRouter", "FGS start failed: ${e.message}")
        false
    }
}
