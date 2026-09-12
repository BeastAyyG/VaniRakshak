package com.vericall.util

import android.app.Activity
import android.app.role.RoleManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.telecom.TelecomManager
import androidx.appcompat.app.AlertDialog
import com.vericall.R

/**
 * Helper to check and request Default Phone Dialer role on Android 10+ and legacy devices.
 */
object VaniDefaultDialerManager {

    const val REQUEST_CODE_SET_DEFAULT_DIALER = 201

    fun isDefaultDialer(context: Context): Boolean {
        val telecomManager = context.getSystemService(Context.TELECOM_SERVICE) as? TelecomManager
            ?: return false
        return telecomManager.defaultDialerPackage == context.packageName
    }

    fun requestDefaultDialer(activity: Activity, requestCode: Int = REQUEST_CODE_SET_DEFAULT_DIALER) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val roleManager = activity.getSystemService(RoleManager::class.java)
            if (roleManager != null && roleManager.isRoleAvailable(RoleManager.ROLE_DIALER)) {
                if (!roleManager.isRoleHeld(RoleManager.ROLE_DIALER)) {
                    val intent = roleManager.createRequestRoleIntent(RoleManager.ROLE_DIALER)
                    activity.startActivityForResult(intent, requestCode)
                    return
                }
            }
        }

        // Fallback for pre-Q or when RoleManager is unavailable
        val telecomManager = activity.getSystemService(Context.TELECOM_SERVICE) as? TelecomManager
        if (telecomManager != null && telecomManager.defaultDialerPackage != activity.packageName) {
            val intent = Intent(TelecomManager.ACTION_CHANGE_DEFAULT_DIALER).apply {
                putExtra(TelecomManager.EXTRA_CHANGE_DEFAULT_DIALER_PACKAGE_NAME, activity.packageName)
            }
            try {
                activity.startActivityForResult(intent, requestCode)
            } catch (_: Exception) {}
        }
    }

    fun promptToSetDefaultDialer(activity: Activity, onConfirmed: (() -> Unit)? = null) {
        if (isDefaultDialer(activity)) {
            onConfirmed?.invoke()
            return
        }

        AlertDialog.Builder(activity)
            .setTitle("Direct In-App Calling")
            .setMessage("To make and receive phone calls directly inside VaniRakshak with live AI voice clone defense, set VaniRakshak as your default phone app.")
            .setPositiveButton("Set as Default Caller") { _, _ ->
                requestDefaultDialer(activity)
                onConfirmed?.invoke()
            }
            .setNegativeButton("Later") { _, _ ->
                onConfirmed?.invoke()
            }
            .show()
    }
}
