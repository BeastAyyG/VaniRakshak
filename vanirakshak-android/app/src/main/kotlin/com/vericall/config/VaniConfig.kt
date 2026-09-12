package com.vericall.config

import android.app.Activity
import android.content.Context
import android.content.SharedPreferences
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog

/**
 * Centralized configuration manager for VaniRakshak.
 * Manages the backend server endpoint and forensic runtime settings.
 */
object VaniConfig {

    private const val PREFS_NAME = "vanirakshak_config"
    private const val KEY_BACKEND_URL = "config_backend_url"
    const val DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

    private fun getPrefs(context: Context): SharedPreferences {
        return context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    /**
     * Retrieves the configured backend base URL (e.g. "http://192.168.1.10:8000" or "http://127.0.0.1:8000").
     */
    fun getBackendUrl(context: Context): String {
        return getPrefs(context).getString(KEY_BACKEND_URL, DEFAULT_BACKEND_URL)
            ?.trim()
            ?.removeSuffix("/")
            ?: DEFAULT_BACKEND_URL
    }

    /**
     * Updates the backend base URL.
     */
    fun setBackendUrl(context: Context, url: String) {
        val cleanUrl = url.trim().removeSuffix("/")
        getPrefs(context).edit().putString(KEY_BACKEND_URL, cleanUrl).apply()
    }

    /**
     * Retrieves the host address (e.g. "127.0.0.1" or "192.168.1.15").
     */
    fun getHost(context: Context): String {
        val url = getBackendUrl(context)
        return try {
            val uri = java.net.URI(url)
            uri.host ?: "127.0.0.1"
        } catch (_: Exception) {
            "127.0.0.1"
        }
    }

    /**
     * Forensic Core HTTP base URL (Port 8000).
     */
    fun getForensicHttpUrl(context: Context): String {
        val host = getHost(context)
        return "http://$host:8000"
    }

    /**
     * Forensic Core WebSocket URL (Port 8000 /ws/stream).
     */
    fun getForensicWsUrl(context: Context): String {
        val host = getHost(context)
        return "ws://$host:8000/ws/stream"
    }

    /**
     * Indic ASR & Scam Shield HTTP base URL (Port 8084).
     */
    fun getIndicHttpUrl(context: Context): String {
        val host = getHost(context)
        return "http://$host:8084"
    }

    /**
     * Indic ASR & Scam Shield WebSocket URL (Port 8084 /ws/live-call-stream).
     */
    fun getIndicWsUrl(context: Context, callId: String, callerPhone: String): String {
        val host = getHost(context)
        val encodedCaller = try {
            java.net.URLEncoder.encode(callerPhone, "UTF-8")
        } catch (_: Exception) {
            callerPhone
        }
        return "ws://$host:8084/ws/live-call-stream?call_id=$callId&caller=$encodedCaller"
    }

    /**
     * Displays a clean configuration dialog for inspecting or updating the backend endpoint.
     */
    fun showServerConfigDialog(activity: Activity, onUpdated: ((String) -> Unit)? = null) {
        val currentUrl = getBackendUrl(activity)
        val input = EditText(activity).apply {
            setText(currentUrl)
            setSelection(currentUrl.length)
            hint = "http://192.168.x.x:8000"
            setPadding(48, 36, 48, 36)
        }

        AlertDialog.Builder(activity)
            .setTitle("⚙️ Backend Server Configuration")
            .setMessage("Set the host machine IP. Both the Forensic Core (Port 8000) and Indic ASR / Scam Shield (Port 8084) will route to this host.\n\nFor physical device over USB, use 127.0.0.1 with 'adb reverse'.")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                val newUrl = input.text.toString().trim()
                if (newUrl.isNotBlank() && (newUrl.startsWith("http://") || newUrl.startsWith("https://"))) {
                    setBackendUrl(activity, newUrl)
                    Toast.makeText(activity, "✓ Backend host set: ${getHost(activity)}", Toast.LENGTH_SHORT).show()
                    onUpdated?.invoke(newUrl)
                } else {
                    Toast.makeText(activity, "Invalid URL. Must start with http:// or https://", Toast.LENGTH_LONG).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .setNeutralButton("Reset Default") { _, _ ->
                setBackendUrl(activity, DEFAULT_BACKEND_URL)
                Toast.makeText(activity, "Reset to default: $DEFAULT_BACKEND_URL", Toast.LENGTH_SHORT).show()
                onUpdated?.invoke(DEFAULT_BACKEND_URL)
            }
            .show()
    }
}
