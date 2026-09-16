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
    private const val KEY_GEMINI_API_KEY = "config_gemini_api_key"

    /**
     * Default backend URL pointing to the reverse-port-forwarded forensic core (Port 8000).
     * Connects via `adb reverse tcp:8000 tcp:8000` and `adb reverse tcp:8084 tcp:8084`.
     */
    const val DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

    /**
     * Known candidate host IPs for seamless auto-fallback (USB reverse proxy vs Wi-Fi LAN IP).
     * 10.1.41.213 is the primary PC Wi-Fi IP; 127.0.0.1 is ADB reverse proxy.
     */
    val CANDIDATE_HOSTS = listOf("127.0.0.1", "10.1.41.213", "192.168.15.1", "192.168.56.1")

    private fun getPrefs(context: Context): SharedPreferences {
        return context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    /**
     * Returns candidate hosts in order of priority:
     * 1. User configured host
     * 2. USB reverse proxy (127.0.0.1)
     * 3. Wi-Fi IP (10.1.41.213)
     */
    fun getCandidateHosts(context: Context): List<String> {
        val configured = getHost(context)
        val list = mutableListOf<String>()
        if (configured.isNotBlank()) list.add(configured)
        for (h in CANDIDATE_HOSTS) {
            if (!list.contains(h)) list.add(h)
        }
        return list
    }

    /**
     * Retrieves the configured backend URL.
     * Defaults to http://127.0.0.1:8000.
     */
    fun getBackendUrl(context: Context): String {
        val saved = getPrefs(context).getString(KEY_BACKEND_URL, null)?.trim()?.removeSuffix("/")
        return if (!saved.isNullOrBlank()) saved else DEFAULT_BACKEND_URL
    }

    /**
     * Returns true if a backend server is reachable/configured.
     * Supports both remote IP/hostnames and local reverse port forwarded 127.0.0.1 / localhost.
     */
    fun hasRemoteBackend(context: Context): Boolean {
        val url = getBackendUrl(context)
        return url.isNotBlank() && (url.startsWith("http://") || url.startsWith("https://"))
    }

    // ─── Gemini API key ────────────────────────────────────────────────────────

    /**
     * Retrieves the Gemini API key configured for claim verification.
     * Returns empty string if not configured — verification falls back to local heuristics.
     */
    fun getGeminiApiKey(context: Context): String {
        return getPrefs(context).getString(KEY_GEMINI_API_KEY, "") ?: ""
    }

    /**
     * Saves the Gemini API key for claim verification and post-call analysis.
     */
    fun setGeminiApiKey(context: Context, apiKey: String) {
        getPrefs(context).edit().putString(KEY_GEMINI_API_KEY, apiKey.trim()).apply()
    }

    /**
     * Returns true if a Gemini API key has been configured.
     */
    fun hasGeminiApiKey(context: Context): Boolean {
        return getGeminiApiKey(context).length > 10
    }

    /**
     * Updates the backend base URL.
     */
    fun setBackendUrl(context: Context, url: String) {
        val cleanUrl = url.trim().removeSuffix("/")
        getPrefs(context).edit().putString(KEY_BACKEND_URL, cleanUrl).apply()
    }

    /**
     * Retrieves the host address from the configured remote backend URL.
     * Returns empty string if no backend is configured.
     */
    fun getHost(context: Context): String {
        val url = getBackendUrl(context)
        if (url.isBlank()) return "127.0.0.1"
        return try {
            val uri = java.net.URI(url)
            uri.host?.takeIf { it.isNotBlank() } ?: "127.0.0.1"
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
            .setTitle("⚙️ Remote Backend Configuration (Optional)")
            .setMessage("VaniRakshak works fully on-device without a remote server.\n\nOptionally configure a remote VaniRakshak backend URL (e.g. https://your-server.example.com) to enable additional cloud-powered forensics.\n\nLeave blank to use on-device processing only.")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                val newUrl = input.text.toString().trim()
                if (newUrl.isBlank()) {
                    setBackendUrl(activity, "")
                    Toast.makeText(activity, "✓ On-device only mode (no remote backend)", Toast.LENGTH_SHORT).show()
                    onUpdated?.invoke("")
                } else if (newUrl.startsWith("http://") || newUrl.startsWith("https://")) {
                    setBackendUrl(activity, newUrl)
                    Toast.makeText(activity, "✓ Remote backend set: ${getHost(activity)}", Toast.LENGTH_SHORT).show()
                    onUpdated?.invoke(newUrl)
                } else {
                    Toast.makeText(activity, "Invalid URL. Must start with http:// or https://", Toast.LENGTH_LONG).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .setNeutralButton("Clear") { _, _ ->
                setBackendUrl(activity, "")
                Toast.makeText(activity, "Remote backend cleared — using on-device processing", Toast.LENGTH_SHORT).show()
                onUpdated?.invoke("")
            }
            .show()
    }
}
