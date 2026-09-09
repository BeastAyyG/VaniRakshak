package com.vericall.util

import android.content.Context

/**
 * Single source of truth for the backend address.
 *
 * Default is the adb-reverse loopback so existing dev flow keeps working,
 * but the host is user-configurable (LAN IP for real devices) and persisted
 * in SharedPreferences. No webhook secrets live here — server owns alerting.
 */
object VaniBackendConfig {

    private const val PREFS = "vani_backend"
    private const val KEY_HOST = "backend_host"
    private const val KEY_PORT = "backend_port"
    private const val KEY_KEYWORD_GATE = "keyword_gate"

    const val DEFAULT_HOST = "127.0.0.1"
    const val DEFAULT_PORT = 8000

    fun getHost(context: Context): String =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(KEY_HOST, DEFAULT_HOST) ?: DEFAULT_HOST

    fun getPort(context: Context): Int =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getInt(KEY_PORT, DEFAULT_PORT)

    fun setHostPort(context: Context, host: String, port: Int) {        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putString(KEY_HOST, host.trim())
            .putInt(KEY_PORT, port.coerceIn(1, 65535))
            .apply()
    }

    fun httpBase(context: Context): String = "http://${getHost(context)}:${getPort(context)}"

    /**
     * Keyword-gated wake-up: when true (default), the guard listens on-device
     * for money-talk and only then opens the WS stream; when false, it streams
     * immediately like before. Auto-bypassed if SpeechRecognizer is unavailable.
     */
    fun isKeywordGate(context: Context): Boolean =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY_KEYWORD_GATE, true)

    fun setKeywordGate(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putBoolean(KEY_KEYWORD_GATE, enabled)
            .apply()
    }

    /** Mirrors docs/PROTOCOL.md §endpoint: ws://host:8000/ws/stream */
    fun wsUrl(context: Context): String = "ws://${getHost(context)}:${getPort(context)}/ws/stream"
}
