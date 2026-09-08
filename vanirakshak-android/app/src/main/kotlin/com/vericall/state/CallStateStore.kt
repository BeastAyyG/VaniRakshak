package com.vericall.state

import android.content.Context

/**
 * Remembers the current call's number across broadcasts.
 *
 * OFFHOOK and IDLE never carry EXTRA_INCOMING_NUMBER, and the receiver process
 * can be recreated between broadcasts, so the number captured at RINGING is
 * persisted rather than held in memory.
 */
object CallStateStore {

    private const val PREFS = "vani_call_state"
    private const val KEY_NUMBER = "last_number"

    fun remember(context: Context, number: String) {
        prefs(context).edit().putString(KEY_NUMBER, number).apply()
    }

    fun recall(context: Context): String? = prefs(context).getString(KEY_NUMBER, null)

    fun clear(context: Context) {
        prefs(context).edit().remove(KEY_NUMBER).apply()
    }

    private fun prefs(context: Context) =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
}
