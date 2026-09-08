package com.vericall.shield

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.json.JSONArray
import org.json.JSONObject

/**
 * Persists live incident forensics, suppressed banking OTP alerts, and
 * coercion overlay trigger events into a lightweight SQLite repository.
 */
object VaniIncidentLogger {

    data class HeldNotification(
        val packageName: String,
        val text: String,
        val timestamp: Long
    )

    data class IncidentEntry(
        val id: Long,
        val callerNumber: String,
        val callerName: String,
        val peakRisk: Int,
        val verdictSummary: String,
        val heldNotificationsJson: String,
        val overlayTriggeredPackagesJson: String,
        val timestamp: Long
    )

    // In-memory active session caches
    private val activeHeldNotifications = mutableListOf<HeldNotification>()
    private val activeOverlayTriggers = mutableListOf<String>()

    private var dbHelper: IncidentDbHelper? = null

    private fun getDb(context: Context): SQLiteDatabase {
        if (dbHelper == null) {
            dbHelper = IncidentDbHelper(context.applicationContext)
        }
        return dbHelper!!.writableDatabase
    }

    @Synchronized
    fun stashHeldNotification(context: Context, packageName: String, text: String, timestamp: Long = System.currentTimeMillis()) {
        val item = HeldNotification(packageName, text, timestamp)
        activeHeldNotifications.add(item)
    }

    @Synchronized
    fun getActiveHeldNotifications(): List<HeldNotification> = activeHeldNotifications.toList()

    @Synchronized
    fun clearActiveHeldNotifications() {
        activeHeldNotifications.clear()
    }

    @Synchronized
    fun logOverlayTrigger(triggeringPackage: String, timestamp: Long = System.currentTimeMillis()) {
        activeOverlayTriggers.add(triggeringPackage)
    }

    @Synchronized
    fun getActiveOverlayTriggers(): List<String> = activeOverlayTriggers.toList()

    @Synchronized
    fun clearActiveOverlayTriggers() {
        activeOverlayTriggers.clear()
    }

    @Synchronized
    fun recordIncident(
        context: Context,
        callerNumber: String,
        callerName: String,
        peakRisk: Int,
        verdictSummary: String
    ): Long {
        val heldJson = JSONArray().apply {
            for (item in activeHeldNotifications) {
                put(JSONObject().apply {
                    put("package", item.packageName)
                    put("text", item.text)
                    put("time", item.timestamp)
                })
            }
        }.toString()

        val triggersJson = JSONArray(activeOverlayTriggers).toString()

        val values = ContentValues().apply {
            put(IncidentDbHelper.COL_CALLER_NUMBER, callerNumber)
            put(IncidentDbHelper.COL_CALLER_NAME, callerName)
            put(IncidentDbHelper.COL_PEAK_RISK, peakRisk)
            put(IncidentDbHelper.COL_VERDICT, verdictSummary)
            put(IncidentDbHelper.COL_HELD_NOTIFS, heldJson)
            put(IncidentDbHelper.COL_TRIGGERS, triggersJson)
            put(IncidentDbHelper.COL_TIMESTAMP, System.currentTimeMillis())
        }

        val id = try {
            getDb(context).insert(IncidentDbHelper.TABLE_INCIDENTS, null, values)
        } catch (e: Exception) {
            -1L
        }

        return id
    }

    private class IncidentDbHelper(context: Context) : SQLiteOpenHelper(context, "vani_incidents.db", null, 1) {
        companion object {
            const val TABLE_INCIDENTS = "incidents"
            const val COL_ID = "id"
            const val COL_CALLER_NUMBER = "caller_number"
            const val COL_CALLER_NAME = "caller_name"
            const val COL_PEAK_RISK = "peak_risk"
            const val COL_VERDICT = "verdict_summary"
            const val COL_HELD_NOTIFS = "held_notifications"
            const val COL_TRIGGERS = "overlay_triggers"
            const val COL_TIMESTAMP = "timestamp"
        }

        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE $TABLE_INCIDENTS (
                    $COL_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    $COL_CALLER_NUMBER TEXT,
                    $COL_CALLER_NAME TEXT,
                    $COL_PEAK_RISK INTEGER,
                    $COL_VERDICT TEXT,
                    $COL_HELD_NOTIFS TEXT,
                    $COL_TRIGGERS TEXT,
                    $COL_TIMESTAMP INTEGER
                )
                """.trimIndent()
            )
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            db.execSQL("DROP TABLE IF EXISTS $TABLE_INCIDENTS")
            onCreate(db)
        }
    }
}
