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
        val timestamp: Long,
        val receiptHash: String? = null,
        val serverTier: String? = null
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

    /**
     * Runs [block] against the cached database, dropping + reopening it once
     * on a stale-connection failure (seen when a cached handle outlives its
     * native connection, e.g. Robolectric sandbox teardown between tests).
     */
    private fun <T> withDb(context: Context, block: (SQLiteDatabase) -> T): T? {
        return try {
            block(getDb(context))
        } catch (e: IllegalStateException) {
            runCatching { dbHelper?.close() }
            dbHelper = null
            runCatching { block(getDb(context)) }.getOrNull()
        } catch (e: Exception) {
            android.util.Log.w("VaniIncidentLogger", "db op failed", e)
            null
        }
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
        verdictSummary: String,
        receiptHash: String? = null,
        serverTier: String? = null
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
            put(IncidentDbHelper.COL_RECEIPT, receiptHash)
            put(IncidentDbHelper.COL_TIER, serverTier)
        }

        val id = withDb(context) { db ->
            db.insert(IncidentDbHelper.TABLE_INCIDENTS, null, values)
        } ?: -1L

        return id
    }

    /** Reads one incident back (receipt + tier included); null when missing/unreadable. */
    fun getIncident(context: Context, id: Long): IncidentEntry? {
        return withDb(context) { db ->
            db.query(
                IncidentDbHelper.TABLE_INCIDENTS, null,
                "${IncidentDbHelper.COL_ID} = ?", arrayOf(id.toString()),
                null, null, null
            ).use { c ->
                if (!c.moveToFirst()) return@withDb null
                fun str(col: String): String = c.getString(c.getColumnIndexOrThrow(col)) ?: ""
                fun optStr(col: String): String? {
                    val i = c.getColumnIndexOrThrow(col)
                    return if (c.isNull(i)) null else c.getString(i)
                }
                IncidentEntry(
                    id = c.getLong(c.getColumnIndexOrThrow(IncidentDbHelper.COL_ID)),
                    callerNumber = str(IncidentDbHelper.COL_CALLER_NUMBER),
                    callerName = str(IncidentDbHelper.COL_CALLER_NAME),
                    peakRisk = c.getInt(c.getColumnIndexOrThrow(IncidentDbHelper.COL_PEAK_RISK)),
                    verdictSummary = str(IncidentDbHelper.COL_VERDICT),
                    heldNotificationsJson = str(IncidentDbHelper.COL_HELD_NOTIFS),
                    overlayTriggeredPackagesJson = str(IncidentDbHelper.COL_TRIGGERS),
                    timestamp = c.getLong(c.getColumnIndexOrThrow(IncidentDbHelper.COL_TIMESTAMP)),
                    receiptHash = optStr(IncidentDbHelper.COL_RECEIPT),
                    serverTier = optStr(IncidentDbHelper.COL_TIER)
                )
            }
        }
    }

    private class IncidentDbHelper(context: Context) : SQLiteOpenHelper(context, "vani_incidents.db", null, 2) {
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
            const val COL_RECEIPT = "receipt_hash"
            const val COL_TIER = "server_tier"
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
                    $COL_TIMESTAMP INTEGER,
                    $COL_RECEIPT TEXT,
                    $COL_TIER TEXT
                )
                """.trimIndent()
            )
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            if (oldVersion < 2) {
                db.execSQL("ALTER TABLE $TABLE_INCIDENTS ADD COLUMN $COL_RECEIPT TEXT")
                db.execSQL("ALTER TABLE $TABLE_INCIDENTS ADD COLUMN $COL_TIER TEXT")
            } else {
                db.execSQL("DROP TABLE IF EXISTS $TABLE_INCIDENTS")
                onCreate(db)
            }
        }
    }
}
