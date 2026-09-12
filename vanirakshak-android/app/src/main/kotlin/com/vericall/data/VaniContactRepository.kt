package com.vericall.data

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.database.Cursor
import android.provider.ContactsContract
import android.util.Log
import androidx.core.content.ContextCompat
import com.vericall.model.CallScenario
import com.vericall.model.VaniContact

object VaniContactRepository {

    private const val TAG = "VaniContactRepository"
    private const val PREF_ENROLLMENTS = "vani_voice_enrollments"
    private const val KEY_ENROLLED_PREFIX = "enrolled_"
    private const val KEY_SEC_PREFIX = "sec_"

    /**
     * Loads contacts from the device's real address book if READ_CONTACTS is granted.
     * Falls back to VaniSampleData if permission is denied or no contacts are stored.
     */
    fun getContacts(context: Context): List<VaniContact> {
        val hasPermission = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.READ_CONTACTS
        ) == PackageManager.PERMISSION_GRANTED

        if (!hasPermission) {
            Log.d(TAG, "READ_CONTACTS permission not granted, returning sample contacts")
            return VaniSampleData.contacts
        }

        val realContacts = queryRealContacts(context)
        return if (realContacts.isNotEmpty()) {
            realContacts
        } else {
            Log.d(TAG, "No contacts found on device, using sample contacts")
            VaniSampleData.contacts
        }
    }

    private fun queryRealContacts(context: Context): List<VaniContact> {
        val contactsList = mutableListOf<VaniContact>()
        val seenPhones = mutableSetOf<String>()
        val prefs = context.getSharedPreferences(PREF_ENROLLMENTS, Context.MODE_PRIVATE)

        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.CONTACT_ID,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.TYPE,
            ContactsContract.CommonDataKinds.Phone.LABEL
        )

        var cursor: Cursor? = null
        try {
            cursor = context.contentResolver.query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                projection,
                null,
                null,
                "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} ASC"
            )

            cursor?.let {
                val idIdx = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.CONTACT_ID)
                val nameIdx = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
                val numIdx = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
                val typeIdx = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.TYPE)
                val labelIdx = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.LABEL)

                while (it.moveToNext()) {
                    val contactId = if (idIdx >= 0) it.getString(idIdx) ?: "" else ""
                    val name = if (nameIdx >= 0) it.getString(nameIdx) ?: "Unknown" else "Unknown"
                    var rawNumber = if (numIdx >= 0) it.getString(numIdx) ?: "" else ""
                    val type = if (typeIdx >= 0) it.getInt(typeIdx) else ContactsContract.CommonDataKinds.Phone.TYPE_OTHER
                    val customLabel = if (labelIdx >= 0) it.getString(labelIdx) else null

                    // Normalize phone number for deduplication
                    val normalizedNum = rawNumber.replace(Regex("[^0-9+]"), "")
                    if (normalizedNum.isBlank() || seenPhones.contains(normalizedNum)) {
                        continue
                    }
                    seenPhones.add(normalizedNum)

                    val relation = getPhoneTypeLabel(type, customLabel)

                    // Check if user has saved a voice enrollment baseline for this contact
                    val isEnrolled = prefs.getBoolean(KEY_ENROLLED_PREFIX + contactId, false) ||
                            prefs.getBoolean(KEY_ENROLLED_PREFIX + normalizedNum, false)
                    val enrollmentSec = prefs.getInt(KEY_SEC_PREFIX + contactId, 15)

                    // Pick an educational default scenario for demonstration
                    val scenario = when {
                        isEnrolled -> CallScenario.GENUINE_CALL
                        name.contains("Aarav", ignoreCase = true) -> CallScenario.AI_CLONE_ATTACK
                        name.contains("Priya", ignoreCase = true) -> CallScenario.GENUINE_CALL
                        else -> CallScenario.SUSPICIOUS_ANOMALY
                    }

                    contactsList.add(
                        VaniContact(
                            id = contactId.ifBlank { "c_${contactsList.size + 1}" },
                            name = name,
                            phone = rawNumber,
                            relation = relation,
                            isVoiceEnrolled = isEnrolled,
                            enrollmentSec = enrollmentSec,
                            defaultScenario = scenario
                        )
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error querying ContactsContract: ${e.message}", e)
        } finally {
            cursor?.close()
        }

        return contactsList
    }

    private fun getPhoneTypeLabel(type: Int, customLabel: String?): String {
        return when (type) {
            ContactsContract.CommonDataKinds.Phone.TYPE_MOBILE -> "Mobile"
            ContactsContract.CommonDataKinds.Phone.TYPE_HOME -> "Home"
            ContactsContract.CommonDataKinds.Phone.TYPE_WORK -> "Work"
            ContactsContract.CommonDataKinds.Phone.TYPE_MAIN -> "Main"
            ContactsContract.CommonDataKinds.Phone.TYPE_CUSTOM -> customLabel ?: "Personal"
            else -> "Personal"
        }
    }

    /**
     * Persists voice enrollment baseline for a contact.
     */
    fun saveEnrollment(context: Context, contact: VaniContact, enrollmentSec: Int = 15) {
        val prefs = context.getSharedPreferences(PREF_ENROLLMENTS, Context.MODE_PRIVATE)
        val normalized = contact.phone.replace(Regex("[^0-9+]"), "")
        prefs.edit()
            .putBoolean(KEY_ENROLLED_PREFIX + contact.id, true)
            .putBoolean(KEY_ENROLLED_PREFIX + normalized, true)
            .putInt(KEY_SEC_PREFIX + contact.id, enrollmentSec)
            .apply()

        contact.isVoiceEnrolled = true
        contact.enrollmentSec = enrollmentSec
        contact.defaultScenario = CallScenario.GENUINE_CALL
    }
}
