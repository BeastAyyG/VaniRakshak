package com.vericall.temp

import android.Manifest
import android.app.role.RoleManager
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.vericall.R
import com.vericall.data.VaniSampleData
import com.vericall.databinding.ActivityTempContactsBinding
import com.vericall.util.VaniBackendConfig
import com.vericall.util.VaniCallRouter
import com.vericall.util.VaniNotificationHelper

class TempContactsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityTempContactsBinding
    private lateinit var adapter: TempContactAdapter

    private val screeningRoleLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { refreshShieldState() }

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { refreshShieldState() }

    private val requiredPermissions: Array<String>
        get() {
            val list = mutableListOf(
                Manifest.permission.READ_PHONE_STATE,
                Manifest.permission.READ_CALL_LOG,
                Manifest.permission.READ_CONTACTS,
                Manifest.permission.RECORD_AUDIO
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                list.add(Manifest.permission.POST_NOTIFICATIONS)
            }
            return list.toTypedArray()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTempContactsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Handle edge-to-edge system insets (mandatory for Android 15 & 16)
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // Initialize notification channels for real-time call risk alerts
        VaniNotificationHelper.createChannels(this)

        // Request real phone call monitoring permissions
        checkAndRequestCallPermissions()

        // Shield status pill is interactive and provides diagnostic/setup overview
        binding.pillShield.setOnClickListener {
            showShieldStatusDialog()
        }

        binding.btnEnableScreening.setOnClickListener { requestScreeningRole() }
        binding.btnEnableAlerts.setOnClickListener { requestNotificationAccess() }
        binding.btnBackendSettings.setOnClickListener { showBackendSettingsDialog() }
        refreshBackendLabel()

        // Runs the real detection path end to end, so the incoming-call alert
        // can be demonstrated without an actual phone call. Opens the guard
        // screen as a LIVE call (the "answer" step): that is what starts the
        // guard service, mic capture and server streaming — otherwise the WS
        // path would never run and the backend would see zero sessions.
        // NOTE: must NOT open the stored c1 profile screen — that path runs
        // the in-process demo analyzer only (isLiveCall=false).
        binding.btnSimulateIncoming.setOnClickListener {
            // Fresh-session semantics: stop any lingering guard and drop the
            // previous call's verdict, or the new screen would replay stale
            // bus state and never start a new server session.
            VaniCallRouter.onEnded(this)
            VaniCallRouter.onRinging(this, SIMULATED_CALLER_NUMBER)
            val caller = VaniCallRouter.resolve(this, SIMULATED_CALLER_NUMBER)
            openLiveGuard(caller)
        }

        adapter = TempContactAdapter(
            context = this,
            contacts = VaniSampleData.contacts,
            onContactClicked = { contact ->
                openDetail(contact.id)
            },
            onEnrollClicked = { contact ->
                TempContactSheet.show(this, contact) {
                    adapter.notifyDataSetChanged()
                }
            }
        )
        binding.listContacts.adapter = adapter

        // Live search over name, phone and relation
        binding.inputSearch.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
            override fun afterTextChanged(s: Editable?) {
                adapter.filter(s?.toString()?.trim() ?: "")
            }
        })

        // One-click demo: simulated AI voice-clone impersonation (Aarav Sharma)
        binding.btnDemoClone.setOnClickListener {
            openDetail("c1")
        }

        // One-click demo: verified human voice (Priya Nair)
        binding.btnDemoGenuine.setOnClickListener {
            openDetail("c2")
        }

        binding.listContacts.setOnItemClickListener { _, _, position, _ ->
            adapter.getItem(position)?.let { openDetail(it.id) }
        }
    }
    fun openDetail(contactId: String) {
        val contact = VaniSampleData.getContactById(contactId) ?: VaniSampleData.contacts[0]
        // Show real-time Android Notification with risk score
        VaniNotificationHelper.showCallRiskNotification(this, contact, contact.defaultScenario)

        startActivity(
            Intent(this, TempContactDetailActivity::class.java).apply {
                putExtra(EXTRA_CONTACT_ID, contactId)
            }
        )
    }

    /**
     * Opens the guard screen as a LIVE call (same intent the ringing-alert
     * notification builds): the detail screen then starts the guard service,
     * mic capture and server streaming.
     */
    fun openLiveGuard(caller: VaniCallRouter.Caller) {
        startActivity(
            Intent(this, TempContactDetailActivity::class.java).apply {
                putExtra(EXTRA_CONTACT_ID, "live_call")
                putExtra(VaniNotificationHelper.EXTRA_CALLER_NAME, caller.name)
                putExtra(VaniNotificationHelper.EXTRA_CALLER_PHONE, caller.phone)
                putExtra(VaniNotificationHelper.EXTRA_CALLER_ENROLLED, caller.enrolled)
            }
        )
    }

    // -------------------------------------------------------------- setup state

    private fun refreshBackendLabel() {
        val gate = if (VaniBackendConfig.isKeywordGate(this)) "keyword-gate ON" else "always-stream"
        binding.tvBackendHost.text =
            "Backend: ${VaniBackendConfig.getHost(this)}:${VaniBackendConfig.getPort(this)} • $gate"
    }

    /** Debug settings: LAN IP for real devices, 127.0.0.1 + adb reverse for emulator. */
    private fun showBackendSettingsDialog() {
        val input = android.widget.EditText(this).apply {
            setText("${VaniBackendConfig.getHost(this@TempContactsActivity)}:${VaniBackendConfig.getPort(this@TempContactsActivity)}")
            hint = "host:port (e.g. 192.168.1.5:8000)"
            inputType = android.text.InputType.TYPE_CLASS_TEXT or
                android.text.InputType.TYPE_TEXT_VARIATION_URI
            setSingleLine()
        }
        val gateCheck = android.widget.CheckBox(this).apply {
            text = "Wake server only on money-talk keywords (saves data/battery)"
            isChecked = VaniBackendConfig.isKeywordGate(this@TempContactsActivity)
        }
        val padded = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(48, 24, 48, 0)
            addView(input)
            addView(gateCheck)
        }
        AlertDialog.Builder(this)
            .setTitle("Backend settings")
            .setMessage("Use your PC's LAN IP on a real device. 127.0.0.1 needs `adb reverse tcp:8000 tcp:8000`. Applies to the next call.")
            .setView(padded)
            .setPositiveButton("Save") { _, _ ->
                val parts = input.text.toString().trim().split(":")
                val host = parts.getOrNull(0)?.trim().orEmpty()
                val port = parts.getOrNull(1)?.trim()?.toIntOrNull()
                    ?: VaniBackendConfig.getPort(this)
                VaniBackendConfig.setKeywordGate(this, gateCheck.isChecked)
                if (host.isNotEmpty()) {
                    VaniBackendConfig.setHostPort(this, host, port)
                    refreshBackendLabel()
                    Toast.makeText(this, "Backend set to $host:$port", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Host cannot be empty", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .setNeutralButton("Reset") { _, _ ->
                VaniBackendConfig.setHostPort(
                    this, VaniBackendConfig.DEFAULT_HOST, VaniBackendConfig.DEFAULT_PORT
                )
                refreshBackendLabel()
            }
            .show()
    }

    private fun hasNotificationAccess(): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            return ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
        }
        return NotificationManagerCompat.from(this).areNotificationsEnabled()
    }

    /** Below Android 10 the OS binds the only available screening service on its own. */
    private fun hasScreeningRole(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return true
        val roleManager = getSystemService(RoleManager::class.java)
        return roleManager?.isRoleHeld(RoleManager.ROLE_CALL_SCREENING) ?: false
    }

    private fun requestScreeningRole() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val intent = getSystemService(RoleManager::class.java)
                ?.createRequestRoleIntent(RoleManager.ROLE_CALL_SCREENING)
            if (intent != null) {
                screeningRoleLauncher.launch(intent)
                return
            }
        }
        Toast.makeText(this, R.string.screening_needed, Toast.LENGTH_LONG).show()
    }

    private fun requestNotificationAccess() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            return
        }
        // Already granted but blocked, or pre-13: send them to the app settings.
        openNotificationSettings()
    }

    /**
     * ACTION_APP_NOTIFICATION_SETTINGS is API 26+, so older devices get the
     * general app details screen instead. Both are wrapped because some OEM
     * builds ship without one or the other.
     */
    private fun openNotificationSettings() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            try {
                startActivity(
                    Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
                        .putExtra(Settings.EXTRA_APP_PACKAGE, packageName)
                )
                return
            } catch (e: ActivityNotFoundException) {
                // Fall through to the generic app details screen.
            }
        }
        try {
            startActivity(
                Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
                    .setData(Uri.fromParts("package", packageName, null))
            )
        } catch (e: ActivityNotFoundException) {
            Toast.makeText(this, R.string.alerts_needed, Toast.LENGTH_LONG).show()
        }
    }

    private fun refreshShieldState() {
        val notificationsOn = hasNotificationAccess()
        val screeningOn = hasScreeningRole()
        val ready = notificationsOn && screeningOn

        val badge = if (ready) R.drawable.bg_badge_enrolled else R.drawable.bg_badge_warning
        val tone = if (ready) R.color.success_deep else R.color.warning_deep

        binding.pillShield.setBackgroundResource(badge)
        binding.viewShieldDot.backgroundTintList = ContextCompat.getColorStateList(this, tone)
        binding.tvShieldText.text = getString(if (ready) R.string.shield_on else R.string.shield_off)
        binding.tvShieldText.setTextColor(ContextCompat.getColor(this, tone))

        binding.rowSetup.visibility = if (ready) View.GONE else View.VISIBLE
        binding.btnEnableScreening.visibility = if (screeningOn) View.GONE else View.VISIBLE
        binding.btnEnableAlerts.visibility = if (notificationsOn) View.GONE else View.VISIBLE
    }

    private fun showShieldStatusDialog() {
        val notificationsOn = hasNotificationAccess()
        val screeningOn = hasScreeningRole()
        val ready = notificationsOn && screeningOn

        val title = if (ready) "🛡️ Call Guard Active" else "⚠️ Shield Setup Needed"
        val message = buildString {
            if (ready) {
                append("All real-time defense layers are armed and monitoring:\n\n")
                append("✓ 16 kHz Audio Forensic Ingestion: Active\n")
                append("✓ Call Screening & Telephony Interception: Active\n")
                append("✓ Heads-up Scam Call Notifications: Enabled\n")
                append("✓ Banking OTP & Financial Coercion Shield: Armed\n\n")
                append("You can simulate incoming clone attacks or genuine calls anytime using the buttons below.")
            } else {
                append("VaniRakshak requires permissions to protect live telephony sessions:\n\n")
                if (!screeningOn) append("• Call Screening Role: Pending (detects incoming calls)\n")
                if (!notificationsOn) append("• Notification Alerts: Disabled (shows real-time risk meter)\n\n")
                append("Tap 'Open Setup' to grant the remaining permissions.")
            }
        }

        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton(if (ready) "Close" else "Open Setup") { _, _ ->
                if (!ready) {
                    if (!screeningOn) requestScreeningRole()
                    else if (!notificationsOn) requestNotificationAccess()
                }
            }
            .show()
    }

    private fun checkAndRequestCallPermissions() {
        val missingPermissions = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missingPermissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                missingPermissions.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            refreshShieldState()
            val allGranted = grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            if (allGranted) {
                Toast.makeText(
                    this,
                    "🛡️ Live Call Shield Active! Incoming phone calls will now be verified.",
                    Toast.LENGTH_LONG
                ).show()
            } else {
                Toast.makeText(
                    this,
                    "⚠️ Phone & Call Log permissions required to detect actual incoming calls.",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        refreshBackendLabel()
        if (::adapter.isInitialized) {
            adapter.notifyDataSetChanged()
        }
        // The user may have granted the role or notification access in Settings.
        refreshShieldState()
    }

    companion object {
        const val EXTRA_CONTACT_ID = "vani.contact.id"
        private const val PERMISSION_REQUEST_CODE = 202

        /** Aarav Sharma from the sample data - resolves as a protected contact. */
        private const val SIMULATED_CALLER_NUMBER = "+91 98765 43210"
    }
}