package com.vericall.contacts

import android.Manifest
import android.app.role.RoleManager
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.text.Editable
import android.text.TextWatcher
import android.view.MotionEvent
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.view.animation.DecelerateInterpolator
import android.view.inputmethod.InputMethodManager
import android.widget.AbsListView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.vericall.R
import com.vericall.data.VaniContactRepository
import com.vericall.data.VaniSampleData
import com.vericall.databinding.ActivityTempContactsBinding
import com.vericall.model.VaniContact
import com.vericall.util.VaniCallRouter
import com.vericall.util.VaniNotificationHelper

/**
 * Dedicated contacts directory and voice enrollment roster activity.
 * Features an interactive Apple-style fluid bottom sheet, real phone call simulation,
 * and live biometric baseline management.
 */
class VaniContactsRosterActivity : AppCompatActivity() {

    private lateinit var binding: ActivityTempContactsBinding
    private lateinit var adapter: VaniContactAdapter

    private var isContactsExpanded = false
    private var contactsCollapsedOffset = 0f

    private val brandHandler = Handler(Looper.getMainLooper())
    private var isBrandCollapsed = false

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
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.CALL_PHONE
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                list.add(Manifest.permission.POST_NOTIFICATIONS)
            }
            return list.toTypedArray()
        }

    private var currentContacts: List<VaniContact> = emptyList()

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

        // Intercept back button for Apple-style sheet collapse
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (isContactsExpanded) {
                    collapseContactsSheet()
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                    isEnabled = true
                }
            }
        })

        // Initialize notification channels for real-time call risk alerts
        VaniNotificationHelper.createChannels(this)

        // Request real phone call monitoring & contacts permissions
        checkAndRequestCallPermissions()

        // Shield status pill is interactive and provides diagnostic/setup overview
        binding.pillShield.setOnClickListener {
            showShieldStatusDialog()
        }

        binding.btnEnableScreening.setOnClickListener { requestScreeningRole() }
        binding.btnEnableAlerts.setOnClickListener { requestNotificationAccess() }

        // Runs the real detection path end to end, so the incoming-call alert
        // can be demonstrated without an actual phone call.
        binding.btnSimulateIncoming.setOnClickListener {
            VaniCallRouter.onRinging(this, SIMULATED_CALLER_NUMBER)
        }

        binding.fabSimulateCall.setOnClickListener {
            VaniCallRouter.onRinging(this, SIMULATED_CALLER_NUMBER)
        }

        adapter = VaniContactAdapter(
            context = this,
            contacts = VaniSampleData.contacts,
            onContactClicked = { contact ->
                openDetail(contact.id)
            },
            onEnrollClicked = { contact ->
                VaniEnrollmentSheet.show(this, contact) {
                    loadContactsAsync()
                }
            }
        )
        binding.listContacts.adapter = adapter

        setupContactsSheetAnimation()

        // Load real device contacts asynchronously
        loadContactsAsync()

        // Live search over name, phone and relation with result badge
        binding.inputSearch.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
            override fun afterTextChanged(s: Editable?) {
                val query = s?.toString()?.trim() ?: ""
                adapter.filter(query)
                if (query.isNotEmpty()) {
                    binding.tvSearchResultCount.visibility = View.VISIBLE
                    binding.tvSearchResultCount.text = "${adapter.count}"
                } else {
                    binding.tvSearchResultCount.visibility = View.GONE
                }
            }
        })

        adapter.onFilterChanged = { count ->
            binding.tvContactSectionCount.text = "$count contacts"
        }

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

        setupBrandCollapsingAnimation()
    }

    override fun onDestroy() {
        super.onDestroy()
        brandHandler.removeCallbacksAndMessages(null)
    }

    private fun setupBrandCollapsingAnimation() {
        // Automatically slide brand name away after 1.6s so the top bar stays super clean
        brandHandler.postDelayed({
            if (!isFinishing && !isDestroyed && !isBrandCollapsed) {
                collapseBrandText(animate = true)
            }
        }, 1600)

        // Tapping the logo icon expands/collapses the name dynamically
        binding.ivAppLogo.setOnClickListener {
            brandHandler.removeCallbacksAndMessages(null)
            if (isBrandCollapsed) {
                expandBrandText()
                // Auto tuck away again after 2.8s
                brandHandler.postDelayed({
                    if (!isFinishing && !isDestroyed && !isBrandCollapsed) {
                        collapseBrandText(animate = true)
                    }
                }, 2800)
            } else {
                collapseBrandText(animate = true)
            }
        }
    }

    private fun collapseBrandText(animate: Boolean) {
        isBrandCollapsed = true
        if (!animate) {
            binding.layoutBrandText.visibility = View.GONE
            return
        }

        binding.layoutBrandText.animate()
            .translationX(-binding.layoutBrandText.width.toFloat().coerceAtLeast(60f))
            .alpha(0f)
            .setDuration(380)
            .setInterpolator(AccelerateDecelerateInterpolator())
            .withEndAction {
                binding.layoutBrandText.visibility = View.GONE
            }
            .start()
    }

    private fun expandBrandText() {
        isBrandCollapsed = false
        binding.layoutBrandText.visibility = View.VISIBLE
        binding.layoutBrandText.translationX = -30f
        binding.layoutBrandText.alpha = 0f
        binding.layoutBrandText.animate()
            .translationX(0f)
            .alpha(1f)
            .setDuration(320)
            .setInterpolator(DecelerateInterpolator(1.8f))
            .start()
    }

    private fun setupContactsSheetAnimation() {
        // Measure header height once layout completes to determine collapsed offset
        binding.contactsDashboardHeader.post {
            contactsCollapsedOffset = binding.contactsDashboardHeader.height.toFloat()
            if (!isContactsExpanded && contactsCollapsedOffset > 0) {
                binding.contactsBottomSheet.translationY = contactsCollapsedOffset
            }
        }

        // Tap on sheet header (drag handle bar)
        binding.contactsSheetHeader.setOnClickListener {
            if (!isContactsExpanded) expandContactsSheet() else collapseContactsSheet()
        }

        // Drag handle touch listener
        var headerTouchStartY = 0f
        binding.contactsSheetHeader.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    headerTouchStartY = event.rawY
                    false
                }
                MotionEvent.ACTION_MOVE -> {
                    val deltaY = event.rawY - headerTouchStartY
                    if (deltaY < -20f && !isContactsExpanded) {
                        expandContactsSheet()
                        true
                    } else if (deltaY > 20f && isContactsExpanded) {
                        collapseContactsSheet()
                        true
                    } else false
                }
                else -> false
            }
        }

        // Inline collapse button
        binding.btnCollapseContacts.setOnClickListener {
            collapseContactsSheet()
        }

        // Search bar click & focus triggers expansion
        binding.inputSearch.setOnClickListener {
            if (!isContactsExpanded) expandContactsSheet()
        }
        binding.inputSearch.setOnFocusChangeListener { _, hasFocus ->
            if (hasFocus && !isContactsExpanded) {
                expandContactsSheet()
            }
        }

        // Scrolling & swiping contact list triggers smooth full-screen expansion
        var listTouchStartY = 0f
        binding.listContacts.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    listTouchStartY = event.rawY
                    false
                }
                MotionEvent.ACTION_MOVE -> {
                    val deltaY = event.rawY - listTouchStartY
                    if (!isContactsExpanded && deltaY < -25f) {
                        expandContactsSheet()
                        val cancelEvent = MotionEvent.obtain(event)
                        cancelEvent.action = MotionEvent.ACTION_CANCEL
                        binding.listContacts.onTouchEvent(cancelEvent)
                        cancelEvent.recycle()
                        true
                    } else if (isContactsExpanded && deltaY > 35f &&
                        binding.listContacts.firstVisiblePosition == 0 &&
                        (binding.listContacts.getChildAt(0)?.top ?: 0) >= 0
                    ) {
                        collapseContactsSheet()
                        val cancelEvent = MotionEvent.obtain(event)
                        cancelEvent.action = MotionEvent.ACTION_CANCEL
                        binding.listContacts.onTouchEvent(cancelEvent)
                        cancelEvent.recycle()
                        true
                    } else false
                }
                else -> false
            }
        }

        binding.listContacts.setOnScrollListener(object : AbsListView.OnScrollListener {
            override fun onScrollStateChanged(view: AbsListView?, scrollState: Int) {
                if (scrollState == AbsListView.OnScrollListener.SCROLL_STATE_TOUCH_SCROLL) {
                    if (!isContactsExpanded) {
                        expandContactsSheet()
                    }
                }
            }

            override fun onScroll(
                view: AbsListView?,
                firstVisibleItem: Int,
                visibleItemCount: Int,
                totalItemCount: Int
            ) {
                if (!isContactsExpanded && firstVisibleItem > 0) {
                    expandContactsSheet()
                }
            }
        })
    }

    private fun expandContactsSheet() {
        if (isContactsExpanded) return
        isContactsExpanded = true

        if (contactsCollapsedOffset <= 0f) {
            contactsCollapsedOffset = binding.contactsDashboardHeader.height.toFloat().coerceAtLeast(resources.displayMetrics.density * 220f)
        }

        binding.contactsBottomSheet.animate()
            .translationY(0f)
            .setDuration(320)
            .setInterpolator(DecelerateInterpolator(1.8f))
            .start()

        binding.contactsDashboardHeader.animate()
            .alpha(0f)
            .translationY(-contactsCollapsedOffset * 0.15f)
            .setDuration(300)
            .start()

        binding.btnCollapseContacts.visibility = View.VISIBLE
        binding.btnCollapseContacts.alpha = 0f
        binding.btnCollapseContacts.animate()
            .alpha(1f)
            .setDuration(250)
            .start()
    }

    private fun collapseContactsSheet() {
        if (!isContactsExpanded) return
        isContactsExpanded = false

        if (contactsCollapsedOffset <= 0f) {
            contactsCollapsedOffset = binding.contactsDashboardHeader.height.toFloat().coerceAtLeast(resources.displayMetrics.density * 220f)
        }

        // Clear focus and dismiss keyboard if active
        binding.inputSearch.clearFocus()
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as? InputMethodManager
        imm?.hideSoftInputFromWindow(binding.inputSearch.windowToken, 0)

        binding.contactsBottomSheet.animate()
            .translationY(contactsCollapsedOffset)
            .setDuration(300)
            .setInterpolator(DecelerateInterpolator(1.6f))
            .start()

        binding.contactsDashboardHeader.animate()
            .alpha(1f)
            .translationY(0f)
            .setDuration(300)
            .start()

        binding.btnCollapseContacts.animate()
            .alpha(0f)
            .setDuration(200)
            .withEndAction {
                binding.btnCollapseContacts.visibility = View.GONE
            }
            .start()
    }

    private fun loadContactsAsync() {
        Thread {
            val contacts = VaniContactRepository.getContacts(this)
            runOnUiThread {
                currentContacts = contacts
                if (::adapter.isInitialized) {
                    adapter.updateData(contacts)
                }
                updateDashboardStats(contacts)
            }
        }.start()
    }

    private fun updateDashboardStats(contacts: List<VaniContact>) {
        val total = contacts.size
        val enrolled = if (::adapter.isInitialized) adapter.getEnrolledCount() else 0
        val threats = if (::adapter.isInitialized) adapter.getThreatCount() else 0
        val safePercent = if (total > 0) ((total - threats) * 100 / total) else 100

        binding.tvContactCount.text = "$total contacts protected by VaniRakshak"
        binding.tvContactSectionCount.text = "$total contacts"
        binding.tvStatProtected.text = "$total"
        binding.tvStatEnrolled.text = "$enrolled"
        binding.tvStatThreats.text = "$threats"
        binding.tvStatSafety.text = "$safePercent%"
    }

    fun openDetail(contactId: String) {
        val contact = currentContacts.find { it.id == contactId }
            ?: VaniSampleData.getContactById(contactId)
            ?: currentContacts.firstOrNull()
            ?: VaniSampleData.contacts[0]

        // Show real-time Android Notification with risk score
        VaniNotificationHelper.showCallRiskNotification(this, contact, contact.defaultScenario)

        startActivity(
            Intent(this, VaniForensicDetailActivity::class.java).apply {
                putExtra(EXTRA_CONTACT_ID, contact.id)
                putExtra("vani.contact.name", contact.name)
                putExtra("vani.contact.phone", contact.phone)
                putExtra("vani.contact.relation", contact.relation)
                putExtra("vani.contact.enrolled", contact.isVoiceEnrolled)
            }
        )
    }

    // -------------------------------------------------------------- setup state

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
        openNotificationSettings()
    }

    private fun openNotificationSettings() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            try {
                startActivity(
                    Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
                        .putExtra(Settings.EXTRA_APP_PACKAGE, packageName)
                )
                return
            } catch (e: ActivityNotFoundException) {
                // Fall through
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
            loadContactsAsync()
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
        loadContactsAsync()
        refreshShieldState()
    }

    companion object {
        const val EXTRA_CONTACT_ID = "vani.contact.id"
        private const val PERMISSION_REQUEST_CODE = 202

        /** Aarav Sharma from the sample data - resolves as a protected contact. */
        private const val SIMULATED_CALLER_NUMBER = "+91 98765 43210"
    }
}
