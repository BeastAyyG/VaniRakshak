package com.vericall

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.view.animation.DecelerateInterpolator
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.AbsListView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.vericall.config.VaniConfig
import com.vericall.contacts.VaniContactAdapter
import com.vericall.contacts.VaniEnrollmentSheet
import com.vericall.contacts.VaniForensicDetailActivity
import com.vericall.data.VaniContactRepository
import com.vericall.data.VaniSampleData
import com.vericall.databinding.ActivityVaniMainBinding
import com.vericall.model.VaniContact
import com.vericall.util.VaniCallRouter
import com.vericall.util.VaniNotificationHelper
import com.vericall.verification.ClaimResultActivity
import com.vericall.verification.ScamResultActivity
import com.vericall.verification.VaniVerificationClient
import com.vericall.verification.VoiceVerificationActivity

class VaniMainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityVaniMainBinding
    private lateinit var contactAdapter: VaniContactAdapter
    private var realContacts: List<VaniContact> = emptyList()

    private enum class Tab { HOME, LIVE, HISTORY, CONTACTS }
    private var currentTab = Tab.HOME

    private var isContactsExpanded = false
    private var contactsCollapsedOffset = 0f

    private val brandHandler = Handler(Looper.getMainLooper())
    private var isBrandCollapsed = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVaniMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

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
                if (currentTab == Tab.CONTACTS && isContactsExpanded) {
                    collapseContactsSheet()
                } else if (currentTab != Tab.HOME) {
                    switchTab(Tab.HOME)
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                    isEnabled = true
                }
            }
        })

        // Initialize core notification channels
        VaniNotificationHelper.createChannels(this)
        requestPermissionsIfMissing()

        // Interactive shield pill allows inspecting & modifying backend server URL
        binding.pillHeaderShield.setOnClickListener {
            VaniConfig.showServerConfigDialog(this)
        }
        binding.pillHeaderShield.setOnLongClickListener {
            VaniConfig.showServerConfigDialog(this)
            true
        }

        setupBottomNavigation()
        setupHomeTab()
        setupLiveTab()
        setupContactsTab()
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

    override fun onResume() {
        super.onResume()
        refreshRecentChecks()
        loadRealContactsAsync()
    }

    // ------------------------------------------------------------- Navigation
    private fun setupBottomNavigation() {
        binding.navBtnHome.setOnClickListener { switchTab(Tab.HOME) }
        binding.navBtnLive.setOnClickListener { switchTab(Tab.LIVE) }
        binding.navBtnHistory.setOnClickListener { switchTab(Tab.HISTORY) }
        binding.navBtnContacts.setOnClickListener { switchTab(Tab.CONTACTS) }
        binding.btnSeeAllHistory.setOnClickListener { switchTab(Tab.HISTORY) }
        switchTab(Tab.HOME)
    }

    private fun switchTab(tab: Tab) {
        currentTab = tab

        val tabs = listOf(
            Triple(Tab.HOME, binding.layoutTabHome, binding.indicatorNavHome to (binding.ivNavHome to binding.tvNavHome)),
            Triple(Tab.LIVE, binding.layoutTabLive, binding.indicatorNavLive to (binding.ivNavLive to binding.tvNavLive)),
            Triple(Tab.HISTORY, binding.layoutTabHistory, binding.indicatorNavHistory to (binding.ivNavHistory to binding.tvNavHistory)),
            Triple(Tab.CONTACTS, binding.layoutTabContacts, binding.indicatorNavContacts to (binding.ivNavContacts to binding.tvNavContacts))
        )

        val brand = ContextCompat.getColor(this, R.color.brand_cobalt)
        val muted = ContextCompat.getColor(this, R.color.text_tertiary)

        tabs.forEach { (t, layout, navViews) ->
            val (indicator, pair) = navViews
            val (iv, tv) = pair
            val isSelected = (t == tab)

            if (isSelected) {
                if (layout.visibility != View.VISIBLE) {
                    layout.alpha = 0f
                    layout.visibility = View.VISIBLE
                    layout.animate().alpha(1f).setDuration(220).start()
                }
                indicator.setBackgroundResource(R.drawable.bg_nav_indicator_active)
                indicator.animate().scaleX(1.05f).scaleY(1.05f).setDuration(150).withEndAction {
                    indicator.animate().scaleX(1f).scaleY(1f).setDuration(100).start()
                }.start()
                iv.setColorFilter(brand)
                tv.setTextColor(brand)
                tv.typeface = Typeface.DEFAULT_BOLD
            } else {
                layout.visibility = View.GONE
                indicator.background = null
                iv.setColorFilter(muted)
                tv.setTextColor(muted)
                tv.typeface = Typeface.DEFAULT
            }
        }

        if (tab == Tab.HISTORY) {
            refreshFullHistory()
        } else if (tab == Tab.CONTACTS) {
            binding.contactsDashboardHeader.post {
                contactsCollapsedOffset = binding.contactsDashboardHeader.height.toFloat()
                if (!isContactsExpanded && contactsCollapsedOffset > 0) {
                    binding.contactsBottomSheet.translationY = contactsCollapsedOffset
                }
            }
        }
    }

    private fun setupHomeTab() {
        binding.layoutTabHome.setOnScrollChangeListener { _, _, scrollY, _, _ ->
            if (scrollY > 15 && !isBrandCollapsed) {
                collapseBrandText(animate = true)
            }
        }

        binding.btnHeroVerify.setOnClickListener {
            startActivity(Intent(this, VoiceVerificationActivity::class.java))
        }

        // Micro touch interactions
        applyMicroTouchFeedback(binding.btnHeroVerify)
        applyMicroTouchFeedback(binding.btnSubmitClaim)

        startHeroPulseGlow()

        binding.btnSubmitClaim.setOnClickListener {
            submitTextQuery()
        }

        binding.inputHomeClaim.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                submitTextQuery()
                true
            } else false
        }

        binding.chipFactCheck.setOnClickListener {
            binding.inputHomeClaim.setText("RBI is going to close all bank accounts without KYC.")
            submitTextQuery()
        }

        binding.chipScamCheck.setOnClickListener {
            binding.inputHomeClaim.setText("Your SBI account will be blocked today. Share your OTP immediately.")
            submitTextQuery()
        }

        binding.chipLiveGuard.setOnClickListener {
            switchTab(Tab.LIVE)
        }

        refreshRecentChecks()
    }

    private fun submitTextQuery() {
        val query = binding.inputHomeClaim.text.toString().trim()
        if (query.isBlank()) {
            Toast.makeText(this, "Please enter a claim or message to check", Toast.LENGTH_SHORT).show()
            return
        }

        binding.inputHomeClaim.text?.clear()
        Toast.makeText(this, "Checking claim against reliable sources...", Toast.LENGTH_SHORT).show()

        VaniVerificationClient.verifyClaim(this, query) { claimResult, scamResult, sourcesJson, redFlagsJson, actionAdvice ->
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                if (scamResult != null) {
                    val intent = Intent(this, ScamResultActivity::class.java).apply {
                        putExtra(ScamResultActivity.EXTRA_STATEMENT, scamResult.suspiciousStatement)
                        putExtra(ScamResultActivity.EXTRA_THREAT_SCORE, scamResult.threatScore)
                        putExtra(ScamResultActivity.EXTRA_SCAM_TYPE, scamResult.scamType)
                        putExtra(ScamResultActivity.EXTRA_RED_FLAGS_JSON, redFlagsJson)
                        putExtra(ScamResultActivity.EXTRA_ACTION_ADVICE, actionAdvice)
                    }
                    startActivity(intent)
                } else if (claimResult != null) {
                    val intent = Intent(this, ClaimResultActivity::class.java).apply {
                        putExtra(ClaimResultActivity.EXTRA_CLAIM, claimResult.claim)
                        putExtra(ClaimResultActivity.EXTRA_VERDICT, claimResult.verdict.name)
                        putExtra(ClaimResultActivity.EXTRA_CONFIDENCE, claimResult.confidence)
                        putExtra(ClaimResultActivity.EXTRA_EXPLANATION, claimResult.explanation)
                        putExtra(ClaimResultActivity.EXTRA_ACTION, claimResult.recommendedAction)
                        putExtra(ClaimResultActivity.EXTRA_SOURCES_JSON, sourcesJson)
                    }
                    startActivity(intent)
                }
            }
        }
    }

    private fun refreshRecentChecks() {
        binding.containerHomeRecentChecks.removeAllViews()
        val history = VaniVerificationClient.getRecentHistory(this).take(3)
        for (item in history) {
            val v = createHistoryRow(item)
            binding.containerHomeRecentChecks.addView(v)
        }
    }

    private fun refreshFullHistory() {
        binding.containerFullHistory.removeAllViews()
        val history = VaniVerificationClient.getRecentHistory(this)
        for (item in history) {
            val v = createHistoryRow(item)
            binding.containerFullHistory.addView(v)
        }
    }

    private fun createHistoryRow(item: VaniVerificationClient.HistoryItem): View {
        val view = LayoutInflater.from(this).inflate(R.layout.item_recent_check, null, false)
        val tvBadge = view.findViewById<TextView>(R.id.tvRecentVerdictBadge)
        val tvTime = view.findViewById<TextView>(R.id.tvRecentTime)
        val tvQuery = view.findViewById<TextView>(R.id.tvRecentQuery)
        val tvDetails = view.findViewById<TextView>(R.id.tvRecentDetails)

        tvTime.text = item.timeLabel
        tvQuery.text = item.query
        tvDetails.text = item.details

        when (item.verdict.uppercase()) {
            "SUPPORTED", "TRUE" -> {
                tvBadge.text = "✓ SUPPORTED"
                tvBadge.setBackgroundResource(R.drawable.bg_badge_supported)
                tvBadge.setTextColor(ContextCompat.getColor(this, R.color.verdict_supported))
            }
            "CONTRADICTED", "FALSE" -> {
                tvBadge.text = "✕ CONTRADICTED"
                tvBadge.setBackgroundResource(R.drawable.bg_badge_contradicted)
                tvBadge.setTextColor(ContextCompat.getColor(this, R.color.verdict_contradicted))
            }
            "SCAM", "FRAUDULENT_CLAIM" -> {
                tvBadge.text = "⚠ HIGH RISK"
                tvBadge.setBackgroundResource(R.drawable.bg_badge_scam)
                tvBadge.setTextColor(ContextCompat.getColor(this, R.color.verdict_scam))
            }
            else -> {
                tvBadge.text = "⚠ UNVERIFIABLE"
                tvBadge.setBackgroundResource(R.drawable.bg_badge_unverifiable)
                tvBadge.setTextColor(ContextCompat.getColor(this, R.color.verdict_unverifiable))
            }
        }

        view.setOnClickListener {
            if (item.verdict == "SCAM") {
                val intent = Intent(this, ScamResultActivity::class.java).apply {
                    putExtra(ScamResultActivity.EXTRA_STATEMENT, "\"${item.query}\"")
                    putExtra(ScamResultActivity.EXTRA_THREAT_SCORE, item.score)
                    putExtra(ScamResultActivity.EXTRA_SCAM_TYPE, item.details)
                }
                startActivity(intent)
            } else {
                val intent = Intent(this, ClaimResultActivity::class.java).apply {
                    putExtra(ClaimResultActivity.EXTRA_CLAIM, item.query)
                    putExtra(ClaimResultActivity.EXTRA_VERDICT, item.verdict)
                    putExtra(ClaimResultActivity.EXTRA_CONFIDENCE, item.score)
                    putExtra(ClaimResultActivity.EXTRA_EXPLANATION, item.details)
                    putExtra(ClaimResultActivity.EXTRA_ACTION, "Always check official portals before acting.")
                }
                startActivity(intent)
            }
        }
        return view
    }

    // ------------------------------------------------------------- Live Guard Tab
    private fun setupLiveTab() {
        binding.btnLiveSimulateCall.setOnClickListener {
            VaniCallRouter.onRinging(this, "+91 98765 43210")
        }

        binding.btnLiveCloneAttack.setOnClickListener {
            openContactDetail("c1")
        }

        binding.btnLiveGenuineCall.setOnClickListener {
            openContactDetail("c2")
        }

        binding.btnLiveFileInspect.setOnClickListener {
            startActivity(Intent(this, com.vericall.verification.FileForensicActivity::class.java))
        }

        applyMicroTouchFeedback(binding.btnLiveSimulateCall)
        applyMicroTouchFeedback(binding.btnLiveCloneAttack)
        applyMicroTouchFeedback(binding.btnLiveGenuineCall)
        applyMicroTouchFeedback(binding.btnLiveFileInspect)
    }

    private fun startHeroPulseGlow() {
        val pulseX = ObjectAnimator.ofFloat(binding.viewHeroPulseGlow, "scaleX", 1.0f, 1.08f, 1.0f).apply {
            duration = 2400
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
            interpolator = AccelerateDecelerateInterpolator()
        }
        val pulseY = ObjectAnimator.ofFloat(binding.viewHeroPulseGlow, "scaleY", 1.0f, 1.08f, 1.0f).apply {
            duration = 2400
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
            interpolator = AccelerateDecelerateInterpolator()
        }
        val alphaAnim = ObjectAnimator.ofFloat(binding.viewHeroPulseGlow, "alpha", 0.65f, 0.95f, 0.65f).apply {
            duration = 2400
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
            interpolator = AccelerateDecelerateInterpolator()
        }
        pulseX.start()
        pulseY.start()
        alphaAnim.start()
    }

    private fun applyMicroTouchFeedback(view: View) {
        view.setOnTouchListener { v, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    v.animate().scaleX(0.975f).scaleY(0.975f).setDuration(100).start()
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    v.animate().scaleX(1.0f).scaleY(1.0f).setDuration(160).setInterpolator(DecelerateInterpolator()).start()
                }
            }
            false
        }
    }

    // ------------------------------------------------------------- Contacts Tab (Production Masterpiece)
    private fun setupContactsTab() {
        contactAdapter = VaniContactAdapter(
            context = this,
            contacts = emptyList(),
            onContactClicked = { contact ->
                openContactDetail(contact.id)
            },
            onEnrollClicked = { contact ->
                VaniEnrollmentSheet.show(this, contact) {
                    loadRealContactsAsync()
                }
            }
        )
        binding.listRealContacts.adapter = contactAdapter

        // Search with live result count badge
        binding.inputContactSearch.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
            override fun afterTextChanged(s: Editable?) {
                val query = s?.toString()?.trim() ?: ""
                contactAdapter.filter(query)
                if (query.isNotEmpty()) {
                    binding.tvSearchResultCount.visibility = View.VISIBLE
                    binding.tvSearchResultCount.text = "${contactAdapter.count}"
                } else {
                    binding.tvSearchResultCount.visibility = View.GONE
                }
            }
        })

        // Callback to update section count when filter changes
        contactAdapter.onFilterChanged = { count ->
            binding.tvContactSectionCount.text = "$count contacts"
        }

        // Quick action cards
        binding.cardContactCloneDemo.setOnClickListener {
            openContactDetail("c1")
        }
        binding.cardContactGenuineDemo.setOnClickListener {
            openContactDetail("c2")
        }

        // FAB: Simulate incoming call
        binding.fabSimulateCall.setOnClickListener {
            VaniCallRouter.onRinging(this, "+91 98765 43210")
        }

        setupContactsSheetAnimation()
        loadRealContactsAsync()
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
        binding.inputContactSearch.setOnClickListener {
            if (!isContactsExpanded) expandContactsSheet()
        }
        binding.inputContactSearch.setOnFocusChangeListener { _, hasFocus ->
            if (hasFocus && !isContactsExpanded) {
                expandContactsSheet()
            }
        }

        // Scrolling & swiping contact list triggers smooth full-screen expansion
        var listTouchStartY = 0f
        binding.listRealContacts.setOnTouchListener { _, event ->
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
                        binding.listRealContacts.onTouchEvent(cancelEvent)
                        cancelEvent.recycle()
                        true
                    } else if (isContactsExpanded && deltaY > 35f &&
                        binding.listRealContacts.firstVisiblePosition == 0 &&
                        (binding.listRealContacts.getChildAt(0)?.top ?: 0) >= 0
                    ) {
                        collapseContactsSheet()
                        val cancelEvent = MotionEvent.obtain(event)
                        cancelEvent.action = MotionEvent.ACTION_CANCEL
                        binding.listRealContacts.onTouchEvent(cancelEvent)
                        cancelEvent.recycle()
                        true
                    } else false
                }
                else -> false
            }
        }

        binding.listRealContacts.setOnScrollListener(object : AbsListView.OnScrollListener {
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
        binding.inputContactSearch.clearFocus()
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as? InputMethodManager
        imm?.hideSoftInputFromWindow(binding.inputContactSearch.windowToken, 0)

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

    private fun loadRealContactsAsync() {
        Thread {
            val contacts = VaniContactRepository.getContacts(this)
            runOnUiThread {
                realContacts = contacts
                contactAdapter.updateData(contacts)

                // Update subtitle
                binding.tvContactCount.text = "${contacts.size} contacts protected by VaniRakshak"

                // Update dashboard stat tiles
                val total = contacts.size
                val enrolled = contactAdapter.getEnrolledCount()
                val threats = contactAdapter.getThreatCount()
                val safetyPct = if (total > 0) ((total - threats) * 100 / total) else 100

                binding.tvStatProtected.text = "$total"
                binding.tvStatEnrolled.text = "$enrolled"
                binding.tvStatThreats.text = "$threats"
                binding.tvStatSafety.text = "${safetyPct}%"

                // Update section header
                binding.tvContactSectionCount.text = "$total contacts"
            }
        }.start()
    }

    private fun openContactDetail(contactId: String) {
        val contact = realContacts.find { it.id == contactId }
            ?: VaniSampleData.getContactById(contactId)
            ?: realContacts.firstOrNull()
            ?: VaniSampleData.contacts[0]

        VaniNotificationHelper.showCallRiskNotification(this, contact, contact.defaultScenario)

        startActivity(
            Intent(this, VaniForensicDetailActivity::class.java).apply {
                putExtra(VaniForensicDetailActivity.EXTRA_CONTACT_ID, contact.id)
                putExtra("vani.contact.name", contact.name)
                putExtra("vani.contact.phone", contact.phone)
                putExtra("vani.contact.relation", contact.relation)
                putExtra("vani.contact.enrolled", contact.isVoiceEnrolled)
            }
        )
    }

    private fun requestPermissionsIfMissing() {
        val permissions = mutableListOf(
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.READ_CALL_LOG,
            Manifest.permission.READ_CONTACTS,
            Manifest.permission.RECORD_AUDIO
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        val missing = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missing.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missing.toTypedArray(), 101)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        loadRealContactsAsync()
    }
}
