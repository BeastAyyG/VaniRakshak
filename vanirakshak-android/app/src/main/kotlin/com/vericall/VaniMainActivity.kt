package com.vericall

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.view.GestureDetector
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.view.animation.DecelerateInterpolator
import android.view.inputmethod.EditorInfo
import android.widget.Toast
import android.provider.ContactsContract
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.recyclerview.widget.LinearLayoutManager
import com.vericall.calling.VaniActiveCallActivity
import com.vericall.calling.VaniCallDetailActivity
import com.vericall.config.VaniConfig
import com.vericall.contacts.VaniContactAdapter
import com.vericall.contacts.VaniContactDetailActivity
import com.vericall.contacts.VaniEnrollmentSheet
import com.vericall.data.VaniCallHistoryRepository
import com.vericall.data.VaniContactRepository
import com.vericall.data.VaniSampleData
import com.vericall.databinding.ActivityVaniMainBinding
import com.vericall.model.CallDirection
import com.vericall.model.VaniCallRecord
import com.vericall.model.VaniContact
import com.vericall.ui.VaniCallHistoryAdapter
import com.vericall.ui.VaniFavoritesAdapter
import com.vericall.ui.VaniProtectedCallAdapter
import com.vericall.util.VaniCallRouter
import com.vericall.util.VaniNotificationHelper
import com.vericall.verification.ClaimResultActivity
import com.vericall.verification.ScamResultActivity
import com.vericall.verification.VaniVerificationClient
import com.vericall.verification.VoiceVerificationActivity

class VaniMainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityVaniMainBinding
    private var protectedCallAdapter: VaniProtectedCallAdapter? = null

    private enum class Tab { FAVORITES, RECENTS, CONTACTS, KEYPAD, VOICEMAIL, SHIELD }
    private var currentTab = Tab.RECENTS

    // Adapters
    private lateinit var favoritesAdapter: VaniFavoritesAdapter
    private lateinit var recentsAdapter: VaniCallHistoryAdapter
    private lateinit var contactAdapter: VaniContactAdapter

    private var realContacts: List<VaniContact> = emptyList()
    private var recentCalls: List<VaniCallRecord> = emptyList()

    // Keypad State
    private val keypadBuffer = StringBuilder()
    private var matchedContact: VaniContact? = null

    // Brand Collapse & Top Bar Vanish
    private val brandHandler = Handler(Looper.getMainLooper())
    private var isBrandCollapsed = false
    private var isTopBarHidden = false
    private var topBarAnimator: ValueAnimator? = null
    private lateinit var gestureDetector: GestureDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVaniMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Apple iOS clean minimalist aesthetic - dark icons on white canvas
        WindowCompat.getInsetsController(window, window.decorView).apply {
            isAppearanceLightStatusBars = true
            isAppearanceLightNavigationBars = true
        }

        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        VaniNotificationHelper.createChannels(this)
        requestPermissionsIfMissing()

        // Shield Status Pill in Header switches to Shield Tab; long-press opens IP config
        binding.pillHeaderShield.setOnClickListener {
            switchTab(Tab.SHIELD)
        }
        binding.pillHeaderShield.setOnLongClickListener {
            VaniConfig.showServerConfigDialog(this)
            true
        }

        setupGestureDetector()
        setupBottomNavigation()
        setupFavoritesTab()
        setupRecentsTab()
        setupContactsTab()
        setupKeypadTab()
        setupVoicemailTab()
        setupShieldTab()
        setupTopBarScrollBehavior()
        setupBrandCollapsingAnimation()
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (::gestureDetector.isInitialized) {
            gestureDetector.onTouchEvent(ev)
        }
        return super.dispatchTouchEvent(ev)
    }

    override fun onResume() {
        super.onResume()
        refreshAllData()
        if (currentTab == Tab.VOICEMAIL) {
            loadProtectedCalls()
        }
    }

    override fun onPause() {
        protectedCallAdapter?.stopPlayback()
        super.onPause()
    }

    override fun onDestroy() {
        protectedCallAdapter?.stopPlayback()
        super.onDestroy()
        brandHandler.removeCallbacksAndMessages(null)
    }

    private fun setupBrandCollapsingAnimation() {
        binding.ivAppLogo.setOnClickListener {
            brandHandler.removeCallbacksAndMessages(null)
            if (isBrandCollapsed) {
                expandBrandText()
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
            .withEndAction { binding.layoutBrandText.visibility = View.GONE }
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

    // ------------------------------------------------------------- Navigation
    private fun setupBottomNavigation() {
        binding.navBtnFavorites.setOnClickListener { switchTab(Tab.FAVORITES) }
        binding.navBtnRecents.setOnClickListener { switchTab(Tab.RECENTS) }
        binding.navBtnContacts.setOnClickListener { switchTab(Tab.CONTACTS) }
        binding.navBtnKeypad.setOnClickListener { switchTab(Tab.KEYPAD) }
        binding.navBtnVoicemail.setOnClickListener { switchTab(Tab.VOICEMAIL) }

        switchTab(Tab.RECENTS)
    }

    private fun switchTab(tab: Tab) {
        currentTab = tab
        showTopBar(animate = false)

        val bottomTabs = listOf(
            Triple(Tab.FAVORITES, binding.layoutTabFavorites, binding.indicatorNavFavorites to (binding.ivNavFavorites to binding.tvNavFavorites)),
            Triple(Tab.RECENTS, binding.layoutTabRecents, binding.indicatorNavRecents to (binding.ivNavRecents to binding.tvNavRecents)),
            Triple(Tab.CONTACTS, binding.layoutTabContacts, binding.indicatorNavContacts to (binding.ivNavContacts to binding.tvNavContacts)),
            Triple(Tab.KEYPAD, binding.layoutTabKeypad, binding.indicatorNavKeypad to (binding.ivNavKeypad to binding.tvNavKeypad)),
            Triple(Tab.VOICEMAIL, binding.layoutTabVoicemail, binding.indicatorNavVoicemail to (binding.ivNavVoicemail to binding.tvNavVoicemail))
        )

        val brand = ContextCompat.getColor(this, R.color.brand_primary)
        val muted = ContextCompat.getColor(this, R.color.text_tertiary)

        bottomTabs.forEach { (t, layout, navViews) ->
            val (indicator, pair) = navViews
            val (iv, tv) = pair
            val isSelected = (t == tab)

            if (isSelected) {
                if (layout.visibility != View.VISIBLE) {
                    layout.alpha = 0f
                    layout.visibility = View.VISIBLE
                    layout.animate().alpha(1f).setDuration(180).start()
                }
                indicator.setBackgroundResource(R.drawable.bg_nav_indicator_active)
                iv.setColorFilter(brand)
                tv.setTextColor(brand)
                tv.typeface = Typeface.DEFAULT_BOLD
            } else {
                layout.visibility = View.GONE
                indicator.background = null
                iv.setColorFilter(muted)
                tv.setTextColor(muted)
                tv.typeface = Typeface.DEFAULT
                if (t == Tab.VOICEMAIL) {
                    protectedCallAdapter?.stopPlayback()
                }
            }
        }

        if (tab == Tab.VOICEMAIL) {
            loadProtectedCalls()
        }

        // Shield is opened via header badge
        if (tab == Tab.SHIELD) {
            binding.layoutTabShield.visibility = View.VISIBLE
            binding.layoutTabShield.alpha = 0f
            binding.layoutTabShield.animate().alpha(1f).setDuration(180).start()
        } else {
            binding.layoutTabShield.visibility = View.GONE
        }
    }

    // ------------------------------------------------------------- 1. Favorites Tab
    private fun setupFavoritesTab() {
        favoritesAdapter = VaniFavoritesAdapter(
            favorites = emptyList(),
            onContactClicked = { contact -> openContactDetail(contact.id) },
            onCallClicked = { contact ->
                VaniActiveCallActivity.startOutgoingCall(this, contact.phone, contact.name)
            }
        )
        binding.recyclerFavorites.layoutManager = LinearLayoutManager(this)
        binding.recyclerFavorites.adapter = favoritesAdapter
    }

    // ------------------------------------------------------------- 2. Recents Tab
    private fun setupRecentsTab() {
        recentsAdapter = VaniCallHistoryAdapter(
            context = this,
            calls = emptyList(),
            onCallClicked = { record ->
                val intent = Intent(this, VaniCallDetailActivity::class.java).apply {
                    putExtra(VaniCallDetailActivity.EXTRA_RECORD_ID, record.id)
                }
                startActivity(intent)
            },
            onQuickCallClicked = { record ->
                VaniActiveCallActivity.startOutgoingCall(this, record.number, record.callerName)
            }
        )
        binding.recyclerRecents.layoutManager = LinearLayoutManager(this)
        binding.recyclerRecents.adapter = recentsAdapter

        binding.btnFilterAll.setOnClickListener {
            recentsAdapter.updateData(recentCalls)
            updateRecentsEmptyState(recentCalls)
        }

        binding.btnFilterMissed.setOnClickListener {
            val missed = recentCalls.filter { it.direction == CallDirection.MISSED }
            recentsAdapter.updateData(missed)
            updateRecentsEmptyState(missed)
        }
    }

    private fun updateRecentsEmptyState(items: List<VaniCallRecord>) {
        binding.layoutRecentsEmpty.visibility = if (items.isEmpty()) View.VISIBLE else View.GONE
    }

    // ------------------------------------------------------------- 3. Contacts Tab
    private fun setupContactsTab() {
        contactAdapter = VaniContactAdapter(
            context = this,
            contacts = emptyList(),
            onContactClicked = { contact -> openContactDetail(contact.id) },
            onEnrollClicked = { contact ->
                VaniEnrollmentSheet.show(this, contact) { refreshAllData() }
            }
        )
        binding.listRealContacts.adapter = contactAdapter

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
    }

    // ------------------------------------------------------------- 4. Keypad Tab
    private fun setupKeypadTab() {
        val keys = listOf(
            binding.key1 to '1', binding.key2 to '2', binding.key3 to '3',
            binding.key4 to '4', binding.key5 to '5', binding.key6 to '6',
            binding.key7 to '7', binding.key8 to '8', binding.key9 to '9',
            binding.keyStar to '*', binding.key0 to '0', binding.keyHash to '#'
        )

        for ((view, char) in keys) {
            view.setOnClickListener {
                it.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                appendKeypadDigit(char)
            }
        }

        // Long press 1 for Voicemail
        binding.key1.setOnLongClickListener {
            it.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
            Toast.makeText(this, "Opening Voicemail...", Toast.LENGTH_SHORT).show()
            switchTab(Tab.VOICEMAIL)
            true
        }

        // Long press 0 for +
        binding.key0.setOnLongClickListener {
            it.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
            appendKeypadDigit('+')
            true
        }

        // Voicemail quick access button
        binding.btnKeypadVoicemail.setOnClickListener {
            it.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
            switchTab(Tab.VOICEMAIL)
        }

        // Add to contacts button
        binding.tvKeypadAddContact.setOnClickListener {
            val number = keypadBuffer.toString().trim()
            if (number.isNotEmpty()) {
                val intent = Intent(Intent.ACTION_INSERT, ContactsContract.Contacts.CONTENT_URI).apply {
                    putExtra(ContactsContract.Intents.Insert.PHONE, number)
                }
                startActivity(intent)
            }
        }

        // Backspace
        binding.btnKeypadBackspace.setOnClickListener {
            it.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
            if (keypadBuffer.isNotEmpty()) {
                keypadBuffer.deleteCharAt(keypadBuffer.length - 1)
                updateKeypadDisplay()
            }
        }

        binding.btnKeypadBackspace.setOnLongClickListener {
            it.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
            keypadBuffer.clear()
            updateKeypadDisplay()
            true
        }

        // Tap on suggested contact match
        binding.cardKeypadContactMatch.setOnClickListener {
            matchedContact?.let { contact ->
                VaniActiveCallActivity.startOutgoingCall(this, contact.phone, contact.name)
            }
        }

        // Call Button
        binding.btnKeypadCall.setOnClickListener {
            it.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
            val number = keypadBuffer.toString().trim()
            if (number.isNotBlank()) {
                val targetName = matchedContact?.name ?: VaniCallRouter.resolve(this, number).name
                VaniActiveCallActivity.startOutgoingCall(this, number, targetName)
            } else {
                Toast.makeText(this, "Enter a phone number to dial", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun appendKeypadDigit(char: Char) {
        keypadBuffer.append(char)
        updateKeypadDisplay()
    }

    private fun updateKeypadDisplay() {
        val text = keypadBuffer.toString()
        binding.tvKeypadDisplayNumber.text = text
        binding.btnKeypadBackspace.visibility = if (text.isNotEmpty()) View.VISIBLE else View.INVISIBLE
        binding.tvKeypadAddContact.visibility = if (text.isNotEmpty()) View.VISIBLE else View.GONE

        // T9 / Number auto-lookup
        if (text.length >= 2) {
            val normalizedQuery = text.replace("+", "")
            val match = realContacts.find {
                it.phone.filter { c -> c.isDigit() }.contains(normalizedQuery) ||
                it.name.contains(normalizedQuery, ignoreCase = true)
            }
            if (match != null) {
                matchedContact = match
                binding.tvKeypadMatchName.text = match.name
                binding.tvKeypadMatchPhone.text = match.phone
                val initial = match.name.firstOrNull { it.isLetter() }?.uppercaseChar() ?: 'V'
                binding.tvKeypadMatchAvatar.text = initial.toString()
                binding.cardKeypadContactMatch.visibility = View.VISIBLE
            } else {
                matchedContact = null
                binding.cardKeypadContactMatch.visibility = View.GONE
            }
        } else {
            matchedContact = null
            binding.cardKeypadContactMatch.visibility = View.GONE
        }
    }

    // ------------------------------------------------------------- 5. Protected Calls / Voicemail Tab
    private fun setupVoicemailTab() {
        protectedCallAdapter = VaniProtectedCallAdapter()
        binding.recyclerProtectedCalls.layoutManager = LinearLayoutManager(this)
        binding.recyclerProtectedCalls.adapter = protectedCallAdapter
        loadProtectedCalls()
    }

    private fun loadProtectedCalls() {
        val allCalls = VaniCallHistoryRepository.getRecents(this)
        if (allCalls.isEmpty()) {
            binding.recyclerProtectedCalls.visibility = View.GONE
            binding.layoutEmptyProtectedCalls.visibility = View.VISIBLE
            binding.badgeSavedCount.text = "0 Saved"
            binding.badgeHighRiskCount.text = "🔒 0 Protected"
        } else {
            binding.recyclerProtectedCalls.visibility = View.VISIBLE
            binding.layoutEmptyProtectedCalls.visibility = View.GONE
            protectedCallAdapter?.updateRecords(allCalls)

            val savedCount = allCalls.size
            val highRiskCount = allCalls.count { it.riskScore >= 50 || it.isProtected }
            binding.badgeSavedCount.text = "$savedCount Saved"
            binding.badgeHighRiskCount.text = "🔒 $highRiskCount Protected"
        }
    }

    // ------------------------------------------------------------- 6. Shield / Settings Tab
    private fun setupShieldTab() {
        startHeroPulseGlow()

        binding.btnHeroVerify.setOnClickListener {
            startActivity(Intent(this, VoiceVerificationActivity::class.java))
        }

        binding.btnSubmitClaim.setOnClickListener { submitTextQuery() }
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

        binding.btnLiveSimulateCall.setOnClickListener {
            VaniCallRouter.onRinging(this, "+91 98765 43210")
        }

        binding.btnLiveCloneAttack.setOnClickListener {
            VaniActiveCallActivity.startOutgoingCall(this, "+91 98765 43210", "Aarav Sharma (Clone Sim)")
        }

        binding.btnLiveGenuineCall.setOnClickListener {
            VaniActiveCallActivity.startOutgoingCall(this, "+91 91234 56780", "Priya Nair (Authentic)")
        }

        binding.btnLiveFileInspect.setOnClickListener {
            startActivity(Intent(this, com.vericall.verification.FileForensicActivity::class.java))
        }

        applyMicroTouchFeedback(binding.btnHeroVerify)
        applyMicroTouchFeedback(binding.btnLiveSimulateCall)
        applyMicroTouchFeedback(binding.btnLiveCloneAttack)
        applyMicroTouchFeedback(binding.btnLiveGenuineCall)
        applyMicroTouchFeedback(binding.btnLiveFileInspect)
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

    private fun refreshAllData() {
        Thread {
            val contacts = VaniContactRepository.getContacts(this)
            val favorites = VaniContactRepository.getFavorites(this)
            val calls = VaniCallHistoryRepository.getRecents(this)

            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                realContacts = contacts
                recentCalls = calls

                // Update Favorites
                favoritesAdapter.updateData(favorites)
                binding.layoutFavoritesEmpty.visibility = if (favorites.isEmpty()) View.VISIBLE else View.GONE

                // Update Recents
                val isMissedSelected = binding.btnFilterMissed.isChecked
                val filteredCalls = if (isMissedSelected) calls.filter { it.direction == CallDirection.MISSED } else calls
                recentsAdapter.updateData(filteredCalls)
                updateRecentsEmptyState(filteredCalls)

                // Update Contacts
                contactAdapter.updateData(contacts)
                binding.tvContactsSubtitle.text = "${contacts.size} contacts protected by VaniRakshak"
            }
        }.start()
    }

    private fun openContactDetail(contactId: String) {
        val intent = Intent(this, VaniContactDetailActivity::class.java).apply {
            putExtra(VaniContactDetailActivity.EXTRA_CONTACT_ID, contactId)
        }
        startActivity(intent)
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
        refreshAllData()
    }

    // ------------------------------------------------------------- Top Bar Gesture & Scroll Collapse
    private fun setupGestureDetector() {
        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onFling(e1: MotionEvent?, e2: MotionEvent, velocityX: Float, velocityY: Float): Boolean {
                if (e1 != null) {
                    val deltaY = e1.y - e2.y
                    if (deltaY > 50 && Math.abs(velocityY) > 120) {
                        hideTopBar(animate = true)
                        return true
                    } else if (deltaY < -50 && Math.abs(velocityY) > 120) {
                        showTopBar(animate = true)
                        return true
                    }
                }
                return false
            }
        })
    }

    private fun setupTopBarScrollBehavior() {
        val rvScrollListener = object : androidx.recyclerview.widget.RecyclerView.OnScrollListener() {
            override fun onScrolled(recyclerView: androidx.recyclerview.widget.RecyclerView, dx: Int, dy: Int) {
                if (dy > 12) {
                    hideTopBar(animate = true)
                } else if (dy < -12) {
                    showTopBar(animate = true)
                }
            }
        }
        binding.recyclerFavorites.addOnScrollListener(rvScrollListener)
        binding.recyclerRecents.addOnScrollListener(rvScrollListener)

        binding.listRealContacts.setOnScrollListener(object : android.widget.AbsListView.OnScrollListener {
            private var lastFirstVisibleItem = 0
            private var lastTop = 0
            override fun onScrollStateChanged(view: android.widget.AbsListView?, scrollState: Int) {}
            override fun onScroll(view: android.widget.AbsListView?, firstVisibleItem: Int, visibleItemCount: Int, totalItemCount: Int) {
                val firstChild = view?.getChildAt(0) ?: return
                val top = firstChild.top
                if (firstVisibleItem > lastFirstVisibleItem || (firstVisibleItem == lastFirstVisibleItem && top < lastTop - 14)) {
                    hideTopBar(animate = true)
                } else if (firstVisibleItem < lastFirstVisibleItem || (firstVisibleItem == lastFirstVisibleItem && top > lastTop + 14)) {
                    showTopBar(animate = true)
                }
                lastFirstVisibleItem = firstVisibleItem
                lastTop = top
            }
        })

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            binding.layoutTabShield.setOnScrollChangeListener { _, _, scrollY, _, oldScrollY ->
                if (scrollY > oldScrollY + 14) {
                    hideTopBar(animate = true)
                } else if (scrollY < oldScrollY - 14) {
                    showTopBar(animate = true)
                }
            }
            binding.recyclerProtectedCalls.addOnScrollListener(object : androidx.recyclerview.widget.RecyclerView.OnScrollListener() {
                override fun onScrolled(recyclerView: androidx.recyclerview.widget.RecyclerView, dx: Int, dy: Int) {
                    if (dy > 14) {
                        hideTopBar(animate = true)
                    } else if (dy < -14) {
                        showTopBar(animate = true)
                    }
                }
            })
        }
    }

    private fun hideTopBar(animate: Boolean = true) {
        if (isTopBarHidden) return
        isTopBarHidden = true
        topBarAnimator?.cancel()
        val header = binding.layoutTopHeader
        val currentHeight = header.height.takeIf { it > 0 } ?: dpToPx(56)
        if (!animate) {
            header.layoutParams.height = 0
            header.alpha = 0f
            header.visibility = View.GONE
            header.requestLayout()
            return
        }
        topBarAnimator = ValueAnimator.ofInt(currentHeight, 0).apply {
            duration = 220
            interpolator = AccelerateDecelerateInterpolator()
            addUpdateListener { va ->
                val h = va.animatedValue as Int
                header.layoutParams.height = h
                header.alpha = h.toFloat() / currentHeight.toFloat()
                header.requestLayout()
            }
            addListener(object : android.animation.AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: android.animation.Animator) {
                    header.visibility = View.GONE
                }
            })
            start()
        }
    }

    private fun showTopBar(animate: Boolean = true) {
        if (!isTopBarHidden) return
        isTopBarHidden = false
        topBarAnimator?.cancel()
        val header = binding.layoutTopHeader
        val targetHeight = dpToPx(56)
        header.visibility = View.VISIBLE
        if (!animate) {
            header.layoutParams.height = targetHeight
            header.alpha = 1f
            header.requestLayout()
            return
        }
        val currentHeight = header.height.coerceAtLeast(0)
        topBarAnimator = ValueAnimator.ofInt(currentHeight, targetHeight).apply {
            duration = 220
            interpolator = DecelerateInterpolator()
            addUpdateListener { va ->
                val h = va.animatedValue as Int
                header.layoutParams.height = h
                header.alpha = h.toFloat() / targetHeight.toFloat()
                header.requestLayout()
            }
            start()
        }
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }
}
