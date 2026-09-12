package com.vericall.shield

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.vericall.contacts.VaniForensicDetailActivity
import com.vericall.databinding.OverlayCoercionWarningBinding

/**
 * Fullscreen fallback coercion warning activity when SYSTEM_ALERT_WINDOW
 * permission is not enabled.
 */
class VaniCoercionActivity : AppCompatActivity() {

    private lateinit var binding: OverlayCoercionWarningBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = OverlayCoercionWarningBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Handle edge-to-edge system insets (mandatory for Android 15 & 16)
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val systemBars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        val pkg = intent.getStringExtra(EXTRA_TRIGGER_PACKAGE) ?: "Financial App"
        binding.tvTriggerApp.text = "Target: $pkg"

        binding.btnEndCallInstead.setOnClickListener {
            val detailIntent = Intent(this, VaniForensicDetailActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
            }
            startActivity(detailIntent)
            finish()
        }

        binding.btnDismiss.setOnClickListener {
            finish()
        }
    }

    companion object {
        const val EXTRA_TRIGGER_PACKAGE = "trigger_package"
    }
}
