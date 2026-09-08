package com.vericall.shield

import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.LayoutInflater
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import com.vericall.R
import com.vericall.temp.TempContactDetailActivity

/**
 * Manages the high-contrast System Alert Window coercion warning when a banking
 * app is opened during an ongoing high-risk call. Falls back cleanly to
 * VaniCoercionActivity if overlay permission is absent.
 */
object VaniCoercionOverlay {

    private var overlayView: View? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    fun show(context: Context, triggeringPackage: String) {
        mainHandler.post {
            if (overlayView != null) return@post // Already visible

            VaniIncidentLogger.logOverlayTrigger(triggeringPackage)

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && Settings.canDrawOverlays(context)) {
                try {
                    val wm = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
                    val view = LayoutInflater.from(context).inflate(R.layout.overlay_coercion_warning, null)

                    view.findViewById<TextView>(R.id.tvTriggerApp).text = "Target: $triggeringPackage"

                    view.findViewById<Button>(R.id.btnDismiss).setOnClickListener {
                        hide(context)
                    }

                    view.findViewById<Button>(R.id.btnEndCallInstead).setOnClickListener {
                        hide(context)
                        val intent = Intent(context, TempContactDetailActivity::class.java).apply {
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                        }
                        context.startActivity(intent)
                    }

                    val layoutType = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                    } else {
                        @Suppress("DEPRECATION")
                        WindowManager.LayoutParams.TYPE_PHONE
                    }

                    val params = WindowManager.LayoutParams(
                        WindowManager.LayoutParams.MATCH_PARENT,
                        WindowManager.LayoutParams.MATCH_PARENT,
                        layoutType,
                        WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                        PixelFormat.TRANSLUCENT
                    )

                    wm.addView(view, params)
                    overlayView = view
                    return@post
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }

            // Fallback to standalone transparent activity if overlay window is unavailable
            val fallbackIntent = Intent(context, VaniCoercionActivity::class.java).apply {
                putExtra(VaniCoercionActivity.EXTRA_TRIGGER_PACKAGE, triggeringPackage)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            }
            context.startActivity(fallbackIntent)
        }
    }

    fun hide(context: Context) {
        mainHandler.post {
            overlayView?.let { view ->
                try {
                    val wm = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
                    wm.removeView(view)
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
            overlayView = null
        }
    }
}
