package com.vericall.ui

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import androidx.core.content.ContextCompat
import com.vericall.R
import com.vericall.model.RiskLevel
import kotlin.math.min

/**
 * Sesame-inspired calm, breathing multi-ring voice visualizer.
 * Reacts smoothly to live microphone amplitude (0.0 .. 1.0) and transitions
 * between authentic (emerald/cobalt), cautionary (amber), and danger (rose) states.
 */
class VoiceVisualizerView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val corePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val ring1Paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private val ring2Paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }

    private val ring3Paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f
    }

    private val auraPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private var targetAmplitude = 0f
    private var currentAmplitude = 0f
    private var idlePhase = 0f
    private var primaryColor = ContextCompat.getColor(context, R.color.brand_cobalt)
    private var isPulseActive = true

    private val idleAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 3200
        repeatCount = ValueAnimator.INFINITE
        interpolator = AccelerateDecelerateInterpolator()
        addUpdateListener {
            idlePhase = it.animatedValue as Float
            // Smoothly interpolate currentAmplitude towards targetAmplitude
            currentAmplitude += (targetAmplitude - currentAmplitude) * 0.25f
            invalidate()
        }
    }

    init {
        updateColors(RiskLevel.SAFE)
        idleAnimator.start()
    }

    fun setAmplitude(amp: Float) {
        targetAmplitude = amp.coerceIn(0f, 1f)
    }

    fun setRiskLevel(level: RiskLevel) {
        updateColors(level)
        invalidate()
    }

    private fun updateColors(level: RiskLevel) {
        primaryColor = when (level) {
            RiskLevel.SAFE -> ContextCompat.getColor(context, R.color.brand_cobalt)
            RiskLevel.WARNING -> ContextCompat.getColor(context, R.color.accent_amber)
            RiskLevel.DANGER -> ContextCompat.getColor(context, R.color.accent_rose)
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val cx = width / 2f
        val cy = height / 2f
        val maxRadius = min(cx, cy) * 0.85f

        // Base breathing factor (0.0 to 1.0 cyclic)
        val breath = kotlin.math.sin(idlePhase * Math.PI * 2.0).toFloat() * 0.5f + 0.5f
        val ampFactor = currentAmplitude * 0.45f + breath * 0.08f

        val coreRadius = maxRadius * (0.36f + ampFactor * 0.25f)

        // 1. Soft glowing outer aura
        auraPaint.color = primaryColor
        auraPaint.alpha = (18 + (ampFactor * 45)).toInt().coerceIn(0, 255)
        canvas.drawCircle(cx, cy, coreRadius * 1.55f, auraPaint)

        // 2. Third outer harmonic ripple ring
        ring3Paint.color = primaryColor
        ring3Paint.alpha = (25 + (breath * 20) + (currentAmplitude * 60)).toInt().coerceIn(0, 255)
        canvas.drawCircle(cx, cy, coreRadius * (1.38f + breath * 0.06f), ring3Paint)

        // 3. Second harmonic ring
        ring2Paint.color = primaryColor
        ring2Paint.alpha = (45 + (breath * 30) + (currentAmplitude * 80)).toInt().coerceIn(0, 255)
        canvas.drawCircle(cx, cy, coreRadius * (1.20f + (1f - breath) * 0.04f), ring2Paint)

        // 4. First tight resonant ring
        ring1Paint.color = primaryColor
        ring1Paint.alpha = (80 + (currentAmplitude * 120)).toInt().coerceIn(0, 255)
        canvas.drawCircle(cx, cy, coreRadius * 1.08f, ring1Paint)

        // 5. Solid calm inner core
        corePaint.color = primaryColor
        corePaint.alpha = 230
        canvas.drawCircle(cx, cy, coreRadius, corePaint)

        // 6. Subtle inner white highlight dot for depth
        corePaint.color = Color.WHITE
        corePaint.alpha = (50 + breath * 40).toInt()
        canvas.drawCircle(cx, cy, coreRadius * 0.22f, corePaint)
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        idleAnimator.cancel()
    }
}
