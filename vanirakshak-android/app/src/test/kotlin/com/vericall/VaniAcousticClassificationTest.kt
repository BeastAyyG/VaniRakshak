package com.vericall

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

class VaniAcousticClassificationTest {

    @Test
    fun rmsOfPureSilenceIsNearZero() {
        val silentBuffer = ShortArray(1600) { 0 }
        var sumSquare = 0.0
        for (sample in silentBuffer) {
            sumSquare += (sample * sample)
        }
        val rms = sqrt(sumSquare / silentBuffer.size)
        assertEquals(0.0, rms, 0.0001)
    }

    @Test
    fun rmsOfSineWaveMatchesExpectedAmplitude() {
        val amp = 10000.0
        val buffer = ShortArray(1600) { i ->
            (amp * sin(2.0 * Math.PI * 440.0 * i / 16000.0)).toInt().toShort()
        }
        var sumSquare = 0.0
        for (sample in buffer) {
            sumSquare += (sample.toDouble() * sample.toDouble())
        }
        val rms = sqrt(sumSquare / buffer.size)
        // Theoretical RMS of sine wave = Amp / sqrt(2) ≈ 7071
        assertEquals(7071.0, rms, 200.0)
    }

    @Test
    fun zeroCrossingRateOf440HzAt16kHzMatchesTheory() {
        val buffer = ShortArray(1600) { i ->
            (5000.0 * sin(2.0 * Math.PI * 440.0 * i / 16000.0)).toInt().toShort()
        }
        var zeroCrossings = 0
        var prevSample = 0
        for (sample in buffer) {
            val s = sample.toInt()
            if ((s >= 0 && prevSample < 0) || (s < 0 && prevSample >= 0)) {
                zeroCrossings++
            }
            prevSample = s
        }
        val zcrRate = zeroCrossings.toDouble() / buffer.size
        // 440 Hz wave has 880 zero crossings per second -> at 16000 Hz, rate ≈ 880/16000 = 0.055
        assertEquals(0.055, zcrRate, 0.008)
        // 0.055 is well within normal biological human pitch range (0.035 - 0.35)
        assertTrue(zcrRate in 0.035..0.35)
    }

    @Test
    fun adaptiveNoiseFloorTracksBackgroundSmoothly() {
        var adaptiveNoiseFloor = 200.0

        // Quiet background noise at 150 RMS
        for (i in 0 until 50) {
            val bgRms = 150.0
            if (bgRms in 30.0..500.0) {
                adaptiveNoiseFloor = 0.96 * adaptiveNoiseFloor + 0.04 * bgRms
            }
        }
        // Should have converged near 150
        assertEquals(150.0, adaptiveNoiseFloor, 10.0)

        val vadThreshold = max(380.0, adaptiveNoiseFloor * 1.6)
        // Normal speech at RMS 1200 should trigger VAD
        val speechRms = 1200.0
        assertTrue("Speech RMS must exceed adaptive VAD threshold", speechRms > vadThreshold)

        // Background noise at 200 RMS should not trigger VAD
        assertFalse("Background noise must not trigger VAD", 200.0 > vadThreshold)
    }
}
