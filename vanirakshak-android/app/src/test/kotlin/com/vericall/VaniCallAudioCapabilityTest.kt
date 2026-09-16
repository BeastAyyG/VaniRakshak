package com.vericall

import com.vericall.audio.AudioSourceType
import com.vericall.audio.VaniCallAudioCapability
import com.vericall.audio.VaniCallAudioSource
import com.vericall.audio.VaniMicrophoneAudioSource
import com.vericall.audio.VaniTelephonyAudioSource
import com.vericall.audio.VaniUnavailableAudioSource
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class VaniCallAudioCapabilityTest {

    @Test
    fun telephonyAudioSourceDeclaresCorrectSourceType() {
        val telephonySource = VaniTelephonyAudioSource()
        assertEquals(AudioSourceType.TELEPHONY_CALL_AUDIO, telephonySource.sourceType)
        assertTrue(telephonySource.sourceType.isTwoWay)
    }

    @Test
    fun microphoneAudioSourceDeclaresCorrectSourceType() {
        val micSource = VaniMicrophoneAudioSource()
        assertEquals(AudioSourceType.MICROPHONE_UPLINK, micSource.sourceType)
        assertFalse(micSource.sourceType.isTwoWay)
    }

    @Test
    fun unavailableAudioSourceNeverAvailable() {
        assertFalse(VaniUnavailableAudioSource.isAvailable())
        assertEquals(AudioSourceType.UNAVAILABLE, VaniUnavailableAudioSource.sourceType)
    }

    @Test
    fun capabilitiesDataClassPreservesProperties() {
        val caps = VaniCallAudioCapability.CallAudioCapabilities(
            oem = "OnePlus",
            model = "CPH2707",
            sdkInt = 36,
            isDefaultDialer = true,
            hasRecordAudioPermission = true,
            hasCaptureAudioOutputPermission = false,
            telephonyCaptureAvailable = false,
            uplinkAvailable = false,
            downlinkAvailable = false,
            twoWayCaptureAvailable = false,
            voiceRecognitionAvailable = true,
            voiceCommunicationAvailable = true,
            standardMicAvailable = true,
            systemCallRecorderDetected = true,
            systemCallRecorderPackage = "com.oplus.callrecorder",
            recommendedSource = "VOICE_RECOGNITION",
            diagnosticSummary = "Summary test"
        )

        assertEquals("OnePlus", caps.oem)
        assertEquals(36, caps.sdkInt)
        assertTrue(caps.isDefaultDialer)
        assertFalse(caps.hasCaptureAudioOutputPermission)
        assertFalse(caps.twoWayCaptureAvailable)
        assertEquals("VOICE_RECOGNITION", caps.recommendedSource)
        assertEquals("com.oplus.callrecorder", caps.systemCallRecorderPackage)
    }
}
