package com.vericall

import com.vericall.data.VaniCallHistoryRepository
import com.vericall.model.CallDirection
import com.vericall.model.TranscriptUtterance
import com.vericall.model.VaniCallRecord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VaniCallHistoryRepositoryTest {

    @Test
    fun testDeriveIntelligence_NormalCall() {
        val transcript = listOf(
            TranscriptUtterance("Rahul", "Hey, can we review the meeting schedule?"),
            TranscriptUtterance("You", "Sure, let's meet at 3 PM today.")
        )
        val (summary, keyPoints, actionItems) = VaniCallHistoryRepository.deriveIntelligence(
            callerName = "Rahul Sharma",
            transcript = transcript,
            riskScore = 12,
            scamType = ""
        )

        assertTrue(summary.contains("Rahul Sharma"))
        assertTrue(summary.contains("schedule") || summary.contains("coordinating"))
        assertEquals(3, keyPoints.size)
        assertTrue(keyPoints[0].contains("Voice biometric verified"))
        assertEquals(1, actionItems.size)
    }

    @Test
    fun testDeriveIntelligence_ScamThreatCall() {
        val transcript = listOf(
            TranscriptUtterance("Unknown", "Your account is frozen. Immediate transfer of 25000 required."),
            TranscriptUtterance("You", "Why is it frozen?")
        )
        val (summary, keyPoints, actionItems) = VaniCallHistoryRepository.deriveIntelligence(
            callerName = "Unknown Caller",
            transcript = transcript,
            riskScore = 88,
            scamType = "Digital Arrest Fraud"
        )

        assertTrue(summary.contains("High-risk call detected"))
        assertTrue(summary.contains("Digital Arrest Fraud"))
        assertTrue(keyPoints.any { it.contains("interlock held") })
        assertTrue(keyPoints.any { it.contains("Coercion pressure") })
        assertTrue(actionItems.any { it.contains("Do not transfer funds") })
    }

    @Test
    fun testVaniCallRecordCreation() {
        val record = VaniCallRecord(
            id = "call_test_123",
            number = "+91 98765 43210",
            callerName = "Rahul Sharma",
            direction = CallDirection.INCOMING,
            timestamp = System.currentTimeMillis(),
            durationSec = 272,
            hasTranscript = true,
            transcript = listOf(
                TranscriptUtterance("Rahul", "Hello!"),
                TranscriptUtterance("You", "Hi Rahul!")
            ),
            aiSummary = "Test conversation",
            keyPoints = listOf("Point 1", "Point 2"),
            actionItems = listOf("Action 1"),
            riskScore = 5,
            isVoiceVerified = true
        )

        assertEquals("call_test_123", record.id)
        assertEquals(CallDirection.INCOMING, record.direction)
        assertEquals(272, record.durationSec)
        assertTrue(record.hasTranscript)
        assertEquals(2, record.transcript.size)
        assertTrue(record.isVoiceVerified)
    }
}
