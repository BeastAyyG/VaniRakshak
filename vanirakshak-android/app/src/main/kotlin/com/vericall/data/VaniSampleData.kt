package com.vericall.data

import com.vericall.model.CallScenario
import com.vericall.model.VaniContact

object VaniSampleData {
    val contacts: MutableList<VaniContact> = mutableListOf(
        VaniContact(
            id = "c1",
            name = "Aarav Sharma",
            phone = "+91 98765 43210",
            relation = "Brother • Family Member",
            isVoiceEnrolled = true,
            enrollmentSec = 18,
            sampleEmbeddingCount = 4,
            defaultScenario = CallScenario.AI_CLONE_ATTACK,
            lastAlertText = "Flagged: Coercive urgent money transfer request",
            isFavorite = true
        ),
        VaniContact(
            id = "c2",
            name = "Priya Nair",
            phone = "+91 91234 56780",
            relation = "SIH Team Lead • Verified Voice",
            isVoiceEnrolled = true,
            enrollmentSec = 15,
            sampleEmbeddingCount = 3,
            defaultScenario = CallScenario.GENUINE_CALL,
            isFavorite = true
        ),
        VaniContact(
            id = "c3",
            name = "Rohan Mehta",
            phone = "+91 94567 89012",
            relation = "Father • Family Member",
            isVoiceEnrolled = true,
            enrollmentSec = 20,
            sampleEmbeddingCount = 5,
            defaultScenario = CallScenario.AI_CLONE_ATTACK,
            lastAlertText = "Flagged: Fake medical emergency scam simulation"
        ),
        VaniContact(
            id = "c4",
            name = "Sara Iyer (HDFC Bank)",
            phone = "+91 80001 23456",
            relation = "Claimed Bank Executive",
            isVoiceEnrolled = false,
            enrollmentSec = 0,
            sampleEmbeddingCount = 0,
            defaultScenario = CallScenario.SUSPICIOUS_ANOMALY,
            lastAlertText = "Unenrolled caller claiming executive authority"
        ),
        VaniContact(
            id = "c5",
            name = "Dev Kumar",
            phone = "+91 90000 11223",
            relation = "College Friend",
            isVoiceEnrolled = true,
            enrollmentSec = 14,
            sampleEmbeddingCount = 3,
            defaultScenario = CallScenario.GENUINE_CALL
        )
    )

    fun getContactById(id: String): VaniContact? = contacts.find { it.id == id }
}
