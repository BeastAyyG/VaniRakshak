package com.vericall

import com.vericall.audio.VaniStreamingClient
import com.vericall.model.RiskLevel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Verdict-frame contract per docs/PROTOCOL.md §5: the server owns tier +
 * interlock and the client renders them without re-deriving.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VaniStreamingClientTest {

    private val client = VaniStreamingClient("ws://127.0.0.1:8000/ws/stream")

    @Test
    fun sessionAckIsSkipped() {
        val ack = """{"type":"session_ack","session_id":"srv-123"}"""
        assertNull(client.parseVerdict(ack))
        assertEquals("srv-123", client.sessionId)
    }

    @Test
    fun blockVerdictIsAuthoritative() {
        val json = """
            {"session_id":"srv-1","risk_score":91,"tier":"BLOCK","p_spoof":0.97,
             "interlock_active":true,"receipt_hash":"abc123",
             "metrics":{"asv_evaluated":true,"asv_consistency":0.12}}
        """.trimIndent()
        val s = client.parseVerdict(json)
        assertNotNull(s)
        s!!
        assertEquals(91, s.risk)
        assertEquals("BLOCK", s.tier)
        assertEquals(RiskLevel.DANGER, s.level)
        assertTrue(s.interlockActive)
        assertTrue(s.serverDriven)
        assertEquals("abc123", s.receiptHash)
        assertEquals(97, s.synthetic)
        assertEquals(12, s.speakerMatch)
        assertEquals("srv-1", s.sessionId)
    }

    @Test
    fun challengeVerdictCarriesPhrase() {
        val json = """
            {"session_id":"srv-2","risk_score":55,"tier":"CHALLENGE","p_spoof":0.4,
             "interlock_active":false,
             "challenge":{"phrase":"Mango, 8, Nadi","instructions":"Repeat the digits"}}
        """.trimIndent()
        val s = client.parseVerdict(json)
        assertNotNull(s)
        s!!
        assertEquals("CHALLENGE", s.tier)
        assertEquals(RiskLevel.WARNING, s.level)
        assertEquals("Mango, 8, Nadi", s.challengePhrase)
        assertEquals("Repeat the digits", s.challengeInstructions)
        assertFalse(s.interlockActive)
        assertNull(s.receiptHash)
    }

    @Test
    fun missingTierFallsBackToRiskBands() {
        val high = client.parseVerdict("""{"risk_score":80,"p_spoof":0.1}""")!!
        assertEquals(RiskLevel.DANGER, high.level)
        assertEquals(RiskLevel.SAFE, client.parseVerdict("""{"risk_score":10}""")!!.level)
        assertEquals(RiskLevel.WARNING, client.parseVerdict("""{"risk_score":50}""")!!.level)
    }

    @Test
    fun malformedOrAckLikeFramesAreIgnored() {
        assertNull(client.parseVerdict("not json"))
        assertNull(client.parseVerdict("""{"hello":"world"}"""))
        assertNull(client.parseVerdict(""))
    }
}
