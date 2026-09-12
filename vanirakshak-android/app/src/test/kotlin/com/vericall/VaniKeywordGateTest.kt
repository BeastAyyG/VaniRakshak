package com.vericall

import com.vericall.audio.VaniKeywordTrigger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Money-talk matcher: fires on banking/transaction words (EN + Hindi),
 * ignores lookalikes via word boundaries ("shopping" must not trip "pin").
 */
class VaniKeywordGateTest {

    private fun find(text: String) = VaniKeywordTrigger.findKeyword(text)

    @Test
    fun moneyWordsTrigger() {
        assertEquals("bank", find("please go to the bank tomorrow"))
        assertEquals("otp", find("tell me your OTP now"))
        assertEquals("upi", find("send it on upi"))
        assertEquals("payment", find("the Payment is pending"))
        assertEquals("account", find("your account is blocked"))
        assertEquals("transfer", find("transfer the amount fast"))
    }

    @Test
    fun hindiWordsTrigger() {
        assertEquals("पैसा", find("मुझे पैसा भेजो"))
        assertEquals("बैंक", find("बैंक खाता बंद हो जाएगा"))
    }

    @Test
    fun shortTokensNeedWordBoundaries() {
        assertNull(find("I love shopping online"))
        assertNull(find("the spinning wheel is nice"))
        assertEquals("pin", find("share your card pin"))
    }

    @Test
    fun innocentChatDoesNotTrigger() {
        assertNull(find("hello beta, did you eat lunch"))
        assertNull(find("the weather is nice today"))
    }

    @Test
    fun keywordSetIsNotEmpty() {
        assert(VaniKeywordTrigger.DEFAULT_KEYWORDS.size >= 20)
    }
}
