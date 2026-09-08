package com.vericall

import com.vericall.util.VaniCallRouter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Covers the caller matching that decides whether an incoming call is a known,
 * voice-enrolled contact or an unknown number. This runs on the JVM with no
 * Android framework, so it only touches [VaniCallRouter.resolve] and
 * [VaniCallRouter.normalize], neither of which needs a Context.
 */
class VaniCallRouterTest {

    @Test
    fun exactStoredNumberResolvesToTheContact() {
        val caller = VaniCallRouter.resolve("+91 98765 43210")
        assertEquals("Aarav Sharma", caller.name)
        assertTrue(caller.known)
        assertTrue(caller.enrolled)
    }

    @Test
    fun numberWithoutCountryCodeStillMatches() {
        assertEquals("Aarav Sharma", VaniCallRouter.resolve("9876543210").name)
    }

    @Test
    fun differentFormattingAndLeadingZeroStillMatches() {
        // Priya is stored as "+91 91234 56780".
        assertEquals("Priya Nair", VaniCallRouter.resolve("09123456780").name)
        assertEquals("Priya Nair", VaniCallRouter.resolve("+919123456780").name)
    }

    @Test
    fun unenrolledContactIsResolvedButNotEnrolled() {
        val caller = VaniCallRouter.resolve("+91 80001 23456")
        assertEquals("Sara Iyer (HDFC Bank)", caller.name)
        assertTrue(caller.known)
        assertFalse(caller.enrolled)
    }

    @Test
    fun unknownNumberFallsBackToTheNumberItself() {
        val caller = VaniCallRouter.resolve("+91 11111 22222")
        assertFalse(caller.known)
        assertFalse(caller.enrolled)
        assertEquals("+91 11111 22222", caller.name)
    }

    @Test
    fun missingOrNonNumericNumberIsReportedAsUnknownCaller() {
        assertEquals("Unknown caller", VaniCallRouter.resolve(null).name)
        assertEquals("Unknown caller", VaniCallRouter.resolve("").name)
        assertEquals("Unknown caller", VaniCallRouter.resolve("   ").name)
        assertEquals("Unknown caller", VaniCallRouter.resolve("Unknown number").name)
    }

    @Test
    fun aSimilarButDistinctNumberDoesNotMatch() {
        // One digit away from Aarav's number.
        assertFalse(VaniCallRouter.resolve("+91 98765 43211").known)
    }

    @Test
    fun normalizeStripsEveryNonDigit() {
        assertEquals("919876543210", VaniCallRouter.normalize("+91 98765 43210"))
        assertEquals("", VaniCallRouter.normalize("no digits here"))
    }
}
