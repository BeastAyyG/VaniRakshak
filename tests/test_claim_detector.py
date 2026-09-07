"""
Test 5 — Non-factual sentence filtering.

Input: "Hello sir, how are you?"
Expected: no verification query — sentence is filtered out.
"""

from backend.services.claim_detector import detect_claims


class TestNonFactualFiltering:
    """Verify that greetings, questions, and acks are NOT treated as claims."""

    def test_greeting_filtered(self):
        sentences = [{"start": 0.0, "end": 2.0, "text": "Hello sir, how are you?"}]
        claims = detect_claims(sentences)
        assert len(claims) == 0, f"Greeting should not be a claim, got: {claims}"

    def test_acknowledgment_filtered(self):
        sentences = [{"start": 0.0, "end": 1.0, "text": "Okay."}]
        claims = detect_claims(sentences)
        assert len(claims) == 0, f"Ack should not be a claim, got: {claims}"

    def test_yes_filtered(self):
        sentences = [{"start": 0.0, "end": 1.0, "text": "Yes."}]
        claims = detect_claims(sentences)
        assert len(claims) == 0

    def test_thanks_filtered(self):
        sentences = [{"start": 0.0, "end": 2.0, "text": "Thank you very much."}]
        claims = detect_claims(sentences)
        assert len(claims) == 0

    def test_question_filtered(self):
        sentences = [{"start": 0.0, "end": 3.0, "text": "Can you hear me properly?"}]
        claims = detect_claims(sentences)
        assert len(claims) == 0

    def test_bye_filtered(self):
        sentences = [{"start": 0.0, "end": 2.0, "text": "Goodbye, take care."}]
        claims = detect_claims(sentences)
        assert len(claims) == 0

    def test_factual_claim_detected(self):
        """Positive case: a factual statement SHOULD be detected."""
        sentences = [
            {"start": 4.0, "end": 8.0, "text": "It is raining heavily in Delhi right now."},
        ]
        claims = detect_claims(sentences)
        assert len(claims) >= 1, "Weather claim should be detected"

    def test_mixed_input(self):
        """Mix of factual and non-factual — only factual should survive."""
        sentences = [
            {"start": 0.0, "end": 2.0, "text": "Hello."},
            {"start": 2.0, "end": 4.0, "text": "How are you?"},
            {"start": 4.0, "end": 8.0, "text": "It is raining heavily in Delhi right now."},
            {"start": 8.0, "end": 9.0, "text": "Okay."},
        ]
        claims = detect_claims(sentences)
        assert len(claims) == 1
        assert "Delhi" in claims[0].text or "raining" in claims[0].text
