"""Dynamic challenge engine (R3 — active anti-replay defence).

The challenge phrase is generated AFTER the call starts, so a
pre-recorded replay of a genuine voice cannot answer it. The phrase
mixes digits with phonetically diverse Hindi + English words and
ends with an instruction to repeat only the numbers, which:

  * a flustered human can do in 5-15 seconds,
  * a replay of the original phrase cannot (the answer was not known
    when the recording was made),
  * a live TTS can do, but the response latency is usually wrong
    (TTS systems are <300 ms or >1500 ms, humans are 600-1200 ms).

This module ships a deterministic phrase generator and a *stub*
response analyzer (string match + latency check). For the hackathon
demo we use string match against the expected answer; for production
you'd wire this to AI4Bharat IndicConformer or Whisper.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


HINDI_WORDS = ["aam", "nadi", "kela", "suraj", "chai", "tara", "kitaab", "ghar"]
ENGLISH_WORDS = ["mango", "river", "blue", "seven", "table", "green", "orange", "cloud"]
DIGITS = list("0123456789")


@dataclass
class Challenge:
    phrase: str
    expected_digits: List[str]
    expected_full: str
    instructions: str
    issued_at_ms: int
    seed: int

    def to_dict(self) -> dict:
        return {
            "phrase": self.phrase,
            "expected_digits": self.expected_digits,
            "expected_full": self.expected_full,
            "instructions": self.instructions,
            "issued_at_ms": self.issued_at_ms,
            "seed": self.seed,
        }


class ChallengeEngine:
    """Phrase generator + response analyzer stub."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed

    def issue(self, *, n_digits: int = 2, n_words: int = 2, lang: str = "mixed") -> Challenge:
        rng = random.Random(self._seed)
        self._seed = (self._seed or 0) + 1
        digits = [rng.choice(DIGITS) for _ in range(n_digits)]
        if lang == "hi":
            words = [rng.choice(HINDI_WORDS) for _ in range(n_words)]
        elif lang == "en":
            words = [rng.choice(ENGLISH_WORDS) for _ in range(n_words)]
        else:
            pool = HINDI_WORDS + ENGLISH_WORDS
            words = [rng.choice(pool) for _ in range(n_words)]

        # interleave "word, digit, word, digit, ..."
        tokens: List[str] = []
        for i, w in enumerate(words):
            tokens.append(w)
            if i < len(digits):
                tokens.append(digits[i])
        phrase = ", ".join(tokens) + "."
        expected_full = " ".join(digits)
        return Challenge(
            phrase=phrase,
            expected_digits=digits,
            expected_full=expected_full,
            instructions="Now repeat ONLY the digits, in order, quickly.",
            issued_at_ms=int(time.time() * 1000),
            seed=self._seed,
        )

    def grade(self, challenge: Challenge, response_text: str, response_latency_ms: int) -> Tuple[bool, str, float]:
        """Return (passed, reason, confidence).

        *passed* is True iff the spoken response looks like a human.
        *confidence* is a [0,1] belief that the answer is genuine.
        """
        # 1. digit-order match
        response_digits = re.findall(r"\d", response_text or "")
        correct_order = response_digits[: len(challenge.expected_digits)] == challenge.expected_digits

        # 2. latency check: humans typically 600-1500 ms; TTS <300 or >1500
        latency_ok = 400 <= response_latency_ms <= 4000

        # 3. crude TTS cue: TTS often answers *too fast* and *without filler
        filler = bool(re.search(r"\b(uh|um|hmm|the|then|ok|okay)\b", (response_text or "").lower()))

        passed = bool(correct_order and latency_ok)
        if not correct_order:
            return False, "digit_order_mismatch", 0.05
        if not latency_ok:
            return False, "latency_suspicious", 0.10
        confidence = 0.6 + (0.2 if filler else 0.0)
        return True, "ok", min(1.0, confidence)
