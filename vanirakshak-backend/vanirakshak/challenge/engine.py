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

This module ships a deterministic phrase generator and the response
grader (digit order + latency). The text it grades is the server's own
transcript of the response audio (``challenge/transcribe.py``, docs/15 N2);
:func:`extract_digits` reads spoken digits in English, romanised Hindi and
Devanagari as well as numerals.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


HINDI_WORDS = ["aam", "nadi", "kela", "suraj", "chai", "tara", "kitaab", "ghar"]
# No number words here: the grader reads spoken digits, so a distractor
# like "seven" would be scored as a digit.
ENGLISH_WORDS = ["mango", "river", "blue", "paper", "table", "green", "orange", "cloud"]
DIGITS = list("0123456789")

# Spoken forms an ASR engine may return for each digit: English, romanised
# Hindi as Whisper actually spells it (docs/14 W2.c: "shuni", "punch",
# "chae", "saat", "no" for nau), and Devanagari, which is matched through
# _devanagari_key below rather than by exact spelling.
_DIGIT_WORDS = {
    "0": ["zero", "oh", "shunya", "shoonya", "sunya", "shuni", "shunni", "shouni", "shuny", "sooni"],
    "1": ["one", "ek", "ik", "ekk"],
    "2": ["two", "do"],
    "3": ["three", "teen", "tin", "deen", "dean", "theen"],
    "4": ["four", "char", "chaar"],
    "5": ["five", "paanch", "panch", "paach", "pach", "punch"],
    "6": ["six", "chhe", "chhah", "chah", "che", "chae", "cheh"],
    "7": ["seven", "saat", "sat", "satt", "saad", "saath"],
    "8": ["eight", "aath", "ath", "aat"],
    "9": ["nine", "nau", "nao", "no"],
}
_WORD_TO_DIGIT = {w: d for d, words in _DIGIT_WORDS.items() for w in words}
_DEVANAGARI_NUMERALS = str.maketrans("०१२३४५६७८९",
                                     "0123456789")
# \w alone drops Devanagari vowel signs (category Mn/Mc), splitting "पाँच".
_TOKEN = re.compile(r"[\wऀ-ॿ]+")

# Devanagari spelling varies a lot in ASR output for the same spoken word
# (शून्य / शूने / शुने for 0, पांच / पाच for 5, साथ for सात). Rather
# than list spellings, reduce a token to a consonant skeleton: drop vowel
# signs, chandrabindu, nukta, virama and visarga; read anusvara as n; fold
# independent vowels into three classes; merge aspirated and retroflex
# consonants into their plain dental pairs and the sibilants into one.
_DEVA_DROP = {chr(c) for c in range(0x093E, 0x094E)} | {"ँ", "़", "ः", "ॢ", "ॣ"}
_DEVA_MAP = {
    "ं": "न",                                     # anusvara -> n
    "अ": "अ", "आ": "अ",               # a, aa
    "इ": "ए", "ई": "ए", "ए": "ए", "ऐ": "ए",  # i, ii, e, ai
    "उ": "ओ", "ऊ": "ओ", "ओ": "ओ", "औ": "ओ",  # u, uu, o, au
    "ख": "क", "घ": "ग", "छ": "च", "झ": "ज",  # kh gh chh jh
    "ठ": "त", "ढ": "द", "ट": "त", "ड": "द",  # Th Dh T D
    "थ": "त", "ध": "द", "फ": "प", "भ": "ब",  # th dh ph bh
    "ण": "न", "श": "स", "ष": "स",                    # N, sh, Sh
}
_DEVA_KEYS = {
    "सनय": "0", "सन": "0",       # shunya
    "एक": "1",       # ek
    "द": "2",                                           # do
    "तन": "3",                                    # teen
    "चर": "4",                                    # char
    "पनच": "5", "पच": "5",       # paanch
    "चह": "6", "च": "6",                    # chhah, chhe
    "सत": "7",                                    # saat
    "अत": "8",                                    # aath
    "न": "9",                    # nau
}


def _devanagari_key(tok: str) -> str:
    key = "".join(_DEVA_MAP.get(c, c) for c in tok if c not in _DEVA_DROP)
    # Collapse doubled letters (gemination): शुन्ने and शुने are one word.
    return "".join(c for i, c in enumerate(key) if i == 0 or c != key[i - 1])


def extract_digits(text: str) -> List[str]:
    """Digits in *text*, in spoken order: numerals ("35", Devanagari numerals)
    and number words in English, romanised Hindi or Devanagari."""
    out: List[str] = []
    for tok in _TOKEN.findall((text or "").lower().translate(_DEVANAGARI_NUMERALS)):
        if tok.isdigit():
            out.extend(tok)
        elif tok in _WORD_TO_DIGIT:
            out.append(_WORD_TO_DIGIT[tok])
        elif any("ऀ" <= c <= "ॿ" for c in tok):
            d = _DEVA_KEYS.get(_devanagari_key(tok))
            if d is not None:
                out.append(d)
    return out


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
        expected_full = " ".join(tokens)
        return Challenge(
            phrase=phrase,
            expected_digits=digits,
            expected_full=expected_full,
            instructions="Now repeat ONLY the digits, in order, quickly.",
            issued_at_ms=int(time.time() * 1000),
            seed=self._seed,
        )

    def grade(
        self,
        challenge: Challenge,
        response_text: str,
        response_latency_ms: int,
        *,
        latency_min_ms: int = 400,
        latency_max_ms: int = 4000,
    ) -> Tuple[bool, str, float]:
        """Return (passed, reason, confidence).

        *passed* is True iff the spoken response looks like a human.
        *confidence* is a [0,1] belief that the answer is genuine.
        """
        # 1. digit-order match
        response_digits = extract_digits(response_text)
        correct_order = response_digits[: len(challenge.expected_digits)] == challenge.expected_digits

        # 2. latency check: humans typically 600-1500 ms; TTS <300 or >1500
        latency_ok = latency_min_ms <= response_latency_ms <= latency_max_ms

        # 3. crude TTS cue: TTS often answers *too fast* and *without filler
        filler = bool(re.search(r"\b(uh|um|hmm|the|then|ok|okay)\b", (response_text or "").lower()))

        passed = bool(correct_order and latency_ok)
        if not correct_order:
            return False, "digit_order_mismatch", 0.05
        if not latency_ok:
            return False, "latency_suspicious", 0.10
        confidence = 0.6 + (0.2 if filler else 0.0)
        return True, "ok", min(1.0, confidence)
