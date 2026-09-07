"""
Hinglish / Romanization Service for VaniRakshak ASR

Converts canonical native Devanagari Hindi transcripts into clean, natural Romanized
Hinglish while strictly PRESERVING English words, banking terminology, and fraud keywords:
- Preserves: KYC, OTP, UPI, PIN, SBI, RBI, bank, account, suspend, expire, headquarters, etc.
- Translates common fraud loan/bank vocabulary to English terms (e.g. बैंक -> bank, खाता/अकाउंट -> account).
- Applies standard conversational Hindi phonetics with schwa-elision.
- Completely deterministic, lightweight, and executes in < 1 millisecond.
"""

import re
from typing import Dict, List, Optional, Tuple

PUNCTUATION_CHARS = ".,!?;:'\"()[]{}/\\-_@#$%*+=<>।“”‘’"

FRAUD_AND_BANKING_LEXICON: Dict[str, str] = {
    # Acronyms & Institutions
    "केवाईसी": "KYC",
    "के-वाई-सी": "KYC",
    "ओटीपी": "OTP",
    "ओ-टी-पी": "OTP",
    "यूपीआई": "UPI",
    "यू-पी-आई": "UPI",
    "पिन": "PIN",
    "एसबीआई": "SBI",
    "आरबीआई": "RBI",
    "पीएनबी": "PNB",
    "एचडीएफसी": "HDFC",
    "आईसीआईसीआई": "ICICI",
    "सीबीआई": "CBI",
    "ईडी": "ED",
    "ट्राई": "TRAI",
    
    # Financial & Account terms
    "बैंक": "bank",
    "अकाउंट": "account",
    "खाता": "account",
    "खाते": "account",
    "बैलेंस": "balance",
    "ट्रांजैक्शन": "transaction",
    "ट्रांजेक्शन": "transaction",
    "क्रेडिट": "credit",
    "डेबिट": "debit",
    "कार्ड": "card",
    "रिफंड": "refund",
    "कैशबैक": "cashback",
    "लिंक": "link",
    "वेरिफिकेशन": "verification",
    "वेरिफाई": "verify",
    "अपडेट": "update",
    "ब्लॉक": "block",
    "ब्लॉक्ड": "blocked",
    "सस्पेंड": "suspend",
    "सस्पेंडेड": "suspended",
    "एक्सपायर": "expire",
    "एक्सपायर्ड": "expired",
    "इनएक्टिव": "inactive",
    "एक्टिवेट": "activate",
    "पासवर्ड": "password",
    
    # Authority & Fraud phrasing
    "हेडक्वार्टर": "headquarters",
    "हेडक्वार्टर्स": "headquarters",
    "कस्टमर": "customer",
    "केयर": "care",
    "ऑफिसर": "officer",
    "मैनेजर": "manager",
    "डिपार्टमेंट": "department",
    "हेल्पलाइन": "helpline",
    "सपोर्ट": "support",
    "सर": "Sir",
    "मैडम": "Madam",
    "कॉल": "call",
    "मैसेज": "message",
    "पुलिस": "police",
    "अरेस्ट": "arrest",
    "डिजिटल": "digital",
    "वारंट": "warrant",
    "बिल": "bill",
    "इलेक्ट्रिसिटी": "electricity",
    "बिजली": "electricity",
    "डिस्कनेक्ट": "disconnect",
    "डिस्कनेक्शन": "disconnection",
    "सर्विस": "service",
    "सिम": "SIM",
    "नंबर": "number",
}

COLLOQUIAL_OVERRIDES: Dict[str, str] = {
    "आपका": "aapka",
    "आपकी": "aapki",
    "आपके": "aapke",
    "आप": "aap",
    "कल": "kal",
    "आज": "aaj",
    "बंद": "band",
    "हो": "ho",
    "जाएगा": "jayega",
    "जाएगी": "jayegi",
    "जाएंगे": "jayenge",
    "गया": "gaya",
    "गई": "gayi",
    "गए": "gaye",
    "है": "hai",
    "हैं": "hain",
    "हूँ": "hoon",
    "हूं": "hoon",
    "था": "tha",
    "थी": "thi",
    "थे": "the",
    "बोल": "bol",
    "रहा": "raha",
    "रही": "rahi",
    "रहे": "rahe",
    "से": "se",
    "मैं": "main",
    "हम": "hum",
    "यह": "yeh",
    "वह": "woh",
    "इस": "is",
    "उस": "us",
    "का": "ka",
    "की": "ki",
    "के": "ke",
    "को": "ko",
    "पर": "par",
    "में": "mein",
    "ने": "ne",
    "तुरंत": "turant",
    "जल्दी": "jaldi",
    "अभी": "abhi",
    "नहीं": "nahi",
    "मत": "mat",
    "करो": "karo",
    "करें": "karein",
    "कर": "kar",
    "कीजिए": "kijiye",
    "कीजिये": "kijiye",
    "दीजिए": "dijiye",
    "दीजिये": "dijiye",
    "बताओ": "batao",
    "बताइए": "bataiye",
    "भेजें": "bhejein",
    "भेजो": "bhejo",
    "साहब": "sahab",
    "नमस्ते": "namaste",
    "धन्यवाद": "dhanyavaad",
    "नमस्कार": "namaskar",
    "क्या": "kya",
    "क्यों": "kyun",
    "कैसे": "kaise",
    "कहाँ": "kahan",
    "यहाँ": "yahan",
    "वहाँ": "wahan",
    "जब": "jab",
    "तब": "tab",
    "अगर": "agar",
    "लेकिन": "lekin",
    "और": "aur",
    "या": "ya",
    "भी": "bhi",
    "तो": "to",
}

# Devanagari character tables for fallback phonetic generation
VOWEL_MAP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}

MATRA_MAP = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "oo",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}

CONSONANT_MAP = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
    "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gya"
}


class HinglishTransliterationService:
    """
    Transforms native Devanagari Hindi text to natural Romanized Hinglish.
    Ensures that technical scam keywords (KYC, OTP, bank account, suspend, etc.)
    remain completely preserved in English.
    """

    def __init__(self):
        self._banking_lexicon = FRAUD_AND_BANKING_LEXICON
        self._colloquial_overrides = COLLOQUIAL_OVERRIDES

    def to_hinglish(self, text: str) -> str:
        """
        Main conversion method:
        Native Hindi -> Romanized Hinglish
        """
        if not text or not text.strip():
            return ""

        # Normalize Hindi danda
        normalized = text.replace("।", " . ")

        # Tokenize by whitespace
        raw_tokens = normalized.split()
        converted_tokens: List[str] = []

        for token in raw_tokens:
            prefix, core, suffix = self._split_punctuation(token)
            
            if not core:
                converted_tokens.append(token)
                continue

            # Check if core is already English / ASCII
            if re.match(r"^[A-Za-z0-9_\-]+$", core):
                converted_tokens.append(f"{prefix}{core}{suffix}")
                continue

            # Check direct fraud/banking lexicon
            if core in self._banking_lexicon:
                val = self._banking_lexicon[core]
                converted_tokens.append(f"{prefix}{val}{suffix}")
                continue

            # Check colloquial overrides
            if core in self._colloquial_overrides:
                val = self._colloquial_overrides[core]
                converted_tokens.append(f"{prefix}{val}{suffix}")
                continue

            # Fallback to phonetic romanization
            romanized = self._phonetic_devanagari(core)
            converted_tokens.append(f"{prefix}{romanized}{suffix}")

        hinglish = " ".join(converted_tokens)
        # Fix spacing around punctuation
        hinglish = re.sub(r"\s+([.,!?;:])", r"\1", hinglish)
        # Normalize double spaces
        hinglish = re.sub(r"\s+", " ", hinglish).strip()
        return hinglish

    def _split_punctuation(self, token: str) -> Tuple[str, str, str]:
        """Splits leading/trailing punctuation from a word token using character membership."""
        i = 0
        while i < len(token) and token[i] in PUNCTUATION_CHARS:
            i += 1
        prefix = token[:i]

        j = len(token)
        while j > i and token[j - 1] in PUNCTUATION_CHARS:
            j -= 1
        suffix = token[j:]
        core = token[i:j]
        return prefix, core, suffix

    def _phonetic_devanagari(self, word: str) -> str:
        """Character-level phonetic transliteration with schwa-elision."""
        chars = list(word)
        n = len(chars)
        out = []

        i = 0
        while i < n:
            ch = chars[i]

            if ch in VOWEL_MAP:
                out.append(VOWEL_MAP[ch])
                i += 1
                continue

            if ch in MATRA_MAP:
                out.append(MATRA_MAP[ch])
                i += 1
                continue

            if ch in ("ं", "ँ"):
                out.append("n")
                i += 1
                continue

            if ch == "ः":
                out.append("h")
                i += 1
                continue

            if ch == "्":
                i += 1
                continue

            if ch in CONSONANT_MAP:
                base = CONSONANT_MAP[ch]
                next_ch = chars[i + 1] if i + 1 < n else None

                if next_ch == "्":
                    out.append(base)
                    i += 2
                    continue
                elif next_ch in MATRA_MAP:
                    out.append(base)
                    i += 1
                    continue
                elif next_ch in ("ं", "ँ", "ः"):
                    out.append(base + "a")
                    i += 1
                    continue
                elif next_ch is None:
                    out.append(base)
                    i += 1
                    continue
                else:
                    out.append(base + "a")
                    i += 1
                    continue

            out.append(ch)
            i += 1

        return "".join(out)
