"""Multilingual support.

A Translator detects the language of a query and translates between
languages. Default implementation uses the configured LLM; a Google
Translate / DeepL adapter can be added without changing strategy code.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..interfaces import LLM


# Common language codes the UI will use
LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese (Simplified)",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ru": "Russian",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
}


@dataclass(frozen=True)
class TranslationResult:
    text: str
    detected_language: str  # ISO 639-1 code
    target_language: str
    used_translator: bool   # False if source == target (no-op)


class Translator(ABC):
    @abstractmethod
    def detect_language(self, text: str) -> str:
        """Return ISO 639-1 code (e.g., 'en', 'hi'). 'unknown' if not detected."""

    @abstractmethod
    def translate(self, text: str, target_lang: str, source_lang: str = "auto") -> TranslationResult: ...


_DETECT_PROMPT = """\
Detect the language of this text. Return ONLY the ISO 639-1 code (e.g., 'en', 'hi', 'es', 'fr', 'de', 'ja', 'zh', 'ar').
If unsure, return 'en'.

Text: \"\"\"{text}\"\"\"

Code:"""


_TRANSLATE_PROMPT = """\
Translate the text below from {source_name} to {target_name}.
Preserve all named entities, numbers, and any inline citation markers like [chunk_id_here].
Return ONLY the translated text, no quotes, no explanation.

Text:
{text}

Translation:"""


# Heuristic Unicode-script detection for common Indian/Asian/RTL scripts.
# Used to short-circuit the LLM call when the language is obvious from the script.
_SCRIPT_RANGES = [
    ("hi", 0x0900, 0x097F),  # Devanagari (Hindi, Marathi)
    ("bn", 0x0980, 0x09FF),  # Bengali
    ("gu", 0x0A80, 0x0AFF),  # Gujarati
    ("ta", 0x0B80, 0x0BFF),  # Tamil
    ("te", 0x0C00, 0x0C7F),  # Telugu
    ("ar", 0x0600, 0x06FF),  # Arabic
    ("zh", 0x4E00, 0x9FFF),  # CJK Unified
    ("ja", 0x3040, 0x309F),  # Hiragana
    ("ru", 0x0400, 0x04FF),  # Cyrillic
]


def _heuristic_language(text: str) -> str | None:
    """Try to detect language from Unicode script. Returns None if no signal."""
    counts: dict[str, int] = {}
    for ch in text[:500]:  # cap for speed
        cp = ord(ch)
        for code, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return None
    # Need at least 3 chars in the script to call it confidently
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0] if best[1] >= 3 else None


class LLMTranslator(Translator):
    """Uses the configured LLM for detection and translation.

    For Devanagari/CJK/etc. text, the script-detection heuristic short-circuits
    the LLM call entirely — saves a round trip and is more reliable than LLM
    self-reporting for short queries.
    """

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def detect_language(self, text: str) -> str:
        # 1. Script heuristic
        h = _heuristic_language(text)
        if h:
            return h
        # 2. If all ASCII, assume English unless very long
        if all(ord(c) < 128 for c in text[:200]):
            return "en"
        # 3. Ask the LLM
        try:
            raw = self.llm.generate(
                prompt=_DETECT_PROMPT.format(text=text[:500]),
                context=[],
                max_tokens=8,
                temperature=0.0,
            )
            code = re.search(r"[a-z]{2}", raw.lower())
            if code:
                return code.group(0)
        except Exception:
            pass
        return "unknown"

    def translate(
        self, text: str, target_lang: str, source_lang: str = "auto"
    ) -> TranslationResult:
        detected = source_lang if source_lang != "auto" else self.detect_language(text)
        if detected == target_lang or target_lang == "auto":
            return TranslationResult(
                text=text, detected_language=detected, target_language=detected, used_translator=False
            )
        source_name = LANGUAGES.get(detected, detected)
        target_name = LANGUAGES.get(target_lang, target_lang)
        try:
            translated = self.llm.generate(
                prompt=_TRANSLATE_PROMPT.format(
                    source_name=source_name, target_name=target_name, text=text
                ),
                context=[],
                max_tokens=min(2048, len(text) * 3),
                temperature=0.0,
            )
            return TranslationResult(
                text=translated.strip(),
                detected_language=detected,
                target_language=target_lang,
                used_translator=True,
            )
        except Exception:
            # If translation fails, fall back to original
            return TranslationResult(
                text=text, detected_language=detected, target_language=target_lang, used_translator=False
            )
