"""Tests for multilingual translator."""
from ragzero.rag.multilingual import LLMTranslator
from ragzero.rag.multilingual.translator import _heuristic_language


def test_heuristic_detects_devanagari():
    # "नमस्ते" = Hindi "namaste"
    assert _heuristic_language("नमस्ते दुनिया") == "hi"


def test_heuristic_detects_arabic():
    assert _heuristic_language("مرحبا بالعالم") == "ar"


def test_heuristic_detects_chinese():
    assert _heuristic_language("你好世界") == "zh"


def test_heuristic_returns_none_for_ascii():
    assert _heuristic_language("Hello world") is None


def test_heuristic_needs_minimum_chars():
    # Single non-ASCII char isn't enough
    assert _heuristic_language("a नm") is None


class StubLLM:
    """LLM stub that records calls and returns programmed responses."""
    def __init__(self, response="en"):
        self.response = response
        self.calls = []
    def generate(self, prompt, context, max_tokens=512, temperature=0.0):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return self.response


def test_translator_short_circuits_on_script_match():
    """For Devanagari text, no LLM call should be needed for detection."""
    llm = StubLLM()
    t = LLMTranslator(llm)
    detected = t.detect_language("नमस्ते दुनिया")
    assert detected == "hi"
    assert len(llm.calls) == 0  # heuristic path


def test_translator_no_op_when_source_equals_target():
    llm = StubLLM()
    t = LLMTranslator(llm)
    result = t.translate("Hello world", target_lang="en", source_lang="en")
    assert not result.used_translator
    assert result.text == "Hello world"
    assert len(llm.calls) == 0


def test_translator_calls_llm_for_cross_language():
    llm = StubLLM(response="Hola mundo")
    t = LLMTranslator(llm)
    result = t.translate("Hello world", target_lang="es", source_lang="en")
    assert result.used_translator
    assert result.text == "Hola mundo"
    assert result.detected_language == "en"
    assert result.target_language == "es"
    assert len(llm.calls) == 1


def test_translator_handles_llm_failure_gracefully():
    class FailingLLM:
        def generate(self, **kwargs):
            raise RuntimeError("API down")
    t = LLMTranslator(FailingLLM())
    result = t.translate("Hello", target_lang="fr", source_lang="en")
    # Should return original text without crashing
    assert result.text == "Hello"
    assert not result.used_translator


def test_translator_detect_ascii_defaults_to_english():
    llm = StubLLM()
    t = LLMTranslator(llm)
    assert t.detect_language("Hello world this is plain English") == "en"
    # Should have short-circuited without calling LLM
    assert len(llm.calls) == 0
