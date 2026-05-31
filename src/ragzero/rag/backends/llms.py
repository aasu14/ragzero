"""LLM backends.

- MockLLM: deterministic, used for tests.
- AnthropicLLM: Claude via the Anthropic SDK.
- OpenAILLM: GPT models via the OpenAI SDK.
- AzureOpenAILLM: GPT models via Azure OpenAI.
- GeminiLLM: Google Gemini via the google-generativeai SDK.
- OpenRouterLLM: Any model on OpenRouter (uses OpenAI-compatible API).
"""
from __future__ import annotations

import os
import re
from typing import Any

from ..interfaces import LLM


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CHUNK_HEADER = re.compile(r"^\[([A-Za-z0-9_:\-\.]+)\][^\n]*$", re.MULTILINE)

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "into",
    "and", "or", "but", "not", "no", "do", "does", "did", "doing", "done",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "have", "has", "had", "can", "could", "would", "should", "may", "might",
    "about", "tell", "me", "you", "your", "my", "our", "us", "we", "i",
    "explain", "describe", "give", "list",
})


def _content_terms(text: str, min_len: int = 3) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9]+", text)
        if len(t) >= min_len and t.lower() not in _STOPWORDS
    }


class MockLLM(LLM):
    """Extractive mock LLM for tests."""

    def generate(self, prompt, context, max_tokens=512, temperature=0.0):
        m = re.search(r"Question: (.+?)\nAnswer:", prompt, re.DOTALL)
        query = m.group(1).strip() if m else ""
        q_terms = _content_terms(query)
        chunks = self._parse_chunks(prompt)
        if not chunks or not q_terms:
            return "[INSUFFICIENT_EVIDENCE]"
        cited_sentences = []
        for chunk_id, text in chunks:
            for sent in _SENT_SPLIT.split(text):
                if q_terms & _content_terms(sent):
                    cited_sentences.append(f"{sent.strip()} [{chunk_id}]")
                    if len(cited_sentences) >= 3:
                        break
            if len(cited_sentences) >= 3:
                break
        if not cited_sentences:
            return "[INSUFFICIENT_EVIDENCE]"
        return " ".join(cited_sentences)

    @staticmethod
    def _parse_chunks(prompt):
        result = []
        current_id = None
        current_text = []
        for line in prompt.splitlines():
            m = _CHUNK_HEADER.match(line)
            if m:
                if current_id:
                    result.append((current_id, " ".join(current_text).strip()))
                current_id = m.group(1)
                current_text = []
            elif current_id is not None:
                if line.startswith("Question: "):
                    break
                current_text.append(line)
        if current_id:
            result.append((current_id, " ".join(current_text).strip()))
        return result


class AnthropicLLM(LLM):  # pragma: no cover
    """Production wrapper around the Anthropic SDK."""

    def __init__(self, model="claude-opus-4-7", api_key=None):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError("anthropic not installed. pip install anthropic") from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Anthropic API key not provided")
        self._client: Any = Anthropic(api_key=key)
        self.model = model

    def generate(self, prompt, context, max_tokens=512, temperature=0.0):
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in resp.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)


class OpenAILLM(LLM):  # pragma: no cover
    """Wrapper around the OpenAI Chat Completions API."""

    def __init__(self, model="gpt-4o-mini", api_key=None, base_url=None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai not installed. pip install openai") from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key not provided")
        kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client: Any = OpenAI(**kwargs)
        self.model = model

    def generate(self, prompt, context, max_tokens=512, temperature=0.0):
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class AzureOpenAILLM(LLM):  # pragma: no cover
    """Azure OpenAI Chat Completions. Uses deployment as the model identifier."""

    def __init__(self, api_key, endpoint, deployment, api_version="2024-02-01"):
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise ImportError("openai not installed. pip install openai") from e
        self._client: Any = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.deployment = deployment

    def generate(self, prompt, context, max_tokens=512, temperature=0.0):
        resp = self._client.chat.completions.create(
            model=self.deployment,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class GeminiLLM(LLM):  # pragma: no cover
    """Google Gemini via google-generativeai SDK.

    Any model name accepted — the SDK validates at call time, not at construction.
    The default suggestions are 'gemini-2.0-flash', 'gemini-2.0-pro', 'gemini-1.5-pro',
    but users can pass any model identifier the API accepts.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "google-generativeai not installed. pip install google-generativeai"
            ) from e
        genai.configure(api_key=api_key)
        self._genai: Any = genai
        self.model_name = model
        self._model: Any = genai.GenerativeModel(model)

    def generate(self, prompt, context, max_tokens=512, temperature=0.0):
        config = self._genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        resp = self._model.generate_content(prompt, generation_config=config)
        # The response shape varies; extract text robustly.
        try:
            return resp.text or ""
        except Exception:
            # Fallback: walk the candidates structure
            parts = []
            for cand in getattr(resp, "candidates", []) or []:
                for part in getattr(cand.content, "parts", []) or []:
                    if hasattr(part, "text"):
                        parts.append(part.text)
            return "".join(parts)


class OpenRouterLLM(LLM):  # pragma: no cover
    """OpenRouter — proxies hundreds of models through one OpenAI-compatible API.

    Users specify the full model identifier including provider prefix, e.g.
    'anthropic/claude-3.5-sonnet', 'meta-llama/llama-3.1-70b-instruct',
    'google/gemini-pro-1.5'. The model name is passed through as-is.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai not installed. pip install openai") from e
        # OpenRouter recommends sending HTTP-Referer and X-Title for analytics
        # and to be eligible for free-tier model access on some routes.
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name
        self._client: Any = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=default_headers or None,
        )
        self.model = model

    def generate(self, prompt, context, max_tokens=512, temperature=0.0):
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
