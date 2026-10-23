"""LLM backends behind one tiny interface: ``complete(system, user) -> str``.

* ``openai``     - any OpenAI-compatible chat endpoint. Covers OpenAI, Groq,
                   Together, Ollama (local) and Gemini's OpenAI-compatible API.
* ``anthropic``  - Anthropic Messages API.
* ``extractive`` - no model at all. Used offline, in tests, and as the
                   "is the LLM actually adding anything?" baseline in eval.
"""
from __future__ import annotations

import time

import requests

DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-5-5"}
DEFAULT_BASE_URLS = {"openai": "https://api.openai.com/v1",
                     "anthropic": "https://api.anthropic.com"}


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"
    model = ""

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        raise NotImplementedError


def _post_with_retry(url, headers, payload, timeout, tries=3):
    last = None
    for attempt in range(tries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:  # network blip
            last = e
        else:
            if r.status_code < 400:
                return r.json()
            last = LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                break
        time.sleep(2 ** attempt)
    raise LLMError(str(last))


class OpenAICompatLLM(BaseLLM):
    name = "openai"

    def __init__(self, model, base_url, api_key, temperature=0.0, timeout=60):
        self.model = model or DEFAULT_MODELS["openai"]
        self.base_url = (base_url or DEFAULT_BASE_URLS["openai"]).rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system, user, max_tokens=400):
        data = _post_with_retry(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key or 'none'}"},
            {"model": self.model, "temperature": self.temperature, "max_tokens": max_tokens,
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user}]},
            self.timeout,
        )
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from e


class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self, model, base_url, api_key, temperature=0.0, timeout=60):
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.base_url = (base_url or DEFAULT_BASE_URLS["anthropic"]).rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system, user, max_tokens=400):
        data = _post_with_retry(
            f"{self.base_url}/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            {"model": self.model, "max_tokens": max_tokens, "temperature": self.temperature,
             "system": system, "messages": [{"role": "user", "content": user}]},
            self.timeout,
        )
        try:
            return "".join(b.get("text", "") for b in data["content"]).strip()
        except KeyError as e:
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from e


class ExtractiveLLM(BaseLLM):
    """Placeholder so the pipeline always has a generator. The extractive
    answer is built in rag.py, which has access to the passages directly."""
    name = "extractive"
    model = "extractive-baseline"

    def complete(self, system, user, max_tokens=400):
        raise LLMError("ExtractiveLLM does not generate text; rag.py handles it.")


def make_llm(cfg) -> BaseLLM:
    p = cfg.llm_provider
    if p == "openai":
        return OpenAICompatLLM(cfg.llm_model, cfg.llm_base_url, cfg.llm_api_key,
                               cfg.llm_temperature, cfg.llm_timeout)
    if p == "anthropic":
        return AnthropicLLM(cfg.llm_model, cfg.llm_base_url, cfg.llm_api_key,
                            cfg.llm_temperature, cfg.llm_timeout)
    if p == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Unknown LLM_PROVIDER '{p}'. Use openai, anthropic or extractive.")
