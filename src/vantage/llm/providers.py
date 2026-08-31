"""Hosted and local model providers.

Four backends, one interface. Each one takes the rendered ``system``/``user``
prompt off the request and returns raw text; parsing, validation and repair are
the graph's job, not the provider's, so a provider that returns prose instead of
JSON produces a critique rather than a crash.

None of these are imported at module load in CI: :mod:`vantage.llm.registry`
only constructs one when the corresponding key is present.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .base import LLMError, LLMRequest, LLMResponse

DEFAULT_TIMEOUT = 60.0
MAX_RETRIES = 2
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class _HttpProvider:
    """Shared retry and error handling for the HTTP-based providers."""

    name: str
    model: str
    timeout: float = DEFAULT_TIMEOUT

    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict:
        last: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
                if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                    time.sleep(0.6 * (2**attempt))
                    continue
                if response.status_code >= 400:
                    raise LLMError(
                        f"{self.name} returned HTTP {response.status_code}: {response.text[:300]}",
                        provider=self.name,
                        retryable=response.status_code in RETRYABLE_STATUS,
                    )
                return response.json()
            except httpx.HTTPError as err:
                last = err
                if attempt < MAX_RETRIES:
                    time.sleep(0.6 * (2**attempt))
                    continue
                raise LLMError(f"{self.name} transport error: {err}", provider=self.name, retryable=True) from err
        raise LLMError(f"{self.name} failed after {MAX_RETRIES + 1} attempts: {last}", provider=self.name)


class OpenAICompatibleClient(_HttpProvider):
    """OpenAI Chat Completions, and anything that speaks the same shape (Groq)."""

    def __init__(self, name: str, model: str, base_url: str, api_key: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        super().__init__(name=name, model=model, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        if request.task != "sql":
            # Every node but the SQL writer returns JSON; the writer returns raw SQL.
            body["response_format"] = {"type": "json_object"}

        data = self._post(
            f"{self.base_url}/chat/completions",
            body,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError(f"{self.name} returned an unexpected body: {str(data)[:300]}", provider=self.name) from err

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class GeminiClient(_HttpProvider):
    """Google Generative Language API."""

    def __init__(self, model: str, api_key: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        super().__init__(name="gemini", model=model, timeout=timeout)
        self.api_key = api_key

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        generation: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_tokens,
        }
        if request.task != "sql":
            generation["responseMimeType"] = "application/json"

        data = self._post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": request.system}]},
                "contents": [{"role": "user", "parts": [{"text": request.user}]}],
                "generationConfig": generation,
            },
            {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError(f"gemini returned an unexpected body: {str(data)[:300]}", provider="gemini") from err

        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text=text,
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
        )


class OllamaClient(_HttpProvider):
    """A locally served model. No key; the host must already be running."""

    def __init__(self, model: str, host: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> None:
        super().__init__(name="ollama", model=model, timeout=timeout)
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": request.temperature, "num_predict": request.max_tokens},
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        if request.task != "sql":
            body["format"] = "json"

        data = self._post(f"{self.host}/api/chat", body, {"Content-Type": "application/json"})
        text = (data.get("message") or {}).get("content", "")
        if not text:
            raise LLMError(f"ollama returned an empty message: {str(data)[:300]}", provider="ollama")
        return LLMResponse(
            text=text,
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
