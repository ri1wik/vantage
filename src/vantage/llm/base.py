"""The model seam.

Every agent node calls exactly one interface, :class:`LLMClient`, with a
:class:`LLMRequest` that carries both a rendered prompt (what a hosted model
reads) and the structured ``payload`` the prompt was rendered from (what the
deterministic baseline reads). Keeping both on the request is what lets
``vantage-bench`` score a hosted model and the offline baseline through the same
graph, the same guardrails and the same metrics.

Every node's contract is JSON, so :func:`parse_json` is deliberately forgiving
about the wrappers models put around it, and strict about everything else.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

TASKS = ("plan", "sql", "critique", "memo")


class LLMError(RuntimeError):
    """Transport, quota or decoding failure from a model provider."""

    def __init__(self, message: str, provider: str = "", retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


@dataclass
class LLMRequest:
    """One call to one node's model."""

    task: str
    system: str
    user: str
    payload: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int = 1400

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; expected one of {TASKS}")


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    """What every provider and the offline baseline must implement."""

    name: str

    def generate(self, request: LLMRequest) -> LLMResponse: ...


_FENCE = re.compile(r"```(?:json|sql)?\s*(.*?)```", re.DOTALL)


def strip_fence(text: str) -> str:
    """Return the body of the first fenced block, or the text unchanged."""
    match = _FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def parse_json(text: str) -> dict[str, Any]:
    """Decode a JSON object from a model response.

    Tolerates markdown fences and leading prose, because those are the two
    failure modes every hosted model produces regardless of instructions. Raises
    :class:`LLMError` when no object can be recovered, so the node can turn it
    into a critique rather than crashing the graph.
    """
    candidate = strip_fence(text)
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"response contained no JSON object: {text[:200]!r}") from None
        try:
            loaded = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as err:
            raise LLMError(f"malformed JSON in response: {err}") from err
    if not isinstance(loaded, dict):
        raise LLMError(f"expected a JSON object, got {type(loaded).__name__}")
    return loaded
