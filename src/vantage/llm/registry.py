"""Model selection.

``mock`` is the default everywhere, including CI, so the full graph runs with no
key and no network. Naming a hosted provider without its key is a hard error
rather than a silent downgrade: a bench row that says ``gemini`` must have been
produced by Gemini.
"""

from __future__ import annotations

import os

from ..warehouse.catalog import Catalog
from .base import LLMClient, LLMError
from .mock import MockAnalyst

#: Provider -> (environment variable holding the key, default model id)
PROVIDERS: dict[str, tuple[str | None, str]] = {
    "mock": (None, "deterministic-baseline"),
    "openai": ("OPENAI_API_KEY", "gpt-4o-mini"),
    "groq": ("GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "gemini": ("GEMINI_API_KEY", "gemini-2.0-flash"),
    "ollama": (None, "llama3.1"),
}


def available_providers() -> list[str]:
    """Providers this machine can actually run right now."""
    out = ["mock"]
    for name, (env_var, _) in PROVIDERS.items():
        if name == "mock":
            continue
        if env_var is None or os.environ.get(env_var):
            out.append(name)
    return out


def get_client(
    provider: str,
    catalog: Catalog,
    model_name: str | None = None,
    **mock_kwargs: object,
) -> LLMClient:
    """Build a client for ``provider``.

    ``mock_kwargs`` (``fault_profile``, ``unfaithful_memo``) are the bench's
    controls and apply only to the deterministic baseline.
    """
    provider = (provider or "mock").lower()
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}; known: {', '.join(sorted(PROVIDERS))}")

    env_var, default_model = PROVIDERS[provider]
    model = model_name or os.environ.get("VANTAGE_MODEL_NAME") or default_model

    if provider == "mock":
        return MockAnalyst(catalog, **mock_kwargs)  # type: ignore[arg-type]

    if env_var and not os.environ.get(env_var):
        raise LLMError(
            f"provider {provider!r} needs {env_var} to be set. "
            f"Run with --model mock for the offline baseline."
        )

    # Imported lazily so a CI run with no keys never constructs an HTTP client.
    from .providers import GeminiClient, OllamaClient, OpenAICompatibleClient

    if provider == "openai":
        return OpenAICompatibleClient("openai", model, "https://api.openai.com/v1", os.environ[env_var])
    if provider == "groq":
        return OpenAICompatibleClient("groq", model, "https://api.groq.com/openai/v1", os.environ[env_var])
    if provider == "gemini":
        return GeminiClient(model, os.environ[env_var])
    return OllamaClient(model)
