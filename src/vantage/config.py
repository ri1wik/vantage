"""Runtime configuration for Vantage, read from the environment (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # pragma: no cover - optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Everything the graph, the API and the bench harness need to boot."""

    model: str = field(default_factory=lambda: _env("VANTAGE_MODEL", "mock"))
    model_name: str | None = field(
        default_factory=lambda: os.environ.get("VANTAGE_MODEL_NAME") or None
    )
    db_path: Path = field(
        default_factory=lambda: (REPO_ROOT / _env("VANTAGE_DB", "data/warehouse.db")).resolve()
    )
    max_attempts: int = field(default_factory=lambda: _env_int("VANTAGE_MAX_ATTEMPTS", 3))
    row_limit: int = field(default_factory=lambda: _env_int("VANTAGE_ROW_LIMIT", 500))
    query_timeout_s: int = field(default_factory=lambda: _env_int("VANTAGE_QUERY_TIMEOUT_S", 10))
    log_dir: Path = field(
        default_factory=lambda: (REPO_ROOT / _env("VANTAGE_LOG_DIR", "logs")).resolve()
    )
    linker_top_k: int = field(default_factory=lambda: _env_int("VANTAGE_LINKER_TOP_K", 4))

    def with_overrides(self, **kwargs: object) -> "Settings":
        """Return a copy with selected fields replaced (used by the bench harness)."""
        from dataclasses import replace

        return replace(self, **kwargs)  # type: ignore[arg-type]


SETTINGS = Settings()
