"""Append-only run log.

One JSON object per run, one file per day. Everything needed to reconstruct a run
without re-running a model is on the line: the plan, the linked tables, every
attempt, the guard verdicts and the fact check.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, log_dir: Path | str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self.log_dir / f"runs-{date.today().isoformat()}.jsonl"

    def write(self, record: dict[str, Any]) -> Path:
        line = json.dumps(record, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return self.path

    def read(self, trace_id: str) -> dict[str, Any] | None:
        """Look a run up by trace id across every log file."""
        for file in sorted(self.log_dir.glob("runs-*.jsonl"), reverse=True):
            for line in file.read_text(encoding="utf-8").splitlines():
                if trace_id not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:  # pragma: no cover - truncated line
                    continue
                if record.get("trace_id") == trace_id:
                    return record
        return None

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # pragma: no cover
                continue
        return out
