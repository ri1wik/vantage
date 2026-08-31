"""Read-only query execution against the warehouse.

The AST guard is the primary control, but a guard that lives in the same process
as the thing it guards is one bug away from being bypassed. This module is the
second, independent layer: the connection is opened read-only, a SQLite
authorizer denies every non-read action code at the driver level, and a progress
handler aborts anything that outruns its wall-clock budget.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Action codes the authorizer permits. Everything else is denied.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


class QueryExecutionError(RuntimeError):
    """Raised when a guarded query still fails at execution time."""

    def __init__(self, message: str, kind: str = "sqlite_error", sql: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.sql = sql


@dataclass
class QueryResult:
    """A materialised result set, capped at ``row_limit`` rows."""

    columns: list[str]
    rows: list[tuple[Any, ...]]
    elapsed_ms: float
    sql: str = ""
    truncated: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        return not self.rows

    def records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]

    def scalar(self) -> Any:
        """The single value of a 1x1 result, or None."""
        if self.rows and self.rows[0]:
            return self.rows[0][0]
        return None

    def normalized(self, precision: int = 2) -> set[tuple]:
        """Order-insensitive, rounding-tolerant view used for answer comparison.

        Column *names* are deliberately excluded: two correct queries may label
        the same number `revenue` or `total_revenue`, and the bench scores the
        answer, not the aliasing.
        """
        out = set()
        for row in self.rows:
            out.add(tuple(_round_cell(c, precision) for c in row))
        return out

    def preview(self, limit: int = 10) -> str:
        head = " | ".join(self.columns)
        body = "\n".join(" | ".join("NULL" if c is None else str(c) for c in r) for r in self.rows[:limit])
        more = f"\n... {self.row_count - limit} more rows" if self.row_count > limit else ""
        return f"{head}\n{body}{more}"

    def as_dict(self) -> dict:
        return {
            "columns": self.columns,
            "rows": [list(r) for r in self.rows],
            "row_count": self.row_count,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "truncated": self.truncated,
        }


def _round_cell(cell: Any, precision: int) -> Any:
    if isinstance(cell, float):
        return round(cell, precision)
    if isinstance(cell, int) and not isinstance(cell, bool):
        return float(cell)
    return cell


def _authorizer(action: int, arg1, arg2, db_name, trigger) -> int:
    """Driver-level allowlist. Denies writes even if the AST guard were bypassed."""
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


class ReadOnlyExecutor:
    """Executes guarded SELECTs. Opens a fresh connection per query."""

    def __init__(self, db_path: Path | str, timeout_s: int = 10, row_limit: int = 500) -> None:
        self.db_path = Path(db_path)
        self.timeout_s = timeout_s
        self.row_limit = row_limit

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise QueryExecutionError(
                f"warehouse not found at {self.db_path}. Run: python -m vantage.warehouse.generate",
                kind="missing_database",
            )
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=self.timeout_s)
        deadline = time.monotonic() + self.timeout_s

        def _watchdog() -> int:
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(_watchdog, 10_000)
        conn.set_authorizer(_authorizer)
        return conn

    def run(self, sql: str) -> QueryResult:
        """Execute ``sql`` and materialise at most ``row_limit`` rows."""
        started = time.perf_counter()
        conn = self._connect()
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(self.row_limit + 1)
            columns = [d[0] for d in cursor.description] if cursor.description else []
        except sqlite3.Error as err:
            # Authorizer denials surface as DatabaseError, timeouts as
            # OperationalError; classify on the message so the critic can tell
            # a repairable typo from a refused operation.
            message = str(err)
            lowered = message.lower()
            if "interrupted" in lowered:
                raise QueryExecutionError(
                    f"query exceeded the {self.timeout_s}s budget", kind="timeout", sql=sql
                ) from err
            if "not authorized" in lowered:
                raise QueryExecutionError(
                    "query attempted a non-read operation and was denied by the authorizer",
                    kind="authorizer_denied",
                    sql=sql,
                ) from err
            raise QueryExecutionError(message, kind="sqlite_error", sql=sql) from err
        finally:
            conn.close()

        truncated = len(rows) > self.row_limit
        return QueryResult(
            columns=columns,
            rows=[tuple(r) for r in rows[: self.row_limit]],
            elapsed_ms=(time.perf_counter() - started) * 1000,
            sql=sql,
            truncated=truncated,
            notes=[f"result truncated to {self.row_limit} rows"] if truncated else [],
        )

    def try_run(self, sql: str) -> tuple[QueryResult | None, QueryExecutionError | None]:
        """Non-raising variant used by the graph, which turns errors into critiques."""
        try:
            return self.run(sql), None
        except QueryExecutionError as err:
            return None, err

    def explain(self, sql: str) -> list[str]:
        """EXPLAIN QUERY PLAN output, used by the critic to spot full scans."""
        conn = self._connect()
        try:
            return [str(r[-1]) for r in conn.execute(f"EXPLAIN QUERY PLAN {sql}")]
        except sqlite3.Error:
            return []
        finally:
            conn.close()
