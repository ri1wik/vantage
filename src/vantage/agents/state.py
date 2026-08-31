"""Graph state and the attempt record.

Every self-correction cycle appends one :class:`Attempt`. Nothing is overwritten,
so a finished run carries the full history: what SQL was tried, what the guard
said about it, what the database said about it, and what the critic decided to do
next. That record is the artifact the bench scores and the API returns, and it is
the only way to tell a model that got it right first time from one that needed
two repairs to get to the same answer.
"""

from __future__ import annotations

import operator
import uuid
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, TypedDict

from ..executor import QueryResult


@dataclass
class Attempt:
    """One pass through write -> guard -> execute -> critique."""

    n: int
    sql: str = ""
    guard_ok: bool = False
    guard_violations: list[str] = field(default_factory=list)
    executed: bool = False
    error: str = ""
    error_kind: str = ""
    row_count: int | None = None
    elapsed_ms: float = 0.0
    verdict: str = ""
    critique: str = ""
    repair_hint: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Critique:
    """The critic's decision about the current attempt."""

    verdict: str            # accept | repair | abandon
    reason: str = ""
    repair_hint: str = ""
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


class VantageState(TypedDict, total=False):
    """What flows through the graph.

    ``attempts`` and ``events`` accumulate rather than replace, so a node that
    only appends does not have to know what came before it.
    """

    question: str
    trace_id: str
    model: str

    plan: dict[str, Any] | None
    linked: dict[str, Any] | None
    sql: str | None
    guard: dict[str, Any] | None
    result: QueryResult | None
    critique: dict[str, Any] | None
    memo: dict[str, Any] | None
    fact_check: dict[str, Any] | None
    refusal: dict[str, Any] | None

    attempt_no: int
    max_attempts: int
    status: str                       # answered | refused | failed
    error: str | None

    #: Appended by the SQL writer, then patched in place by the critic once it
    #: knows the verdict, so it is a replace field rather than an accumulator.
    attempts: list[dict[str, Any]]
    events: Annotated[list[dict[str, Any]], operator.add]


def event(node: str, message: str, **extra: Any) -> dict[str, Any]:
    """A single line of run narration, surfaced by the API and the CLI."""
    return {"node": node, "message": message, **extra}
