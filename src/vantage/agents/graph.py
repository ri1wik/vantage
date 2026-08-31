"""The LangGraph wiring, and the facade the rest of the system talks to.

    planner --> schema_linker --> sql_writer --> executor --> critic
                                      ^                          |
                                      +-- repair (bounded) ------+
                                                                 |
                                                        accept --+--> memo_composer

The repair edge is what makes this self-correcting, and the budget on it is what
keeps it honest: a loop that can retry forever will eventually stumble onto
something that runs, which is not the same as being right. ``max_attempts``
caps it, every attempt is logged, and the bench scores how many were needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from ..config import SETTINGS, Settings
from ..executor import ReadOnlyExecutor
from ..guardrails.sql_guard import SqlGuard
from ..llm.base import LLMClient
from ..llm.registry import get_client
from ..retrieval.linker import SchemaLinker
from ..run_log import RunLogger
from ..warehouse.catalog import get_catalog
from . import nodes
from .nodes import NodeContext
from .state import VantageState, new_trace_id


@dataclass
class Answer:
    """What a completed run returns."""

    question: str
    trace_id: str
    status: str                       # answered | refused | failed
    model: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    memo: dict[str, Any] | None = None
    memo_text: str = ""
    refusal: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    linked: dict[str, Any] | None = None
    guard: dict[str, Any] | None = None
    critique: dict[str, Any] | None = None
    fact_check: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def self_corrected(self) -> bool:
        """True when an earlier attempt failed and a later one succeeded."""
        return self.status == "answered" and self.attempt_count > 1

    @property
    def faithfulness(self) -> float:
        return float((self.fact_check or {}).get("faithfulness", 1.0))

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "trace_id": self.trace_id,
            "status": self.status,
            "model": self.model,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "memo": self.memo,
            "memo_text": self.memo_text,
            "refusal": self.refusal,
            "plan": self.plan,
            "linked": self.linked,
            "guard": self.guard,
            "critique": self.critique,
            "fact_check": self.fact_check,
            "attempts": self.attempts,
            "events": self.events,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "attempt_count": self.attempt_count,
            "self_corrected": self.self_corrected,
        }


def _bind(fn: Callable[[VantageState, NodeContext], dict], ctx: NodeContext):
    """Adapt a two-argument node to the single-argument signature LangGraph wants."""

    def wrapper(state: VantageState) -> dict:
        return fn(state, ctx)

    wrapper.__name__ = fn.__name__
    return wrapper


def build_graph(ctx: NodeContext):
    """Compile the agent graph for a given context."""
    graph = StateGraph(VantageState)
    graph.add_node("planner", _bind(nodes.planner, ctx))
    graph.add_node("schema_linker", _bind(nodes.schema_linker, ctx))
    graph.add_node("sql_writer", _bind(nodes.sql_writer, ctx))
    graph.add_node("executor", _bind(nodes.execute, ctx))
    graph.add_node("critic", _bind(nodes.critic, ctx))
    graph.add_node("memo_composer", _bind(nodes.memo_composer, ctx))

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        lambda s: "stop" if s.get("status") in ("refused", "failed") else "continue",
        {"stop": END, "continue": "schema_linker"},
    )
    graph.add_edge("schema_linker", "sql_writer")
    graph.add_edge("sql_writer", "executor")
    graph.add_edge("executor", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"repair": "sql_writer", "accept": "memo_composer", "abandon": END},
    )
    graph.add_edge("memo_composer", END)
    return graph.compile()


def _route_after_critic(state: VantageState) -> str:
    verdict = (state.get("critique") or {}).get("verdict", "abandon")
    if verdict == "repair":
        # Belt and braces: the critic already respects the budget, but the router
        # must never be the reason a loop runs away.
        if int(state.get("attempt_no", 1)) >= int(state.get("max_attempts", 3)):
            return "abandon"
        return "repair"
    return verdict if verdict in ("accept", "abandon") else "abandon"


class VantageAnalyst:
    """Ask questions of the warehouse.

    >>> analyst = VantageAnalyst()
    >>> answer = analyst.ask("Total revenue by product category in 2024")
    >>> answer.status
    'answered'
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: LLMClient | None = None,
        db_path: Path | str | None = None,
        strict_memo: bool = True,
        log_runs: bool = True,
        **client_kwargs: object,
    ) -> None:
        self.settings = settings or SETTINGS
        path = Path(db_path) if db_path else self.settings.db_path
        self.catalog = get_catalog(str(path))
        self.client = client or get_client(
            self.settings.model, self.catalog, self.settings.model_name, **client_kwargs
        )
        self.ctx = NodeContext(
            catalog=self.catalog,
            client=self.client,
            linker=SchemaLinker(self.catalog, top_k=self.settings.linker_top_k),
            guard=SqlGuard(self.catalog, row_limit=self.settings.row_limit),
            executor=ReadOnlyExecutor(path, self.settings.query_timeout_s, self.settings.row_limit),
            max_attempts=self.settings.max_attempts,
            strict_memo=strict_memo,
        )
        self.graph = build_graph(self.ctx)
        self.logger = RunLogger(self.settings.log_dir) if log_runs else None

    def ask(self, question: str, max_attempts: int | None = None) -> Answer:
        started = time.perf_counter()
        initial: VantageState = {
            "question": question,
            "trace_id": new_trace_id(),
            "model": getattr(self.client, "name", self.settings.model),
            "attempt_no": 0,
            "max_attempts": max_attempts or self.settings.max_attempts,
            "attempts": [],
            "events": [],
            "status": "",
        }
        # The recursion limit is a hard stop behind the attempt budget: six nodes
        # per repair cycle, plus the entry and exit nodes.
        state = self.graph.invoke(initial, {"recursion_limit": 6 * initial["max_attempts"] + 8})
        answer = self._to_answer(state, (time.perf_counter() - started) * 1000)
        if self.logger:
            self.logger.write(answer.as_dict())
        return answer

    def _to_answer(self, state: dict, latency_ms: float) -> Answer:
        result = state.get("result")
        status = state.get("status") or ("answered" if result is not None else "failed")
        return Answer(
            question=state.get("question", ""),
            trace_id=state.get("trace_id", ""),
            status=status,
            model=state.get("model", ""),
            sql=state.get("sql"),
            columns=list(result.columns) if result is not None else [],
            rows=[list(r) for r in result.rows] if result is not None else [],
            row_count=result.row_count if result is not None else 0,
            memo=state.get("memo"),
            memo_text=nodes.render_memo(state.get("memo")),
            refusal=state.get("refusal"),
            plan=state.get("plan"),
            linked=state.get("linked"),
            guard=state.get("guard"),
            critique=state.get("critique"),
            fact_check=state.get("fact_check"),
            attempts=list(state.get("attempts") or []),
            events=list(state.get("events") or []),
            error=state.get("error"),
            latency_ms=latency_ms,
        )
