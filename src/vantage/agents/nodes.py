"""The five agents.

Each node is a plain function over :class:`VantageState`, so any of them can be
tested on a hand-written state without standing up the graph. The nodes never
raise on model or database failure: a failure becomes a critique, and the critic
decides whether it is repairable. That is the whole point of the loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from ..executor import QueryExecutionError, ReadOnlyExecutor
from ..guardrails.sql_guard import SqlGuard
from ..llm.base import LLMClient, LLMError, LLMRequest, parse_json
from ..plan import Plan
from ..retrieval.linker import SchemaLinker
from ..verify import facts
from ..warehouse.catalog import Catalog
from . import prompts
from .state import Attempt, Critique, VantageState, event


@dataclass
class NodeContext:
    """Everything the nodes share. Built once per analyst, reused per question."""

    catalog: Catalog
    client: LLMClient
    linker: SchemaLinker
    guard: SqlGuard
    executor: ReadOnlyExecutor
    max_attempts: int = 3
    strict_memo: bool = True

    def catalog_summary(self) -> str:
        """Compact schema for the planner: names, grain and columns, no samples."""
        lines = []
        for name in self.catalog.table_names:
            table = self.catalog.tables[name]
            lines.append(f"{name} ({table.row_count:,} rows) - {table.description}")
            lines.append("  columns: " + ", ".join(table.column_names))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 1. Planner
# --------------------------------------------------------------------------
def planner(state: VantageState, ctx: NodeContext) -> dict:
    question = state["question"]
    request = LLMRequest(
        task="plan",
        system=prompts.PLANNER_SYSTEM,
        user=prompts.planner_user(question, ctx.catalog_summary()),
        payload={"question": question},
    )
    try:
        raw = ctx.client.generate(request).text
        plan = Plan.from_dict(parse_json(raw))
    except LLMError as err:
        return {
            "status": "failed",
            "error": f"planner failed: {err}",
            "events": [event("planner", f"model error: {err}")],
        }

    if plan.is_refusal:
        refusal = plan.refusal.to_dict() if plan.refusal else {
            "category": "ambiguous", "reason": plan.rationale, "suggestion": ""
        }
        return {
            "plan": plan.to_dict(),
            "refusal": refusal,
            "status": "refused",
            "events": [event("planner", f"refused: {refusal['category']}", **refusal)],
        }

    return {
        "plan": plan.to_dict(),
        "events": [event("planner", plan.rationale or "plan ready", tables=plan.tables())],
    }


# --------------------------------------------------------------------------
# 2. Schema linker
# --------------------------------------------------------------------------
def schema_linker(state: VantageState, ctx: NodeContext) -> dict:
    """Retrieve the table subset, and record whether it covered the plan.

    The coverage check is the live linker-recall signal: if the planner needs a
    table retrieval did not surface, that is a retrieval miss, and it is recorded
    on the trace rather than silently patched over.
    """
    question = state["question"]
    linked = ctx.linker.link(question)
    plan = Plan.from_dict(state.get("plan") or {})

    plan_tables = plan.tables()
    missed = [t for t in plan_tables if t not in linked.tables]
    scope = sorted(set(linked.tables) | set(plan_tables))
    if missed:
        # Widen the scope to whatever the plan legitimately needs, plus the
        # bridges to reach it, and keep the miss visible on the trace.
        scope = sorted(set(scope) | ctx.catalog.fk_closure(set(scope), hops=1))

    payload = linked.as_dict()
    payload["scope"] = scope
    payload["plan_tables"] = plan_tables
    payload["missed_tables"] = missed
    payload["recall"] = 1.0 if not plan_tables else (len(plan_tables) - len(missed)) / len(plan_tables)
    payload["ddl"] = ctx.catalog.render(scope)

    return {
        "linked": payload,
        "events": [
            event(
                "schema_linker",
                f"linked {len(linked.tables)} table(s); scope {len(scope)}",
                tables=linked.tables,
                recall=payload["recall"],
            )
        ],
    }


# --------------------------------------------------------------------------
# 3. SQL writer (writes, then guards)
# --------------------------------------------------------------------------
def sql_writer(state: VantageState, ctx: NodeContext) -> dict:
    attempt_no = int(state.get("attempt_no", 0)) + 1
    linked = state.get("linked") or {}
    critique = state.get("critique") or {}

    payload = {
        "question": state["question"],
        "plan": state.get("plan") or {},
        "ddl": linked.get("ddl", ""),
        "tables": linked.get("scope", []),
        "attempt": attempt_no,
        "critique": critique,
        "previous_sql": state.get("sql") or "",
    }
    request = LLMRequest(
        task="sql",
        system=prompts.SQL_SYSTEM,
        user=prompts.sql_user(payload),
        payload=payload,
    )

    try:
        sql = ctx.client.generate(request).text
    except LLMError as err:
        attempt = Attempt(n=attempt_no, guard_ok=False, error=str(err), error_kind="model_error")
        return {
            "attempt_no": attempt_no,
            "sql": "",
            "guard": {"ok": False, "violations": [{"code": "MODEL_ERROR", "message": str(err)}]},
            "attempts": [*(state.get("attempts") or []), attempt.as_dict()],
            "events": [event("sql_writer", f"model error: {err}")],
        }

    report = ctx.guard.check(sql, allowed_tables=linked.get("scope") or None)
    attempt = Attempt(
        n=attempt_no,
        sql=report.sql or sql,
        guard_ok=report.ok,
        guard_violations=[str(v) for v in report.violations],
    )
    return {
        "attempt_no": attempt_no,
        "sql": report.sql if report.ok else sql,
        "guard": report.as_dict(),
        "attempts": [*(state.get("attempts") or []), attempt.as_dict()],
        "events": [event("sql_writer", f"attempt {attempt_no}: guard {report.summary()}")],
    }


# --------------------------------------------------------------------------
# 4. Executor (a node, not an agent: no model call)
# --------------------------------------------------------------------------
def execute(state: VantageState, ctx: NodeContext) -> dict:
    guard = state.get("guard") or {}
    if not guard.get("ok"):
        return {"result": None, "events": [event("executor", "skipped: query did not pass the guard")]}

    started = time.perf_counter()
    result, error = ctx.executor.try_run(state["sql"] or "")
    elapsed = (time.perf_counter() - started) * 1000

    if error is not None:
        return {
            "result": None,
            "error": str(error),
            "events": [event("executor", f"{error.kind}: {error}", elapsed_ms=round(elapsed, 2))],
        }
    return {
        "result": result,
        "error": None,
        "events": [
            event("executor", f"{result.row_count} row(s) in {result.elapsed_ms:.0f}ms", rows=result.row_count)
        ],
    }


# --------------------------------------------------------------------------
# 5. Critic
# --------------------------------------------------------------------------
#: Guard codes that a rewrite can plausibly fix, and the hint to send back.
REPAIR_HINTS = {
    "UNKNOWN_COLUMN": "Use only the columns listed in the schema. Replace the invalid column.",
    "UNKNOWN_TABLE": "Use only the tables listed in the schema. Replace the invalid table.",
    "PARSE_ERROR": "The SQL did not parse. Re-emit a single valid SQLite SELECT.",
    "MULTIPLE_STATEMENTS": "Emit exactly one statement, with no trailing statement after the semicolon.",
    "NON_SELECT": "Emit a SELECT. Vantage is read-only.",
    "WRITE_OPERATION": "Emit a SELECT. Writes, DDL and PRAGMA are rejected.",
    "BANNED_FUNCTION": "Remove the filesystem or extension function and use plain SQL.",
    "EMPTY_SQL": "No SQL was produced. Emit a single SELECT for the plan.",
    "MODEL_ERROR": "The model call failed. Re-emit the query.",
}


def critic(state: VantageState, ctx: NodeContext) -> dict:
    """Decide whether the current attempt is an answer, a repair or a dead end.

    Deliberately rule-based rather than a model call. The critic's job is to be
    right about mechanical facts (did it parse, did it run, does the shape match
    the plan), and a rule is both cheaper and more reliable at that than a model.
    A hosted model is consulted only for the caveats it adds to the memo.
    """
    attempt_no = int(state.get("attempt_no", 1))
    max_attempts = int(state.get("max_attempts", ctx.max_attempts))
    guard = state.get("guard") or {}
    result = state.get("result")
    plan = Plan.from_dict(state.get("plan") or {})
    exhausted = attempt_no >= max_attempts

    critique: Critique
    if not guard.get("ok"):
        codes = [v["code"] for v in guard.get("violations", []) if v.get("severity") == "error"]
        hint = " ".join(REPAIR_HINTS.get(code, "") for code in codes).strip()
        detail = "; ".join(
            v["message"] for v in guard.get("violations", []) if v.get("severity") == "error"
        )
        critique = Critique(
            verdict="abandon" if exhausted or not hint else "repair",
            reason=f"guard rejected the query: {detail}",
            repair_hint=hint,
        )
    elif state.get("error"):
        critique = Critique(
            verdict="abandon" if exhausted else "repair",
            reason=f"execution failed: {state['error']}",
            repair_hint=f"The database rejected the query: {state['error']}. Correct it against the schema.",
        )
    elif result is None:
        critique = Critique(
            verdict="abandon",
            reason="no result was produced",
            repair_hint="",
        )
    else:
        critique = _shape_critique(plan, result, state.get("sql") or "", exhausted)

    caveats = list(critique.caveats)
    for violation in guard.get("violations", []):
        if violation.get("code") in ("LIMIT_INJECTED", "LIMIT_CLAMPED"):
            caveats.append(violation["message"])
    if result is not None and result.truncated:
        caveats.append(f"Result truncated to {result.row_count} rows.")
    for note in plan.notes:
        caveats.append(note)
    for flt in plan.filters:
        if flt.description:
            caveats.append(f"Scope: {flt.description}.")
    critique.caveats = list(dict.fromkeys(caveats))

    # Backfill the verdict onto the attempt that earned it.
    attempts = [dict(a) for a in (state.get("attempts") or [])]
    if attempts:
        last = attempts[-1]
        last.update(
            {
                "executed": result is not None,
                "error": state.get("error") or "",
                "row_count": result.row_count if result is not None else None,
                "elapsed_ms": round(result.elapsed_ms, 2) if result is not None else 0.0,
                "verdict": critique.verdict,
                "critique": critique.reason,
                "repair_hint": critique.repair_hint,
            }
        )

    return {
        "critique": critique.as_dict(),
        "attempts": attempts,
        "events": [event("critic", f"{critique.verdict}: {critique.reason}")],
    }


def _missing_group_by(sql: str) -> bool:
    """An aggregate projected beside a bare column with no GROUP BY.

    SQLite accepts this and quietly returns one arbitrary row, which is the
    nastiest failure in this pipeline: the query runs, the guard is happy, the
    shape is right and the number is wrong. Only the AST catches it.
    """
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # pragma: no cover - already parsed by the guard
        return False
    if not isinstance(tree, exp.Select) or tree.args.get("group"):
        return False
    projections = tree.selects
    has_aggregate = any(p.find(exp.AggFunc) for p in projections)
    has_bare_column = any(p.find(exp.Column) and not p.find(exp.AggFunc) for p in projections)
    return has_aggregate and has_bare_column


def _shape_critique(plan: Plan, result, sql: str, exhausted: bool) -> Critique:
    """Does the result look like the plan that asked for it?"""
    if plan.dimensions and plan.measure and _missing_group_by(sql):
        return Critique(
            verdict="abandon" if exhausted else "repair",
            reason="the query aggregates beside an ungrouped column, so SQLite returned one arbitrary row",
            repair_hint="Add GROUP BY covering every projected dimension.",
        )
    expected_columns = len(plan.dimensions) + (1 if plan.measure else 0)
    if expected_columns and len(result.columns) < expected_columns:
        return Critique(
            verdict="abandon" if exhausted else "repair",
            reason=(
                f"the plan asked for {expected_columns} output column(s) "
                f"but the query returned {len(result.columns)}"
            ),
            repair_hint=(
                "Project every dimension alongside the measure and GROUP BY all of them."
            ),
        )
    if plan.dimensions and result.row_count == 1 and not plan.limit:
        return Critique(
            verdict="accept",
            reason="single grouped row returned",
            caveats=["Only one group matched the filters."],
        )
    if result.is_empty:
        return Critique(
            verdict="accept",
            reason="query ran but matched no rows",
            caveats=["No rows matched the filters, so no figures are reported."],
        )
    return Critique(verdict="accept", reason=f"{result.row_count} row(s) returned")


# --------------------------------------------------------------------------
# 6. Memo composer (+ fact verification)
# --------------------------------------------------------------------------
def memo_composer(state: VantageState, ctx: NodeContext) -> dict:
    result = state.get("result")
    critique = state.get("critique") or {}
    if result is None:
        return {"status": "failed", "events": [event("memo_composer", "no result to summarise")]}

    payload = {
        "question": state["question"],
        "columns": result.columns,
        "rows": [list(r) for r in result.rows[:20]],
        "row_count": result.row_count,
        "caveats": critique.get("caveats", []),
    }
    request = LLMRequest(
        task="memo",
        system=prompts.MEMO_SYSTEM,
        user=prompts.memo_user(payload),
        payload=payload,
    )

    try:
        memo = parse_json(ctx.client.generate(request).text)
    except LLMError as err:
        return {
            "status": "failed",
            "error": f"memo composer failed: {err}",
            "events": [event("memo_composer", f"model error: {err}")],
        }

    memo.setdefault("caveats", [])
    memo["caveats"] = list(dict.fromkeys([*memo["caveats"], *critique.get("caveats", [])]))

    check = facts.check_memo(memo, result, sql=state.get("sql") or "")
    if not check.ok and ctx.strict_memo:
        # Strip rather than reject: a memo with an unsupported figure removed is
        # still useful, and the trace records exactly what was pulled.
        memo["headline"] = facts.redact(str(memo.get("headline", "")), check)
        for claim in memo.get("claims", []) or []:
            claim["text"] = facts.redact(str(claim.get("text", "")), check)
        memo["caveats"].append(
            f"{len(check.unverified)} unverified figure(s) were removed by the facts checker."
        )
        memo["claims"] = [
            c for c in memo.get("claims", []) or [] if "[unverified]" not in str(c.get("text", ""))
        ]

    return {
        "memo": memo,
        "fact_check": check.as_dict(),
        "status": "answered",
        "events": [
            event(
                "memo_composer",
                f"memo composed; faithfulness {check.faithfulness:.0%}",
                unverified=len(check.unverified),
            )
        ],
    }


def render_memo(memo: dict | None) -> str:
    """Plain-text memo for the CLI and the API."""
    if not memo:
        return ""
    lines = [str(memo.get("headline", "")).strip()]
    for claim in memo.get("claims", []) or []:
        text = str(claim.get("text", "")).strip()
        if text:
            lines.append(f"  - {text}")
    caveats = memo.get("caveats") or []
    if caveats:
        lines.append("Caveats:")
        lines += [f"  * {c}" for c in caveats]
    return "\n".join(line for line in lines if line.strip())


def dumps(obj: object) -> str:  # pragma: no cover - debugging helper
    return json.dumps(obj, indent=2, default=str)
