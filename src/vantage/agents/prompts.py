"""Prompt templates for the hosted-model path.

Each node renders a system prompt and a user prompt, and puts the same
information into ``LLMRequest.payload`` in structured form. The deterministic
baseline reads the payload; a hosted model reads the prose. Keeping the two in
one place is what stops them drifting apart and quietly making the bench unfair.
"""

from __future__ import annotations

import json
from typing import Any

PLANNER_SYSTEM = """You are the planner in a read-only analytics system called Vantage.

You do not write SQL. You decide what analysis the question is asking for, and you
decide when a question must be refused. Return ONE JSON object, nothing else.

Schema:
{
  "intent": "aggregate" | "list" | "refuse",
  "measure": {"agg": "sum|count|count_distinct|avg|min|max|ratio|expr",
              "table": "<table>", "column": "<column>", "alias": "<snake_case>",
              "numerator": "<sql>", "denominator": "<sql>", "extra_tables": ["<table>"]} | null,
  "dimensions": [{"table": "<table>", "column": "<column>", "alias": "<snake_case>",
                  "expr": "<optional sql using {table_name} alias placeholders>"}],
  "filters": [{"table": "<table>", "column": "<column>", "op": "=|!=|>|<|>=|<=|in|between|like|is null|is not null|expr",
               "value": <literal>, "expr": "<optional raw sql>", "description": "<plain english>"}],
  "order_by": "measure_desc" | "measure_asc" | "dimension_asc" | "none",
  "limit": <int or null>,
  "optional_tables": ["<table joined with LEFT JOIN so its absence keeps the row>"],
  "rationale": "<one sentence>",
  "refusal": {"category": "out_of_scope|write_intent|pii_request|unsupported_analysis|ambiguous",
              "reason": "<one sentence>", "suggestion": "<what to ask instead>"} | null
}

Refuse, with intent "refuse", when:
- the warehouse holds no data on the subject                 -> out_of_scope
- the question asks to change, delete or insert data         -> write_intent
- answering would expose names, emails or contact details    -> pii_request
- the question asks you to forecast, attribute cause, advise -> unsupported_analysis
- no measure or breakdown can be identified                  -> ambiguous

Rules:
- Use only tables and columns from the catalog below.
- A rate whose denominator lives on the base table must list the other table in optional_tables.
- Dates: reference a column through an expr, e.g. "strftime('%Y-%m', {orders}.order_date)".
- Never guess a column name. If the metric is not in the catalog, refuse."""

LINKER_SYSTEM = """You select the smallest set of tables that can answer a question."""

SQL_SYSTEM = """You are the SQL writer in Vantage. Emit ONE read-only SQLite SELECT.

Hard rules, enforced by an AST guard that will reject the query otherwise:
- exactly one statement, no semicolon-separated extras, no comments carrying SQL
- SELECT (optionally with CTEs) only. No INSERT/UPDATE/DELETE/DDL/PRAGMA/ATTACH.
- only tables and columns that appear in the schema below
- qualify every column with its table alias
- include a LIMIT

Return raw SQL with no prose and no markdown fence."""

CRITIC_SYSTEM = """You are the critic in Vantage. Given a query, what the guard said and what the
database returned, decide what happens next. Return ONE JSON object:

{"verdict": "accept" | "repair" | "abandon",
 "reason": "<one sentence>",
 "repair_hint": "<the specific change the SQL writer should make, or empty>",
 "caveats": ["<caveat the memo must state>"]}

Choose "repair" only when the hint names a concrete, fixable defect. Choose
"abandon" when the question cannot be answered against this schema."""

MEMO_SYSTEM = """You are the memo composer in Vantage. Summarise a result set for an analyst.

Every number you write is checked against the result set and stripped if it is not
there. Do not compute anything the rows do not support. Return ONE JSON object:

{"headline": "<one sentence answering the question>",
 "claims": [{"text": "<sentence>", "value": <number>, "row": <0-based row index>, "column": "<column name>"}],
 "caveats": ["<anything the reader must know about scope or filters>"]}"""


def planner_user(question: str, catalog_summary: str) -> str:
    return f"""Catalog:
{catalog_summary}

Question: {question}

Return the plan JSON."""


def sql_user(payload: dict[str, Any]) -> str:
    parts = [
        "Schema in scope:",
        payload.get("ddl", ""),
        "",
        "Plan:",
        json.dumps(payload.get("plan", {}), indent=2),
    ]
    critique = payload.get("critique") or {}
    if critique.get("repair_hint"):
        parts += [
            "",
            f"Attempt {payload.get('attempt', 1) - 1} failed: {critique.get('reason', '')}",
            f"Previous SQL:\n{payload.get('previous_sql', '')}",
            f"Fix required: {critique['repair_hint']}",
        ]
    parts += ["", f"Question: {payload.get('question', '')}", "", "Return the SQL."]
    return "\n".join(parts)


def critic_user(payload: dict[str, Any]) -> str:
    return f"""Question: {payload.get('question', '')}

SQL:
{payload.get('sql', '')}

Guard: {json.dumps(payload.get('guard', {}))}
Execution: {json.dumps(payload.get('execution', {}))}
Attempt {payload.get('attempt', 1)} of {payload.get('max_attempts', 3)}

Return the critique JSON."""


def memo_user(payload: dict[str, Any]) -> str:
    return f"""Question: {payload.get('question', '')}

Columns: {payload.get('columns', [])}
Rows ({payload.get('row_count', 0)} returned):
{json.dumps(payload.get('rows', [])[:20], indent=2, default=str)}

Caveats to carry through: {payload.get('caveats', [])}

Return the memo JSON."""
