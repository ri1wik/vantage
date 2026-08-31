"""MCP server: Vantage as a tool for another agent.

    python -m vantage.mcp_server            # stdio
    python -m vantage.mcp_server --http     # streamable HTTP on :8765

Six tools, and the split between them is the point. ``vantage_ask`` is the whole
graph: plan, retrieve, write, guard, execute, critique, verify. ``vantage_run_sql``
is the raw execution path, and it still refuses anything the AST guard rejects,
so an agent holding this tool cannot write to the warehouse whatever it sends.

Requires ``mcp>=2``. It is an optional dependency: nothing else in Vantage
imports this module.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from .agents.graph import VantageAnalyst
from .config import SETTINGS
from .guardrails.sql_guard import SqlGuard
from .warehouse.catalog import get_catalog

server = MCPServer(
    name="vantage",
    version="0.1.0",
    instructions=(
        "Vantage answers plain-English questions about a read-only retail warehouse "
        "(orders, order lines, products, categories, suppliers, customers, stores, "
        "payments, shipments, returns). Prefer vantage_ask, which plans, writes "
        "AST-guarded SQL, self-corrects on failure and verifies every figure in its "
        "memo against the result set. Use vantage_run_sql only when you already have "
        "the exact query. Every path is read-only and refuses writes."
    ),
)


@lru_cache(maxsize=1)
def _analyst() -> VantageAnalyst:
    return VantageAnalyst()


@lru_cache(maxsize=1)
def _guard() -> SqlGuard:
    return SqlGuard(get_catalog(str(SETTINGS.db_path)), row_limit=SETTINGS.row_limit)


@server.tool(
    description=(
        "Answer a plain-English analytics question. Returns the memo, the executed SQL, "
        "the rows, and the full attempt trace including any self-corrections."
    )
)
def vantage_ask(question: str, max_attempts: int | None = None, include_rows: bool = True) -> dict[str, Any]:
    answer = _analyst().ask(question, max_attempts=max_attempts)
    payload: dict[str, Any] = {
        "status": answer.status,
        "memo": answer.memo_text,
        "sql": answer.sql,
        "columns": answer.columns,
        "row_count": answer.row_count,
        "attempts": answer.attempt_count,
        "self_corrected": answer.self_corrected,
        "faithfulness": answer.faithfulness,
        "trace_id": answer.trace_id,
        "latency_ms": round(answer.latency_ms, 2),
    }
    if answer.refusal:
        payload["refusal"] = answer.refusal
    if include_rows:
        payload["rows"] = answer.rows[:100]
    if answer.error:
        payload["error"] = answer.error
    return payload


@server.tool(description="The warehouse schema: tables, grain, row counts, columns and foreign keys.")
def vantage_schema(table: str | None = None) -> dict[str, Any]:
    catalog = get_catalog(str(SETTINGS.db_path))
    names = [table.lower()] if table else list(catalog.table_names)
    missing = [n for n in names if not catalog.has_table(n)]
    if missing:
        return {"error": f"unknown table(s): {missing}", "known_tables": list(catalog.table_names)}
    return {
        "total_rows": catalog.total_rows(),
        "tables": [
            {
                "name": catalog.tables[n].name,
                "description": catalog.tables[n].description,
                "grain": catalog.tables[n].grain,
                "rows": catalog.tables[n].row_count,
                "ddl": catalog.tables[n].ddl(),
            }
            for n in names
        ],
    }


@server.tool(
    description=(
        "Check a SQL string against the AST guard without executing it. Reports every "
        "violation with a code and a message. Use this to understand why a query was refused."
    )
)
def vantage_validate_sql(sql: str, allowed_tables: list[str] | None = None) -> dict[str, Any]:
    return _guard().check(sql, allowed_tables=allowed_tables).as_dict()


@server.tool(
    description=(
        "Execute a read-only SELECT against the warehouse. The query is guarded first: "
        "writes, DDL, PRAGMA, unknown tables and unknown columns are refused, and a row "
        "limit is enforced."
    )
)
def vantage_run_sql(sql: str, row_limit: int | None = None) -> dict[str, Any]:
    report = _guard().check(sql)
    if not report.ok:
        return {"ok": False, "refused": True, "violations": report.as_dict()["violations"]}
    executor = _analyst().ctx.executor
    result, error = executor.try_run(report.sql)
    if error is not None:
        return {"ok": False, "refused": False, "error": str(error), "kind": error.kind}
    limit = min(row_limit or result.row_count, result.row_count)
    return {
        "ok": True,
        "sql": report.sql,
        "columns": result.columns,
        "rows": [list(r) for r in result.rows[:limit]],
        "row_count": result.row_count,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "warnings": [str(v) for v in report.warnings],
    }


@server.tool(
    description=(
        "Show which tables the schema linker retrieves for a question, with scores. "
        "Useful for diagnosing why an answer used the tables it did."
    )
)
def vantage_link(question: str) -> dict[str, Any]:
    return _analyst().ctx.linker.link(question).as_dict()


@server.tool(description="Headline metrics from the most recent vantage-bench run, if one has been saved.")
def vantage_bench_summary() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "bench" / "results" / "results.json"
    if not path.exists():
        return {"available": False, "hint": "run: python -m bench.runner --model mock --out bench/results"}
    data = json.loads(path.read_text())
    return {
        "available": True,
        "model": data.get("model"),
        "warehouse_rows": data.get("warehouse_rows"),
        "summary": data.get("summary"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Vantage MCP server.")
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if args.http:
        server.run(transport="streamable-http", port=args.port)
    else:
        server.run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
