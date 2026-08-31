"""FastAPI surface.

    uvicorn vantage.api:app --port 8000

The API returns the whole trace, not just the answer: the plan, the linked
tables, every attempt with its guard verdict, and the facts check. An analyst who
cannot see the query has to take the number on trust, which is the thing this
system exists to avoid.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agents.graph import VantageAnalyst
from .config import SETTINGS
from .guardrails.sql_guard import SqlGuard
from .llm.registry import available_providers
from .warehouse.catalog import get_catalog

app = FastAPI(
    title="Vantage",
    version="0.1.0",
    description="Self-correcting multi-agent data analyst over a read-only warehouse.",
)


@lru_cache(maxsize=1)
def get_analyst() -> VantageAnalyst:
    """One analyst per process: the TF-IDF index and the graph are reusable."""
    return VantageAnalyst()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    max_attempts: int | None = Field(None, ge=1, le=6)
    include_rows: bool = True


class ValidateRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=20_000)
    allowed_tables: list[str] | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    analyst = get_analyst()
    return {
        "status": "ok",
        "model": getattr(analyst.client, "name", SETTINGS.model),
        "database": str(SETTINGS.db_path),
        "tables": len(analyst.catalog.tables),
        "rows": analyst.catalog.total_rows(),
        "max_attempts": SETTINGS.max_attempts,
        "providers_available": available_providers(),
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    catalog = get_analyst().catalog
    return {
        "tables": [
            {
                "name": table.name,
                "description": table.description,
                "grain": table.grain,
                "rows": table.row_count,
                "columns": [
                    {"name": c.name, "type": c.type, "description": c.description, "primary_key": c.is_pk}
                    for c in table.columns
                ],
                "foreign_keys": [
                    {"column": fk.from_column, "references": f"{fk.to_table}.{fk.to_column}"}
                    for fk in table.foreign_keys
                ],
            }
            for table in (catalog.tables[n] for n in catalog.table_names)
        ],
        "total_rows": catalog.total_rows(),
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    answer = get_analyst().ask(request.question, max_attempts=request.max_attempts)
    payload = answer.as_dict()
    if not request.include_rows:
        payload["rows"] = []
    return payload


@app.post("/sql/validate")
def validate_sql(request: ValidateRequest) -> dict[str, Any]:
    """Run the AST guard on a query without executing it."""
    catalog = get_catalog(str(SETTINGS.db_path))
    guard = SqlGuard(catalog, row_limit=SETTINGS.row_limit)
    return guard.check(request.sql, allowed_tables=request.allowed_tables).as_dict()


@app.get("/link")
def link(question: str) -> dict[str, Any]:
    """What the schema linker would retrieve for a question. Useful for debugging recall."""
    return get_analyst().ctx.linker.link(question).as_dict()


@app.get("/traces/{trace_id}")
def trace(trace_id: str) -> dict[str, Any]:
    analyst = get_analyst()
    if analyst.logger is None:
        raise HTTPException(status_code=503, detail="run logging is disabled")
    record = analyst.logger.read(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no run logged with trace id {trace_id}")
    return record


@app.get("/traces")
def recent_traces(limit: int = 20) -> dict[str, Any]:
    analyst = get_analyst()
    if analyst.logger is None:
        raise HTTPException(status_code=503, detail="run logging is disabled")
    records = analyst.logger.tail(min(limit, 200))
    return {
        "count": len(records),
        "traces": [
            {
                "trace_id": r.get("trace_id"),
                "question": r.get("question"),
                "status": r.get("status"),
                "attempts": r.get("attempt_count"),
                "latency_ms": r.get("latency_ms"),
            }
            for r in records
        ],
    }
