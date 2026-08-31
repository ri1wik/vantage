"""The analysis plan: the contract between the planner and the SQL writer.

Both the deterministic baseline and a hosted model emit this same JSON shape, so
a plan can be inspected, diffed, logged and replayed without re-running a model.
Anything the planner cannot express here is out of scope by construction, which
is what keeps the SQL writer from inventing joins to answer a question the
planner never authorised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Aggregations the SQL writer knows how to compile.
AGGREGATIONS = ("sum", "count", "avg", "min", "max", "count_distinct", "ratio", "expr")

#: Why Vantage declined. The bench scores refusal category, not just the refusal.
REFUSAL_CATEGORIES = (
    "out_of_scope",       # the warehouse holds no such data
    "write_intent",       # the user asked to change data
    "pii_request",        # the answer would expose personally identifying fields
    "unsupported_analysis",  # forecasting, causal or advisory questions
    "ambiguous",          # under-specified beyond a safe default
)


@dataclass
class Measure:
    """What is being counted or summed."""

    agg: str
    table: str
    column: str
    alias: str
    expr: str | None = None          # raw SQL for ratio/expr aggregations
    numerator: str | None = None     # used when agg == "ratio"
    denominator: str | None = None
    extra_tables: list[str] = field(default_factory=list)  # tables a raw expr needs

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [])}


@dataclass
class Dimension:
    """What the measure is broken down by."""

    table: str
    column: str
    alias: str
    expr: str | None = None          # e.g. strftime('%Y-%m', o.order_date)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Filter:
    """One WHERE predicate, in structured form so it can be audited."""

    table: str
    column: str
    op: str
    value: Any = None
    expr: str | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}


@dataclass
class Refusal:
    category: str
    reason: str
    suggestion: str = ""

    def to_dict(self) -> dict:
        return {"category": self.category, "reason": self.reason, "suggestion": self.suggestion}


@dataclass
class Plan:
    """A fully specified analysis, or a refusal."""

    intent: str = "aggregate"                      # aggregate | list | refuse
    measure: Measure | None = None
    dimensions: list[Dimension] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    order_by: str = "measure_desc"                 # measure_desc | measure_asc | dimension_asc | none
    limit: int | None = None
    rationale: str = ""
    refusal: Refusal | None = None
    notes: list[str] = field(default_factory=list)
    #: Tables that must be LEFT JOINed, so rows without a match survive. Needed
    #: for any rate whose denominator lives on the base table (return rate,
    #: delivery coverage): an inner join would silently drop the denominator.
    optional_tables: list[str] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        return self.intent == "refuse" or self.refusal is not None

    def tables(self) -> list[str]:
        """Every table the plan touches, in a stable order."""
        seen: list[str] = []
        measure_tables: list[str] = []
        if self.measure:
            measure_tables = [self.measure.table, *self.measure.extra_tables]
        for table in (
            measure_tables + [d.table for d in self.dimensions] + [f.table for f in self.filters]
        ):
            if table and table not in seen:
                seen.append(table)
        return seen

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "measure": self.measure.to_dict() if self.measure else None,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "filters": [f.to_dict() for f in self.filters],
            "order_by": self.order_by,
            "limit": self.limit,
            "rationale": self.rationale,
            "refusal": self.refusal.to_dict() if self.refusal else None,
            "notes": self.notes,
            "optional_tables": self.optional_tables,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        """Build a plan from model output, dropping fields it got wrong.

        Deliberately lenient: a hosted model that emits an unknown aggregation or
        a half-formed dimension should produce a plan the critic can reject, not
        an exception that kills the run.
        """
        refusal_raw = data.get("refusal")
        refusal = None
        if isinstance(refusal_raw, dict) and refusal_raw.get("category"):
            refusal = Refusal(
                category=str(refusal_raw.get("category")),
                reason=str(refusal_raw.get("reason", "")),
                suggestion=str(refusal_raw.get("suggestion", "")),
            )

        measure_raw = data.get("measure")
        measure = None
        if isinstance(measure_raw, dict) and measure_raw.get("agg"):
            measure = Measure(
                agg=str(measure_raw["agg"]).lower(),
                table=str(measure_raw.get("table", "")),
                column=str(measure_raw.get("column", "")),
                alias=str(measure_raw.get("alias") or "value"),
                expr=measure_raw.get("expr"),
                numerator=measure_raw.get("numerator"),
                denominator=measure_raw.get("denominator"),
                extra_tables=list(measure_raw.get("extra_tables", []) or []),
            )

        dimensions = [
            Dimension(
                table=str(d.get("table", "")),
                column=str(d.get("column", "")),
                alias=str(d.get("alias") or d.get("column") or "dimension"),
                expr=d.get("expr"),
            )
            for d in data.get("dimensions", []) or []
            if isinstance(d, dict) and (d.get("column") or d.get("expr"))
        ]

        filters = [
            Filter(
                table=str(f.get("table", "")),
                column=str(f.get("column", "")),
                op=str(f.get("op", "=")),
                value=f.get("value"),
                expr=f.get("expr"),
                description=str(f.get("description", "")),
            )
            for f in data.get("filters", []) or []
            if isinstance(f, dict) and (f.get("column") or f.get("expr"))
        ]

        limit = data.get("limit")
        if isinstance(limit, str) and limit.isdigit():
            limit = int(limit)
        if not isinstance(limit, int):
            limit = None

        return cls(
            intent=str(data.get("intent", "aggregate")),
            measure=measure,
            dimensions=dimensions,
            filters=filters,
            order_by=str(data.get("order_by", "measure_desc")),
            limit=limit,
            rationale=str(data.get("rationale", "")),
            refusal=refusal,
            notes=list(data.get("notes", []) or []),
            optional_tables=list(data.get("optional_tables", []) or []),
        )
