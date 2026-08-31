"""Compile a :class:`~vantage.plan.Plan` into SQLite SQL.

Joins are never guessed. Every table beyond the base is attached by walking the
catalog's foreign-key graph, so the compiler physically cannot emit a join
predicate that does not exist in the schema. That is the property that makes the
deterministic baseline a fair control for a hosted model: both are held to the
same plan, and only one of them can invent a relationship.
"""

from __future__ import annotations

from dataclasses import dataclass

from .plan import Dimension, Filter, Measure, Plan
from .warehouse.catalog import Catalog


class CompileError(ValueError):
    """The plan cannot be expressed against this schema."""


@dataclass
class CompiledQuery:
    sql: str
    tables: list[str]
    aliases: dict[str, str]


def _alias_for(table: str, taken: set[str]) -> str:
    """Short, stable, collision-free table alias (orders -> o, order_items -> oi)."""
    parts = table.split("_")
    candidate = "".join(p[0] for p in parts) or table[:2]
    if candidate not in taken:
        return candidate
    for n in range(2, 10):
        probe = f"{candidate}{n}"
        if probe not in taken:
            return probe
    raise CompileError(f"could not allocate an alias for {table}")  # pragma: no cover


def _literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


class SqlCompiler:
    """Turns a validated plan into a single guarded-shape SELECT."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    # -- join construction -------------------------------------------------
    def _order_tables(self, plan: Plan) -> list[str]:
        """Base table first, then the rest in the order the plan mentions them."""
        wanted = plan.tables()
        if not wanted:
            raise CompileError("plan references no tables")
        base = plan.measure.table if plan.measure and plan.measure.table else wanted[0]
        rest = [t for t in wanted if t != base]
        return [base, *rest]

    def _build_from(
        self, tables: list[str], optional: set[str] | None = None
    ) -> tuple[str, dict[str, str]]:
        """FROM/JOIN clause covering ``tables``, plus every bridge table needed.

        Tables named in ``optional`` are attached with LEFT JOIN so base rows with
        no match are preserved; everything downstream of an optional table is
        LEFT JOINed too, since an inner join below a left join undoes it.
        """
        optional = optional or set()
        for table in tables:
            if not self.catalog.has_table(table):
                raise CompileError(f"unknown table '{table}'")

        base = tables[0]
        aliases: dict[str, str] = {base: _alias_for(base, set())}
        joined = [base]
        soft: set[str] = set()
        clause = [f"FROM {base} {aliases[base]}"]

        for target in tables[1:]:
            if target in joined:
                continue
            path = self._path_from_joined(joined, target)
            if not path:
                raise CompileError(f"no foreign-key path connects '{target}' to {joined}")
            for fk in path:
                new = fk.to_table if fk.from_table in joined else fk.from_table
                anchor = fk.from_table if new == fk.to_table else fk.to_table
                if new in joined:
                    continue
                aliases[new] = _alias_for(new, set(aliases.values()))
                left = f"{aliases[anchor]}.{fk.from_column if anchor == fk.from_table else fk.to_column}"
                right = f"{aliases[new]}.{fk.to_column if new == fk.to_table else fk.from_column}"
                kind = "LEFT JOIN" if (new in optional or anchor in soft) else "JOIN"
                if kind == "LEFT JOIN":
                    soft.add(new)
                clause.append(f"{kind} {new} {aliases[new]} ON {left} = {right}")
                joined.append(new)
        return "\n".join(clause), aliases

    def _path_from_joined(self, joined: list[str], target: str):
        """Shortest FK path from any already-joined table to ``target``."""
        best = None
        for source in joined:
            path = self.catalog.join_path(source, target)
            if path and (best is None or len(path) < len(best)):
                best = path
        return best

    # -- clause rendering --------------------------------------------------
    def _ref(self, aliases: dict[str, str], table: str, column: str) -> str:
        if table not in aliases:
            raise CompileError(f"table '{table}' is not in the FROM clause")
        if not self.catalog.has_column(table, column):
            raise CompileError(f"column '{column}' does not exist on '{table}'")
        return f"{aliases[table]}.{column}"

    def _dimension_sql(self, aliases: dict[str, str], dim: Dimension) -> str:
        if dim.expr:
            return dim.expr.format(**aliases)
        return self._ref(aliases, dim.table, dim.column)

    def _measure_sql(self, aliases: dict[str, str], measure: Measure) -> str:
        agg = measure.agg.lower()
        if agg == "expr" and measure.expr:
            return measure.expr.format(**aliases)
        if agg == "ratio":
            if not (measure.numerator and measure.denominator):
                raise CompileError("ratio measure needs a numerator and a denominator")
            num = measure.numerator.format(**aliases)
            den = measure.denominator.format(**aliases)
            return f"ROUND(1.0 * {num} / NULLIF({den}, 0), 4)"
        if agg == "count" and measure.column in ("*", ""):
            return "COUNT(*)"
        ref = self._ref(aliases, measure.table, measure.column)
        if agg == "count_distinct":
            return f"COUNT(DISTINCT {ref})"
        if agg in ("sum", "avg"):
            return f"ROUND({agg.upper()}({ref}), 2)"
        if agg in ("count", "min", "max"):
            return f"{agg.upper()}({ref})"
        raise CompileError(f"unsupported aggregation '{measure.agg}'")

    def _filter_sql(self, aliases: dict[str, str], flt: Filter) -> str:
        if flt.expr:
            return flt.expr.format(**aliases)
        ref = self._ref(aliases, flt.table, flt.column)
        op = flt.op.lower()
        if op in ("is null", "is not null"):
            return f"{ref} {op.upper()}"
        if op == "in":
            values = flt.value if isinstance(flt.value, (list, tuple)) else [flt.value]
            return f"{ref} IN ({', '.join(_literal(v) for v in values)})"
        if op == "between" and isinstance(flt.value, (list, tuple)) and len(flt.value) == 2:
            return f"{ref} BETWEEN {_literal(flt.value[0])} AND {_literal(flt.value[1])}"
        if op == "like":
            return f"{ref} LIKE {_literal(flt.value)}"
        if op in ("=", "!=", "<>", ">", "<", ">=", "<="):
            return f"{ref} {op} {_literal(flt.value)}"
        raise CompileError(f"unsupported operator '{flt.op}'")

    # -- entry point -------------------------------------------------------
    def compile(self, plan: Plan) -> CompiledQuery:
        if plan.is_refusal:
            raise CompileError("a refusal plan has no SQL")

        tables = self._order_tables(plan)
        from_clause, aliases = self._build_from(tables, set(plan.optional_tables))

        projections: list[str] = []
        group_by: list[str] = []
        for i, dim in enumerate(plan.dimensions, start=1):
            projections.append(f"{self._dimension_sql(aliases, dim)} AS {dim.alias}")
            group_by.append(str(i))

        measure_alias = None
        if plan.measure:
            measure_alias = plan.measure.alias
            projections.append(f"{self._measure_sql(aliases, plan.measure)} AS {measure_alias}")
        if not projections:
            raise CompileError("plan produced no projections")

        parts = ["SELECT " + ",\n       ".join(projections), from_clause]

        predicates = [self._filter_sql(aliases, f) for f in plan.filters]
        if predicates:
            parts.append("WHERE " + "\n  AND ".join(predicates))
        if group_by and plan.measure:
            parts.append("GROUP BY " + ", ".join(group_by))

        order = self._order_clause(plan, measure_alias)
        if order:
            parts.append(order)
        if plan.limit:
            parts.append(f"LIMIT {int(plan.limit)}")

        return CompiledQuery(sql="\n".join(parts), tables=list(aliases), aliases=aliases)

    def _order_clause(self, plan: Plan, measure_alias: str | None) -> str:
        mode = (plan.order_by or "").lower()
        if mode in ("", "none"):
            return ""
        if mode.startswith("measure") and measure_alias:
            direction = "ASC" if mode.endswith("asc") else "DESC"
            return f"ORDER BY {measure_alias} {direction}"
        if mode.startswith("dimension") and plan.dimensions:
            direction = "DESC" if mode.endswith("desc") else "ASC"
            return f"ORDER BY {plan.dimensions[0].alias} {direction}"
        return ""
