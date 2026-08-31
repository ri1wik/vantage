"""Plan compilation. Joins come from the FK graph, never from a guess."""

from __future__ import annotations

import pytest

from vantage.plan import Dimension, Filter, Measure, Plan
from vantage.sql_compiler import CompileError


def test_a_single_table_aggregate_compiles(compiler, executor) -> None:
    plan = Plan(
        measure=Measure("sum", "returns", "refund_amount", "refund_amount"),
        dimensions=[Dimension("returns", "reason", "reason")],
    )
    sql = compiler.compile(plan).sql
    assert "FROM returns" in sql and "GROUP BY" in sql
    assert executor.run(sql).row_count == 6


def test_multi_hop_joins_are_derived_from_foreign_keys(compiler) -> None:
    plan = Plan(
        measure=Measure("sum", "order_items", "line_total", "revenue"),
        dimensions=[Dimension("categories", "category_name", "category")],
    )
    sql = compiler.compile(plan).sql
    # products is never named in the plan; the FK path requires it.
    assert "JOIN products" in sql
    assert "JOIN categories" in sql


def test_optional_tables_become_left_joins(compiler) -> None:
    plan = Plan(
        measure=Measure(
            "ratio", "order_items", "quantity", "return_rate",
            numerator="SUM(COALESCE({returns}.quantity, 0))",
            denominator="SUM({order_items}.quantity)",
            extra_tables=["returns"],
        ),
        dimensions=[Dimension("categories", "department", "department")],
        optional_tables=["returns"],
    )
    sql = compiler.compile(plan).sql
    assert "LEFT JOIN returns" in sql
    assert "JOIN products" in sql and "LEFT JOIN products" not in sql


def test_filters_and_ordering_render(compiler, executor) -> None:
    plan = Plan(
        measure=Measure("count_distinct", "orders", "order_id", "order_count"),
        dimensions=[Dimension("orders", "channel", "channel")],
        filters=[Filter("orders", "status", "=", "completed", description="completed only")],
        order_by="measure_asc",
        limit=3,
    )
    sql = compiler.compile(plan).sql
    assert "WHERE o.status = 'completed'" in sql
    assert "ORDER BY order_count ASC" in sql and "LIMIT 3" in sql
    # Literals are escaped, so a value containing a quote cannot break out.
    quoted = Plan(
        measure=Measure("count", "stores", "store_id", "n"),
        filters=[Filter("stores", "city", "=", "O'Fallon")],
    )
    assert "'O''Fallon'" in compiler.compile(quoted).sql
    result = executor.run(sql)
    assert result.row_count == 3
    assert [r[1] for r in result.rows] == sorted(r[1] for r in result.rows)


def test_impossible_plans_raise_rather_than_emit_bad_sql(compiler) -> None:
    """A plan the schema cannot express must fail loudly, not compile to something
    that runs and answers a different question."""
    for plan, message in [
        (Plan(measure=Measure("sum", "nope", "x", "v")), "unknown table"),
        (Plan(measure=Measure("sum", "orders", "not_a_column", "v")), "does not exist"),
        (Plan(measure=Measure("median", "order_items", "line_total", "v")), "unsupported aggregation"),
        (Plan(intent="refuse"), "refusal plan"),
    ]:
        with pytest.raises(CompileError, match=message):
            compiler.compile(plan)
