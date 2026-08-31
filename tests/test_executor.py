"""Read-only execution: the layer that holds even if the guard is bypassed."""

from __future__ import annotations

from vantage.executor import QueryResult, ReadOnlyExecutor


def test_a_select_returns_columns_and_rows(executor) -> None:
    result = executor.run("SELECT status, COUNT(*) AS n FROM orders GROUP BY 1")
    assert result.columns == ["status", "n"]
    assert result.row_count == 4
    assert result.elapsed_ms > 0


def test_the_authorizer_denies_writes_even_without_the_guard(executor) -> None:
    """The guard never sees this query; the driver must still refuse it."""
    result, error = executor.try_run("DELETE FROM orders")
    assert result is None
    assert error is not None and error.kind == "authorizer_denied"


def test_sqlite_errors_are_classified_not_raised(executor) -> None:
    _, error = executor.try_run("SELECT nope FROM orders")
    assert error is not None and error.kind == "sqlite_error"
    assert "no such column" in str(error)


def test_results_are_truncated_at_the_row_limit(warehouse) -> None:
    executor = ReadOnlyExecutor(warehouse, row_limit=5)
    result = executor.run("SELECT product_id FROM products")
    assert result.row_count == 5
    assert result.truncated
    assert result.notes


def test_a_missing_database_is_reported_clearly(tmp_path) -> None:
    _, error = ReadOnlyExecutor(tmp_path / "absent.db").try_run("SELECT 1")
    assert error is not None and error.kind == "missing_database"
    assert "vantage.warehouse.generate" in str(error)


def test_normalized_rows_ignore_ordering_and_integer_float_mismatch() -> None:
    a = QueryResult(columns=["k", "v"], rows=[("x", 1), ("y", 2.004)], elapsed_ms=1.0)
    b = QueryResult(columns=["label", "total"], rows=[("y", 2.0), ("x", 1.0)], elapsed_ms=1.0)
    assert a.normalized() == b.normalized()
    assert a.records()[0] == {"k": "x", "v": 1}
    assert QueryResult(columns=["n"], rows=[(7,)], elapsed_ms=1.0).scalar() == 7
