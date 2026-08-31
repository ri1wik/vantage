"""The AST guard. Everything here is a query that must, or must not, get through."""

from __future__ import annotations

SAFE = "SELECT o.status, COUNT(*) AS n FROM orders o GROUP BY 1 LIMIT 10"


def test_a_clean_select_passes_untouched(guard) -> None:
    report = guard.check(SAFE)
    assert report.ok
    assert report.violations == []
    assert report.tables == ["orders"]


def test_write_and_ddl_statements_are_refused(guard) -> None:
    for sql in [
        "DELETE FROM orders",
        "UPDATE products SET list_price = 0",
        "DROP TABLE orders",
        "INSERT INTO orders (order_id) VALUES (1)",
        "PRAGMA table_info(orders)",
        "CREATE TABLE x (a INT)",
        "ATTACH DATABASE 'other.db' AS o",
    ]:
        report = guard.check(sql)
        assert not report.ok, sql
        assert "WRITE_OPERATION" in {v.code for v in report.errors}, sql


def test_statement_counting_is_done_on_the_tree_not_on_semicolons(guard) -> None:
    """A stacked query is refused; a trailing semicolon or comment is not one."""
    stacked = guard.check("SELECT 1; DROP TABLE orders")
    assert not stacked.ok
    assert "MULTIPLE_STATEMENTS" in {v.code for v in stacked.errors}
    assert guard.check("SELECT store_name FROM stores LIMIT 1; -- done").ok
    assert guard.check("SELECT store_name FROM stores /* note */ LIMIT 1").ok


def test_unknown_identifiers_are_refused_and_the_real_ones_are_offered(guard) -> None:
    """The message is the repair hint the SQL writer gets back, so it carries the schema."""
    table = guard.check("SELECT * FROM sales_facts")
    assert not table.ok
    assert "order_items" in next(v for v in table.errors if v.code == "UNKNOWN_TABLE").message

    column = guard.check("SELECT o.total_amount FROM orders o")
    assert not column.ok
    assert "order_date" in next(v for v in column.errors if v.code == "UNKNOWN_COLUMN").message


def test_a_bare_column_with_no_from_clause_is_refused(guard) -> None:
    """sqlglot happily parses garbage as an expression; the catalog is the check."""
    assert not guard.check("SELECT this is not sql").ok
    assert guard.check("SELECT 1").ok


def test_filesystem_functions_are_refused(guard) -> None:
    report = guard.check("SELECT load_extension('evil.so')")
    assert not report.ok
    assert "BANNED_FUNCTION" in {v.code for v in report.errors}


def test_a_missing_limit_is_injected(guard) -> None:
    report = guard.check("SELECT status FROM orders")
    assert report.ok
    assert report.limit_injected
    assert "LIMIT 500" in report.sql
    assert "LIMIT_INJECTED" in {v.code for v in report.warnings}


def test_an_oversized_limit_is_clamped(guard) -> None:
    report = guard.check("SELECT status FROM orders LIMIT 100000")
    assert report.ok
    assert "LIMIT 500" in report.sql
    assert "LIMIT_CLAMPED" in {v.code for v in report.warnings}


def test_ctes_and_set_operations_are_allowed(guard) -> None:
    cte = "WITH t AS (SELECT order_id, SUM(line_total) v FROM order_items GROUP BY 1) SELECT AVG(v) FROM t"
    assert guard.check(cte).ok
    assert guard.check("SELECT region FROM stores UNION SELECT region FROM customers").ok


def test_markdown_fences_are_stripped_before_parsing(guard) -> None:
    assert guard.check("```sql\nSELECT COUNT(*) FROM orders\n```").ok


def test_out_of_scope_tables_warn_but_do_not_block(guard) -> None:
    report = guard.check("SELECT COUNT(*) FROM payments", allowed_tables=["orders"])
    assert report.ok
    assert "OUT_OF_SCOPE_TABLE" in {v.code for v in report.warnings}


def test_unparseable_sql_fails_closed(guard) -> None:
    report = guard.check("SELEKT * FROM (((")
    assert not report.ok
    assert report.sql == ""
