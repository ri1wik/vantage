"""The generated warehouse is the ground truth every other test leans on."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from vantage.warehouse.generate import ROW_PLAN, TOTAL_ROWS, build


def test_total_row_count_is_exactly_258k(warehouse: Path) -> None:
    conn = sqlite3.connect(warehouse)
    total = sum(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ROW_PLAN)
    conn.close()
    assert total == TOTAL_ROWS == 258_000


def test_per_table_counts_match_the_plan(warehouse: Path) -> None:
    conn = sqlite3.connect(warehouse)
    actual = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ROW_PLAN}
    conn.close()
    assert actual == ROW_PLAN


def test_generation_is_deterministic(tmp_path: Path) -> None:
    """Same seed, same bytes. Without this the bench gold answers drift."""
    first = build(tmp_path / "a.db")
    second = build(tmp_path / "b.db")
    assert first == second
    conn = sqlite3.connect(tmp_path / "a.db")
    checksum_a = conn.execute("SELECT SUM(line_total), COUNT(*) FROM order_items").fetchone()
    conn.close()
    conn = sqlite3.connect(tmp_path / "b.db")
    checksum_b = conn.execute("SELECT SUM(line_total), COUNT(*) FROM order_items").fetchone()
    conn.close()
    assert checksum_a == checksum_b


def test_foreign_keys_have_no_orphans(warehouse: Path) -> None:
    conn = sqlite3.connect(warehouse)
    conn.execute("PRAGMA foreign_keys = ON")
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    conn.close()
    assert violations == []


def test_every_order_has_at_least_one_line(warehouse: Path) -> None:
    conn = sqlite3.connect(warehouse)
    orphans = conn.execute(
        "SELECT COUNT(*) FROM orders o WHERE NOT EXISTS "
        "(SELECT 1 FROM order_items oi WHERE oi.order_id = o.order_id)"
    ).fetchone()[0]
    conn.close()
    assert orphans == 0


def test_dates_fall_inside_the_declared_window(warehouse: Path) -> None:
    conn = sqlite3.connect(warehouse)
    lo, hi = conn.execute("SELECT MIN(order_date), MAX(order_date) FROM orders").fetchone()
    conn.close()
    assert lo >= "2023-01-01"
    assert hi <= "2025-12-31"
