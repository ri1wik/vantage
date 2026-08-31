"""Catalog introspection and the foreign-key graph the compiler joins over."""

from __future__ import annotations


def test_all_ten_tables_are_discovered(catalog) -> None:
    assert len(catalog.tables) == 10
    assert "order_items" in catalog.table_names


def test_row_counts_are_read_from_the_database(catalog) -> None:
    assert catalog.total_rows() == 258_000
    assert catalog.table("order_items").row_count == 118_000


def test_glossary_descriptions_and_synonyms_are_merged(catalog) -> None:
    line_total = catalog.table("order_items").column("line_total")
    assert "net revenue" in line_total.description.lower()
    assert "revenue" in line_total.synonyms


def test_low_cardinality_columns_get_sample_values(catalog) -> None:
    statuses = set(catalog.table("orders").column("status").sample_values)
    assert {"completed", "cancelled"} <= statuses
    # High-cardinality and identifier columns are deliberately not sampled.
    assert catalog.table("customers").column("email").sample_values == ()


def test_neighbours_walk_the_fk_graph_in_both_directions(catalog) -> None:
    assert catalog.neighbours("order_items") == {"orders", "products", "returns"}
    assert "order_items" in catalog.neighbours("returns")


def test_join_path_finds_the_shortest_route(catalog) -> None:
    path = catalog.join_path("returns", "categories")
    assert [(fk.from_table, fk.to_table) for fk in path] == [
        ("returns", "order_items"),
        ("order_items", "products"),
        ("products", "categories"),
    ]
    assert catalog.join_path("orders", "orders") == []


def test_ddl_rendering_includes_columns_and_foreign_keys(catalog) -> None:
    ddl = catalog.table("order_items").ddl()
    assert "CREATE TABLE order_items" in ddl
    assert "line_total" in ddl
    assert "REFERENCES orders(order_id)" in ddl
