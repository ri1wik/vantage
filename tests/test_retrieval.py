"""TF-IDF index and the hybrid schema linker."""

from __future__ import annotations

from vantage.retrieval.tfidf import TfidfIndex, normalize


def test_normalize_folds_identifiers_and_plurals() -> None:
    assert normalize("Order_Items") == "order item"
    assert normalize("categories") == "category"
    assert normalize("Which STORES sold most?") == "which store sold most"


def test_tfidf_ranks_the_matching_document_first() -> None:
    index = TfidfIndex(
        labels=["a", "b"],
        documents=["shipments carrier delivery parcel", "payments processor gateway charge"],
    )
    assert index.top("which carrier delivered late", k=1)[0][0] == "a"
    assert index.score("")["a"] == 0.0
    assert index.vocabulary_size > 0


RECALL_CASES = [
    ("What was total revenue by product category in 2024?", ["order_items", "orders", "products", "categories"]),
    ("Which carrier has the most lost shipments?", ["shipments"]),
    ("Refund amount by return reason", ["returns"]),
    ("Top 10 customers by net revenue", ["order_items", "orders", "customers"]),
    ("Average lead time by supplier region", ["suppliers"]),
    ("Revenue by currency", ["order_items", "orders"]),
    ("Return rate by product department", ["order_items", "returns", "products", "categories"]),
    ("Payment failure rate by processor", ["payments"]),
]


def test_linker_recall_on_representative_questions(linker) -> None:
    """Recall is the metric that matters: a table the linker misses is a table the
    SQL writer cannot use, and no amount of downstream repair recovers it."""
    for question, required in RECALL_CASES:
        assert linker.recall(question, required) == 1.0, question


def test_temporal_anchor_adds_the_header_holding_the_date(linker) -> None:
    """order_items carries revenue but no date, so a monthly question needs orders."""
    without_time = linker.link("Total revenue by category")
    with_time = linker.link("Monthly revenue by category in 2025")
    assert "orders" not in without_time.tables
    assert "orders" in with_time.tables


def test_column_ownership_beats_the_ranking_without_over_including(linker) -> None:
    """`currency` lives only on orders and would otherwise be buried by `revenue`.

    The same pass must not over-fire: "refund amount" names
    `returns.refund_amount`, so the longer phrase claims those words and
    `payments.amount` is not dragged in.
    """
    assert "orders" in linker.link("Revenue by currency").tables
    assert linker.link("Refund amount by return reason").tables == ["returns"]


def test_geography_resolves_to_the_entity_the_question_names(linker) -> None:
    assert "stores" in linker.link("Revenue by store region").tables
    assert "suppliers" in linker.link("Average lead time by supplier region").tables
    assert "customers" in linker.link("Revenue by customer region").tables


def test_bridges_keep_the_selected_tables_joinable(catalog, linker) -> None:
    linked = linker.link("Return rate by product department")
    for table in linked.tables:
        assert catalog.has_table(table)
    # Every non-seed table is on a join path, so the compiler can always spell it.
    assert set(linked.bridges) <= set(linked.tables)
    assert "products" in linked.tables


def test_link_result_serialises_for_the_trace(linker) -> None:
    payload = linker.link("Revenue by sales channel").as_dict()
    assert set(payload) == {"tables", "seeds", "bridges", "lexicon_hits", "scores"}
    assert payload["tables"]
