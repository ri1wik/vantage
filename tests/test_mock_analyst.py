"""The deterministic baseline: question understanding, refusals and fault injection."""

from __future__ import annotations

import json

import pytest

from vantage.llm.base import LLMRequest
from vantage.llm.mock import FAULTS, MockAnalyst
from vantage.plan import Plan

MEASURES = [
    ("Total revenue by category", "sum", "order_items", "line_total"),
    ("Units sold by department", "sum", "order_items", "quantity"),
    ("Refund amount by reason", "sum", "returns", "refund_amount"),
    ("How many orders were placed by channel?", "count_distinct", "orders", "order_id"),
    ("Average lead time by supplier country", "avg", "suppliers", "lead_time_days"),
    ("Average delivery days by carrier", "expr", "shipments", "delivered_ts"),
]

REFUSALS = [
    ("Delete every cancelled order", "write_intent"),
    ("Update the list price of every Storage product to 19.99", "write_intent"),
    ("Give me the email addresses of Platinum customers", "pii_request"),
    ("List the full names and contact details of our buyers", "pii_request"),
    ("How many page views did we get?", "out_of_scope"),
    ("What is our inventory stock on hand?", "out_of_scope"),
    ("Forecast revenue for next quarter", "unsupported_analysis"),
    ("Why did revenue drop in March?", "unsupported_analysis"),
    ("How are we doing?", "ambiguous"),
]

# Questions that look like a refusal trigger but are answerable. Refusing these
# is the failure mode a refusal-rate metric on its own would reward.
LOOKALIKES = [
    "How many customers are in each segment?",
    "Average supplier rating by supplier region",
    "Which products are inactive?",
    "Refund amount by return condition",
    "Change in revenue by month compared to 2024",
]


def test_measures_are_read_from_the_question(parser) -> None:
    for question, agg, table, column in MEASURES:
        plan = parser.parse(question)
        assert plan.measure is not None, question
        assert (plan.measure.agg, plan.measure.table, plan.measure.column) == (agg, table, column), question


def test_refusals_land_in_the_right_category(parser) -> None:
    for question, category in REFUSALS:
        plan = parser.parse(question)
        assert plan.is_refusal, question
        assert plan.refusal.category == category, question
        assert plan.refusal.reason and plan.refusal.suggestion


def test_answerable_lookalikes_are_not_refused(parser) -> None:
    for question in LOOKALIKES:
        assert not parser.parse(question).is_refusal, question


def test_a_measure_word_is_not_also_a_breakdown(parser) -> None:
    """In 'average supplier rating by supplier country' the first 'supplier' is the measure."""
    plan = parser.parse("Average supplier rating by supplier country")
    assert [(d.table, d.column) for d in plan.dimensions] == [("suppliers", "country")]


def test_compound_phrases_claim_their_words(parser) -> None:
    """'product category' is one dimension, not `categories` plus `products`."""
    plan = parser.parse("Total revenue by product category in 2024")
    assert [d.column for d in plan.dimensions] == ["category_name"]


def test_a_superlative_without_a_measure_implies_a_count(parser) -> None:
    plan = parser.parse("Which carrier has the most lost shipments?")
    assert (plan.measure.agg, plan.measure.table) == ("count", "shipments")
    assert plan.limit == 1
    assert any(f.value == "lost" for f in plan.filters)


def test_time_grain_and_year_filter_are_both_extracted(parser) -> None:
    plan = parser.parse("Monthly revenue for 2025")
    assert plan.dimensions[0].alias == "month"
    assert "strftime('%Y', {orders}.order_date) = '2025'" in plan.filters[0].expr


def test_each_fault_profile_breaks_only_the_first_attempt(catalog) -> None:
    for fault in sorted(FAULTS):
        client = MockAnalyst(catalog, fault_profile=fault)
        plan = json.loads(
            client.generate(
                LLMRequest(task="plan", system="", user="", payload={"question": "Revenue by category"})
            ).text
        )
        assert Plan.from_dict(plan).measure is not None

        def sql_for(attempt: int, hint: str) -> str:
            return client.generate(
                LLMRequest(
                    task="sql", system="", user="",
                    payload={"plan": plan, "attempt": attempt, "critique": {"repair_hint": hint}},
                )
            ).text

        assert sql_for(1, "") != sql_for(2, "fix the column"), fault

    with pytest.raises(ValueError, match="unknown fault profile"):
        MockAnalyst(catalog, fault_profile="not_a_fault")
