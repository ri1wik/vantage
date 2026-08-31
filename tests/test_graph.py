"""End-to-end runs through the compiled LangGraph."""

from __future__ import annotations

from vantage.agents.graph import VantageAnalyst
from vantage.llm.mock import FAULTS


def test_a_normal_question_answers_in_one_attempt(analyst) -> None:
    answer = analyst.ask("Total revenue by product category in 2024")
    assert answer.status == "answered"
    assert answer.attempt_count == 1
    assert not answer.self_corrected
    assert answer.row_count == 40
    assert answer.faithfulness == 1.0
    assert answer.trace_id


def test_a_refusal_stops_before_any_sql_is_written(analyst) -> None:
    answer = analyst.ask("Delete all cancelled orders")
    assert answer.status == "refused"
    assert answer.refusal["category"] == "write_intent"
    assert answer.sql is None
    assert answer.attempts == []


def test_every_injected_fault_is_diagnosed_and_repaired(settings) -> None:
    """Seven distinct first-attempt faults, all recovered inside the budget.

    `missing_group_by` is the one that matters most: it produces SQL that parses,
    passes the guard and runs, and is still wrong. Only the critic's AST check
    catches it.
    """
    for fault in sorted(FAULTS):
        analyst = VantageAnalyst(settings=settings, log_runs=False, fault_profile=fault)
        answer = analyst.ask("Total revenue by product category in 2024")
        assert answer.status == "answered", (fault, answer.attempts)
        assert answer.self_corrected, fault
        assert answer.attempts[0]["verdict"] == "repair", fault
        assert answer.attempts[-1]["verdict"] == "accept", fault
        assert answer.row_count == 40, fault


def test_the_attempt_budget_is_enforced(settings) -> None:
    """A model that never repairs must stop at the budget, not loop."""
    analyst = VantageAnalyst(settings=settings, log_runs=False, fault_profile="unknown_table")
    answer = analyst.ask("Total revenue by category", max_attempts=1)
    assert answer.status != "answered"
    assert answer.attempt_count == 1
    assert answer.critique["verdict"] == "abandon"


def test_the_trace_is_recorded_and_retrievable_by_id(settings, tmp_path) -> None:
    analyst = VantageAnalyst(settings=settings.with_overrides(log_dir=tmp_path), log_runs=True)
    answer = analyst.ask("Monthly revenue for 2025")

    assert answer.plan["measure"]["column"] == "line_total"
    assert set(answer.linked["scope"]) >= {"order_items", "orders"}
    assert answer.linked["recall"] == 1.0
    assert [a["n"] for a in answer.attempts] == [1]
    assert any(e["node"] == "schema_linker" for e in answer.events)

    record = analyst.logger.read(answer.trace_id)
    assert record is not None and record["question"] == "Monthly revenue for 2025"
    assert analyst.logger.read("no-such-trace") is None


def test_an_unfaithful_memo_is_stripped_not_shipped(settings) -> None:
    analyst = VantageAnalyst(settings=settings, log_runs=False, unfaithful_memo=True)
    answer = analyst.ask("Revenue by store region")
    assert answer.status == "answered"
    assert answer.fact_check["unverified"]
    assert "42.7" not in answer.memo_text
    assert any("unverified figure" in c for c in answer.memo["caveats"])
