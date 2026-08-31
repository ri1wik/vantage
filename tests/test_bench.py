"""The benchmark harness itself, including the controls that prove it measures."""

from __future__ import annotations

import collections

from bench.metrics import CaseScore, answers_match, percentile, summarize
from bench.runner import BenchRunner, load_cases
from vantage.executor import QueryResult


def test_the_suite_is_60_cases_across_four_tiers() -> None:
    cases = load_cases()
    assert len(cases) == 60
    assert collections.Counter(c["tier"] for c in cases) == {
        "semantic_accuracy": 24,
        "self_correction": 12,
        "refusal": 12,
        "memo_faithfulness": 12,
    }
    assert len({c["id"] for c in cases}) == 60


def test_answer_comparison_ignores_aliasing_and_row_order() -> None:
    gold = QueryResult(columns=["k", "v"], rows=[("a", 1.0), ("b", 2.0)], elapsed_ms=1.0)
    match, _ = answers_match(gold, ["label", "total"], [["b", 2.0], ["a", 1.0]])
    assert match
    miss, reason = answers_match(gold, ["label", "total"], [["a", 1.0], ["b", 9.0]])
    assert not miss and "missing" in reason


def test_refusal_metrics_punish_refusing_everything() -> None:
    """A system that refuses every question must not score well on this tier."""
    scores = [
        CaseScore("r1", "refusal", "q", True, detail={"expected_refusal": True, "refused": True, "category_match": True}),
        CaseScore("r2", "refusal", "q", False, detail={"expected_refusal": False, "refused": True, "category_match": False}),
        CaseScore("r3", "refusal", "q", False, detail={"expected_refusal": False, "refused": True, "category_match": False}),
    ]
    metrics = summarize(scores)["refusal"]
    assert metrics["recall"] == 1.0
    assert metrics["precision"] < 0.5
    assert metrics["false_positive"] == 2


def test_percentiles_use_nearest_rank() -> None:
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([], 50) == 0.0


def test_the_null_agent_fails_every_tier_the_baseline_passes(settings) -> None:
    """The control that makes the rest of the numbers mean something.

    A do-nothing agent emits valid SQL and confident prose. If it can pass a
    tier, that tier is measuring the harness rather than the system.
    """
    cases = [c for c in load_cases() if c["id"] in {"t1-01", "t2-01", "t3-01", "t4-01"}]

    null = BenchRunner(model="null", db_path=settings.db_path).run(cases)
    assert null["summary"]["passed"] == 0
    assert null["controls_injected"] is False

    baseline = BenchRunner(model="mock", db_path=settings.db_path).run(cases)
    assert baseline["summary"]["passed"] == len(cases)
    assert baseline["summary"]["linker_recall"]["mean"] == 1.0
