"""Scoring for vantage-bench.

Answers are compared as row *sets* with floats rounded to two decimals, never as
SQL strings. Two correct queries can differ in aliasing, join order and row
order; only the numbers have to agree. Where ordering is part of the question
("top 5"), the gold query carries the ORDER BY and LIMIT and the comparison is
made on the ordered prefix.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from vantage.executor import QueryResult


@dataclass
class CaseScore:
    """One case's outcome, in enough detail to explain a failure without a rerun."""

    id: str
    tier: str
    question: str
    passed: bool
    reason: str = ""
    status: str = ""
    attempts: int = 0
    self_corrected: bool = False
    latency_ms: float = 0.0
    linker_recall: float | None = None
    faithfulness: float | None = None
    sql: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "tier": self.tier,
            "question": self.question,
            "passed": self.passed,
            "reason": self.reason,
            "status": self.status,
            "attempts": self.attempts,
            "self_corrected": self.self_corrected,
            "latency_ms": round(self.latency_ms, 2),
            "linker_recall": self.linker_recall,
            "faithfulness": self.faithfulness,
            "sql": self.sql,
            "detail": self.detail,
        }


def normalize_rows(columns: list[str], rows: list[list[Any]], precision: int = 2) -> set[tuple]:
    """Order-insensitive, rounding-tolerant row set. Column names are ignored."""
    out = set()
    for row in rows:
        out.add(tuple(_cell(c, precision) for c in row))
    return out


def _cell(cell: Any, precision: int) -> Any:
    if isinstance(cell, bool):
        return cell
    if isinstance(cell, (int, float)):
        return round(float(cell), precision)
    return cell


def ordered_rows(rows: list[list[Any]], precision: int = 2) -> list[tuple]:
    return [tuple(_cell(c, precision) for c in row) for row in rows]


def answers_match(
    gold: QueryResult,
    got_columns: list[str],
    got_rows: list[list[Any]],
    ordered: bool = False,
    precision: int = 2,
) -> tuple[bool, str]:
    """Compare an answer to the gold result. Returns (match, human explanation)."""
    gold_rows = [list(r) for r in gold.rows]
    if len(gold.columns) != len(got_columns):
        return False, f"expected {len(gold.columns)} column(s), got {len(got_columns)}"

    if ordered:
        want, have = ordered_rows(gold_rows, precision), ordered_rows(got_rows, precision)
        if want == have:
            return True, ""
        return False, f"ordered rows differ; first mismatch at index {_first_diff(want, have)}"

    want, have = normalize_rows(gold.columns, gold_rows, precision), normalize_rows(got_columns, got_rows, precision)
    if want == have:
        return True, ""
    missing, extra = want - have, have - want
    parts = []
    if missing:
        parts.append(f"{len(missing)} gold row(s) missing, e.g. {sorted(missing, key=str)[0]}")
    if extra:
        parts.append(f"{len(extra)} unexpected row(s), e.g. {sorted(extra, key=str)[0]}")
    return False, "; ".join(parts)


def _first_diff(want: list, have: list) -> int:
    for i, (a, b) in enumerate(zip(want, have)):
        if a != b:
            return i
    return min(len(want), len(have))


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: rank = ceil(p/100 * N).

    Not interpolated. A 60-case run is small enough that interpolation would
    report a p95 that no case actually took.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(p / 100 * len(ordered))))
    return ordered[rank - 1]


@dataclass
class TierSummary:
    tier: str
    passed: int
    total: int

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def summarize(scores: list[CaseScore]) -> dict[str, Any]:
    """Roll case scores up into the headline metrics."""
    by_tier: dict[str, list[CaseScore]] = {}
    for score in scores:
        by_tier.setdefault(score.tier, []).append(score)

    latencies = [s.latency_ms for s in scores if s.latency_ms > 0]
    recalls = [s.linker_recall for s in scores if s.linker_recall is not None]
    faiths = [s.faithfulness for s in scores if s.faithfulness is not None]

    refusal = _refusal_metrics(by_tier.get("refusal", []))
    correction = _correction_metrics(by_tier.get("self_correction", []))

    return {
        "cases": len(scores),
        "passed": sum(1 for s in scores if s.passed),
        "pass_rate": round(sum(1 for s in scores if s.passed) / len(scores), 4) if scores else 0.0,
        "tiers": {
            tier: {
                "passed": sum(1 for s in group if s.passed),
                "total": len(group),
                "rate": round(sum(1 for s in group if s.passed) / len(group), 4),
            }
            for tier, group in sorted(by_tier.items())
        },
        "semantic_execution_accuracy": _rate(by_tier.get("semantic_accuracy", [])),
        "self_correction": correction,
        "refusal": refusal,
        "memo_faithfulness": {
            "mean": round(statistics.fmean(faiths), 4) if faiths else 1.0,
            "clean_memos": sum(1 for f in faiths if f >= 1.0),
            "checked": len(faiths),
        },
        "linker_recall": {
            "mean": round(statistics.fmean(recalls), 4) if recalls else 1.0,
            "perfect": sum(1 for r in recalls if r >= 1.0),
            "measured": len(recalls),
        },
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
    }


def _rate(group: list[CaseScore]) -> float:
    return round(sum(1 for s in group if s.passed) / len(group), 4) if group else 0.0


def _correction_metrics(group: list[CaseScore]) -> dict[str, Any]:
    """Recovery rate, and the attempt cost of recovering."""
    if not group:
        return {"recovery_rate": 0.0, "recovered": 0, "total": 0, "mean_attempts": 0.0}
    recovered = [s for s in group if s.passed]
    return {
        "recovery_rate": round(len(recovered) / len(group), 4),
        "recovered": len(recovered),
        "total": len(group),
        "mean_attempts": round(statistics.fmean([s.attempts for s in group]), 2),
    }


def _refusal_metrics(group: list[CaseScore]) -> dict[str, Any]:
    """Precision, recall and F1 over "should refuse".

    Refusal rate alone is gameable in the most obvious way possible: a system that
    refuses everything scores 100%. The traps in this tier are the negative class,
    so precision is what stops that.
    """
    tp = sum(1 for s in group if s.detail.get("expected_refusal") and s.detail.get("refused"))
    fp = sum(1 for s in group if not s.detail.get("expected_refusal") and s.detail.get("refused"))
    fn = sum(1 for s in group if s.detail.get("expected_refusal") and not s.detail.get("refused"))
    tn = sum(1 for s in group if not s.detail.get("expected_refusal") and not s.detail.get("refused"))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    category_hits = [s for s in group if s.detail.get("expected_refusal")]
    correct_category = sum(1 for s in category_hits if s.detail.get("category_match"))
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "category_accuracy": round(correct_category / len(category_hits), 4) if category_hits else 0.0,
    }
